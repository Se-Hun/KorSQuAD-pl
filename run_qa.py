import os
import json
import argparse
import platform
from glob import glob

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.data.processors.squad import SquadResult
from transformers.data.metrics.squad_metrics import compute_predictions_logits, compute_predictions_log_probs, squad_evaluate

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils.models import MODEL_CLASSES, get_model
from metrics.evaluate_korquad_v1 import evaluate_with_hf_examples as korquad_v1_evaluate
from dataset import QuestionAnsweringDataModule, DATA_NAMES, is_squad_version_2


class QuestionAnswering(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # save hyper parameters
        self.save_hyperparameters("model_type", "model_name_or_path", "data_name", "lang_id",
                                  "n_best_size", "max_answer_length", "null_score_diff_threshold",
                                  "weight_decay", "learning_rate", "adam_epsilon")

        # prepare model
        model = get_model(self.hparams.model_type, self.hparams.model_name_or_path)
        self.model = model
        if ("uncased" in self.hparams.model_name_or_path) or (self.hparams.model_type in ["albert", "electra"]):
            self.do_lower_case = True
        else:
            self.do_lower_case = False

        # for processing Impossible Question
        self.version_2_with_negative = is_squad_version_2(self.hparams.data_name)

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        if self.hparams.model_type in ["xlm", "roberta", "distilbert", "distilkobert"]:
            del inputs["token_type_ids"]

        if self.hparams.model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
            if self.version_2_with_negative:
                inputs.update({"is_impossible": batch[7]})
            if hasattr(self.model, "config") and hasattr(self.model.config, "lang2id"):
                inputs.update(
                    {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * self.hparams.lang_id)}
                )

        outputs = self(inputs)

        loss = outputs[0]
        result = {"loss": loss}
        return result

    def validation_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        if self.hparams.model_type in ["xlm", "roberta", "distilbert", "distilkobert"]:
            del inputs["token_type_ids"]

        if self.hparams.model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
            if self.version_2_with_negative:
                inputs.update({"is_impossible": batch[7]})
            if hasattr(self.model, "config") and hasattr(self.model.config, "lang2id"):
                inputs.update(
                    {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * self.hparams.lang_id)}
                )

        outputs = self(inputs)

        loss = outputs[0]
        result = {"loss": loss}
        return result

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }

        if self.hparams.model_type in ["xlm", "roberta", "distilbert", "distilkobert"]:
            del inputs["token_type_ids"]

        example_indices = batch[3]

        # XLNet and XLM use more arguments for their predictions
        if self.hparams.model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
            # for lang_id-sensitive xlm models
            if hasattr(self.model, "config") and hasattr(self.model.config, "lang2id"):
                inputs.update(
                    {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * self.hparams.lang_id)}
                )

        outputs = self(inputs)

        if len(list(outputs.keys())) >= 5:
            start_logits = outputs[0]
            start_top_index = outputs[1]
            end_logits = outputs[2]
            end_top_index = outputs[3]
            cls_logits = outputs[4]
            result = {"example_indices": example_indices, "start_logits": start_logits, "start_top_index": start_top_index,
                      "end_logits": end_logits, "end_top_index": end_top_index, "cls_logits": cls_logits}
        else:
            start_logits = outputs[0]
            end_logits = outputs[1]
            result = {"example_indices": example_indices, "start_logits": start_logits, "end_logits": end_logits}

        return result

    def test_epoch_end(self, outputs):
        example_indices = torch.cat([x["example_indices"] for x in outputs]).detach().cpu().tolist()
        start_logits = torch.cat([x["start_logits"] for x in outputs]).detach().cpu().tolist()
        end_logits = torch.cat([x["end_logits"] for x in outputs]).detach().cpu().tolist()

        if "cls_logits" in list(outputs[0].keys()):
            start_top_index = torch.cat([x["start_top_index"] for x in outputs]).detach().cpu().tolist()
            end_top_index = torch.cat([x["end_top_index"] for x in outputs]).detach().cpu().tolist()
            cls_logits = torch.cat([x["cls_logits"] for x in outputs]).detach().cpu().tolist()

        examples = self.trainer.datamodule.test_examples
        features = self.trainer.datamodule.test_features

        all_results = []
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index]
            unique_id = int(eval_feature.unique_id)

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            if "cls_logits" in list(outputs[0].keys()):
                result = SquadResult(
                    unique_id,
                    start_logits[i],
                    end_logits[i],
                    start_top_index=start_top_index[i],
                    end_top_index=end_top_index[i],
                    cls_logits=cls_logits[i],
                )
            # Other models only use 2 arguments.
            else:
                result = SquadResult(unique_id, start_logits[i], end_logits[i])

            all_results.append(result)

        # Compute predictions
        output_prediction_file = os.path.join(self.trainer.checkpoint_callback.dirpath, "predictions_eval.json")
        output_nbest_file = os.path.join(self.trainer.checkpoint_callback.dirpath, "nbest_predictions_eval.json")

        if self.version_2_with_negative:
            output_null_log_odds_file = os.path.join(self.trainer.checkpoint_callback.dirpath, "null_odds_eval.json")
        else:
            output_null_log_odds_file = None

        # XLNet and XLM use a more complex post-processing procedure
        if self.hparams.model_type in ["xlnet", "xlm"]:
            start_n_top = self.model.config.start_n_top if hasattr(self.model, "config") else self.model.module.config.start_n_top
            end_n_top = self.model.config.end_n_top if hasattr(self.model, "config") else self.model.module.config.end_n_top

            predictions = compute_predictions_log_probs(
                examples,
                features,
                all_results,
                self.hparams.n_best_size,
                self.hparams.max_answer_length,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                start_n_top,
                end_n_top,
                self.version_2_with_negative,
                self.trainer.datamodule.tokenizer,
                False # Not want to do verbose logging
            )
        else:
            predictions = compute_predictions_logits(
                examples,
                features,
                all_results,
                self.hparams.n_best_size,
                self.hparams.max_answer_length,
                self.do_lower_case,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                False, # Not want to do verbose logging
                self.version_2_with_negative,
                self.hparams.null_score_diff_threshold,
                self.trainer.datamodule.tokenizer
            )

        # Perform evaluation about KorQuAD
        if self.hparams.data_name == "korquad_v1.0":
            results = korquad_v1_evaluate(examples, predictions)
        # Perform evaluation about SQuAD
        else:
            results = squad_evaluate(examples, predictions)

        # Dump evaluation result file
        result_file = os.path.join(self.trainer.checkpoint_callback.dirpath, "result.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            print("Result file is dumped at ", result_file)

        print(json.dumps(results, indent=4))
        return

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        t_total = len(self.train_dataloader()) * self.trainer.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Question Answering")

        parser.add_argument("--null_score_diff_threshold", default=0.0, type=float,
                            help="If null_score - best_non_null is greater than the threshold predict null.")
        parser.add_argument("--max_seq_length", default=384, type=int,
                            help="The maximum total input sequence length after WordPiece tokenization."
                                 "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
        parser.add_argument("--doc_stride", default=128, type=int,
                            help="When splitting up a long document into chunks, how much stride to take between chunks.")
        parser.add_argument("--max_query_length", default=64, type=int,
                            help="The maximum number of tokens for the question."
                                 "Questions longer than this will be truncated to this length.")
        parser.add_argument("--weight_decay", default=0.01, type=float,
                            help="The weight decay to apply (if not zero) to all layers except all bias and "
                                 "LayerNorm weights in AdamW optimizer of huggingface transformers.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="The epsilon hyperparameter for AdamW optimizer of huggingface transformers.")
        parser.add_argument('--learning_rate', default=3e-5, type=float,
                            help="Optimizer for learning rate.")
        parser.add_argument("--n_best_size", default=20, type=int,
                            help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
        parser.add_argument("--max_answer_length", default=30, type=int,
                            help="The maximum length of an answer that can be generated."
                                 "This is needed because the start and end predictions are not conditioned on one another.")
        parser.add_argument("--lang_id", default=0, type=int,
                            help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)")

        return parent_parser


def main():
    # Argument Setting -------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list")

    # Other parameters
    parser.add_argument("--data_name", default="squad_v2.0", type=str,
                        help="Data Name selected in the list: " + ", ".join(DATA_NAMES))

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument("--num_train_epochs", default=3, type=int, help="Epochs at train time.")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--gpu_ids", default="0", type=str,
                        help="gpu device ids. e.g.) `0` : GPU 0, `0,3` : GPU 0 and 3")

    parser.add_argument("--seed", default=42, type=int, help="Seed Number")

    parser = pl.Trainer.add_argparse_args(parser)
    parser = QuestionAnswering.add_model_specific_args(parser)
    args = parser.parse_args()
    # ------------------------------------------------------------------------------------------------------------------

    # Set seed
    pl.seed_everything(args.seed)

    # Validation for "doc_stride" Arg ----------------------------------------------------------------------------------
    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        print("WARNING - You've set a doc stride which may be superior to the document length in some "
              "examples. This could result in errors when building features from the examples. Please reduce the doc "
              "stride or increase the maximum length to ensure the features are correctly built.")

    # Dataset ----------------------------------------------------------------------------------------------------------
    args.model_type = args.model_type.lower()
    args.model_name_or_path = args.model_name_or_path.lower()
    args.data_name = args.data_name.lower()

    dm = QuestionAnsweringDataModule(args)
    dm.prepare_data()
    # ------------------------------------------------------------------------------------------------------------------

    # Callbacks and Loggers --------------------------------------------------------------------------------------------
    model_folder = './model/{}/{}'.format(args.data_name, args.model_name_or_path.replace("/", "-"))
    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=model_folder,
        filename='{epoch:02d}-{val_loss:.3f}'
    )

    tensorboard_logger = TensorBoardLogger(
        save_dir=model_folder, name=''  # <-- if experiment name(=name) is empty, subdirectory is not made.
    )
    # ------------------------------------------------------------------------------------------------------------------

    # Trainer ----------------------------------------------------------------------------------------------------------
    trainer = pl.Trainer(
        gpus=args.gpu_ids if platform.system() != 'Windows' else 1,  # <-- for dev. pc
        accelerator="ddp" if "," in args.gpu_ids else None,
        logger=tensorboard_logger,
        callbacks=[model_checkpoint_callback],
        max_epochs=args.num_train_epochs
    )
    # ------------------------------------------------------------------------------------------------------------------

    # Do train !
    if args.do_train:
        model = QuestionAnswering(**vars(args))
        dm.setup('fit')
        trainer.fit(model, dm)

    # Do eval !
    if args.do_eval:
        assert (trainer.num_gpus < 2), "At test mode, Use single gpu for preventing collision !"

        model_files = glob(os.path.join(trainer.checkpoint_callback.dirpath, "*.ckpt"))
        best_fn = sorted(model_files, key=lambda fn: fn.split("=")[-1])[0]
        print("[Evaluation] Best Model File name is {}".format(best_fn))

        model = QuestionAnswering.load_from_checkpoint(best_fn, **vars(args))
        dm.setup('test')
        trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    main()

