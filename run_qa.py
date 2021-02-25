import os
import argparse
import platform
from glob import glob

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from utils.models import MODEL_CLASSES, get_model
from dataset import DATA_NAMES, is_squad_version_2

class QuestionAnswering(pl.LightningModule):
    def __init__(self,
                 data_name,
                 model_type,
                 model_name_or_path,
                 do_lower_case,
                 lang_id,
                 n_best_size,
                 max_answer_length,
                 null_score_diff_threshold,
                 learning_rate: float=5e-5):

        super().__init__()
        self.save_hyperparameters()

        # prepare model
        model = get_model(self.hparams.model_type, self.hparams.model_name_or_path)

        self.model = model

        # for SQuAD, KorQuAD
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
            # models only use two.
            from transformers.data.processors.squad import SquadResult
            if "cls_logits" in list(outputs[0].keys()):
                result = SquadResult(
                    unique_id,
                    start_logits[i],
                    end_logits[i],
                    start_top_index=start_top_index[i],
                    end_top_index=end_top_index[i],
                    cls_logits=cls_logits[i],
                )

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

            from transformers.data.metrics.squad_metrics import compute_predictions_log_probs
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
            from transformers.data.metrics.squad_metrics import compute_predictions_logits
            predictions = compute_predictions_logits(
                examples,
                features,
                all_results,
                self.hparams.n_best_size,
                self.hparams.max_answer_length,
                self.hparams.do_lower_case,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                False, # Not want to do verbose logging
                self.version_2_with_negative,
                self.hparams.null_score_diff_threshold,
                self.trainer.datamodule.tokenizer
            )

        # Compute the F1 and exact scores.
        from transformers.data.metrics.squad_metrics import squad_evaluate
        results = squad_evaluate(examples, predictions)
        return results

    def configure_optimizers(self):
        from transformers import AdamW

        # param_optimizer = list(self.named_parameters())
        # no_decay = ['bias', 'gamma', 'beta']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        #      'weight_decay_rate': 0.01},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        #      'weight_decay_rate': 0.0}
        # ]
        # optimizer = AdamW(
        #     optimizer_grouped_parameters,
        #     lr=self.hparams.learning_rate,
        # )
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=1e-8)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=5e-5)
        return parser


def main():
    pl.seed_everything(42) # set seed

    # Argument Setting -------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    # Other parameters
    parser.add_argument("--data_name", default="squad_v2.0", type=str,
                        help="Data Name selected in the list: " + ", ".join(DATA_NAMES))
    parser.add_argument("--null_score_diff_threshold", type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")

    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")  # In case of Uncased Model, Set this flag!!!

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument("--batch_size", help="batch_size", default=32, type=int)
    parser.add_argument("--gpu_id", help="gpu device id", default="0")

    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

    parser.add_argument("--lang_id", default=0, type=int,
                        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)")

    parser = pl.Trainer.add_argparse_args(parser)
    parser = QuestionAnswering.add_model_specific_args(parser)
    args = parser.parse_args()
    # ------------------------------------------------------------------------------------------------------------------

    # Validation For "doc_stride" Arg ----------------------------------------------------------------------------------
    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        print("WARNING - You've set a doc stride which may be superior to the document length in some "
              "examples. This could result in errors when building features from the examples. Please reduce the doc "
              "stride or increase the maximum length to ensure the features are correctly built.")

    # Dataset ----------------------------------------------------------------------------------------------------------
    from dataset import QuestionAnswering_Data_Module
    args.model_type = args.model_type.lower()
    dm = QuestionAnswering_Data_Module(args.data_name, args.model_type, args.model_name_or_path, args.do_lower_case,
                                       args.max_seq_length, args.doc_stride, args.max_query_length,
                                       args.batch_size)
    dm.prepare_data()
    # ------------------------------------------------------------------------------------------------------------------

    # Model Checkpoint -------------------------------------------------------------------------------------------------
    from pytorch_lightning.callbacks import ModelCheckpoint
    model_folder = './model/{}/{}'.format(args.data_name, args.model_type)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          mode='min', # loss --> minimize ! if you wanna monitor acc, you should change mode is 'max'.
                                          dirpath=model_folder,
                                          filename='{epoch:02d}-{val_loss:.2f}')
    # ------------------------------------------------------------------------------------------------------------------

    # Early Stopping ---------------------------------------------------------------------------------------------------
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True
    )
    # ------------------------------------------------------------------------------------------------------------------

    # Trainer ----------------------------------------------------------------------------------------------------------
    trainer = pl.Trainer(
        gpus=args.gpu_id if platform.system() != 'Windows' else 1,  # <-- for dev. pc
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stop_callback]
    )
    # ------------------------------------------------------------------------------------------------------------------

    # Do train !
    if args.do_train:
        model = QuestionAnswering(args.data_name, args.model_type, args.model_name_or_path, args.do_lower_case, args.lang_id,
                                  args.n_best_size, args.max_answer_length, args.null_score_diff_threshold)
        dm.setup('train')
        trainer.fit(model, dm)

    # Do eval !
    if args.do_eval:
        model_files = glob(os.path.join(model_folder, '*.ckpt'))
        best_fn = model_files[-1]
        print("[Evaluation] Best Model File name is {}".format(best_fn))
        model = QuestionAnswering.load_from_checkpoint(best_fn)
        dm.setup('test')
        trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    main()

