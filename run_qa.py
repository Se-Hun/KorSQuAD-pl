import os
import argparse
import platform
from glob import glob

import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score
from datasets import load_metric
# from transformers.data.metrics import squad_metrics as metric

class QuestionAnswering(pl.LightningModule):
    def __init__(self,
                 data_name,
                 text_reader,
                 learning_rate: float=5e-5):
        super().__init__()
        self.save_hyperparameters()

        # prepare text reader
        from utils.readers import get_text_reader
        text_reader = get_text_reader(self.hparams.text_reader, self.hparams.data_name)
        self.text_reader = text_reader

        # prepare metric
        self.metric = load_metric(data_name)

    # def forward(self, input_ids, token_type_ids, attention_mask, start_positions, end_positions):
    #     outputs = self.text_reader(input_ids=input_ids,
    #                                token_type_ids=token_type_ids,
    #                                attention_mask=attention_mask,
    #                                start_positions=start_positions,
    #                                end_positions=end_positions)
    #
    #     return outputs # (loss, logits) --> logits : [batch_size, num_labels]

    def forward(self, x):
        # if text_reader model is BERT, x values are consist of
        # input_ids, token_type_ids, attention_mask, start_positions, end_positions !

        return self.text_reader(**x)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)

        loss = outputs[0]

        result = {"loss": loss}
        return result

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)

        loss = outputs[0]
        start_logits = outputs[1]
        end_logits = outputs[2]

        start_preds = start_logits.argmax(dim=1)
        end_preds = end_logits.argmax(dim=1)

        start_labels = batch["start_positions"]
        end_labels = batch["end_positions"]

        result = {"loss": loss, "start_preds": start_preds, "end_preds": end_preds,
                  "start_labels": start_labels, "end_labels": end_labels}
        return result

    def validation_epoch_end(self, outputs):
        start_preds = torch.cat([x["start_preds"] for x in outputs]).cpu().numpy()
        end_preds = torch.cat([x["end_preds"] for x in outputs]).cpu().numpy()
        start_labels = torch.cat([x["start_labels"] for x in outputs]).cpu().numpy()
        end_labels = torch.cat([x["end_labels"] for x in outputs]).cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        # start_labels, end_labels = np.array(start_labels), np.array(end_labels)
        val_acc_start = accuracy_score(start_labels, start_preds)
        val_acc_end = accuracy_score(end_labels, end_preds)
        val_acc = (val_acc_start + val_acc_end) / 2

        # self.metric.compute(predictions=predictions, references=label_ids)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch)

        start_logits = outputs[1]
        end_logits = outputs[2]

        start_preds = start_logits.argmax(dim=1)
        end_preds = end_logits.argmax(dim=1)

        start_labels = batch["start_positions"]
        end_labels = batch["end_positions"]

        result = {"start_preds": start_preds, "end_preds": end_preds,
                  "start_labels": start_labels, "end_labels": end_labels}
        return result

    def test_epoch_end(self, outputs):
        start_preds = torch.cat([x["start_preds"] for x in outputs]).cpu().numpy()
        end_preds = torch.cat([x["end_preds"] for x in outputs]).cpu().numpy()
        start_labels = torch.cat([x["start_labels"] for x in outputs]).cpu().numpy()
        end_labels = torch.cat([x["end_labels"] for x in outputs]).cpu().numpy()

        correct_count = torch.sum(labels == preds)
        test_acc = correct_count.float() / float(len(labels))

        # scores per class
        class_scores = classification_report(labels.cpu().numpy(), preds.cpu().numpy(), digits=4)
        print(class_scores)
        matrix = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
        print(matrix)
        class_accuracy = matrix.diagonal() / matrix.sum(axis=1)
        print(class_accuracy)

        # dump predicted outputs
        predicted_outputs_fn = os.path.join(self.trainer.callbacks[1].dirpath, 'predicted_outputs.txt')
        predicted_outputs = labels.cpu().tolist()
        with open(predicted_outputs_fn, "w", encoding='utf-8') as f:
            for output in predicted_outputs:
                print(output, file=f)
            print("Predicted Outputs are dumped at {}".format(predicted_outputs_fn))

        self.log("test_acc", test_acc, prog_bar=True)
        return test_acc

    def configure_optimizers(self):
        from transformers import AdamW

        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
        )
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

    # mode specific --------------------------------------------------------------------------------
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to train QA model.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to predict on dataset.")

    # model specific -------------------------------------------------------------------------------
    parser.add_argument("--text_reader", help="bert, kobert, koelectra, others, ...", default="bert")

    # data name ------------------------------------------------------------------------------------
    parser.add_argument("--data_name", help="squad_v2, korquad_v2", default="squad_v2")

    # experiment settings --------------------------------------------------------------------------
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")  # bert has 512 tokens.
    parser.add_argument("--batch_size", help="batch_size", default=32, type=int)
    parser.add_argument("--gpu_id", help="gpu device id", default="0")

    parser = pl.Trainer.add_argparse_args(parser)
    parser = QuestionAnswering.add_model_specific_args(parser)
    args = parser.parse_args()
    # ------------------------------------------------------------------------------------------------------------------

    # Dataset ----------------------------------------------------------------------------------------------------------
    from dataset import QuestionAnswering_Data_Module
    dm = QuestionAnswering_Data_Module(args.data_name, args.text_reader, args.max_seq_length, args.batch_size)
    dm.prepare_data()
    # ------------------------------------------------------------------------------------------------------------------

    # Model Checkpoint -------------------------------------------------------------------------------------------------
    from pytorch_lightning.callbacks import ModelCheckpoint
    model_name = '{}'.format(args.text_reader)
    model_folder = './model/{}/{}'.format(args.data_name, model_name)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
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
        model = QuestionAnswering(args.data_name, args.text_reader)
        trainer.fit(model, dm)

    # Do eval !
    if args.do_eval:
        model_files = glob(os.path.join(model_folder, '*.ckpt'))
        best_fn = model_files[-1]
        model = QuestionAnswering.load_from_checkpoint(best_fn)
        trainer.test(model, test_dataloaders=[dm.test_dataloader()])

if __name__ == '__main__':
    main()