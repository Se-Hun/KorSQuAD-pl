import os
from typing import Optional, Union, Dict
from argparse import Namespace

from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from utils.models import get_tokenizer

DATA_NAMES = ["squad_v1.1", "korquad_v1.0", "squad_v2.0", "korquad_v2.0"]


# Data Utils -----------------------------------------------------------------------------------------------------------
def is_squad_version_2(data_name):
    if data_name in ["squad_v2.0"]:
        return True
    elif data_name in ["squad_v1.1", "korquad_v1.0", "korquad_v2.0"]:
        return False
    else:
        raise KeyError(data_name)


# Data Module ----------------------------------------------------------------------------------------------------------
class QuestionAnsweringDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args: Optional[Union[Dict, Namespace]] = None
    ):
        super().__init__()

        # configurations
        self.data_name = args.data_name
        if self.data_name not in DATA_NAMES:
            raise NotImplementedError(args.data_name)

        self.model_type = args.model_type
        self.model_name_or_path = args.model_name_or_path
        if ("uncased" in self.model_name_or_path) or (self.model_type in ["albert", "electra"]):
            self.do_lower_case = True
        else:
            self.do_lower_case = False

        self.max_seq_length = args.max_seq_length
        self.doc_stride = args.doc_stride
        self.max_query_length = args.max_query_length

        self.version_2_with_negative = is_squad_version_2(self.data_name)

        self.num_threads_for_features = 4
        self.batch_size = args.batch_size

        # for balancing between CPU and GPU
        self.num_workers = 4 * torch.cuda.device_count()

    def prepare_data(self):
        # prepare tokenizer
        self.tokenizer = get_tokenizer(self.model_type, self.model_name_or_path, self.do_lower_case)

        # store data configurations
        self.data_dir = os.path.join("./data", self.data_name)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_dataset = self.load_squad_examples(mode="train")  # Comment out this code for debugging.
            val_dataset = self.load_squad_examples(mode="dev")

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            test_dataset, test_examples, test_features = self.load_squad_examples(mode="test")

            self.test_dataset = test_dataset

            # For Evaluating with Formal SQuAD and KorQuAD Metrics
            self.test_examples = test_examples
            self.test_features = test_features

    def load_squad_examples(self, mode="train"):
        if self.data_dir:
            processor = SquadV2Processor() if self.version_2_with_negative else SquadV1Processor()
            if mode == "train":
                examples = processor.get_train_examples(self.data_dir, filename="train.json")
            elif mode == "dev":
                examples = processor.get_train_examples(self.data_dir, filename="dev.json") # for obtaining start positions and end positions
            elif mode == "test":
                examples = processor.get_dev_examples(self.data_dir, filename="dev.json")
            else:
                raise KeyError(mode)

            # for debugging -- to small set
            # Uncomment out below code for debugging.
            # N = 10
            # examples = examples[:N]
            # --------------------------------------

            is_training = mode != "test" # for obtaining start positions and end positions
            features, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
                doc_stride=self.doc_stride,
                max_query_length=self.max_query_length,
                is_training=is_training,
                return_dataset="pt", # Return DataType is Pytorch Tensor !
                threads=self.num_threads_for_features
            )

        if not is_training:
            return dataset, examples, features

        return dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

