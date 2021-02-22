import os

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor

DATA_NAMES = ["squad_v1", "korquad_v1", "squad_v2", "korquad_v2"]

# Data Utils -----------------------------------------------------------------------------------------------------------
def is_squad_version_2(data_name):
    if data_name in ["squad_v2", "korquad_v2"]:
        return True
    elif data_name in ["squad_v1", "korquad_v1"]:
        return False
    else:
        raise KeyError(data_name)


# Data Module ----------------------------------------------------------------------------------------------------------
class QuestionAnswering_Data_Module(pl.LightningDataModule):
    def __init__(self,
                 data_name,
                 model_type,
                 model_name_or_path,
                 do_lower_case,
                 max_seq_length,
                 doc_stride,
                 max_query_length,
                 batch_size):

        super().__init__()

        self.data_name = data_name
        if self.data_name not in DATA_NAMES:
            raise NotImplementedError(data_name) # validation about dataset name

        # prepare tokenizer
        from utils.models import get_tokenizer
        self.tokenizer = get_tokenizer(model_type, model_name_or_path, do_lower_case)

        # store data configurations
        self.data_dir = os.path.join("./data", self.data_name)

        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length

        self.version_2_with_negative = is_squad_version_2(self.data_name) # for SQuAD, KorQuAD

        # store batch size
        self.batch_size = batch_size

    def prepare_data(self):
        train_dataset = self.load_squad_examples(mode="train") # Comment out this code for debugging.
        val_dataset = self.load_squad_examples(mode="dev")
        test_dataset, test_examples, test_features = self.load_squad_examples(mode="test")

        self.train_dataset = train_dataset # Comment out this code for debugging.
        self.val_dataset = val_dataset
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
                threads=2,
            )

        if not is_training:
            return dataset, examples, features

        return dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)