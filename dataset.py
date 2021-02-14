import os
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# Dataset Builders -----------------------------------------------------------------------------------------------------
class SquadDataset(Dataset):
    def __init__(self, contexts, questions, answers, tokenizer, max_seq_len):
        self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len

        # for debugging -- to small set
        # N = 70
        # contexts = contexts[:N]
        # questions = questions[:N]
        # answers = answers[:N]

        # tokenize about context-question pairs
        encodings = self.tokenizer(contexts, questions, truncation=True, padding="max_length", max_length=self.max_seq_len)
        # encodings = self.tokenizer(contexts, questions, truncation=True, padding=True)

        # Next, we need to convert our character start/end positions to token start/end positions !
        self.encodings = self.add_token_positions(encodings, answers)

    def add_token_positions(self, encodings, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.max_seq_len

            # if end position is None, the 'char_to_token' function points to the space before the correct token - > add + 1
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length # 이거 고쳤는데 맞는지 확인할것!
                # end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] + 1) # ??
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

        return encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# Data Modules ---------------------------------------------------------------------------------------------------------
class QuestionAnswering_Data_Module(pl.LightningDataModule):
    def __init__(self, data_name, text_reader, max_seq_length, batch_size):
        super().__init__()

        self.data_name = data_name

        # prepare tokenizer
        from utils.readers import get_tokenizer
        self.tokenizer = get_tokenizer(text_reader, self.data_name)

        # data configs
        self.data_dir = os.path.join("./data", self.data_name)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def prepare_data(self):
        # read data
        train_contexts, train_questions, train_answers = self._read_squad(os.path.join(self.data_dir, "train.json"))
        val_contexts, val_questions, val_answers = self._read_squad(os.path.join(self.data_dir, "dev.json"))
        test_contexts, test_questions, test_answers = self._read_squad(os.path.join(self.data_dir, "dev.json")) # now, dev and test is same dataset.

        # add end index
        train_answers, train_contexts = self._add_end_idx(train_answers, train_contexts)
        val_answers, val_contexts = self._add_end_idx(val_answers, val_contexts)
        test_answers, test_contexts = self._add_end_idx(test_answers, test_contexts)

        # building dataset
        # dataset = task_to_dataset[self.task] -- Now, fixed at Squad Version 2.0

        self.train_dataset = SquadDataset(train_contexts, train_questions, train_answers, self.tokenizer, self.max_seq_length)
        self.val_dataset = SquadDataset(val_contexts, val_questions, val_answers, self.tokenizer, self.max_seq_length)
        self.test_dataset = SquadDataset(test_contexts, test_questions, test_answers, self.tokenizer, self.max_seq_length)

    def _read_squad(self, path):
        path = Path(path)
        with open(path, 'rb') as f:
            squad_dict = json.load(f)

        contexts = []
        questions = []
        answers = []
        for group in squad_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)

        return contexts, questions, answers

    def _add_end_idx(self, answers, contexts):
        for answer, context in zip(answers, contexts):
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)

            # sometimes squad answers are off by a character or two – fix this
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx - 1:end_idx - 1] == gold_text:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1  # When the gold label is off by one character
            elif context[start_idx - 2:end_idx - 2] == gold_text:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2  # When the gold label is off by two characters

        return answers, contexts

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
