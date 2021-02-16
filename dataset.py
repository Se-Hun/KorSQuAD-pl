import os
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# Dataset Builders -----------------------------------------------------------------------------------------------------
class SquadDataset(Dataset):
    def __init__(self, contexts, questions, question_ids, answers, tokenizer, max_seq_len, is_eval=False):
        self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len
        self.doc_stride = 128

        # self.is_eval = is_eval

        # for debugging -- to small set
        # N = 70
        # contexts = contexts[:N]
        # questions = questions[:N]
        # answers = answers[:N]

        # tokenize about context-question pairs
        # Also, To work with any kind of models,
        # we need to account for the special case where the model expects padding on the left
        # ( in which case we switch the order of the question and the context):
        self.pad_on_right = tokenizer.padding_side == "right"
        if self.pad_on_right:
            tokenized_examples = self.tokenizer(
                questions,
                contexts,
                truncation="only_second",
                max_length=self.max_seq_len,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length"
            )
        else:
            tokenized_examples = self.tokenizer(
                contexts,
                questions,
                truncation="only_first",
                max_length=self.max_seq_len,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        self.features = self.add_answer_position(tokenized_examples, answers, sample_mapping, offset_mapping)

        # if self.is_eval:
        #     # storing example ids
        #     self.features = self.add_example_id(tokenized_examples, question_ids, sample_mapping)
        # else:
        #     # The offset mappings will give us a map from token to character position in the original context. This will
        #     # help us compute the start_positions and end_positions.
        #     offset_mapping = tokenized_examples.pop("offset_mapping")
        #     self.features = self.add_answer_position(tokenized_examples, answers, sample_mapping, offset_mapping)


        # encodings = self.tokenizer(contexts, questions, truncation=True, padding="max_length", max_length=self.max_seq_len)
        # encodings = self.tokenizer(contexts, questions, truncation=True, padding=True)

        # Next, we need to convert our character start/end positions to token start/end positions !
        # self.encodings = self.add_token_positions(encodings, answers)

    def add_answer_position(self, tokenized_examples, answer_examples, sample_mapping, offset_mapping):
        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = answer_examples[sample_index]
            # If no answers are given, set the cls_index as answer.
            if answers["answer_start"] == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"]
                end_char = start_char + len(answers["text"])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def add_example_id(self, tokenized_examples, question_ids, sample_mapping):
        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(question_ids[sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
            # tokenized_examples["offset_mapping"][i] = [
            #     (o if sequence_ids[k] == context_index else (-1, -1))
            #     for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            # ]

        return tokenized_examples

    # def add_token_positions(self, encodings, answers):
    #     start_positions = []
    #     end_positions = []
    #     for i in range(len(answers)):
    #         start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
    #         end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))
    #
    #         # if start position is None, the answer passage has been truncated
    #         if start_positions[-1] is None:
    #             start_positions[-1] = self.max_seq_len
    #
    #         # if end position is None, the 'char_to_token' function points to the space before the correct token - > add + 1
    #         if end_positions[-1] is None:
    #             end_positions[-1] = self.tokenizer.model_max_length # 이거 고쳤는데 맞는지 확인할것!
    #             # end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] + 1) # ??
    #     encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
    #
    #     return encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.features.items()}
        # if self.is_eval:
        #     item = {}
        #     for key, val in self.features.items():
        #         if key != "example_id" and key != "offset_mapping":
        #             item[key] = torch.tensor(val[idx])
        #     example_id = self.features["example_id"][idx]
        #     offset_mapping = self.features["offset_mapping"][idx]
        #     return item
        # else:
        #     return {key: torch.tensor(val[idx]) for key, val in self.features.items()}

        # item = {}
        # for key, val in self.features.items():
        #     if key != "example_id" and key != "offset_mapping":
        #         item[key] = torch.tensor(val[idx])
        #     # if key == "example_id" or key == "offset_mapping":
        #     #     item[key] = val[idx]
        #     # else:
        #     #     item[key] = torch.tensor(val[idx])
        # return item

        # if self.is_eval:
        #     item = {}
        #     for key, val in self.features.items():
        #         if key != "example_id" and key != "offset_mapping":
        #             item[key] = torch.tensor(val[idx])
        #     return item
        # else:
        #     return {key: torch.tensor(val[idx]) for key, val in self.features.items()}
        # return {key: torch.tensor(val[idx]) for key, val in self.features.items()}

        # if self.is_train:
        #     return {key: torch.tensor(val[idx]) for key, val in self.features.items()}
        # else:
        #     item = {}
        #     for key, val in self.features.items():
        #         if key != "example_id" and key != "offset_mapping":
        #             item[key] = torch.tensor(val[idx])
        #     return item

        # return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.features.input_ids)
        # return len(self.encodings.input_ids)

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

        # data collator For data loader
        from transformers import default_data_collator
        self.data_collator = default_data_collator

    def prepare_data(self):
        # read data
        train_contexts, train_questions, train_question_ids, train_answers = self._read_squad(os.path.join(self.data_dir, "train.json"))
        val_contexts, val_questions, val_question_ids, val_answers = self._read_squad(os.path.join(self.data_dir, "dev.json"))
        test_contexts, test_questions, test_question_ids, test_answers = self._read_squad(os.path.join(self.data_dir, "dev.json")) # now, dev and test is same dataset.

        # add end index
        # train_answers, train_contexts = self._add_end_idx(train_answers, train_contexts)
        # val_answers, val_contexts = self._add_end_idx(val_answers, val_contexts)
        # test_answers, test_contexts = self._add_end_idx(test_answers, test_contexts)

        # building dataset
        # dataset = task_to_dataset[self.task] -- Now, fixed at Squad Version 2.0

        self.train_dataset = SquadDataset(train_contexts, train_questions, train_question_ids, train_answers,
                                          self.tokenizer, self.max_seq_length)
        self.val_dataset = SquadDataset(val_contexts, val_questions, val_question_ids, val_answers,
                                        self.tokenizer, self.max_seq_length)
        self.test_dataset = SquadDataset(test_contexts, test_questions, test_question_ids, test_answers,
                                         self.tokenizer, self.max_seq_length, is_eval=True)

    def _read_squad(self, path):
        path = Path(path)
        with open(path, 'rb') as f:
            squad_dict = json.load(f)

        contexts = []
        questions = []
        question_ids = []
        answers = []
        for group in squad_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    question_id = qa["id"]
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        question_ids.append(question_id)
                        answers.append(answer)

        return contexts, questions, question_ids, answers

    # def _add_end_idx(self, answers, contexts):
    #     for answer, context in zip(answers, contexts):
    #         gold_text = answer['text']
    #         start_idx = answer['answer_start']
    #         end_idx = start_idx + len(gold_text)
    #
    #         # sometimes squad answers are off by a character or two – fix this
    #         if context[start_idx:end_idx] == gold_text:
    #             answer['answer_end'] = end_idx
    #         elif context[start_idx - 1:end_idx - 1] == gold_text:
    #             answer['answer_start'] = start_idx - 1
    #             answer['answer_end'] = end_idx - 1  # When the gold label is off by one character
    #         elif context[start_idx - 2:end_idx - 2] == gold_text:
    #             answer['answer_start'] = start_idx - 2
    #             answer['answer_end'] = end_idx - 2  # When the gold label is off by two characters
    #
    #     return answers, contexts

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)
