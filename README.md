# KorSQuAD-pl

[Korean(한국어)](./README_KOR.md)

**KorSQuAD-pl** provides code that enables transfer learning experiments on [KorQuAD](https://korquad.github.io/category/1.0_KOR.html) and [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), which are Korean and English Question Answering task datasets.

---

![](figs/SQuADExample.png)

---

![](figs/korquad_example.PNG)

---

**KorSQuAD-pl** has the following features.
* KorSQuAD-pl use pre-trained language models distributed in [Huggingface Transformers](https://github.com/huggingface/transformers).
* KorSQuAD-pl is implemented through [PyTorch Lightning](https://www.pytorchlightning.ai/)'s code style.

## Dependencies
* torch>=1.9.0
* pytorch-lihgtning==1.3.8
* transformers>=4.8.0
* kobert-transformers==0.5.1
* sentencepiece>=0.1.96
* scikit-learn
* numpy

## Usage

### 1. Download Dataset

The datasets supported by **KorSQuAD-pl** are as follows.

| Dataset       | Link                                                | 
| ------------- | --------------------------------------------------- |
| KorQuAD 1.0   | [LINK](https://korquad.github.io/KorQuad%201.0/)    |
| KorQuAD 2.0 (Preparing) |  [LINK](https://korquad.github.io/)                 |
| SQuAD 1.1     | [LINK](https://rajpurkar.github.io/SQuAD-explorer/) |
| SQuAD 2.0     | [LINK](https://rajpurkar.github.io/SQuAD-explorer/) |

* In the case of KorQuAD dataset, if you run the following command, the dataset is automatically downloaded and saved in the `./data` path.
  (**Currently, code for downloading KorQuAD 2.0 dataset is not yet complete. I will update it as soon as possible.**)
    ```bash
    python download_korquad.py --download_dir ./data
    ```
* In the case of SQuAD dataset, if you run the following command, the dataset is automatically downloaded and saved in the `./data` path.
    ```bash
    python download_squad.py --download_dir ./data
    ```

### 2. Training and Evaluation

Transfer learning is performed and evaluating through the following command.

* `--model_type` : type of model, e.g., `bert`
* `--model_name_or_path` : name or path of the model, e.g., `bert-base-uncased`
* `--data_name` : dataset name to use training and evaluating, e.g., `korquad_v1.0`, `korquad_v2.0`, `squad_v1.1`, `squad_v2.0`
* `--do_train` : training run
* `--do_eval` : evaluating run
* `--gpu_ids` : GPU ids to be used when performing transfer learning, e.g., `0` mean using GPU 0, `0,3` mean using GPU 0 and 3
* `--max_seq_length` : the maximum total input sequence length after WordPiece tokenization
* `--num_train_epochs` : number of epochs in training
* `--batch_size` : batch size at training
* `--learning_rate` : optimizer for learning rate.
* `--adam_epsilon` : huggingface AdamW optimizer's epsilon value

```bash
python3 run_qa.py --model_type bert \
                  --model_name_or_path bert-base-uncased \
                  --data_name squad_v2.0 \
                  --do_train \
                  --do_eval \
                  --gpu_ids 0 \
                  --max_seq_length 384 \
                  --num_train_epochs 2 \
                  --batch_size 16 \
                  --learning_rate 3e-5 \
                  --adam_epsilon 1e-8
```

### 3. Distributed Training and Evaluation

If you want to perform distributed training, use the following command.

```bash
python3 run_qa.py --model_type bert \
                  --model_name_or_path bert-large-uncased-whole-word-masking \
                  --data_name squad_v2.0 \
                  --do_train \
                  --gpu_ids 0,1,2,3 \
                  --max_seq_length 384 \
                  --num_train_epochs 2 \
                  --batch_size 4 \
                  --learning_rate 3e-5 \
                  --adam_epsilon 1e-8
```

When performing distributed training, crash take place when evaluating. So, you need to command to change single GPU as follows. In other words, if you use a multi GPU, training and evaluating cannot be performed at the same time. 

```bash
python3 run_qa.py --model_type bert \
                  --model_name_or_path bert-large-uncased-whole-word-masking \
                  --data_name squad_v2.0 \
                  --do_eval \
                  --gpu_ids 0 \
                  --max_seq_length 384 \
                  --num_train_epochs 2 \
                  --batch_size 4 \
                  --learning_rate 3e-5 \
                  --adam_epsilon 1e-8
```

### 4. Tensorboard with.PyTorch Lightning

All checkpoint and tensorboard log files are stored in the `./model` folder. 

Therefore, you can use tensorboard by specifying `--logdir` as follows.

```bash
tensorboard --logdir ./model/squad_v2.0/bert-base-uncased/
```

## Experiment Settings

Hyper parameters and GPU settings for experiments are as follows:

* For models of `small` and `base` size, experiments are performed using a single GPU.
* For models of `large` size, experiments are performed through distributed training. In other words, this cases are performed at Multi GPU environment.(Specifically, 4 GPUs of 16GB were used.)

### 1. KorQuAD
| Hyper Parameter             | Value                                    | 
| --------------------------- | :--------------------------------------: |
| `null_score_diff_threshold` | 0.0                                      |
| `max_seq_length`            | 512                                      |
| `doc_stride`                | 128                                      |
| `max_query_length`          | 64                                       |
| `n_best_size`               | 20                                       |
| `max_answer_length`         | 30                                       |
| `batch_size`                | 16(small size, base size), 4(large size) |
| `num_train_epochs`          | 3                                        |
| `weight_decay`              | 0.01                                     |
| `adam_epsilon`              | 1e-6(KoELECTRA), 1e-8(others)            |
| `learning_rate`             | 5e-5                                     |

### 2. SQuAD
| Hyper Parameter             | Value                                        | 
| --------------------------- | :------------------------------------------: |
| `null_score_diff_threshold` | 0.0                                          |
| `max_seq_length`            | 384                                          |
| `doc_stride`                | 128                                          |
| `max_query_length`          | 64                                           |
| `n_best_size`               | 20                                           |
| `max_answer_length`         | 30                                           |
| `batch_size`                | 16(small size, base size), 4(large size)     |
| `num_train_epochs`          | 3                                            |
| `weight_decay`              | 0.01                                         |
| `adam_epsilon`              | 1e-6(ALBERT, RoBERTa, ELECTRA), 1e-8(others) |
| `learning_rate`             | 3e-5                                         |


## Result of Experiments

### 1. KorQuAD 1.0
| Model Type   | model_name_or_path                                                                                            | Model Size | Exact Match (%) | F1 Score (%) |
| ------------ | ------------------------------------------------------------------------------------------------------------- | :--------: | :-------------: | :----------: |
| BERT         | [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)                           | Base       | 66.92           | 87.18        |
| KoBERT       | [monologg/kobert](https://huggingface.co/monologg/kobert)                                                     | Base       | 47.73           | 75.12        |
| DistilBERT   | [distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased)               | Small      | 62.91           | 83.28        |
| DistilKoBERT | [monologg/distilkobert](https://huggingface.co/monologg/distilkobert)                                         | Small      | 54.78           | 78.85        |
| KoELECTRA    | [monologg/koelectra-small-v2-discriminator](https://huggingface.co/monologg/koelectra-small-v2-discriminator) | Small      | **81.45**       | 90.09        |
|              | [monologg/koelectra-base-v2-discriminator](https://huggingface.co/monologg/koelectra-base-v2-discriminator)   | Base       | **83.94**       | 92.20        |
|              | [monologg/koelectra-small-v3-discriminator](https://huggingface.co/monologg/koelectra-small-v3-discriminator) | Small      | 81.13           | **90.70**    |
|              | [monologg/koelectra-base-v3-discriminator](https://huggingface.co/monologg/koelectra-base-v3-discriminator)   | Base       | 83.92           | **92.92**    |

### 2. KorQuAD 2.0 (Preparing)
| Model Type   | model_name_or_path                                                                                            | Model Size | Exact Match (%) | F1 Score (%) |
| ------------ | ------------------------------------------------------------------------------------------------------------- | :--------: | :-------------: | :----------: |
| BERT         | [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)                           | Base       |                 |              |
| KoBERT       | [monologg/kobert](https://huggingface.co/monologg/kobert)                                                     | Base       |                 |              |
| DistilBERT   | [distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased)               | Small      |                 |              |
| DistilKoBERT | [monologg/distilkobert](https://huggingface.co/monologg/distilkobert)                                         | Small      |                 |              |
| KoELECTRA    | [monologg/koelectra-small-v2-discriminator](https://huggingface.co/monologg/koelectra-small-v2-discriminator) | Small      |                 |              |
|              | [monologg/koelectra-base-v2-discriminator](https://huggingface.co/monologg/koelectra-base-v2-discriminator)   | Base       |                 |              |
|              | [monologg/koelectra-small-v3-discriminator](https://huggingface.co/monologg/koelectra-small-v3-discriminator) | Small      |                 |              |
|              | [monologg/koelectra-base-v3-discriminator](https://huggingface.co/monologg/koelectra-base-v3-discriminator)   | Base       |                 |              |

### 3. SQuAD 1.1
| Model Type | model_name_or_path                                                                                    | Model Size | Exact Match (%) | F1 Score (%) |
| ---------- | ----------------------------------------------------------------------------------------------------- | :--------: | :-------------: | :----------: |
| BERT       | [bert-base-cased](https://huggingface.co/bert-base-cased)                                             | Base       | 80.38           | 87.99        |
|            | [bert-base-uncased](https://huggingface.co/bert-base-uncased)                                         | Base       | 80.03           | 87.52        |
|            | [bert-large-uncased-whole-word-masking](https://huggingface.co/bert-large-uncased-whole-word-masking) | Large      | 85.51           | 91.88        |
| DistilBERT | [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)                                 | Small      | 75.94           | 84.30        |
|            | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                             | Small      | 76.72           | 84.78        |
| ALBERT     | [albert-base-v1](https://huggingface.co/albert-base-v1)                                               | Base       | 79.46           | 87.70        |
|            | [albert-base-v2](https://huggingface.co/albert-base-v2)                                               | Base       | 79.25           | 87.34        |
| RoBERTa    | [roberta-base](https://huggingface.co/roberta-base)                                                   | Base       | 83.04           | 90.48        |
|            | [roberta-large](https://huggingface.co/roberta-large)                                                 | Large      | 85.18           | 92.25        |
| ELECTRA    | [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)       | Small      | **77.11**       | **85.41**    |
|            | [google/electra-base-discriminator](https://huggingface.co/google/electra-base-discriminator)         | Base       | **84.70**       | **91.30**    |
|            | [google/electra-large-discriminator](https://huggingface.co/google/electra-large-discriminator)       | Large      | **87.14**       | **93.41**    |

### 4. SQuAD 2.0
| Model Type | model_name_or_path                                                                                    | Model Size | Exact Match (%) | F1 Score (%) |
| ---------- | ----------------------------------------------------------------------------------------------------- | :--------: | :-------------: | :----------: |
| BERT       | [bert-base-cased](https://huggingface.co/bert-base-cased)                                             | Base       | 70.52           | 73.79        |
|            | [bert-base-uncased](https://huggingface.co/bert-base-uncased)                                         | Base       | 72.02           | 75.35        |
|            | [bert-large-uncased-whole-word-masking](https://huggingface.co/bert-large-uncased-whole-word-masking) | Large      | 78.97           | 82.14        |
| DistilBERT | [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)                                 | Small      | 63.89           | 66.97        |
|            | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                             | Small      | 65.40           | 68.03        |
| ALBERT     | [albert-base-v1](https://huggingface.co/albert-base-v1)                                               | Base       | 74.75           | 77.77        |
|            | [albert-base-v2](https://huggingface.co/albert-base-v2)                                               | Base       | 76.48           | 79.92        |
| RoBERTa    | [roberta-base](https://huggingface.co/roberta-base)                                                   | Base       | **78.91**       | **82.20**    |
|            | [roberta-large](https://huggingface.co/roberta-large)                                                 | Large      | 80.83           | 84.29        |
| ELECTRA    | [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)       | Small      | **70.55**       | **73.64**    |
|            | [google/electra-base-discriminator](https://huggingface.co/google/electra-base-discriminator)         | Base       | 78.70           | 82.17        |

## TODO list

- [ ] add KorQuAD 2.0 

## References

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [KorQuAD](https://korquad.github.io/category/1.0_KOR.html)
- [KorQuAD by graykode](https://github.com/graykode/KorQuAD-beginner)
- [KorQuAD by lyeoni](https://github.com/lyeoni/KorQuAD)
- [KoBert shows low performance on KorQuad](https://github.com/SKTBrain/KoBERT/issues/1)
- [KoBERT-KorQuAD by monologg](https://github.com/monologg/KoBERT-KorQuAD)
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [KoELECTRA by monologg](https://github.com/monologg/KoELECTRA)

---

If you have any additional questions, please register an issue in this repository or contact sehunhu5247@gmail.com.
