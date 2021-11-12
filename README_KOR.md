# KorSQuAD-pl

[English(영어)](./README.md)

**KorSQuAD-pl**은 한국어와 영어 Question Answering 테스크 데이터셋인 [KorQuAD](https://korquad.github.io/category/1.0_KOR.html)와 [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)에 대한 전이학습 실험을 할 수 있게 해주는 코드를 제공합니다.

---

![](figs/korquad_example.PNG)

---

![](figs/SQuADExample.png)

---

**KorSQuAD-pl**은 다음과 같은 특징들을 가집니다.
* [Huggingface Transformers](https://github.com/huggingface/transformers)의 models에 배포된 Pre-trained Language Model들을 사용
* [PyTorch Lightning](https://www.pytorchlightning.ai/)의 코드 스타일을 통한 코드 구현

## Dependencies
* torch>=1.9.0
* pytorch-lihgtning==1.3.8
* transformers>=4.8.0
* kobert-transformers==0.5.1
* sentencepiece>=0.1.96
* scikit-learn
* numpy

## Usage

### 1. Dataset 다운로드

**KorSQuAD-pl**에서 지원하는 데이터셋들은 다음과 같습니다.

| 데이터셋        | 링크                                                 | 
| ------------- | --------------------------------------------------- |
| KorQuAD 1.0   | [LINK](https://korquad.github.io/KorQuad%201.0/)    |
| KorQuAD 2.0 (준비중) |  [LINK](https://korquad.github.io/)                 |
| SQuAD 1.1     | [LINK](https://rajpurkar.github.io/SQuAD-explorer/) |
| SQuAD 2.0     | [LINK](https://rajpurkar.github.io/SQuAD-explorer/) |

* KorQuAD 데이터셋의 경우, 아래와 같은 명령어를 실행하면 데이터셋을 자동으로 다운로드 받아 `./data` 경로에 저장됩니다.
  (**현재 KorQuAD 2.0는 아직 완료하지 못했습니다. 빠른 시일 내에 업데이트하도록 하겠습니다.**)
    ```bash
    python download_korquad.py --download_dir ./data
    ```
* SQuAD의 경우, 아래와 같은 명령어를 실행하면 데이터셋을 자동으로 다운로드 받아 `./data` 경로에 저장됩니다.
    ```bash
    python download_squad.py --download_dir ./data
    ```

### 2. Training and Evaluation
다음과 같은 명령어를 통해 전이학습을 수행하고 이에 대한 평가를 수행합니다.

* `--model_type` : 모델의 유형 ex) `bert`
* `--model_name_or_path` : 모델의 이름 또는 경로 ex) `bert-base-uncased`
* `--data_name` : 전이학습에 사용할 데이터셋 이름 ex) `korquad_v1.0`, `korquad_v2.0`, `squad_v1.1`, `squad_v2.0`
* `--do_train` : 훈련 모드 수행
* `--do_eval` : 평가 모드 수행
* `--gpu_ids` : 전이학습 수행시 사용할 GPU의 ID들 ex) `0` : 0번 GPU 사용, `0,3` : 0번과 3번 GPU 사용
* `--max_seq_length` : 최대 시퀀스의 길이
* `--num_train_epochs` : 훈련시의 에폭 수
* `--batch_size` : 훈련시의 배치 크기
* `--learning_rate` : 학습률
* `--adam_epsilon` : AdamW 옵티마이저의 epsilon 값

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

Distributed Training을 수행하고 싶다면 다음과 같은 명령어를 사용합니다. 

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

Distributed Training을 수행할 때에는 Multi GPU를 이용하여 평가를 하면 충돌이 발생합니다. 따라서, 다음과 같이 Single GPU로 명령어를 변경하여 학습과 별개로 사용해야 합니다.

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

`./model` 폴더에 모든 Checkpoint와 Tensorboard Log 파일들을 저장하도록 해두었습니다.

따라서, 다음과 같이 Tensorboard의 `--logdir`를 지정하여 사용하시면 됩니다.

```bash
tensorboard --logdir ./model/squad_v2.0/bert-base-uncased/
```

## Experiment Settings

실험의 하이퍼파라미터 및 GPU 사용은 다음과 같습니다.

* `small`, `base` 모델의 경우, 단일 GPU를 통해 실험 수행
* `large` 모델의 경우, Distributed Training을 통해 Multi GPU 환경에서 실험 수행(구체적으로는 16GB짜리 GPU 4개를 사용)

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

### 2. KorQuAD 2.0 (준비중)
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

- [ ] KorQuAD 2.0 업데이트

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

추가적으로 문의 사항이 있으시면 해당 repository의 issue를 등록해주시거나 sehunhu5247@gmail.com으로 문의해주시면 감사하겠습니다.