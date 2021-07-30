[한국어](./README.md) | [English](./README_EN.md)

# KorSQuAD-pl

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

**KorSQuAD-pl**에서 제공하는 데이터셋들은 다음과 같습니다.

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

### 2. Training
다음과 같은 명령어를 통해 전이학습을 수행합니다.

* `--model_type` : 모델의 유형 ex) `bert`
* `--model_name_or_path` : 모델의 이름 또는 경로 ex) `bert-base-uncased`
* `--do_lower_case` : 대문자를 모두 소문자로 바꿀지(uncased model)
* `--data_name` : 전이학습에 사용할 데이터셋 이름 ex) `squad_v1.1`, `korquad_v1.0`, `squad_v2.0`, `korquad_v2.0`
* `--do_train` : 훈련 모드 수행
* `--gpu_ids` : 전이학습 수행시 사용할 GPU의 ID들 ex) `0` : 0번 GPU 사용, `0,3` : 0번과 3번 GPU 사용
* `--batch_size` : 훈련시의 배치 크기
* `--learning_rate` : 학습률

```bash
python3 run_qa.py --model_type bert \
                  --model_name_or_path bert-base-uncased \
                  --do_lower_case \
                  --data_name squad_v2.0 \
                  --do_train \
                  --gpu_ids 0 \
                  --batch_size 12 \
                  --learning_rate 3e-5
```

### 3. Evaluation
다음과 같은 명령어를 통해 전이학습에 대한 평가를 수행합니다.

* `--model_type` : 모델의 유형 ex) `bert`
* `--model_name_or_path` : 모델의 이름 또는 경로 ex) `bert-base-uncased`
* `--do_lower_case` : 대문자를 모두 소문자로 바꿀지(uncased model)
* `--data_name` : 전이학습에 사용할 데이터셋 이름 ex) `squad_v1.1`, `korquad_v1.0`, `squad_v2.0`, `korquad_v2.0`
* `--do_eval` : 평가 모드 수행
* `--gpu_ids` : 전이학습 수행시 사용할 GPU의 ID들 ex) `0` : 0번 GPU 사용, `0,3` : 0번과 3번 GPU 사용
* `--batch_size` : 훈련시의 배치 크기
* `--learning_rate` : 학습률

```bash
python3 run_qa.py --model_type bert \
                  --model_name_or_path bert-base-uncased \
                  --do_lower_case \
                  --data_name squad_v2.0 \
                  --do_eval \
                  --gpu_ids 0 \
                  --batch_size 12 \
                  --learning_rate 3e-5
```

### 4. Training and Evaluation
전이학습을 수행한 후에 곧바로 평가를 수행하고자 하면 다음과 같이 `--do_train` 옵션과 `--do_eval` 옵션을 함께 키면 됩니다.

```bash
python3 run_qa.py --model_type bert \
                  --model_name_or_path bert-base-uncased \
                  --do_lower_case \
                  --data_name squad_v2.0 \
                  --do_train \
                  --do_eval \
                  --gpu_ids 0 \
                  --batch_size 12 \
                  --learning_rate 3e-5
```

### 5. Distributed Training and Evaluation

Distributed Training을 수행하고 싶다면 다음과 같은 명령어를 사용합니다. 

```bash
python3 run_qa.py --model_type bert \
                  --model_name_or_path bert-large-uncased-whole-word-masking \
                  --do_lower_case \
                  --data_name squad_v2.0 \
                  --do_train \
                  --gpu_ids 0,1,2,3 \
                  --batch_size 4 \
                  --learning_rate 3e-5
```

Distributed Training을 수행할 때에는 Multi GPU를 이용하여 평가를 하면 충돌이 발생합니다. 따라서, 다음과 같이 Single GPU로 명령어를 변경하여 학습과 별개로 사용해야 합니다.

```bash
python3 run_qa.py --model_type bert \
                  --model_name_or_path bert-large-uncased-whole-word-masking \
                  --do_lower_case \
                  --data_name squad_v2.0 \
                  --do_eval \
                  --gpu_ids 0 \
                  --batch_size 4 \
                  --learning_rate 3e-5
```


### 6. Formal Evaluation for KorQuAD 1.0

KorQuAD 1.0에 대한 공식 Evaluation Script를 사용하려면 다음과 같은 명령어를 사용합니다.

(SQuAD의 Evaluation Script와 KorQuAD의 Evaluation Script가 차이가 있기 때문에 KorQuAD에 대해 정확한 평가를 하고자 한다면 반드시 다음의 명령어를 통해 평가해야합니다.)

```bash
python3 evaluate_korquad_v1.py --dataset_file ./data/korquad_v1.0/dev.json \
                               --prediction_file ./model/korquad_v1.0/{$model_type}/predictions_eval.json
```

### 7. Tensorboard with.PyTorch Lightning

`./model` 폴더에 모든 Checkpoint와 Tensorboard Log 파일들을 저장하도록 해두었습니다.

따라서, 다음과 같이 Tensorboard의 `--logdir`를 지정하여 사용하시면 됩니다.

```bash
tensorboard --logdir ./model/squad_v2.0/bert-base-uncased/
```

## Result of Experiments

실험의 세부사항은 다음과 같습니다.

* `small`, `base` 모델의 경우, 단일 GPU를 통해 실험 수행
* `large` 모델의 경우, Distributed Training을 통해 실험 수행

### 1. KorQuAD 1.0
| Model Type   | model_name_or_path                                                                                            | Exact Match (%) | F1 Score (%) |
| ------------ | ------------------------------------------------------------------------------------------------------------- | --------------- | ------------ |
| BERT         | [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)                           |  65.62          |  73.36       |
| KoBERT       | [monologg/kobert](https://huggingface.co/monologg/kobert)                                                     |  50.91          |  60.99       |
| DistilBERT   | [distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased)               |  62.20          |  70.90       |
| DistilKoBERT | [monologg/distilkobert](https://huggingface.co/monologg/distilkobert)                                         |  55.88          |  64.06       |
| KoELECTRA    | [monologg/koelectra-small-v2-discriminator](https://huggingface.co/monologg/koelectra-small-v2-discriminator) |  80.51          |  86.25       |
|              | [monologg/koelectra-base-v2-discriminator](https://huggingface.co/monologg/koelectra-base-v2-discriminator)   |  83.44          |  88.67       |
|              | [monologg/koelectra-small-v3-discriminator](https://huggingface.co/monologg/koelectra-small-v3-discriminator) |  81.39          |  87.05       |
|              | [monologg/koelectra-base-v3-discriminator](https://huggingface.co/monologg/koelectra-base-v3-discriminator)   |  **84.48**      |  **89.64**   |

### 2. KorQuAD 2.0 (준비중)
| Model Type   | model_name_or_path                                                                                            | Exact Match (%) | F1 Score (%) |
| ------------ | ------------------------------------------------------------------------------------------------------------- | --------------- | ------------ |
| BERT         | [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)                           |                 |              |
| KoBERT       | [monologg/kobert](https://huggingface.co/monologg/kobert)                                                     |                 |              |
| DistilBERT   | [distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased)               |                 |              |
| DistilKoBERT | [monologg/distilkobert](https://huggingface.co/monologg/distilkobert)                                         |                 |              |
| KoELECTRA    | [monologg/koelectra-small-v2-discriminator](https://huggingface.co/monologg/koelectra-small-v2-discriminator) |                 |              |
|              | [monologg/koelectra-base-v2-discriminator](https://huggingface.co/monologg/koelectra-base-v2-discriminator)   |                 |              |
|              | [monologg/koelectra-small-v3-discriminator](https://huggingface.co/monologg/koelectra-small-v3-discriminator) |                 |              |
|              | [monologg/koelectra-base-v3-discriminator](https://huggingface.co/monologg/koelectra-base-v3-discriminator)   |                 |              |

### 3. SQuAD 1.1
| Model Type | model_name_or_path                                                                                    | Exact Match (%) | F1 Score (%) |
| ---------- | ----------------------------------------------------------------------------------------------------- | --------------- | ------------ |
| BERT       | [bert-base-cased](https://huggingface.co/bert-base-cased)                                             |  77.94          |  85.77       |
|            | [bert-base-uncased](https://huggingface.co/bert-base-uncased)                                         |  78.13          |  86.00       |
|            | [bert-large-uncased-whole-word-masking](https://huggingface.co/bert-large-uncased-whole-word-masking) |  **83.10**      |  **90.00**   |
| DistilBERT | [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)                                 |  74.23          |  82.49       |
|            | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                             |  74.67          |  82.95       |
| ALBERT     | [albert-base-v1](https://huggingface.co/albert-base-v1)                                               |  75.58          |  84.47       |
|            | [albert-base-v2](https://huggingface.co/albert-base-v2)                                               |  74.46          |  83.60       |
| XLNet      | [xlnet-base-cased](https://huggingface.co/xlnet-base-cased)                                           |                 |              |
|            | [xlnet-large-cased](https://huggingface.co/xlnet-large-cased)                                         |                 |              |
| ELECTRA    | [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)       |  68.52          |  77.29       |
|            | [google/electra-base-discriminator](https://huggingface.co/google/electra-base-discriminator)         |  74.82          |  82.76       |
|            | [google/electra-large-discriminator](https://huggingface.co/google/electra-large-discriminator)       |                 |              |

### 4. SQuAD 2.0
| Model Type | model_name_or_path                                                                                    | Exact Match (%) | F1 Score (%) |
| ---------- | ----------------------------------------------------------------------------------------------------- | --------------- | ------------ |
| BERT       | [bert-base-cased](https://huggingface.co/bert-base-cased)                                             |  68.03          |  71.23       |
|            | [bert-base-uncased](https://huggingface.co/bert-base-uncased)                                         |  69.68          |  72.89       |
|            | [bert-large-uncased-whole-word-masking](https://huggingface.co/bert-large-uncased-whole-word-masking) |  71.95          |  75.72       |
| DistilBERT | [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)                                 |  63.89          |  66.97       |
|            | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                             |  65.40          |  68.03       |
| ALBERT     | [albert-base-v1](https://huggingface.co/albert-base-v1)                                               |  72.12          |  75.54       |
|            | [albert-base-v2](https://huggingface.co/albert-base-v2)                                               |  68.75          |  72.37       |
| XLNet      | [xlnet-base-cased](https://huggingface.co/xlnet-base-cased)                                           |                 |              |
|            | [xlnet-large-cased](https://huggingface.co/xlnet-large-cased)                                         |                 |              |
| ELECTRA    | [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)       |  62.14          |  64.57       |
|            | [google/electra-base-discriminator](https://huggingface.co/google/electra-base-discriminator)         |  63.68          |  67.30       |
|            | [google/electra-large-discriminator](https://huggingface.co/google/electra-large-discriminator)       |                 |              |

## TODO list

- [ ] KorQuAD 2.0 업데이트
- [x] 모델 추가
- [ ] 모든 실험 결과 종합
- [x] ReadME 작성
- [ ] ReadME EN 작성

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