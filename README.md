# SOTA_QA
Question & Answering Task에 대한 SOTA 성능을 가졌던 모델들을 Fine-Tuning해보기 위한 Repository입니다.

구현에 있어서는 다음과 같은 두 가지 특징이 있습니다.
* [Huggingface의 Transformers](https://github.com/huggingface/transformers)를 이용하여 구현
* [Pytorch Lightning](https://www.pytorchlightning.ai/)을 이용하여 구현

## Installations
1. Pytorch 설치 : [Pytorch 홈페이지](https://pytorch.org/get-started/locally/)를 방문하여 자신의 환경에 맞는 Command로 설치
2. Pytorch Lightning 설치 : `conda install pytorch-lightning -c conda-forge`
3. Transformers 설치 : `pip install transformers`
4. Scikit Learn 설치 : `conda install scikit-learn`
5. Numpy 설치 : `conda install numpy`

## Dataset For Question & Answering (MRC)
한국어 MRC 데이터셋은 KorQuAD를 사용하였으며, 영어 MRC 데이터셋은 SQuAD를 사용하였습니다.

* 한국어 데이터 셋 : [KorQuAD 1.0](https://korquad.github.io/category/1.0_KOR.html), [KorQuAD 2.0](https://korquad.github.io/)
* 영어 데이터 셋 : [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/), [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)

### 1. KorQuAD Data Download
KorQuAD 데이터셋은 현재 Script를 지원하지 않습니다.

아래의 Page들에서 직접 다운로드한 후, `./data`의 경로에 직접 저장합니다.

[KorQuAD 1.0](https://korquad.github.io/category/1.0_KOR.html), [KorQuAD 2.0](https://korquad.github.io/)

### 2. SQuAD Data Download
```bash
$ python download_squad.py --download_dir ./data
```

위의 명령어를 실행하면 `./data/squad_v1.1`과 `./data/squad_v2.0`의 경로에 데이터가 다운로드 받아짐을 확인할 수 있습니다.

## Usage
코드의 경우 Huggingface Transformers의 `run_qa.py` 코드의 일부를 수정하여 작성했습니다.

아래의 명령어들은 모두 SQuAD 2.0 데이터셋에 대해 사전학습된 BERT 모델로 Fine-Tuning을 진행하고 Evaluation을 하도록 하는 예시코드입니다.

### 1. Training
```bash
$ python3 run_squad.py --model_type bert \
                       --model_name_or_path bert-base-uncased \
                       --do_lower_case \
                       --data_name squad_v2.0 \
                       --do_train \
                       --gpu_id 0 \
                       --batch_size 12 \
                       --learning_rate 3e-5
```

### 2. Evaluation
```bash
$ python3 run_squad.py --model_type bert \
                       --model_name_or_path bert-base-uncased \
                       --do_lower_case \
                       --data_name squad_v2.0 \
                       --do_eval \
                       --gpu_id 0 \
                       --batch_size 12 \
                       --learning_rate 3e-5
```

### 3. Evaluation about KorQuAD 1.0

KorQuAD 1.0에 대한 공식 Evaluation Script를 사용하려면 다음과 같은 명령어를 사용합니다.

**SQuAD의 Evaluation Script와 KorQuAD의 Evaluation Script가 차이가 있기 때문에 KorQuAD에 대해 정확한 평가를 하고자 한다면 반드시 다음의 명령어를 통해 평가해야합니다.**

```bash
$ python3 evaluate_v1_0.py ./data/korquad_v1.0/dev.json \
                           ./model/korquad_v1.0/{$model_type}/predictions_eval.json
```

## Results

### 1. KorQuAD 1.0
|                                       | Exact Match (%) | F1 Score (%) |
| ------------------------------------- | --------------- | ------------ |
| bert-base-multilingual-cased          | 66.45           | 86.45        |
| distilbert-base-multilingual-cased    | ~~              | ~~           |
| monologg/kobert                       | ~~              | ~~           |
| monologg/koelectra-base-discriminator | ~~              | ~~           |

### 2. KorQuAD 2.0
|                                       | Exact Match (%) | F1 Score (%) |
| ------------------------------------- | --------------- | ------------ |
| bert-base-multilingual-cased          | ~~              | ~~           |
| distilbert-base-multilingual-cased    | ~~              | ~~           |
| monologg/kobert                       | ~~              | ~~           |
| monologg/koelectra-base-discriminator | ~~              | ~~           |

### 3. SQuAD 1.1
|                                         | Exact Match (%) | F1 Score (%) |
| --------------------------------------- | --------------- | ------------ |
| bert-base-uncased                       | 76.20           | 84.79        |
| distilbert-base-uncased                 | 74.13           | 82.71        |
| bert-large-uncased                      | ~~              | ~~           |
| distilbert-base-uncased-distilled-squad | ~~              | ~~           |

### 4. SQuAD 2.0
|                                         | Exact Match (%) | F1 Score (%) |
| --------------------------------------- | --------------- | ------------ |
| bert-base-uncased                       | 71.68           | 68.80        |
| distilbert-base-uncased                 | 66.92           | 64.36        |
| bert-large-uncased                      | ~~              | ~~           |
| distilbert-base-uncased-distilled-squad | ~~              | ~~           |

## TODO list

- [x] SQuAD 데이터셋 다운로드 코드
- [ ] KorQuAD 데이터셋 다운로드 코드
- [x] SQuAD 1.1, 2.0
- [x] KorQuAD 1.0
- [ ] KorQuAD 2.0
- [ ] 모든 실험 결과 종합
- [x] ReadME 작성

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

