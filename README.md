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
```bash
~~~
```

### 2. SQuAD Data Download
```bash
$ python download_squad.py --download_dir ./data
```

## Usage
코드의 경우 Huggingface Transformers의 example 코드의 일부를 수정하여 작성했습니다.

아래의 명령어들은 모두 SQuAD 2.0 데이터셋에 대해 사전학습된 BERT 모델로 Fine-Tuning을 진행하고 Evaluation을 하도록 하는 예시코드입니다.

### 1. Training
```bash
$ python3 run_squad.py --model_type bert \
                       --model_name_or_path bert-base-uncased \
                       --do_lower_case \
                       --data_dir data \
                       --train_file KorQuAD_v1.0_train.json \
                       --predict_file KorQuAD_v1.0_dev.json \
                       --evaluate_during_training \
                       --per_gpu_train_batch_size 8 \
                       --per_gpu_eval_batch_size 8 \
                       --max_seq_length 512 \
                       --logging_steps 4000 \
                       --save_steps 4000 \
                       --do_train
```

### 2. Evaluation


## Results

## TODO list
- [x] SQuAD 데이터셋 다운로드 코드
- [ ] KorQuAD 데이터셋 다운로드 코드
- [x] SQuAD 1.1, 2.0
- [x] KorQuAD 1.0
- [ ] KorQuAD 2.0
- [ ] 모든 실험 결과 종합
- [ ] ReadME 작성

## References
