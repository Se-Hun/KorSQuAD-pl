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
6. datasets 라이브러리 설치 : `pip install datasets`

## Dataset For Question & Answering (MRC)
해당 프로젝트에서는 다음과 같은 한국어 QA Dataset과 영어 Dataset 각각 한 개씩을 사용합니다.

* 한국어 데이터 셋 : [KorQuAD 2.0](https://korquad.github.io/)
* 영어 데이터 셋 : [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)

