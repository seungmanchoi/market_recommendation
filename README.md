# Wholesale Recommendation System

## Overview

이 프로젝트는 소매업체에 대해 도매업체를 추천하는 시스템을 개발하는 것을 목표로 합니다. 데이터를 기반으로, 소매업체가 도매업체로부터 구매할 가능성을 예측하고, 이 예측에 기반하여 도매업체를 추천합니다.

사용되는 주요 알고리즘은 `SGDClassifier`이며, 이 모델은 대규모 데이터셋을 효율적으로 처리하고 빠르게 학습할 수 있는 특성을 가지고 있습니다. 데이터는 CSV 파일로부터 읽어들이며, 배치 단위로 처리하여 모델을 학습시킵니다.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Details](#project-details)
    - [Model Training](#model-training)
    - [Recommendation](#recommendation)
- [Evaluation](#evaluation)
- [Files](#files)
- [Notes](#notes)

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib

## Installation

필수 패키지를 설치하려면, 다음 명령어를 실행하세요:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

## Usage
모델 학습 및 도매업체 추천을 수행하려면, 다음 명령어를 실행하세요:

1. 모델 학습 및 저장:
```bash
python train_model.py
```
2. 모델 로드 및 도매업체 추천:
```bash
python recommend_wholesale.py
```

## Project Details
### Model Training
모델 학습은 train_model.py에서 수행되며, 주된 작업은 다음과 같습니다:

1. CSV 파일로부터 데이터를 배치(batch) 단위로 로드합니다.
2. 데이터 전처리:
    - 카테고리 데이터를 멀티라벨로 인코딩
    - 필요 없는 열 제거 및 새로운 피처 생성
3. SGDClassifier를 사용하여 모델을 학습합니다.
4. 학습된 모델, 스케일러, 멀티라벨 바이너라이저를 파일로 저장합니다.


### Recommendation
추천은 recommend_wholesale.py에서 수행되며, 주요 작업은 다음과 같습니다:
1. 저장된 모델과 관련된 객체들을 로드합니다.
2. 새로운 데이터에 대해 피처 일관성을 유지하고 누락된 피처를 처리합니다.
3. 각 소매업체에 대해 도매업체를 추천하고, 결과를 출력합니다.

## Evaluation
모델의 성능은 ROC 커브와 AUC (Area Under Curve) 값으로 평가됩니다. 높은 AUC 값은 모델이 구매 예측을 잘 수행하고 있음을 나타냅니다.

- True Positive (TP): 모델이 구매할 것으로 예측했고, 실제로도 구매한 경우.
- False Positive (FP): 모델이 구매할 것으로 예측했지만, 실제로는 구매하지 않은 경우.
- True Negative (TN): 모델이 구매하지 않을 것으로 예측했고, 실제로도 구매하지 않은 경우.
- False Negative (FN): 모델이 구매하지 않을 것으로 예측했지만, 실제로는 구매한 경우.



