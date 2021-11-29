# Transfer learning tutorial

## 개요

---

- 빠른 시간 내에 높은 정확도를 얻을 수 있음
- "사전 학습된 모델(pre-trained model)"을 사용
- 사전 학습된 모델이란, 유사한 문제에 대해 큰 데이터로 이미 학습된 모델을 의미함
- 해당 조건을 만족하는 사전 학습된 모델들이 공개되어 있음(e.g VGG, MobileNet, ResNet 등)

## ResNet과 전이 학습을 사용한 CIFAR-10 분류기 학습

---

1. 사전 학습된 모델 불러오기

```python
# Resnet18 모델 불러오기
model_ft = torchvision.models.resnet18(pretrained=True)
```

2. 자신의 문제에 맞게 모델 변형

```python
# fc layer의 입력 features 크기
num_ftrs = model_ft.fc.in_features

# 마지막 fc layer의 output shape를 변형
model_ft.fc = nn.Linear(num_ftrs, 10)
```

- CIFAR-10 데이터 셋의 경우 10개의 카테고리를 갖고 있음
- 마지막 레이어의 출력 크기를 10개로 맞춤
- 또는, 출력 크기를 변형하는 fc 레이어를 이어 붙일 수도 있음

3. Fine-tuning
    1. 전체 모델 학습
        - base model의 가중치를 초기 가중치로 설정하고 새로운 데이터 셋을 사용해 전체적인 모델을 다시 학습
        - 학습에 걸리는 시간이 길고, 큰 사이즈의 데이터 셋이 필요
    2. 일부 고정, 일부 재학습
        - base model의 일부분을 고정하여 학습에서 제외
        - 나머지 계층과 분류기 부분 만을 재학습
        - 학습에 걸리는 시간이 감소함
    3. 분류기만 재학습
        - base model의 대부분을 고정
        - 분류기만 재학습하기 때문에 학습 시간이 적고 데이터 셋이 적을 때 적합

---

출처:

- [https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html](https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html)
- [https://jeinalog.tistory.com/13](https://jeinalog.tistory.com/13)