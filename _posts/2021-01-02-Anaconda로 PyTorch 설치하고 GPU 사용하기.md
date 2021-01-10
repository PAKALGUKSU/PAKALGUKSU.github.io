---
title: "Anaconda로 PyTorch 설치하고 GPU 사용하기"
categories:
    - Development
tags:
    - Development
toc: True
---

기존에 Tensorflow(Keras)를 이용하다가 새로 PyTorch 환경을 설정해야 하는 일이 생겨, anaconda 환경에서 GPU를 이용하는 PyTorch 설치 방법을 정리해봤다.

### Conda 가상환경 생성
```shell
conda create -n pytorch_env python=3.7
conda activate pytorch_env
```
Python 3.7버전을 사용하는 pytorch_env라는 가상환경을 생성한 이후 활성화시킨다.

### PyTorch, CUDA 설치
[PyTorch 공식 설치 홈페이지](https://pytorch.org/get-started/locally/) 의 안내에 따라 PyTorch와 CUDA를 설치한다. 위 홈페이지에서 자신의 환경에 맞는 PyTorch 설정을 선택하면 설치 명령어가 표시되는데, PyTorch 버전에 맞는 CUDA 및 cuDNN도 같이 설치된다.

위 사진의 경우 linux의 conda 환경에서 CUDA 10.2에 해당하는 PyTorch 설치를 선택했을 때 실행해야 할 명렁어이다.
```shell
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

### PyTorch에서 GPU 사용 확인
설치를 완료했으면, 다음과 같이 PyTorch에서 GPU를 잘 인식하고 있는지 확인할 수 있다.
쉘에 다음 명렁을 입력해 python을 실행한다.
```shell
python
```

python을 실행한 이후, 다음 코드의 실행 결과를 확인한다.

```python
import torch
torch.cuda.is_available()
```
결과로 True가 출력된다면 PyTorch에서 정상적으로 GPU를 인식하고 있는 것이다.