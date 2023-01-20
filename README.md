# slang-stt

## 모델 설명
![image](https://user-images.githubusercontent.com/68102387/213361376-2fd26a2c-285a-4692-abd8-59cd8a436f4c.png)

slang-stt(speech to text) 모델은 은어∙속어를 음성을 인식하여 텍스트로 변환하는 한국어 음성모델입니다. 은어와 속어가 포함되어 있는 연령대별 특징적 발화 음성 데이터를 전사하여 텍스트로 나타냅니다. 

## 모델 아키텍처
![image](https://user-images.githubusercontent.com/68102387/213350380-cbbedef0-aac6-40e7-a440-ec1c11b19273.jpg)

wav2vec 2.0은 2020년 페이스북에서 개발하였으며, 입력한 원시 음성데이터를 기반으로 자기지도학습을 거쳐 데이터를 보다 정확하게 인식하는 음성 모델입니다. 한국어를 포함한 51개의 언어로 pre-trained 되어 있으며, 적은 양의 데이터로도 높은 정확도를 보이는 음성 인식 모델을 구축할 수 있습니다. 기존의 VQ wav2vec보다 안정적인 아키텍처를 가졌으며, 학습된 모델을 다양한 작업에 활용할 수 있습니다. wav2vec 2.0은 일정 거리에 위치한 벡터를 예측하는 CPC(Contrastive Predictive Coding) 방법론과 일정 부분이 가려진 데이터를 트랜스포머 인코더에 입력한 후 그 부분이 무엇인지를 예측하는 mask prediction을 수행하는 MLM(Masked Language Modeling) pre-training 방법론을 사용하여 사전학습을 진행하였습니다. 사전학습 모델에 원하는 작업을 수행하도록 fine-tuning하여 모델을 구성할 수 있습니다.

## 모델 입출력
● 입력: 음성 데이터 (16000 Hz)   
● 출력: 텍스트 데이터  
 
## 모델 태스크
음성 인식 

## 테스트 시스템 사양
```
Windows 10 Pro
Python 3.8.15
Torch 1.13.1
CUDA 11.7
cuDnn 8.5.0
Docker 20.10.21
```

## 학습 데이터셋
연령대별 특징적 발화(은어∙속어 등) 원천 데이터   

## 하이퍼파라미터
● num_train_epochs: epoch 개수  
● batch_size: batch 사이즈  
● weight_decay: 가중치 감쇠, 기존 값 0.05  
● learning_rate: 학습률, 기존 값 5e-5  
● step_size: 학습 중 체크포인트를 저장할 스텝 단위, 기존 값 500

## 실행 방법
### 데이터전처리리
```
python preprocessing.py 
--data_path ./data \
```  
### 모델 학습 
```
python train.py \
--model_final ./model_final \
--basemodel_path ./basemodel_path \
--num_train_epochs 3 \
--batch_size 4 \
--weight_decay 0.005 \
--learning_rate 1e-4 \
--step_size 500
```  
### 모델로 예측 
```
python prediction.py 
--data_path ./data \
--model_path ./model_path
```  

## 평가 기준
음성 인식의 정확도를 측정하는 지표인 CER(Character Error Rate)을 사용하여 성능을 측정하였으며, 모델의 CER 결과는 12.5입니다.   

## License
MIT License

Copyright (c) Deep Loading, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
