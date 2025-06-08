# FinSight
인공지능개론 기말 프로젝트

## 스크립트 설명 및 사용 법
1. 데이터셋 준비 : [[AIHub - 어류 개체 촬영 영상]](href="https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=154, "dataset link")에서 데이터셋을 다운받아 압축을 푼 후, [하위 폴더](model_training/datasets)에 넣어줍니다. 
(데이터셋의 크기가 크기 때문에 압축을 푼 후 dtset1과 gbt_fish_dtset1.json만 사용함)
```
model_training
└── datasets
    └── 어류 개체 촬영 영상
        └── dtset1
        └── gbt_fish_dtset1.json
```
2. 의존성 패키지 설치 
[requirements.txt](requirements.txt)에서 의존성 패키지를 설치합니다.

3. CUDA를 사용할 수 있는지 확인합니다.
[check_cuda.py](model_training/scripts/check_cuda.py)

4. resnet18 출력층만 교체하는 학습
[resnet18_pretrained.py](model_training/scripts/resnet18_pretrained.py)

5. resnet18 fine-tuning (Layer-4와 출력층만 학습)
[resnet18_finetune.py](model_training/scripts/resnet18_finetune.py)

6. resnet18 전 층 학습 (Layer-4, Layer-3, Layer-2, Layer-1, 출력층 학습)
[resnet18_train_all.py](model_training/scripts/resnet18_train_all.py)

7. 결과물 실행 
a) [Fish_image](Service/Fish_image/)에 테스트 할 이미지 넣기
b) [FinSight.py](Service/FinSight.py) 실행

8. Confusion matrix와 학습 결과 확인
[test.py](model_training/scripts/test.py)

번외 : 디렉토리 구조 및 설명
```
root
└── model_training
    └── scripts(실행 코드)
    └── datasets
        └── 어류 개체 촬영 영상 (원본 데이터셋)
        └── trained_model (학습된 모델)
        └── test_data (구글 테스트 데이터 50장)
        └── processed_dt (전처리 후 데이터셋)
└── Service
    └── Fish_image (테스트 할 이미지)
    └── FishSight.py (실행코드)
    └── resnet18_train_all_2.pth (학습이 완료된 모델)
```
