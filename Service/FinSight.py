import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ✅ 현재 파일 경로 기준으로 상대 경로 사용
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 모델 정의 함수 - 전체 층 학습 버전 (resnet18_train_all.py에서 가져옴)
def get_train_all_model(num_classes):
    # 사전 학습된 가중치로 모델 초기화
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # 모든 층을 학습 가능하도록 설정
    for param in model.parameters():
        param.requires_grad = True
    
    # 마지막 FC 레이어를 타겟 클래스 수에 맞게 변경
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

# ✅ 클래스 이름과 클래스 수 직접 정의
class_names = ["넙치","조피볼락","참돔","감성돔","돌돔"]
num_classes = len(class_names)

# ✅ 모델 로딩
model_path = os.path.join(BASE_DIR, "resnet18_train_all_2.pth")
model = get_train_all_model(num_classes) # num_classes를 사용하여 모델 초기화
model.load_state_dict(torch.load(model_path, map_location='cpu'))  # GPU 사용 시 'cuda'
model.eval()

# ✅ 이미지 폴더 위치
image_folder = os.path.join(BASE_DIR, "Fish_image")

# ✅ 이미지 전처리 (모델 입력 크기에 맞게 수정 필요)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 모델 입력 크기에 맞게 수정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 정규화 추가
])

# ✅ 이미지 예측
for img_name in os.listdir(image_folder):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    img_path = os.path.join(image_folder, img_name)
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]
        print(f"{img_name}: 예측된 어종은 {label}")
