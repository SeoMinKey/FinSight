# 필요한 라이브러리 임포트
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
import os
from tqdm import tqdm

# 경로 설정
BASE_PATH = "C:/Users/서민기/Desktop/인공지능개론시험준비/기말프로젝트/FinSight/model_training/datasets"
SAMPLE_IMAGE_PATH = os.path.join(BASE_PATH, "processed_dt/img_cropped_sample")
SAMPLE_JSON_PATH = os.path.join(BASE_PATH, "processed_dt/sample_analysis_new.json")
TEST_DATA_PATH = os.path.join(BASE_PATH, "test_data")
TEST_JSON_PATH = os.path.join(BASE_PATH, "test_data/test_label.json")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "trained_models")
MODEL_NAME = 'resnet18_finetune.pth'

# 커스텀 데이터셋 클래스 정의
class FishDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.img_dir = img_dir
        self.transform = transform
        self.categories = list(self.data['categories'].values())
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        self.images = []
        self.labels = []
        for img_key, img_info in self.data['sample_images'].items():
            filename_base = img_info['filename'].replace('./', '')
            base_name = os.path.splitext(filename_base)[0]
            cropped_img_filename = f"{base_name}_cropped.jpg"
            img_path = os.path.join(self.img_dir, cropped_img_filename)

            if os.path.exists(img_path):
                self.images.append(img_path)
                if img_info['categories']:
                    self.labels.append(self.category_to_idx[img_info['categories'][0]])
                else:
                    print(f"경고: {img_info['filename']} 에 레이블 없음")
                    self.images.pop()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 모델 정의 함수
def get_finetune_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 평가 함수
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

# 학습 장비 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 데이터 로드
dataset = FishDataset(SAMPLE_JSON_PATH, SAMPLE_IMAGE_PATH, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 모델 준비
num_classes = len(dataset.categories)
model = get_finetune_model(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# 학습 루프
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

# 모델 저장
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
print(f"모델 저장 완료: {MODEL_NAME}")
