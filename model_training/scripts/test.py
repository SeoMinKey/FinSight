import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# 경로 설정
TEST_JSON_PATH = 'C:/Users/서민기/Desktop/인공지능개론시험준비/기말프로젝트/FinSight/model_training/datasets/test_data/test_label.json'
TEST_DATA_PATH = 'C:/Users/서민기/Desktop/인공지능개론시험준비/기말프로젝트/FinSight/model_training/datasets/test_data'
MODEL_PATH = 'C:/Users/서민기/Desktop/인공지능개론시험준비/기말프로젝트/FinSight/model_training/datasets/trained_models/resnet18_finetune.pth'

# Transform 정의 (학습과 동일해야 함)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 커스텀 데이터셋 클래스
class FishDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.image_dir = image_dir
        self.transform = transform
        self.categories = list(data["categories"].values())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.categories)}

        self.samples = []
        for img_info in data["sample_images"].values():
            label_names = img_info["categories"]
            if not label_names:
                continue
            label_name = label_names[0]
            if label_name in self.class_to_idx:
                label = self.class_to_idx[label_name]
                image_path = os.path.join(image_dir, img_info["filename"])
                if os.path.exists(image_path):
                    self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 평가 함수
def evaluate_model(model, dataloader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # 클래스별 통계용 초기화
    num_classes = len(class_names)
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes

    if len(dataloader.dataset) == 0:
        print("⚠️ 테스트 데이터셋이 비어 있습니다.")
        return 0.0, 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 클래스별 통계 업데이트
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    correct_per_class[label] += 1
                total_per_class[label] += 1

    # 클래스별 정확도 출력
    print("\n📌 클래스별 정확도:")
    for i in range(num_classes):
        accuracy = correct_per_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0.0
        print(f"  - {class_names[i]}: {accuracy:.4f}")

    return running_loss / total, correct / total

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    if len(dataloader.dataset) == 0:
        print("⚠️ 테스트 데이터셋이 비어 있습니다.")
        return 0.0, 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

# 메인 실행
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 로드
    test_dataset = FishDataset(TEST_JSON_PATH, TEST_DATA_PATH, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 클래스 수 추출
    num_classes = len(test_dataset.class_to_idx)

    # 모델 정의 및 로드
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    print(f"✅ 테스트 샘플 수: {len(test_loader.dataset)}")
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, test_dataset.categories)
    print(f"\n📊 테스트 손실: {test_loss:.4f} | 정확도: {test_acc:.4f}")

if __name__ == "__main__":
    main()
