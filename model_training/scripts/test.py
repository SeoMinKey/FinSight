import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


# 경로 설정
TEST_JSON_PATH = 'C:/Users/서민기/Desktop/인공지능개론시험준비/기말프로젝트/FinSight/model_training/datasets/test_data/test_label.json'
TEST_DATA_PATH = 'C:/Users/서민기/Desktop/인공지능개론시험준비/기말프로젝트/FinSight/model_training/datasets/test_data'
MODEL_PATH = 'C:/Users/서민기/Desktop/인공지능개론시험준비/기말프로젝트/FinSight/model_training/datasets/trained_models/resnet18_train_all_2.pth' 
#resnet18_finetune.pth   resnet18_pretrained_fish
#resnet18_finetune_weight_1

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
    all_labels = []
    all_preds = []

    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)

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

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    print("\n📌 클래스별 정확도:")
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"  {class_name}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"  {class_name}: 데이터 없음")

    # 기존 혼동 행렬 출력 부분 그대로 유지
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

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
