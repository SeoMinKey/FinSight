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
import numpy as np
import argparse

# 하이퍼파라미터 설정
HYPERPARAMS = {
    'batch_size': 64,         # 배치 크기 (전체 층 학습을 위해 감소)
    'learning_rate': 1e-4,   # 학습률 (전체 층 학습을 위해 감소)
    'epochs': 15,            # 에폭 수 (전체 층 학습을 위해 증가)
    'weight_type': 'manual', # 가중치 계산 방식 ('inverse', 'balanced', 'manual')
    'num_workers': 8,        # 데이터 로딩에 사용할 워커 수
}

# 경로 설정
CODE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(CODE_PATH, '..', 'datasets')
SAMPLE_IMAGE_PATH = os.path.join(BASE_PATH, "processed_dt/img_cropped_sample")
SAMPLE_JSON_PATH = os.path.join(BASE_PATH, "processed_dt/sample_analysis_new.json")
TEST_DATA_PATH = os.path.join(BASE_PATH, "test_data")
TEST_JSON_PATH = os.path.join(BASE_PATH, "test_data/test_label.json")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "trained_models")
MODEL_NAME = 'resnet18_train_all_3.pth'

# 학습용 전처리 정의 (데이터 증강 추가)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 약간 크게 리사이즈
    transforms.RandomCrop(224),    # 랜덤 크롭
    transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
    transforms.RandomRotation(15),  # 회전
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 색상 변화
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 검증용 전처리 정의 (증강 없음)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

    # 클래스별 샘플 수 계산 메서드 추가
    def get_class_counts(self):
        class_counts = np.zeros(len(self.categories))
        for label in self.labels:
            class_counts[label] += 1
        return class_counts

class BalancedFishDataset(Dataset):
    def __init__(self, dataset, augment_classes=None, augment_factor=2):
        self.dataset = dataset
        self.augment_classes = augment_classes or []  # 추가 증강할 클래스 인덱스 리스트
        self.augment_factor = augment_factor  # 증강 배수
        
        # 원본 인덱스 리스트 생성
        self.indices = list(range(len(dataset)))
        
        # 증강할 클래스의 인덱스 추가
        self.augmented_indices = []
        for idx in self.indices:
            _, label = dataset[idx]
            if label in self.augment_classes:
                # 증강 배수만큼 인덱스 추가
                self.augmented_indices.extend([idx] * (augment_factor - 1))
        
        # 최종 인덱스 리스트 생성
        self.indices.extend(self.augmented_indices)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 원본 데이터셋에서 해당 인덱스의 데이터 가져오기
        return self.dataset[self.indices[idx]]

# 클래스 가중치 계산 함수 수정
def calculate_class_weights(dataset, weight_type='manual'):
    """클래스별 가중치를 계산하는 함수
    
    Args:
        dataset: 데이터셋 객체
        weight_type: 가중치 계산 방식 ('inverse', 'balanced', 'manual')
    
    Returns:
        torch.Tensor: 클래스별 가중치
    """
    class_counts = dataset.get_class_counts()
    num_samples = sum(class_counts)
    num_classes = len(class_counts)
    
    if weight_type == 'inverse':
        # 클래스 빈도의 역수를 가중치로 사용
        class_weights = num_samples / (class_counts * num_classes)
    elif weight_type == 'balanced':
        # 균형 잡힌 가중치 (sklearn의 'balanced' 옵션과 유사)
        class_weights = num_samples / (class_counts * num_classes)
    elif weight_type == 'manual':
        # 수동으로 지정한 가중치 (카테고리 매핑에 맞게 조정)
        # 혼동 행렬에서 Black porgy가 Rock bream으로 잘못 분류되는 경우가 많으므로 가중치 증가
        class_weights = torch.tensor([
            1.0,  # Olive flounder (인덱스 0) - 가중치 증가
            1.0,  # Korea rockfish (인덱스 1) - 가중치 증가
            1.0,  # Red seabream (인덱스 2)
            4.0,  # Black porgy (인덱스 3) - 가중치 더 증가
            1.0   # Rock bream (인덱스 4) - 가중치 감소
        ])
    else:
        # 기본값: 모든 클래스에 동일한 가중치
        class_weights = torch.ones(num_classes)
    
    return torch.tensor(class_weights, dtype=torch.float)

# 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 모델 정의 함수 - 전체 층 학습 버전
def get_train_all_model(num_classes):
    # 사전 학습된 가중치로 모델 초기화
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # 모든 층을 학습 가능하도록 설정 (파인튜닝과 다른 부분)
    for param in model.parameters():
        param.requires_grad = True
    
    # 마지막 FC 레이어를 타겟 클래스 수에 맞게 변경
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
    
    # ZeroDivisionError 방지
    if total == 0:
        return 0.0, 0.0
    
    return running_loss / total, correct / total

# Focal Loss 구현 (어려운 샘플에 더 집중)
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 명령줄 인자 파싱 함수
def parse_args():
    parser = argparse.ArgumentParser(description='ResNet18 전체 층 학습')
    parser.add_argument('--batch_size', type=int, default=HYPERPARAMS['batch_size'],
                        help='배치 크기 (기본값: 64)')
    parser.add_argument('--lr', type=float, default=HYPERPARAMS['learning_rate'],
                        help='학습률 (기본값: 0.0001)')
    parser.add_argument('--epochs', type=int, default=HYPERPARAMS['epochs'],
                        help='에폭 수 (기본값: 15)')
    parser.add_argument('--weight_type', type=str, default=HYPERPARAMS['weight_type'],
                        choices=['inverse', 'balanced', 'manual', 'none'],
                        help='가중치 계산 방식 (기본값: balanced)')
    parser.add_argument('--num_workers', type=int, default=HYPERPARAMS['num_workers'],
                        help='데이터 로딩에 사용할 워커 수 (기본값: 8)')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Focal Loss 사용 여부')
    return parser.parse_args()

def main():
    # 명령줄 인자 파싱 또는 기본값 사용
    try:
        args = parse_args()
    except:
        # 명령줄 인자가 없는 경우 기본값 사용
        class Args:
            def __init__(self):
                self.batch_size = HYPERPARAMS['batch_size']
                self.lr = HYPERPARAMS['learning_rate']
                self.epochs = HYPERPARAMS['epochs']
                self.weight_type = HYPERPARAMS['weight_type']
                self.num_workers = HYPERPARAMS['num_workers']
                self.use_focal_loss = False
        args = Args()

    # 학습 장비 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 학습 데이터 로드
    dataset = FishDataset(SAMPLE_JSON_PATH, SAMPLE_IMAGE_PATH, transform=None)  # transform은 나중에 적용
    
    # 데이터셋 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 각 데이터셋에 적절한 transform 적용
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # 카테고리 매핑 정보 출력
    print("\n=== 카테고리 매핑 정보 ===")
    for cat, idx in dataset.category_to_idx.items():
        print(f"카테고리: {cat}, 인덱스: {idx}")
    
    # 클래스별 가중치 계산
    class_weights = calculate_class_weights(dataset, weight_type=args.weight_type)
    print(f"클래스별 가중치 ({args.weight_type}): {class_weights}")
    
    # 문제가 있는 클래스들의 인덱스 확인
    black_porgy_idx = dataset.category_to_idx["Black porgy"]
    olive_flounder_idx = dataset.category_to_idx["Olive flounder"]
    korea_rockfish_idx = dataset.category_to_idx["Korea rockfish"]
    
    # 균형 잡힌 데이터셋 생성 (문제가 있는 클래스들 증강)
    balanced_train_dataset = BalancedFishDataset(
        train_dataset, 
        augment_classes=[black_porgy_idx, olive_flounder_idx, korea_rockfish_idx], 
        augment_factor=3
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        balanced_train_dataset,  # 균형 잡힌 데이터셋 사용
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    # 모델 준비 - 전체 층 학습 모델 사용
    num_classes = len(dataset.categories)
    model = get_train_all_model(num_classes).to(device)

    # 가중치를 적용한 손실 함수 정의
    if args.use_focal_loss:
        # Focal Loss 감마 파라미터 증가
        criterion = FocalLoss(weight=class_weights.to(device), gamma=3.0)  # 감마 값 증가
        print("Focal Loss 사용")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Cross Entropy Loss 사용")
    
    # 옵티마이저 정의 - 전체 층 학습을 위한 설정
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)  # 가중치 감쇠 추가
    
    # 학습률 스케줄러 추가
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # 학습 루프
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        # 학습률 스케줄러 업데이트
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"best_{MODEL_NAME}"))
            print(f"새로운 최고 성능 모델 저장: best_{MODEL_NAME} (정확도: {val_acc:.4f})")

    # 최종 모델 저장
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
    print(f"최종 모델 저장 완료: {MODEL_NAME}")
    print(f"사용된 하이퍼파라미터: 배치 크기={args.batch_size}, 학습률={args.lr}, 에폭 수={args.epochs}, 가중치 타입={args.weight_type}")
    print(f"최고 검증 정확도: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()

# Black porgy에 특화된 데이터 증강 추가
class BlackPorgyAugmentedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.black_porgy_indices = []
        self.other_indices = []
        
        # Black porgy와 다른 클래스 인덱스 분리
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label == dataset.dataset.category_to_idx["Black porgy"]:
                self.black_porgy_indices.append(idx)
            else:
                self.other_indices.append(idx)
        
        # 최종 인덱스 리스트 생성 (Black porgy는 5배 증강)
        self.indices = self.other_indices + self.black_porgy_indices * 5
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        
        # Black porgy인 경우 추가 증강 적용
        if label == self.dataset.dataset.category_to_idx["Black porgy"]:
            # 랜덤하게 추가 증강 적용
            if random.random() > 0.5:
                img = transforms.functional.adjust_contrast(img, random.uniform(1.5, 2.0))
            if random.random() > 0.5:
                img = transforms.functional.adjust_brightness(img, random.uniform(1.2, 1.5))
        
        return img, label

# Black porgy에 특화된 데이터 증강 정의
black_porgy_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),  # 수직 반전 추가
    transforms.RandomRotation(30),  # 회전 각도 증가
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # 어파인 변환 추가
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변화 강화
    transforms.RandomGrayscale(p=0.1),  # 흑백 변환 추가
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 모델 정의 함수 수정 - 특징 추출 레이어 추가
def get_enhanced_model(num_classes):
    # 사전 학습된 가중치로 모델 초기화
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # 모든 층을 학습 가능하도록 설정
    for param in model.parameters():
        param.requires_grad = True
    
    # 특징 추출 레이어 추가
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    return model