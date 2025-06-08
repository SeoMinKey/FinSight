# 필요한 라이브러리 임포트
import torch
import torch.nn as nn  # 신경망 모듈
import torch.optim as optim  # 최적화 알고리즘
from torch.utils.data import Dataset, DataLoader  # 데이터 처리 도구
from torchvision import transforms, models  # 이미지 변환 및 사전학습 모델
from PIL import Image  # 이미지 처리
import json
import os
import glob
from tqdm import tqdm  # 진행률 표시 바

# 프로젝트 경로 설정
CODE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(CODE_PATH, '..', 'datasets')

SAMPLE_IMAGE_PATH = os.path.join(BASE_PATH, "processed_dt/img_cropped_sample")  # 크롭된 이미지 경로
SAMPLE_JSON_PATH = os.path.join(BASE_PATH, "processed_dt/sample_analysis_new.json")  # 어노테이션 파일 경로

MODEL_SAVE_PATH = os.path.join(BASE_PATH, "trained_models")  # 학습된 모델 저장 경로
MODEL_NAME = 'resnet18_pretrained_fish.pth' # 모델 파일 이름

# 전처리한 테스트 데이터 저장 경로 추가
TEST_DATA_PATH = os.path.join(BASE_PATH, "test_data")  # 테스트 데이터 경로
TEST_JSON_PATH = os.path.join(BASE_PATH, "test_data/test_label.json")  # 테스트 데이터 경로


# 커스텀 데이터셋 클래스 정의
class FishDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        """데이터셋 초기화
        Args:
            json_path (str): 어노테이션 JSON 파일 경로
            img_dir (str): 이미지 디렉토리 경로
            transform: 이미지 전처리 변환 함수
        """
        # JSON 파일 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform
        
        # 카테고리(어종) 정보를 숫자 인덱스로 매핑
        self.categories = list(self.data['categories'].values())
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # 이미지 경로와 레이블 준비
        self.images = []
        self.labels = []
        for img_info in self.data['sample_images'].values():
            # 크롭된 이미지 파일명 생성
            img_path = os.path.join(self.img_dir, img_info['filename'].replace('./', '').replace('.JPG', '_cropped.jpg'))
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.labels.append(self.category_to_idx[img_info['categories'][0]])
    
    def __len__(self):
        """데이터셋의 총 샘플 수 반환"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """특정 인덱스의 이미지와 레이블 반환"""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 이미지 로드 및 전처리
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 이미지 전처리 파이프라인 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 입력 크기로 조정
    transforms.ToTensor(),  # PIL 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 통계치로 정규화
])

# 데이터셋 및 데이터로더 초기화
dataset = FishDataset(SAMPLE_JSON_PATH, SAMPLE_IMAGE_PATH, transform=transform)
# 학습:검증 데이터 8:2 비율로 분할
try:
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 데이터로더 생성 (배치 처리를 위함)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
except ValueError as e:
    print(f"데이터셋 로드 중 오류 발생: {e}. 데이터셋이 비어 있거나 유효한 이미지가 없습니다.")
    print("annotation_test.py를 실행하여 데이터셋을 생성했는지 확인해주세요.")
    exit() # 스크립트 종료

def get_pretrained_model(num_classes):
    """사전학습된 ResNet18 모델을 가져와서 파인튜닝 없이 마지막 레이어만 수정"""
    # 사전학습된 ResNet18 모델 로드
    model = models.resnet18(pretrained=True)
    
    # 모든 파라미터를 고정 (파인튜닝 없음)
    for param in model.parameters():
        param.requires_grad = False
    
    # 마지막 완전연결층만 교체하여 학습 가능하게 설정
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def evaluate_model(model, dataloader, criterion, device):
    """모델 평가 함수"""
    model.eval()  # 평가 모드 설정
    running_loss = 0.0
    running_corrects = 0
    
    # 배치 단위로 처리
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 순전파
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # 통계 업데이트
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    # 손실과 정확도 계산
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc

# img_sample 폴더의 이미지를 분류하는 함수 추가
def classify_original_samples(model, device, categories):
    """img_sample 폴더의 원본 이미지를 분류하고 클래스별 정확도를 출력하는 함수"""
    print("\n원본 샘플 이미지 분류 시작...")
    
    # 이미지 전처리 파이프라인
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 모델을 평가 모드로 설정
    model.eval()
    
    # img_sample 폴더의 모든 이미지 파일 가져오기
    image_files = glob.glob(os.path.join(TEST_DATA_PATH, "*.JPG")) + \
                 glob.glob(os.path.join(TEST_DATA_PATH, "*.jpg"))
    
    # 각 클래스별 예측 결과 저장을 위한 딕셔너리
    class_correct = {cat: 0 for cat in categories}
    class_total = {cat: 0 for cat in categories}

    # JSON 파일에서 실제 레이블 로드
    with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
    
    # 파일명과 실제 레이블 매핑
    file_to_label = {}
    for img_info in sample_data['sample_images'].values():
        filename_base = img_info['filename'].replace('./', '')
        # 원본 이미지 파일명 (크롭되지 않은)을 키로 사용
        original_filename = filename_base.replace('.JPG', '.JPG') if '.JPG' in filename_base else filename_base.replace('.jpg', '.jpg')
        file_to_label[original_filename] = img_info['categories'][0]

    # 각 이미지 분류
    for img_path in tqdm(image_files):
        try:
            file_name = os.path.basename(img_path)
            # 실제 레이블 가져오기
            true_label_name = file_to_label.get(file_name)
            if true_label_name is None:
                print(f"경고: {file_name} 에 대한 레이블 정보를 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            # 이미지 로드 및 전처리
            img = Image.open(img_path).convert('RGB')
            img_tensor = img_transform(img).unsqueeze(0).to(device)  # 배치 차원 추가
            
            # 예측
            with torch.no_grad():
                outputs = model(img_tensor)
                _, preds = torch.max(outputs, 1)
                pred_class_idx = preds.item()
                pred_class_name = categories[pred_class_idx]
            
            # 정확도 계산을 위한 통계 업데이트
            class_total[true_label_name] += 1
            if pred_class_name == true_label_name:
                class_correct[true_label_name] += 1
            
        except Exception as e:
            print(f"이미지 {img_path} 처리 중 오류 발생: {str(e)}")
    
    # 클래스별 정확도 출력
    print("\n클래스별 정확도:")
    print("-" * 40)
    print(f"{'클래스':<20} {'정확도':<10}")
    print("-" * 40)
    
    overall_correct = 0
    overall_total = 0

    for class_name in categories:
        correct = class_correct.get(class_name, 0)
        total = class_total.get(class_name, 0)
        overall_correct += correct
        overall_total += total

        if total > 0:
            accuracy = 100 * correct / total
            print(f"{class_name:<20} {accuracy:.2f}%")
        else:
            print(f"{class_name:<20} {'N/A'}")
    
    print("-" * 40)
    if overall_total > 0:
        overall_accuracy = 100 * overall_correct / overall_total
        print(f"{'전체 정확도':<20} {overall_accuracy:.2f}%")
    else:
        print(f"{'전체 정확도':<20} {'N/A'}")

def main():
    """메인 실행 함수"""
    # 모델 저장 디렉토리 생성
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # 모델 초기화 (클래스 수는 데이터셋의 어종 수)
    num_classes = len(dataset.categories)
    model = get_pretrained_model(num_classes)
    
    # GPU 사용 가능하면 GPU 사용, 아니면 CPU 사용
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)

    # 기존 모델이 있으면 로드, 없으면 새로 학습
    if os.path.exists(model_path):
        print(f"기존 모델 로드: {MODEL_NAME}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("새 모델 학습 시작")
        # 손실 함수 및 옵티마이저 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

        # 학습 수행 (마지막 레이어만 학습)
        print("학습 시작 - 마지막 레이어만 학습")
        num_epochs = 10
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            # 학습 단계
            model.train()  # 학습 모드 설정
            running_loss = 0.0
            running_corrects = 0
            
            # 배치 단위로 처리
            for inputs, labels in tqdm(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 그래디언트 초기화
                optimizer.zero_grad()
                
                # 순전파 + 역전파 + 최적화
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # 통계 업데이트
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # 에폭당 학습 손실과 정확도 계산
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 검증 단계
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
            print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # 학습된 모델 저장
        torch.save(model.state_dict(), model_path)
        print("학습 완료 및 모델 저장됨 - 파인튜닝 없이 마지막 레이어만 학습")
    
    # 모델 성능 평가 (기존 모델 로드 또는 새 모델 학습 후 모두 실행)
    print("\n최종 모델 평가:")
    # criterion이 정의되지 않았을 수 있으므로, 여기에 다시 정의합니다.
    criterion = nn.CrossEntropyLoss() 
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
    print(f'검증 데이터 Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    
    # 어종별 매핑 정보 출력
    print("\n어종 분류 매핑 정보:")
    for idx, category in enumerate(dataset.categories):
        print(f"클래스 {idx}: {category}")
    
    # img_sample 폴더의 이미지 분류 (기존 모델 로드 또는 새 모델 학습 후 모두 실행)
    classify_original_samples(model, device, dataset.categories)

if __name__ == '__main__':
    main()