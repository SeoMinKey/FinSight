import json
import os
from collections import defaultdict
import shutil

def analyze_json_structure(json_path, limit=20):
    """JSON 파일의 구조를 분석하고 데이터를 정리하는 함수"""
    print("\n=== JSON 파일 분석 시작 ===")
    
    # JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. 카테고리 정보 분석
    '''데이터셋의 구조변환 필요 
    "categories":
    [{
    "name":"Olive flounder",
    "supercategory":"fish",
    "id":1}


    필요한 데이터 : id, name(어종ID, 어종명)
    최종적으로 id:name 형식으로 변환 필요
    '''
    categories = {cat['id']: cat['name'] for cat in data['categories']} #categories 리스트를 돌면서  id:name으로 새로운 딕셔너리 생성
    print("\n=== 어종 카테고리 정보 ===")
    for cat_id, cat_name in categories.items(): #items() 메서드는 딕셔너리의 키와 값을 튜플로 묶은 객체를 반환
        print(f"ID: {cat_id}, 이름: {cat_name}")
    
    # 2. 이미지 정보 분석 및 조건에 맞는 이미지 선택
    valid_images = []
    for ann in data['annotations']:
        # bbox가 있고 diseases_exist가 false인 이미지만 선택
        if ann['bbox'] and not ann['diseases_exist']:
            img_id = ann['image_id']
            # 해당 이미지 정보 찾기
            for img in data['images']:
                if img['id'] == img_id:
                    valid_images.append({
                        'id': img_id,
                        'file_name': img['file_name'],
                        'category_id': ann['category_id'],
                        'bbox': ann['bbox']  # bbox 정보 추가
                    })
                    break
            # limit이 0이면 모든 이미지 선택, 아니면 limit 개수만큼만 선택
            if limit > 0 and len(valid_images) >= limit:
                break
    
    # 최종 데이터 구조 생성 (limit이 0이면 모든 이미지 사용, 아니면 limit 개수만큼 사용)
    sample_images = valid_images if limit == 0 else valid_images[:limit]
    
    # 6. 최종 데이터 구조 생성
    sample_data = {
        'categories': categories,
        'sample_images': {str(img_data['id']): {
            'filename': img_data['file_name'],
            'categories': [categories[img_data['category_id']]],
            'bbox': img_data['bbox']  # bbox 정보 추가
        } for img_data in sample_images}
    }
    
    return sample_data

# 기본 경로
BASE_PATH = "C:/Users/서민기/Desktop/인공지능개론시험준비/기말프로젝트/FinSight/model_training/datasets"

# 원본 데이터 경로
ORIGINAL_JSON_PATH = os.path.join(BASE_PATH, "어류 개체 촬영 영상/Training/gbt_fish_dtset1.json")
ORIGINAL_DATASET_PATH = os.path.join(BASE_PATH, "어류 개체 촬영 영상/Training/dtset1")

# 새로운 샘플 데이터 경로
SAMPLE_OUTPUT_PATH = os.path.join(BASE_PATH, "New_sample/전처리 샘플 데이터/sample_analysis_new.json")
SAMPLE_IMAGE_PATH = os.path.join(BASE_PATH, "New_sample/전처리 샘플 데이터/img_sample")


# 기존 경로 상수에 크롭 이미지 경로 추가
SAMPLE_CROPPED_PATH = os.path.join(BASE_PATH, "New_sample/전처리 샘플 데이터/img_cropped_sample")


# 파일 상단에 PIL import 추가
from PIL import Image

# 파일 상단에 import 추가
import cv2
import numpy as np

def crop_image(image_path, bbox, output_path):
    """이미지를 bbox 좌표에 따라 크롭하는 함수"""
    try:
        # 이미지 열기
        with Image.open(image_path) as img:
            # bbox 좌표 추출 [x, y, width, height]
            x, y, w, h = map(int, bbox)
            # 크롭 영역 계산 (left, top, right, bottom)
            crop_area = (x, y, x + w, y + h)
            # 이미지 크롭
            cropped_img = img.crop(crop_area)
            # 크롭된 이미지 저장
            cropped_img.save(output_path)
            return True
    except Exception as e:
        print(f"이미지 크롭 중 오류 발생: {str(e)}")
        return False

def apply_clahe(image):
    """CLAHE를 적용하여 이미지의 대비를 개선하는 함수"""
    try:
        # BGR 이미지를 LAB 색공간으로 변환
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # L 채널에 대해 CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[...,0] = clahe.apply(lab[...,0])
        
        # LAB에서 BGR로 다시 변환
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
    except Exception as e:
        print(f"CLAHE 적용 중 오류 발생: {str(e)}")
        return image

def crop_image(image_path, bbox, output_path):
    """이미지를 bbox 좌표에 따라 크롭하고 CLAHE를 적용하는 함수"""
    try:
        # 한글 경로 처리를 위한 수정
        img = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("이미지를 읽을 수 없습니다.")
            
        # bbox 좌표 추출 [x, y, width, height]
        x, y, w, h = map(int, bbox)
        
        # 이미지 크롭
        cropped = img[y:y+h, x:x+w]
        
        # CLAHE 적용
        enhanced = apply_clahe(cropped)
        
        # 한글 경로 처리를 위한 이미지 저장
        _, img_encoded = cv2.imencode('.jpg', enhanced)
        img_encoded.tofile(output_path)
        return True
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {str(e)}")
        return False

# main 함수 내 파일 복사 부분 수정
def main():
    try:
        # JSON 파일에서 데이터 읽기 (limit=0으로 설정하면 모든 이미지 선택)
        sample_data = analyze_json_structure(ORIGINAL_JSON_PATH, limit=0)
        
        # 결과 저장
        with open(SAMPLE_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        print(f"\n분석 결과가 {SAMPLE_OUTPUT_PATH}에 저장되었습니다.")
        
        # 파일 복사 및 크롭 부분
        for img_id, info in sample_data["sample_images"].items():
            file_name = info['filename'].replace('./', '')
            src_path = os.path.join(ORIGINAL_DATASET_PATH, file_name)
            dst_path = os.path.join(SAMPLE_IMAGE_PATH, file_name)
            
            # 크롭된 이미지 경로 생성
            cropped_file_name = os.path.splitext(file_name)[0] + "_cropped" + os.path.splitext(file_name)[1]
            cropped_path = os.path.join(SAMPLE_CROPPED_PATH, cropped_file_name)
            
            # 원본 이미지 디렉토리와 크롭 이미지 디렉토리 생성
            os.makedirs(SAMPLE_IMAGE_PATH, exist_ok=True)
            os.makedirs(SAMPLE_CROPPED_PATH, exist_ok=True)
            
            if os.path.exists(src_path):
                # 원본 이미지 복사
                shutil.copy2(src_path, dst_path)
                print(f"복사 완료: {src_path} -> {dst_path}")
                
                # 이미지 크롭
                if crop_image(dst_path, info['bbox'], cropped_path):
                    print(f"크롭 완료: {cropped_path}")
                else:
                    print(f"크롭 실패: {file_name}")
            else:
                print(f"파일 없음: {src_path}")
    except Exception as e:
        print(f"파일 복사 오류 발생: {str(e)}")









if __name__ == '__main__':
    main()


