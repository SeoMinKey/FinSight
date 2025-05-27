import json
import os
from collections import defaultdict
import shutil

def analyze_json_structure(json_path):
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
    
    # 2. 이미지 정보 분석 
    limit = 20 #20개만 사용
    '''이미지 데이터 구조
    "images":
    [{
    "light":"L02",
    "file_name":".\/560364_objt_kr_2021-02-05_11-42-12-33_002.JPG","width":2704,
    "date_captured":"2021-02-09 15:12:47",
    "id":0,
    "height":1520},
    '''
    # 필요한 데이터 : file_name, id (파일명, 이미지 ID) --> id:file_name 형식으로 변환 필요

    images = {img['id']: img['file_name'] for img in data['images'][:limit]}
    print(f"\n총 이미지 수: {len(data['images'])}")
    print(f"\n추린 이미지 수: {len(images)}")
    for img_id, img_name in images.items():
        print(f"ID: {img_id}, 파일명: {img_name}")
    
    # 3. 어노테이션 정보 분석 
    '''
    어노테이션 데이터 구조 (물고기의 병을 찾기 위한 bbox와 병의 종류를 찾기 위한 category_id 등 포함)
    "annotations":
    [   
        {"diseases_bbox":[],
        "category_id":2,
        "iscrowd":1
        ,"bbox":[383,292,1012,398],
        "id":0,
        "attribute":"objt",
        "diseases_desc":"",
        "image_id":0,
        "keypoints":[385,405,1272,595,755,306,689,578],
        "gd":245,
        "diseases_exist":false},
    '''
    # 필요한 데이터 : category_id, image_id (물고기 종류, 이미지 ID)

    annotations = defaultdict(list) #딕셔너리 추가용
    for ann in data['annotations']:
        if ann['image_id'] in images:  # 위에서 추린 이미지 라벨에서 존재하는 이미지 ID만 처리
            annotations[ann['image_id']].append(ann['category_id']) #이미지ID : 어종 ID 형식으로 변환
    
    print(f"\n총 어노테이션 수: {len(data['annotations'])} ")
    print(f"\n추린 어노테이션 수: {len(annotations)}")
    for img_id, cat_ids in annotations.items():
        print(f"이미지 ID: {img_id}, 어종 ID: {cat_ids}")
    
    # 4. 카테고리별 이미지 수 계산 
    category_counts = defaultdict(int)
    for img_id, cat_ids in annotations.items():
        for cat_id in cat_ids:
            category_counts[cat_id] += 1    #각 어종 키가 검출되면 카운트 증가
    
    print("\n=== 카테고리별 이미지 수  ===")
    for cat_id, count in sorted(category_counts.items()):
        print(f"카테고리 {categories[cat_id]}: {count}개")
    
    # 5. 이미지 ID와 파일명 매핑 검증
    print("\n=== 이미지 ID와 파일명 매핑 검증 ===")
    for img_id, img_name in images.items():
        if img_id in annotations:
            print(f"이미지 ID: {img_id}")
            print(f"파일명: {img_name}")
            print(f"카테고리: {[categories[cat_id] for cat_id in annotations[img_id]]}\n")  #img_id에 해당하는 어종 카테고리 출력
    
    # 6. 최종 데이터 구조 생성
    sample_data = {
        'categories': categories,
        'sample_images': {str(img_id): {
            'filename': img_name,
            'categories': [categories[cat_id] for cat_id in annotations[img_id]]
        } for img_id, img_name in images.items() if img_id in annotations} 
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

def main():
    try:
        # JSON 파일에서 데이터 읽기
        sample_data = analyze_json_structure(ORIGINAL_JSON_PATH)
            
        # 결과 저장
        with open(SAMPLE_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        print(f"\n분석 결과가 {SAMPLE_OUTPUT_PATH}에 저장되었습니다.")
        
        # 파일 복사 부분
        for img_id, info in sample_data["sample_images"].items():
            file_name = info['filename'].replace('./', '')
            src_path = os.path.join(ORIGINAL_DATASET_PATH, file_name)
            dst_path = os.path.join(SAMPLE_IMAGE_PATH, file_name)
            
            os.makedirs(SAMPLE_IMAGE_PATH, exist_ok=True)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"복사 완료: {src_path} -> {dst_path}")
            else:
                print(f"파일 없음: {src_path}")
    except Exception as e:
        print(f"파일 복사 오류 발생: {str(e)}")


            

if __name__ == '__main__':
    main()