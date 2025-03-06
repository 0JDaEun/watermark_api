import os
import sys
import torch
from app.models.fgsm import fgsm_attack
from app.utils.image_processing import load_image, save_image
from app.utils.model_utils import generate_noise

# 작업 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)  # 모듈 경로 추가

# 전역 변수 설정
INPUT_DIR = os.path.join(BASE_DIR, 'archive')  # 원본 데이터셋 디렉토리
OUTPUT_DIR = os.path.join(BASE_DIR, 'noise_archive')  # 노이즈 추가된 데이터셋 디렉토리
WATERMARK_STRENGTH = 0.1

def generate_noise(shape):
    """랜덤 노이즈 생성"""
    return torch.randn(shape)

def apply_watermark(image_path, output_path):
    """이미지에 노이즈를 추가하고 저장"""
    try:
        print(f"Processing: {image_path}")
        image = load_image(image_path)  # 이미지 로드
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
        
        print(f"Image loaded: {image.shape}")
        
        watermark = generate_noise(image.shape)  # 노이즈 생성
        perturbed_image = image + WATERMARK_STRENGTH * watermark  # 노이즈 추가
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # 값 제한
        
        save_image(perturbed_image, output_path)  # 이미지 저장
        print(f"Saved watermarked image to: {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def process_directory(input_dir, output_dir):
    """디렉터리 내 모든 이미지를 처리"""
    for root, dirs, files in os.walk(input_dir):  # 모든 하위 디렉토리를 순회
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                
                # 출력 디렉토리가 없으면 생성
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 파일명에 '_noise' 추가
                output_file_name = f"{os.path.splitext(file)[0]}_noise{os.path.splitext(file)[1]}"
                output_file_path = os.path.join(os.path.dirname(output_path), output_file_name)
                
                print(f"Input Path: {input_path}")
                print(f"Output Path: {output_file_path}")
                
                apply_watermark(input_path, output_file_path)

if __name__ == '__main__':
    print("Starting to add noise to the dataset...")
    
    # train 디렉터리 처리
    train_input = os.path.join(INPUT_DIR, 'train')
    train_output = os.path.join(OUTPUT_DIR, 'train')
    if os.path.exists(train_input):
        process_directory(train_input, train_output)
    
    # val 디렉터리 처리
    val_input = os.path.join(INPUT_DIR, 'val')
    val_output = os.path.join(OUTPUT_DIR, 'val')
    if os.path.exists(val_input):
        process_directory(val_input, val_output)
    
    print("Noise addition process completed.")
