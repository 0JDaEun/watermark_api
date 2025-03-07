import os
import torch
import numpy as np
from flask import Flask, jsonify, send_from_directory
from PIL import Image

app = Flask(__name__)

INPUT_DIR = 'input_images'  # 이미지가 저장된 디렉토리
RESULT_DIR = 'result_images'  # 처리된 이미지가 저장될 디렉토리
WATERMARK_STRENGTH = 0.5  # 워터마크 강도

# 이미지 로드 및 저장 함수
def load_image(image_path):
    image = Image.open(image_path)
    image = np.array(image) / 255.0  # 이미지 값 정규화
    return torch.tensor(image, dtype=torch.float32)

def save_image(image, path):
    # torch tensor -> numpy array -> PIL 이미지로 변환
    image = image.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)로 변환
    image = (image * 255).astype(np.uint8)  # 이미지 값 0~255로 변환
    image = Image.fromarray(image)
    image.save(path)

# 노이즈 생성 함수 (torch.randn 사용)
def generate_noise(image_shape):
    # 이미지 크기(shape)에 맞는 노이즈 생성 (정규 분포에서 샘플링)
    noise = torch.randn(image_shape)  # 표준 정규 분포에서 샘플링
    return noise

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the watermark API!"})

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/apply_watermark', methods=['GET'])
def apply_watermark():
    image_name = 'sample1.jpg'
    image_path = os.path.join(INPUT_DIR, image_name)

    print(f"Looking for image at: {image_path}")

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return jsonify({'error': f'Image {image_name} not found in {INPUT_DIR}'}), 404

    try:
        print("Loading image...")
        image = load_image(image_path)
        print(f"Image loaded successfully. Shape: {image.shape}")

        print("Generating noise...")
        noise = generate_noise(image.shape)
        print(f"Noise generated successfully. Shape: {noise.shape}")

        print("Applying noise...")
        perturbed_image = image + WATERMARK_STRENGTH * noise
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # 이미지 값이 0~1 범위에 있도록 제한
        print("Noise applied successfully")

        result_path = os.path.join(RESULT_DIR, f'noised_{image_name}')
        noise_path = os.path.join(RESULT_DIR, f'noise_{image_name}')
        
        print(f"Saving result to {result_path}")
        save_image(perturbed_image, result_path)
        save_image(noise, noise_path)
        print("Result and noise saved successfully")

        return jsonify({'result': 'Noise applied successfully', 'path': result_path})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
