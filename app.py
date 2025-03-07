import os
import torch
import numpy as np
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

WATERMARK_STRENGTH = 0.5  # 워터마크 강도

# 이미지 로드 및 처리 함수
def process_image(image):
    image_np = np.array(image) / 255.0  # 이미지 값 정규화
    image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)로 변환
    return image_tensor

def apply_noise(image_tensor):
    noise = torch.randn_like(image_tensor)  # 이미지와 같은 shape의 노이즈 생성
    perturbed_image = image_tensor + WATERMARK_STRENGTH * noise
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # 이미지 값이 0~1 범위에 있도록 제한
    return perturbed_image

def tensor_to_pil(tensor):
    image = tensor.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)로 변환
    image = (image * 255).astype(np.uint8)  # 이미지 값 0~255로 변환
    return Image.fromarray(image)

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the watermark API!"})

@app.route('/apply_watermark', methods=['GET', 'POST'])
def apply_watermark():
    if request.method == 'GET':
        return jsonify({'message': 'This endpoint requires a POST request with an image file.'})
    elif request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        try:
            image = Image.open(image_file)
            image_tensor = process_image(image)
            
            perturbed_image_tensor = apply_noise(image_tensor)
            perturbed_image = tensor_to_pil(perturbed_image_tensor)
            
            output = io.BytesIO()
            perturbed_image.save(output, format='JPEG')
            output.seek(0)
            
            return send_file(output, mimetype='image/jpeg')
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
