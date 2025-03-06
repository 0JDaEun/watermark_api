from flask import Flask, request, jsonify
import torch
import numpy as np
from app.models.fgsm import fgsm_attack
from app.utils.image_processing import load_image, save_image
from app.utils.model_utils import generate_noise
from app.utils.watermark_detection import detect_watermark, is_watermark_present
from app.config import LATENT_DIM, EPSILON
import os

app = Flask(__name__)

# 전역 변수 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'input')
RESULT_DIR = os.path.join(BASE_DIR, 'data', 'results')
WATERMARK_STRENGTH = 0.1

# 결과 디렉토리가 없으면 생성
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def generate_noise(shape):
    return torch.randn(shape)

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

        print("Generating watermark...")
        watermark = generate_noise(image.shape)
        print(f"Watermark generated successfully. Shape: {watermark.shape}")

        print("Applying watermark...")
        image.requires_grad = True
        perturbed_image = image + WATERMARK_STRENGTH * watermark
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        print("Watermark applied successfully")

        result_path = os.path.join(RESULT_DIR, f'watermarked_{image_name}')
        watermark_path = os.path.join(RESULT_DIR, f'watermark_{image_name}')
        
        print(f"Saving result to {result_path}")
        save_image(perturbed_image, result_path)
        save_image(watermark, watermark_path)
        print("Result and watermark saved successfully")

        return jsonify({'result': 'Watermark applied successfully', 'path': result_path})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_watermark', methods=['GET'])
def detect_watermark_route():
    image_name = 'sample1.jpg'
    original_path = os.path.join(INPUT_DIR, image_name)
    watermarked_path = os.path.join(RESULT_DIR, f'watermarked_{image_name}')
    watermark_path = os.path.join(RESULT_DIR, f'watermark_{image_name}')

    if not all(os.path.exists(path) for path in [original_path, watermarked_path, watermark_path]):
        return jsonify({'error': 'One or more required files not found'}), 404

    try:
        similarity = detect_watermark(original_path, watermarked_path, watermark_path)
        is_present = is_watermark_present(similarity)

        return jsonify({
            'watermark_detected': is_present,
            'similarity': similarity
        })
    except Exception as e:
        print(f"Error in watermark detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
