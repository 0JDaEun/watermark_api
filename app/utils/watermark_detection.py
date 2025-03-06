import torch
import torch.nn.functional as F
from app.utils.image_processing import load_image

def detect_watermark(original_image_path, watermarked_image_path, watermark_path):
    original_image = load_image(original_image_path)
    watermarked_image = load_image(watermarked_image_path)
    watermark = load_image(watermark_path)

    # 워터마크된 이미지와 원본 이미지의 차이 계산
    difference = watermarked_image - original_image

    # 차이와 워터마크 사이의 유사도 계산
    similarity = F.cosine_similarity(difference.flatten(), watermark.flatten(), dim=0)

    return similarity.item()

def is_watermark_present(similarity, threshold=0.3):
    return similarity > threshold
