# watermark_api/app/config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'data', 'models')
SAMPLE_DIR = os.path.join(BASE_DIR, 'data', 'sample_images')

DISCRIMINATOR_PATH = os.path.join(MODEL_DIR, 'gan_discriminator.pth')

LATENT_DIM = 100
EPSILON = 0.01
