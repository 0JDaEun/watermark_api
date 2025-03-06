# watermark_api/app/utils/model_utils.py
import torch

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def generate_noise(image_shape):
    return torch.randn(image_shape)
