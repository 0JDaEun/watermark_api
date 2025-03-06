from PIL import Image
import torch
import torchvision.transforms as transforms

def load_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def save_image(tensor, path):
    image = transforms.ToPILImage()(tensor.squeeze(0))
    image.save(path)
