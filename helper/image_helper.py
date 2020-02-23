import pathlib
import torch
from torchvision import transforms
from PIL import Image

class ImageTransform():
    def __init__(self, resize=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)

def transform_image(path, phase):
    current_dir = pathlib.Path(__file__).resolve().parent 
    img = Image.open(str(current_dir) + '/../data/data_new/train/' + path)
    if img.mode == 'L':
        img = img.convert('RGB')
    return ImageTransform()(img, phase)

def fetch_images(paths, phase):
    images = torch.Tensor(len(paths), 3, 224, 224)
    for i, path in enumerate(paths):
        images[i] = transform_image(path, phase)

    return images

