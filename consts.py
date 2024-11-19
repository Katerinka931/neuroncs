# Константы
from enum import Enum

import torch
import albumentations as A

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cuda'
TRAIN_PATH = "data/train/*/*.jpg"
TEST_PATH = "data/test/*/*.jpg"
VALID_PATH = "data/valid/*/*.jpg"
BATCH_SIZE = 128

# Объединение несколько преобразований изображений в один конвейер
TRANSFORMS = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.1),
    A.GaussNoise(p=0.1),
    A.Resize(224, 224),
    A.Normalize(),
])

class ModelType(Enum):
    RESNET_RBF = 1
    RESNET_LINEAR = 2
    EFFICIENT = 3
    TEST = 4
    # ALEXNET_RBF = 3
    # ALEXNET_LINEAR = 4