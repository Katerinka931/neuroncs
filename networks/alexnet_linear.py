import numpy as np
import torch
import torchvision
from tqdm import tqdm

from dataset import ImageDataset
from consts import DEVICE, TRAIN_PATH, VALID_PATH, BATCH_SIZE, TRANSFORMS
from utils import calculate_accuracy, train_model


def train_alexnet_model_with_linear(learning_rate, epochs):
    """ RBF"""
    train_dataset = ImageDataset(TRAIN_PATH, transform=TRANSFORMS)
    valid_dataset = ImageDataset(VALID_PATH, transform=TRANSFORMS, mode="train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    custom_output_layer = torch.nn.Linear(4096, len(train_dataset.all_folder_name))
    # Замените выходной слой на кастомный
    model.classifier = torch.nn.Sequential(
        *list(model.classifier.children())[:-1],  # Удалите последний слой
        custom_output_layer  # Добавьте кастомный слой
    )
    model.to(DEVICE)
    # model.load_state_dict(torch.load("ready_models/model_1800.pth", torch.device(DEVICE)))
    train_accuracy, valid_accuracy, train_loss, valid_loss = train_model(model, epochs, train_dataloader,
                                                                         valid_dataloader, learning_rate)

    torch.save(model.state_dict(), f'ready_models/model_alexnet_linear_{epochs}.pth')


def check_alexnet_model_with_linear(epochs):
    """проверка готовой модели RBF"""
    train_dataset = ImageDataset(TRAIN_PATH, transform=TRANSFORMS)
    test_dataset = ImageDataset(TRAIN_PATH, transform=TRANSFORMS)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    custom_output_layer = torch.nn.Linear(4096, len(train_dataset.all_folder_name))
    # Замените выходной слой на кастомный
    model.classifier = torch.nn.Sequential(
        *list(model.classifier.children())[:-1],  # Удалите последний слой
        custom_output_layer  # Добавьте кастомный слой
    )
    model.to(DEVICE)
    model.load_state_dict(torch.load(f"ready_models/model_alexnet_linear_{epochs}.pth", map_location=torch.device(DEVICE)))
    result = calculate_accuracy(model, test_dataloader)
    print(result)
