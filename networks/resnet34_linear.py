import torch
import torchvision

from consts import DEVICE, TRAIN_PATH, VALID_PATH, BATCH_SIZE, TRANSFORMS
from dataset import ImageDataset
from utils import calculate_accuracy, train_model


def train_resnet_model_with_linear(learning_rate, epochs):
    """Res-Net-34 Linear"""
    train_dataset = ImageDataset(TRAIN_PATH, transform=TRANSFORMS)
    valid_dataset = ImageDataset(VALID_PATH, transform=TRANSFORMS, mode="train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(512, len(train_dataset.all_folder_name))
    model.to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    train_accuracy, valid_accuracy, train_loss, valid_loss = train_model(model, epochs, train_dataloader,
                                                                         valid_dataloader, learning_rate)

    torch.save(model.state_dict(), f'ready_models/model_resnet_linear_{epochs}.pth')


def check_resnet_model_with_linear(epochs):
    """проверка готовой модели LINEAR"""
    train_dataset = ImageDataset(TRAIN_PATH, transform=TRANSFORMS)
    test_dataset = ImageDataset(TRAIN_PATH, transform=TRANSFORMS)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(512, len(train_dataset.all_folder_name))
    model.to(DEVICE)
    model.load_state_dict(torch.load(f"ready_models/model_resnet_linear_{epochs}.pth", map_location=torch.device(DEVICE)))
    result = calculate_accuracy(model, test_dataloader)
    print(result)
