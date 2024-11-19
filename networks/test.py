# import numpy as np
# import torch
# import torchvision
# from tqdm import tqdm
# from dataset import ImageDataset
# from rbf_layer import RBFLayer, rbf_gaussian, l_norm
# from consts import DEVICE, TRAIN_PATH, VALID_PATH, BATCH_SIZE, TRANSFORMS
# from utils import calculate_accuracy, train_model
#
#
# def train_test(learning_rate, epochs):
#     """Res-Net-34 RBF"""
#     train_dataset = ImageDataset(TRAIN_PATH, transform=TRANSFORMS)
#     valid_dataset = ImageDataset(VALID_PATH, transform=TRANSFORMS, mode="train")
#     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
#
#     model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
#     model.fc = RBFLayer(in_features_dim=512,
#                         num_kernels=150,
#                         out_features_dim=len(train_dataset.all_folder_name),
#                         radial_function=rbf_gaussian,
#                         norm_function=l_norm,
#                         normalization=True)
#     model.to(DEVICE)
#     # model.load_state_dict(torch.load("ready_models/model_1800.pth", torch.device(DEVICE)))
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = torch.nn.CrossEntropyLoss()
#
#     train_accuracy_list = []
#     valid_accuracy_list = []
#     train_loss_list = []
#     valid_loss_list = []
#     for epoch in range(epochs):
#         model.train()
#         train_losses = []
#         valid_losses = []
#         train_predictions = []
#         valid_predictions = []
#         train_labels = []
#         valid_labels = []
#         for images, labels in tqdm(train_dataloader):
#             images = images.to(DEVICE)
#             labels = labels.to(DEVICE)
#             labels = torch.argmax(labels, dim=1)
#             out = model(images)
#             loss = criterion(out, labels)
#             train_losses.append(loss.item())
#             train_predictions.extend(torch.argmax(out, dim=1).cpu().numpy())
#             train_labels.extend(labels.cpu().numpy())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         model.eval()
#         for images, labels in valid_dataloader:
#             images = images.to(DEVICE)
#             labels = labels.to(DEVICE)
#             labels = torch.argmax(labels, dim=1)
#             out = model(images)
#             valid_predictions.extend(torch.argmax(out, dim=1).cpu().numpy())
#             valid_labels.extend(labels.cpu().numpy())
#             loss = criterion(out, labels)
#             valid_losses.append(loss.item())
#
#         train_accuracy = np.round(
#             np.mean(np.array(train_predictions) == torch.argmax(torch.tensor(train_labels)).cpu().numpy()), 3)
#         valid_accuracy = np.round(
#             np.mean(np.array(valid_predictions) == torch.argmax(torch.tensor(valid_labels)).cpu().numpy()), 3)
#         train_loss = np.round(np.mean(train_losses), 3)
#         valid_loss = np.round(np.mean(valid_losses), 3)
#
#         train_accuracy_list.append(train_accuracy)
#         valid_accuracy_list.append(valid_accuracy)
#         train_loss_list.append(train_loss)
#         valid_loss_list.append(valid_loss)
#         print(f"epoch {epoch + 1} loss_train: {train_loss} loss_valid: {valid_loss}, "
#               f"accuracy : {np.round(valid_accuracy, 3)}")
#
#     torch.save(model.state_dict(), f'ready_models/model_test_rbf_{epochs}.pth')
#
#
# def check_test(epochs):
#     """проверка готовой модели RBF"""
#     train_dataset = ImageDataset(TRAIN_PATH, transform=TRANSFORMS)
#     test_dataset = ImageDataset(TRAIN_PATH, transform=TRANSFORMS)
#     test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
#
#     model = torchvision.models.densenet121(weights=torchvision.models.De.IMAGENET1K_V1)
#     model.fc = RBFLayer(in_features_dim=512,
#                         num_kernels=150,
#                         out_features_dim=len(train_dataset.all_folder_name),
#                         radial_function=rbf_gaussian,
#                         norm_function=l_norm,
#                         normalization=True)
#     model.to(DEVICE)
#     model.load_state_dict(torch.load(f"ready_models/model_test_rbf_{epochs}.pth", map_location=torch.device(DEVICE)))
#     result = calculate_accuracy(model, test_dataloader)
#     print(result)

import numpy as np
import torch
import torchvision

from dataset import ImageDataset
from rbf_layer import RBFLayer, rbf_gaussian, l_norm
from consts import DEVICE, TRAIN_PATH, VALID_PATH, TRANSFORMS
from utils import calculate_accuracy, train_model

BATCH_SIZE = 150

def train_test(learning_rate, epochs):
    """Res-Net-34 RBF"""
    train_dataset = ImageDataset(TRAIN_PATH, transform=TRANSFORMS)
    valid_dataset = ImageDataset(VALID_PATH, transform=TRANSFORMS, mode="train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    model.fc = RBFLayer(in_features_dim=2048,
                        num_kernels=150,
                        out_features_dim=len(train_dataset.all_folder_name),
                        radial_function=rbf_gaussian,
                        norm_function=l_norm,
                        normalization=True)
    model.to(DEVICE)
    # model.load_state_dict(torch.load("ready_models/model_1800.pth", torch.device(DEVICE)))

    train_accuracy, valid_accuracy, train_loss, valid_loss = train_model(model, epochs, train_dataloader,
                                                                         valid_dataloader, learning_rate)

    torch.save(model.state_dict(), f'ready_models/model_resnext50_32x4d_rbf_{epochs}.pth')


def check_test(epochs):
    """проверка готовой модели RBF"""
    train_dataset = ImageDataset(TRAIN_PATH, transform=TRANSFORMS)
    test_dataset = ImageDataset(TRAIN_PATH, transform=TRANSFORMS)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = torchvision.models.a(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = RBFLayer(in_features_dim=2048,
                        num_kernels=150,
                        out_features_dim=len(train_dataset.all_folder_name),
                        radial_function=rbf_gaussian,
                        norm_function=l_norm,
                        normalization=True)
    model.to(DEVICE)
    model.load_state_dict(torch.load(f"ready_models/model_resnext50_32x4d_rbf_{epochs}.pth", map_location=torch.device(DEVICE)))
    result = calculate_accuracy(model, test_dataloader)
    print(result)
