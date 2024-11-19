import os

import cv2
import torch
import numpy as np
import albumentations as A
from tqdm import tqdm
from consts import DEVICE


def train_model(model, epochs, train_dataloader, valid_dataloader, learning_rate):
    """
    :param model: модель нерйонной сети
    :param epochs: колчиество эпох обучения
    :param train_dataloader:
    :param valid_dataloader:
    :param learning_rate:
    :return: значения аккуратности и потерь по каждой эпохе
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    train_accuracy_list = []
    valid_accuracy_list = []
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        valid_losses = []
        train_predictions = []
        valid_predictions = []
        train_labels = []
        valid_labels = []
        for images, labels in tqdm(train_dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            out = model(images)
            loss = criterion(out, labels)
            train_losses.append(loss.item())
            train_predictions.extend(torch.argmax(out, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        for images, labels in valid_dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            out = model(images)
            valid_predictions.extend(torch.argmax(out, dim=1).cpu().numpy())
            valid_labels.extend(labels.cpu().numpy())
            loss = criterion(out, labels)
            valid_losses.append(loss.item())

        train_accuracy = np.round(np.mean(np.array(train_predictions) == torch.argmax(torch.tensor(train_labels), dim=1).cpu().numpy()), 3)
        valid_accuracy = np.round(np.mean(np.array(valid_predictions) == torch.argmax(torch.tensor(valid_labels), dim=1).cpu().numpy()), 3)
        train_loss = np.round(np.mean(train_losses), 3)
        valid_loss = np.round(np.mean(valid_losses), 3)

        train_accuracy_list.append(train_accuracy)
        valid_accuracy_list.append(valid_accuracy)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        print(f"epoch {epoch + 1} loss_train: {train_loss} loss_valid: {valid_loss}, "
              f"accuracy : {np.round(valid_accuracy, 3)}")
    return train_accuracy_list, valid_accuracy_list, train_loss_list, valid_loss_list


def predict_image(model, train_dataset, image_path):
    """"data/test/Golden Orb Weaver/1.jpg"""
    class_name = image_path.split("\\")[-2]
    test_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
    ])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = test_transforms(image=image)["image"]
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)[None]
    model.eval()
    predict = model(image.to(DEVICE))
    label_no = torch.argmax(predict, dim=1).cpu().numpy()[0]
    class_dict = {v: k for k, v in train_dataset.dict_name_no.items()}

    print(f"true label: {class_name} predict: {class_dict[label_no]}")
    return class_name == class_dict[label_no]


def calculate_accuracy(model, dataloader):
    model.eval()
    all_valid_predictions = []
    all_real_labels = []
    index = 0
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        out = model(images)
        all_valid_predictions.extend(torch.argmax(out, dim=1).cpu().numpy())
        all_real_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())

        index += 1
        print(index)
    accuracy = np.round(np.mean(np.array(all_valid_predictions) == np.array(all_real_labels)), 3)
    return accuracy


def calculate_accuracy_for_test_data(model, train_dataset):
    """временный метод проверки точности на тестовом наборе данных"""
    rootdir = 'D:\\Projects\\MasterNeuralNetwork\\data\\test\\'
    count = 0
    trues = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            res = predict_image(model, train_dataset, os.path.join(subdir, file))
            count += 1
            trues = trues + 1 if res else trues
    return trues / count
