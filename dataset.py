import torch
import cv2
import glob
import os
import matplotlib.pyplot as plt


class ImageDataset:

    def __init__(self, path, mode="train", transform=None):
        self.path = path
        self.transform = transform
        self.mode = mode
        self.imgs_path = glob.glob(self.path)

        if mode == "train":
            sub_folder_path = self.path.split("*")[0]
            self.all_folder_name = os.listdir(sub_folder_path)
            self.dict_name_no = {
                self.all_folder_name[i]: i for i in range(len(self.all_folder_name))
            }

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        if self.mode == "train":
            class_no = self.dict_name_no[self.imgs_path[index].split("\\")[-2]]
            class_vector = torch.zeros(len(self.dict_name_no))
            class_vector[class_no] = 1

            return img, class_vector

        return img

    def __len__(self):
        return len(self.imgs_path)

    def show(self, index):
        img = cv2.imread(self.imgs_path[index])
        class_name = self.imgs_path[index].split("\\")[-2]
        plt.imshow(img)
        plt.title(class_name)
        plt.show()


if __name__ == "__main__":
    dataset = ImageDataset("data/train/*/*.jpg")
    # dataset.show(1000)
