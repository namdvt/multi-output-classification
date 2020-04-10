import librosa
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.datasets import DatasetFolder
import os
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as tf


IMAGE_EXTENSION = '.jpg'

color_classes = ('black', 'blue', 'brown', 'green', 'red', 'white')
idx_to_color = dict(enumerate(color_classes))
color_to_idx = {char: idx for idx, char in idx_to_color.items()}

category_classes = ('dress', 'pants', 'shirt', 'shoes', 'shorts')
idx_to_category = dict(enumerate(category_classes))
category_to_idx = {char: idx for idx, char in idx_to_category.items()}



def default_loader(path):
    image = Image.open(path)
    return image


class ApparelImagesDataset(DatasetFolder):
    def __init__(self, root, loader=default_loader, transforms=None):
        self.root = root
        self.loader = loader
        self.samples = make_dataset(self.root, color_to_idx, category_to_idx)
        self.transform = transforms

    def __getitem__(self, index):
        path, target_color, target_category = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target_color, target_category

    def __len__(self):
        return len(self.samples)


def make_dataset(dir, color_class_to_idx, category_class_to_idx):
    images = []
    for f in os.listdir(dir):
        color, category = f.split('_')
        path = os.path.join(dir, f)
        for file_name in os.listdir(path):
            item = (os.path.join(path, file_name), color_class_to_idx[color], category_class_to_idx[category])
            images.append(item)

    return images


def collate_fn(batch):
    images = list()
    color = list()
    category = list()

    for b in batch:
        images.append(b[0])
        color.append(b[1])
        category.append(b[2])

    return torch.stack(images), torch.tensor(color), torch.tensor(category)


def get_loader(root, batch_size, shuffle):
    transforms = tf.Compose([
        tf.Resize([100, 100]),
        tf.ToTensor(),
        tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ApparelImagesDataset(root=root, transforms=transforms)
    num_train = int(len(dataset) * 0.9)
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              collate_fn=collate_fn,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn,
                            drop_last=True)
    return train_loader, val_loader
