import numpy as np
from sklearn.model_selection import train_test_split

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2 as transforms

import os
import random
from .path import get_image_sources, directory_from_files


def split_dataset(src: str, dst: str, label: str) -> None:
    sources = get_image_sources(f"{src}/{label}")
    train, test = train_test_split(sources, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    directory_from_files(f"{dst}/train/{label}", train)
    directory_from_files(f"{dst}/test/{label}", test)
    directory_from_files(f"{dst}/val/{label}", val)


def get_img_dataset_normalisation(dataset):
    data = np.array([np.array(img) for img, _ in dataset])
    mean = data.mean(axis=(0, 1, 2)) / 255
    stdev = data.std(axis=(0, 1, 2)) / 255
    return mean, stdev


class MaskedImageDataset(Dataset):
    def __init__(self, folder: str, transform=None, select_channel: int | None = None, fourier_domain: bool = False):
        self.image_paths = list(map(lambda path: os.path.join(folder, path), os.listdir(folder)))

        self.transforms = transform
        self.mask_transform = transforms.RandomErasing(p=1, scale=(0.3, 0.3))
        self.select_channel = select_channel
        self.fourier_domain = fourier_domain

        if self.fourier_domain:
            assert isinstance(self.select_channel, int)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])

        if self.transforms is not None:
            image = self.transforms(image)
        if self.select_channel is not None:
            image = image[self.select_channel]

        t_image = self.mask_transform(image)

        if self.fourier_domain:
            t_image_fft = np.fft.fft2(t_image)
            image_fft = np.fft.fft2(image)
            t_image = np.array([t_image, t_image_fft.real, t_image_fft.imag])
            image = np.array([image, image_fft.real, image_fft.imag])

        return t_image, image


class CocoBinaryClassificationDataset(Dataset):
    def __init__(self, root: str, classes: set[int], transform=None, select_channel: int | None = None, **kwargs):
        self.coco_dataset = torchvision.datasets.CocoDetection(root, root + "_annotations.coco.json", **kwargs)
        self.classes = classes

        self.transforms = transform
        self.select_channel = select_channel

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, index):
        image, boxes = self.coco_dataset[index]

        if self.transforms is not None:
            image = self.transforms(image)
        if self.select_channel is not None:
            image = image[self.select_channel]

        class_ = any(box["category_id"] in self.classes for box in boxes)
        return image, class_


class SiameseDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

        targets = np.array([s[1] for s in self.dataset])
        self.classes = tuple(set(targets))

        self.grouped_examples = {}
        for cls in self.classes:
            self.grouped_examples[cls] = np.where(targets == cls)[0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        selected_class = random.choice(self.classes)
        index_1 = np.random.choice(self.grouped_examples[selected_class])
        image_1, _ = self.dataset[int(index_1)]

        # positive example
        if index % 2 == 0:
            while (index_2 := np.random.choice(self.grouped_examples[selected_class])) == index_1:
                pass
            target = torch.tensor(1, dtype=torch.float)

        # negative example
        else:
            while (other_selected_class := random.choice(self.classes)) == selected_class:
                pass
            index_2 = np.random.choice(self.grouped_examples[other_selected_class])
            target = torch.tensor(0, dtype=torch.float)

        image_2, _ = self.dataset[int(index_2)]
        return image_1, image_2, target


__all__ = ["split_dataset", "get_img_dataset_normalisation",
           "MaskedImageDataset", "CocoBinaryClassificationDataset", "SiameseDataset"]
