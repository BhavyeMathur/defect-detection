import numpy as np
from sklearn.model_selection import train_test_split

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

import os
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
    def __init__(self, folder: str, transform=None, select_channel: int | None =None):
        self.image_paths = list(map(lambda path: os.path.join(folder, path), os.listdir(folder)))

        self.transforms = transform
        self.mask_transform = transforms.RandomErasing(p=1, scale=(0.3, 0.3))
        self.select_channel = select_channel

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image = self.transforms(image)

        if self.select_channel:
            image = image[self.select_channel]

        t_image = self.mask_transform(image)
        return t_image, image


__all__ = ["split_dataset", "get_img_dataset_normalisation", "MaskedImageDataset"]
