import torchvision
import numpy as np
from sklearn.model_selection import train_test_split

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


__all__ = ["split_dataset", "get_img_dataset_normalisation"]
