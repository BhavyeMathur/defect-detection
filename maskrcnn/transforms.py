import torch
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            _, height, width = image.shape  # Get dimensions of the tensor
            bbox = target["boxes"]
            bbox = torch.tensor(bbox)  # Convert bbox to torch.Tensor if not already
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # Corrected width access
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-1)
        return image, target


__all__ = ["Compose", "ToTensor", "RandomHorizontalFlip"]
