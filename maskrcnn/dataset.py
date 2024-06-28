import os
import json

import numpy as np

import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root, transforms=None, max_samples=None):
        self.root = root
        self.transforms = transforms

        # TODO why is there a sorted call?
        self.imgs = [f for f in sorted(os.listdir(root)) if f.endswith(".jpg") or f.endswith(".png")]
        
        if max_samples:
            self.imgs = self.imgs[:max_samples]  # TODO shouldn't this be seeded and randomised?
            
        with open(os.path.join(root, "_annotations.coco.json")) as f:
            self.annotations = json.load(f)
            
        self.image_id_to_annotations = self._get_image_id_to_annotations()

    def _get_image_id_to_annotations(self):
        image_id_to_annotations = {}
        for annotation in self.annotations["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in image_id_to_annotations:
                image_id_to_annotations[image_id] = []
            image_id_to_annotations[image_id].append(annotation)
        return image_id_to_annotations

    def _get_img_info(self, img_name):
        for img_info in self.annotations["images"]:
            if img_info["file_name"] == img_name:
                return img_info
        return None

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, img_name)
        img_info = self._get_img_info(img_name)
        img_id = img_info["id"]

        img = Image.open(img_path).convert("RGB")

        anns = self.image_id_to_annotations.get(img_id, [])

        boxes = []
        labels = []
        masks = []

        for obj in anns:
            xmin, ymin, width, height = obj["bbox"]
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(obj["category_id"])
            mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
            if obj["segmentation"]:
                for seg in obj["segmentation"]:
                    if isinstance(seg, list):
                        poly = np.array(seg).reshape((len(seg) // 2, 2))
                        poly = poly.astype(int)
                        cv2.fillPoly(mask, [poly], 1)
                    else:
                        raise ValueError("Segmentation format not supported.")
            masks.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        target = {"boxes": boxes, "labels": labels, "masks": masks}

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


__all__ = ["CustomDataset"]
