from typing import Literal

import numpy as np
from ultralytics import YOLO

from PIL.Image import Image
from skimage import feature
from skimage import transform
from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# MacOS plotting fix
import sys

if sys.platform == "darwin":
    import matplotlib

    matplotlib.use('TkAgg')

TESTED_ANGLES = np.linspace(-np.pi / 4, np.pi / 4, 61, endpoint=False)


def crop_image(image: np.ndarray | Image, bbox: tuple[int, int, int, int], greyscale: bool = True) -> np.ndarray:
    x, y, w, h = bbox

    if isinstance(image, Image):
        cropped = image.crop((x, y, x + w, y + h))
        cropped = np.array(cropped)
    else:
        cropped = image[x:x + w, y:y + h]

    if greyscale:
        cropped = cropped.mean(axis=-1)
    return cropped


class ShampooThenConditionerResults:
    def __init__(self, yolo, classifications, crops):
        self.yolo = yolo
        self.classifications = classifications
        self.crops = crops

    def plot(self) -> plt.Figure:
        fig = plt.figure()
        plt.imshow(255 - self.yolo.orig_img)

        classifications = iter(self.classifications)

        for x0, y0, x1, y1 in self.yolo.boxes.xyxy:
            x0, y0, x1, y1 = map(lambda tensor: tensor.cpu(), (x0, y0, x1, y1))
            plt.gca().add_patch(
                patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  facecolor="none", edgecolor="red" if next(classifications) else "yellow"))

        plt.show()
        return fig


class ShampooThenConditioner:
    def __init__(self, path: str):
        self.model = YOLO(path)

    def __call__(self, src: str, gradient_filter: Literal["gradient", "laplacian", "sobel"] | None = None,
                 sigma: float = 2, low: float | None = None, high: float | None = None, threshold: float | None = 80):
        yolo = self.model(src, classes=[4], verbose=False)[0]
        classifications = []
        crops = []

        for x0, y0, x1, y1 in yolo.boxes.xyxy:
            x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
            w = x1 - x0
            h = y1 - y0

            crop = 255 - crop_image(yolo.orig_img, (y0, x0, h, w))
            defective = self._predict_defective(crop, sigma, low, high, threshold, gradient_filter)
            classifications.append(defective)
            crops.append(crop)

        return ShampooThenConditionerResults(yolo, classifications, crops)

    @staticmethod
    def _predict_defective(image: np.ndarray, sigma, low, high, threshold, gradient_filter) -> bool:
        if gradient_filter == "gradient":
            image = np.gradient(image)[1]
        elif gradient_filter == "laplacian":
            image = ndimage.laplace(image)
        elif gradient_filter == "sobel":
            image = ndimage.sobel(image, 1)
        elif gradient_filter is None:
            pass
        else:
            raise ValueError(f"Unknown gradient filter '{gradient_filter}'")

        image = feature.canny(image, sigma=sigma, low_threshold=low, high_threshold=high)
        hough = transform.hough_line(image, theta=TESTED_ANGLES)
        peaks = transform.hough_line_peaks(*hough, threshold=threshold, num_peaks=1)[0]

        return len(peaks) == 0

    def to(self, device):
        self.model.to(device)


__all__ = ["ShampooThenConditioner"]
