from typing import Iterable
from tqdm import tqdm

import os
import shutil


def _is_image(f: str) -> bool:
    return any(f.lower().endswith(ext) for ext in {".png", ".jpg", ".jpeg", ".bmp"})


def get_image_sources(root: str, n: int | None = None) -> list[str]:
    files = os.listdir(root)
    files = map(lambda f: os.path.join(root, f), files)
    files = list(filter(_is_image, files))

    if n is not None:
        files = files[:min(len(files), n)]
    return files


def directory_from_files(dst: str, srcs: Iterable[str], verbose: bool = True) -> None:
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)

    for src in (tqdm(srcs) if verbose else srcs):
        shutil.copy(src, f"{dst}/{os.path.basename(src)}")


__all__ = ["get_image_sources", "directory_from_files"]

