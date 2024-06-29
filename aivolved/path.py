import os


def _is_image(f: str) -> bool:
    return any(f.lower().endswith(ext) for ext in {".png", ".jpg", ".jpeg"})


def get_image_sources(root: str, n: int | None = None) -> list[str]:
    files = os.listdir(root)
    files = map(lambda f: os.path.join(root, f), files)
    files = list(filter(_is_image, files))

    if n is not None:
        files = files[:min(len(files), n)]
    return files


__all__ = ["get_image_sources"]

