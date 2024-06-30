from sklearn.model_selection import train_test_split

from .path import get_image_sources, directory_from_files


def split_dataset(src: str, dst: str, label: str) -> None:
    sources = get_image_sources(f"{src}/{label}")
    train, test = train_test_split(sources, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    directory_from_files(f"{dst}/train/{label}", train)
    directory_from_files(f"{dst}/test/{label}", test)
    directory_from_files(f"{dst}/val/{label}", val)


__all__ = ["split_dataset"]
