import os
import random
import pandas as pd

from typing import List, Optional, Union
from pathlib import PosixPath
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from .decoders import ImageDataDecoder


class ImageDataset(Dataset):
    """
    Class to load images from a directory or a list of directories.
    This dataset is made for Self-Supervised Learning tasks, as no labels are loaded.

    Args:
        root: str or list of str, the path to the directory or a list of directories where the images are stored.
        transform: callable, a function/transform.
        path_preserved: list of str or str, the path or list of paths that will be preserved for validation.
        frac: float, the fraction of images that will be preserved for validation.
        is_valid: bool, if True, the images will be checked for validity.

    Returns:
        image: torch.Tensor, the image at the given index.
    """

    def __init__(
        self,
        root: Union[str, List[str], PosixPath],
        transform=None,
        path_preserved: Optional[Union[List[str], ListConfig[str], str]] = None,
        frac: float = 0.1,
        is_valid: bool = True,
    ):
        self.root = root
        self.transform = transform
        self.path_preserved = list(path_preserved) if path_preserved else []
        self.frac = frac
        self.preserved_images = []
        self.is_valid = is_valid
        self.images_list = self._get_images(root)

    def _get_images(self, path: str):
        """
        Function to retrieves images from a directory or a list of directories.
        """
        images = []
        match path:
            case str() | PosixPath():
                p = path
                preserve = p in self.path_preserved
                try:
                    images.extend(
                        self._retrieve_from_path(p, preserve=preserve, frac=self.frac, is_valid=self.is_valid)
                    )
                except OSError:
                    print(f"the path indicated at {p} cannot be found.")

            case list() | ListConfig():
                for p in path:
                    images.extend(self._get_images(p))

            case _:
                raise SyntaxError("The entry is neither a list or a str")

        return images

    def _retrieve_from_path(self, path: str, is_valid: bool = True, preserve: bool = False, frac: float = 0.1):
        """
        Function to retrieve images from a directory. If there are subdirectories, the function will retrieve the images
        from them as well.

        Args:
            path: str, the path to the directory.
            is_valid: bool, if True, the images will be checked for validity.
            preserve: bool, if True, a fraction of the images of each subdirectories will be preserved for validation.
            frac: float, the fraction of images that will be preserved for validation.
        """
        images_ini = len(self.preserved_images)
        images = []
        for root, _, files in os.walk(path):
            images_dir = []
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
                    im = os.path.join(root, file)
                    if is_valid:
                        try:
                            _ = self._get_image_data(im)
                            images_dir.append(im)

                        except OSError:
                            print(f"Image at path {im} could not be opened.")

                    else:
                        images_dir.append(im)

            if preserve:
                random.seed(24)
                random.shuffle(images_dir)
                split_index = int(len(images_dir) * frac)
                self.preserved_images.extend(images_dir[:split_index])
                images.extend(images_dir[split_index:])

            else:
                images.extend(images_dir)

        images_end = len(self.preserved_images)
        if preserve:
            print(f"{images_end - images_ini} images have been retrieved for the dataset at path {path}")

        return images

    def _get_image_data(self, path: str):
        with open(path, "rb") as f:
            image_data = f.read()

        image = ImageDataDecoder(image_data).decode()
        return image

    def __len__(self) -> int:
        return len(self.images_list)

    def __getitem__(self, index: int):
        try:
            path = self.images_list[index]
            image = self._get_image_data(path)

        except Exception as e:
            raise RuntimeError(f"Can nor read image for sample {index}") from e
        if self.transform is not None:
            image = self.transform(image)

        return image


class LabelledDataset(Dataset):
    """
    Class to load images from a directory.
    This dataset is made for Supervised Learning tasks, as labels are expected.

    Args:
        data_path: str or list of str, the path to the directory or a csv file indicating where the images are at.
        root: Optional[str], if paths are not absolutes, the root directory where the images are stored.
        transform: callable, a function/transform.

    Returns:
        image: torch.Tensor, the image at the given index.
    """

    def __init__(
        self,
        data_path: str,
        root: Optional[str] = None,
        transform=None,
    ):
        self.root = root
        self.transform = transform
        self.images_list, self.labels = self._get_images_and_labels(data_path)
        self.translate_dict = {}

    def _get_images_and_labels(self, data_path: str) -> tuple[list]:
        match data_path:
            case str() | PosixPath() as s if str(s).endswith(".csv"):
                df = pd.read_csv(data_path)
                images_list, labels = df["names"].tolist(), df["pseudo_labels"].tolist()

                if self.root:
                    images_list = [os.path.join(self.root, im.split("/")[-1]) for im in images_list]

            case str() | PosixPath() as s if os.path.isdir(s):
                images_list, labels = [], []
                folders = [e for e in os.listdir(s) if os.path.isdir(os.path.join(s, e))]
                for f in folders:
                    images = [
                        os.path.join(s, f, im)
                        for im in os.listdir(os.path.join(s, f))
                        if im.endswith((".png", ".jpg", ".jpeg", ".tiff"))
                    ]
                    images_list.extend(images)
                    labels.extend([f] * len(images))

            case _:
                raise SyntaxError("The data_path format isn't recognized.")

        return images_list, labels

    def _get_image_data(self, path: str):
        with open(path, "rb") as f:
            image_data = f.read()

        image = ImageDataDecoder(image_data).decode()
        return image

    def __len__(self) -> int:
        return len(self.images_list)

    def __getitem__(self, index: int):
        try:
            path = self.images_list[index]
            image = self._get_image_data(path)
            label = self.translate_dict[self.labels[index]]

        except Exception as e:
            raise RuntimeError(f"can not read image @ {path}") from e

        if self.transform is not None:
            image = self.transform(image)

        return image, label
