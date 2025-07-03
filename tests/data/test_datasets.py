import os
import torchvision.transforms as transforms

from pathlib import Path
from torch.utils.data import DataLoader
from dinov2.data.datasets import ImageDataset, LabelledDataset
from dinov2.data.transforms import make_classification_train_transform


def test_single_path():
    path_dataset_test = Path(__file__).parent / "dataset1"
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageDataset(path_dataset_test, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32)
    for i in dataloader:
        assert len(i) == 32
        break


def test_several_paths():
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.226]),
        ]
    )

    dirs = [Path(__file__).parent / d for d in ["dataset1", "dataset2"]]
    dataset = ImageDataset(root=dirs, transform=transform)
    expected_length = len(
        [
            f
            for d in dirs
            for _, _, files in os.walk(d)
            for f in files
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))
        ]
    )
    assert len(dataset) == expected_length, f"expected length is {expected_length}, dataset length is {len(dataset)}"
    dataloader = DataLoader(dataset, batch_size=32)
    for i in dataloader:
        assert len(i) == 32
        break


def test_labelled_dataset():
    path_dataset_test = Path(__file__).parent / "dataset1"
    transform = make_classification_train_transform()
    dataset = LabelledDataset(path_dataset_test, transform=transform)
    expected_length = len(
        [
            f
            for _, _, files in os.walk(path_dataset_test)
            for f in files
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))
        ]
    )
    assert len(dataset) == expected_length, f"expected length is {expected_length}, dataset length is {len(dataset)}"
    dataloader = DataLoader(dataset, batch_size=32)
    for ims, _ in dataloader:
        assert len(ims) == 32, f"batchsize should be 32, it is {len(ims)}"
        break


if __name__ == "__main__":
    test_single_path()
    test_several_paths()
    test_labelled_dataset()
