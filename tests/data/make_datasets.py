from pathlib import Path
import shutil
import os
from collections import defaultdict


def main():
    base = Path("/home/manon/classification/data/Single_cells")

    dataset_path1 = Path(os.getcwd()) / "dataset1"
    dataset_path2 = Path(os.getcwd()) / "dataset2"

    os.makedirs(dataset_path1, exist_ok=True)
    os.makedirs(dataset_path2, exist_ok=True)

    for folder, path in zip(["barcelona", "rabin"], [dataset_path1, dataset_path2]):
        data = defaultdict()
        for dirpath, _, f in os.walk(Path(base) / folder):
            files = [im for im in f if im.endswith((".png", ".jpeg", ".jpg", ".tiff"))][:64]
            if (len(files) > 0) & (not "ipynb" in dirpath):
                data[Path(dirpath).name] = files
        print(path)
        for k, v in data.items():
            print(k)
            print(path / k)
            os.mkdir(path / k)
            for im in v:
                shutil.copy(base / folder / k / im, path / k / im)


if __name__ == "__main__":
    main()
