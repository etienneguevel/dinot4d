import os
import shutil

from pathlib import Path


HERE_DIR = Path(__file__).parent
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "logs" / "memory_probing"


def main():
    for exp in os.listdir(DATA_DIR):
        if "metrics.csv" in os.listdir(DATA_DIR / exp):
            shutil.copy(DATA_DIR / exp / "metrics.csv", HERE_DIR / (exp + "_metrics.csv"))

        else:
            print(f"No metrics file found for {exp}")


if __name__ == "__main__":
    main()
