import argparse
import csv
import logging
import os
import shutil
import yaml
from argparse import Namespace

import torch

from dinov2 import ROOT_DIR
from dinov2.train.train import do_train
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.utils.config import setup


logger = logging.getLogger("dinov2")


def get_args_parser():

    parser = argparse.ArgumentParser("DINO", add_help=False)

    # Choose the dataset
    parser.add_argument(
        "--dataset",
        type=str,
    )

    # Choose the output dir
    parser.add_argument(
        "--output_dir",
        type=str,
    )

    # Model parameters
    parser.add_argument(
        "--arch",
        default="vit_base_patch16_384",
        type=str,
        choices=["vit_tiny", "vit_small", "vit_base", "vit_large"],
    )

    # Choose the batch size
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
    )

    # Choose the number of epochs
    parser.add_argument(
        "--num_epochs",
        default=100,
        type=int,
    )

    # Indicate the number of GPUs used
    parser.add_argument(
        "--num_gpus",
        type=int,
    )

    return parser


def write_yaml(arch: str, batch_size: int, num_epochs: int, train_dataset: str, output_dir: str):

    cont = {
        "student": {
            "arch": arch,
        },
        "train": {
            "batch_size_per_gpu": batch_size,
            "OFFICIAL_EPOCH_LENGTH": num_epochs,
            "dataset_path": train_dataset,
            "path_preserved": [],
        },
        "optim": {
            "epochs": 1,
            "warmup_epochs": 0,
        },
    }

    output_path = output_dir / "config_added.yaml"
    with open(output_path, "w") as f:
        logger.info(f"Writing config file at {output_path}")
        yaml.dump(cont, f)

    return output_path


def main():
    # Get the arguments
    parser = get_args_parser()
    args = parser.parse_args()

    # Create the output_dir
    output_dir = ROOT_DIR / args.output_dir / f"{args.arch}_bs{args.batch_size}_gpus{args.num_gpus}"
    if os.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Writing to {output_dir}")

    # Create the config file with the arguments and save it
    config_path = write_yaml(args.arch, args.batch_size, args.num_epochs, args.dataset, output_dir)

    # Merge the yaml file with the default one
    args = Namespace(
        output_dir=output_dir,
        config_file=config_path,
        opts=[],
    )

    cfg = setup(args)

    # Load the model
    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    # logger.info("Model:\n{}".format(model))

    # Train the model
    metrics = do_train(cfg, model)

    # Save the file
    csv_file = output_dir / "metrics.csv"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


if __name__ == "__main__":
    main()
