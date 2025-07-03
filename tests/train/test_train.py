from argparse import Namespace
import logging
import pytest
from pathlib import Path

import torch

from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.train.train import do_train
from dinov2.utils.config import setup


logger = logging.getLogger("dinov2")


def test_one_epoch():
    if torch.cuda.is_available():
        config_path = Path(__file__).parent / "config.yaml"
        output_path = Path(__file__).parent / "logs"

        # Make an args object
        args = Namespace(
            output_dir=output_path,
            config_file=config_path,
            opts=[],
        )

        # Setup
        cfg = setup(args)

        # Setup the correct paths for data loading and output
        dataset_path = Path(__file__).parent.parent / "data" / "dataset1"

        cfg.train.dataset_path = dataset_path
        cfg.train.output_dir = output_path

        # Setup the model
        model = SSLMetaArch(cfg).to(torch.device("cuda"))
        model.prepare_for_distributed_training()

        logger.info("Model:\n {}".format(model))
        do_train(cfg, model, resume=False)

    else:
        warning_msg = "No CUDA devices found, skipping test,"
        logger.warning(warning_msg)
        pytest.skip(warning_msg)


if __name__ == "__main__":
    test_one_epoch()
