import logging
import pytest
from pathlib import Path

import torch
from omegaconf import OmegaConf


from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.train.train import do_train

logger = logging.getLogger("dinov2")


def test_one_epoch():
    if torch.cuda.is_available():
        cfg = OmegaConf.load(Path(__file__).parent / "config.yaml")

        # Setup the correct paths for data loading and output
        dataset_path = Path(__file__).parent.parent / "data" / "dataset1"
        output_path = Path(__file__).parent / "logs"

        cfg.train.dataset_path = dataset_path
        cfg.train.output_dir = output_path

        # Setup the model
        model = SSLMetaArch(cfg).to(torch.device("cuda"))

        logger.info("Model:\n {}".format(model))
        do_train(cfg, model, resume=False)
    else:
        warning_msg = "No CUDA devices found, skipping test,"
        logger.warning(warning_msg)
        pytest.skip(warning_msg)


if __name__ == "__main__":
    test_one_epoch()
