import logging
import os
import pytest

from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.configs import load_and_merge_config
from dinov2.logging import setup_logging


global logger
setup_logging(level=logging.INFO)
logger = logging.getLogger("dinov2")


def test_basic_loading():
    cfg = load_and_merge_config("tests/config")
    model = SSLMetaArch(cfg)
    logger.info("Model: \n{}".format(model))


def test_pretrained_loading():
    cfg = load_and_merge_config("train/vitl_dainobloom")
    if os.path.exists(cfg.student.pretrained_weights):
        model = SSLMetaArch(cfg)
        logger.info("Model: \n{}".format(model))

    else:
        warning_msg = "Path to the pretrained weights non-existent, test skipped."
        logger.warning(warning_msg)
        pytest.skip("Path to the pretrained weights non-existent, test skipped.")


if __name__ == "__main__":
    test_basic_loading()
    test_pretrained_loading()
