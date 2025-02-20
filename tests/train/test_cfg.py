import logging

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
    model = SSLMetaArch(cfg)
    logger.info("Model: \n{}".format(model))


if __name__ == "__main__":
    test_basic_loading()
    test_pretrained_loading()
