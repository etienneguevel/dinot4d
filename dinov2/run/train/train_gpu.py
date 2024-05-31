import logging
import os
import sys

import torch
import torch.multiprocessing as mp
from dinov2.logging import setup_logging
from dinov2.train.train import main as train_main
from dinov2.run.submit import get_args_parser


logger = logging.getLogger("dinov2")


def main():
    setup_logging()

    args = get_args_parser.parse_args()
    assert os.path.exists(args.config_file), "Configuration file does not exist!"

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(train_main, nprocs=WORLD_SIZE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
