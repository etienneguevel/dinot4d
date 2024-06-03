import logging
import os
import sys

import torch
import torch.multiprocessing as mp
from dinov2.logging import setup_logging
from dinov2.train import main as train_main, get_args_parser as get_train_args_parser
from dinov2.run.submit import get_args_parser


logger = logging.getLogger("dinov2")

def wrapper(rank, args):
    train_main(*[args])

def main():
    setup_logging()

    description = "Local launcher for DINOV2 training"
    train_args_parser = get_train_args_parser(add_help=False)
    args_parser = get_args_parser(description=description, parents=[train_args_parser])
    args = args_parser.parse_args()
    assert os.path.exists(args.config_file), "Configuration file does not exist!"

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(wrapper, args=[args], nprocs=WORLD_SIZE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
