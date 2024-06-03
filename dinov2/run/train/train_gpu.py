import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dinov2.logging import setup_logging
from dinov2.train import get_args_parser as get_train_args_parser, SSLMetaArch
from dinov2.run.submit import get_args_parser
from dinov2.utils.config import get_cfg_from_args

logger = logging.getLogger("dinov2")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.barrier()


class Trainer:
    def __init__(self, rank, args, world_size):
        self.rank = rank
        self.args = args
        self.world_size = world_size

    def __call__(self):

        setup(self.rank, self.world_size)
        cfg = get_cfg_from_args(self.args)

        model = SSLMetaArch(cfg).to(torch.device(self.rank))
        model.prepare_for_distributed_training()
        logger.info("Model:\n{}".format(model))


def main():
    setup_logging()

    description = "Local launcher for DINOV2 training"
    train_args_parser = get_train_args_parser(add_help=False)
    args_parser = get_args_parser(description=description, parents=[train_args_parser])
    args = args_parser.parse_args()
    description = "Local launcher for DINOV2 training"
    train_args_parser = get_train_args_parser(add_help=False)
    args_parser = get_args_parser(description=description, parents=[train_args_parser])
    args = args_parser.parse_args()
    assert os.path.exists(args.config_file), "Configuration file does not exist!"

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(Trainer, args=[args], nprocs=WORLD_SIZE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
