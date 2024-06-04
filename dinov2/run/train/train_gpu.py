import logging
import os
import sys
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dinov2.logging import setup_logging
from dinov2.train import get_args_parser as get_train_args_parser, SSLMetaArch
from dinov2.run.submit import get_args_parser
from dinov2.utils.config import get_cfg_from_args
from dinov2.distributed import setup_gpu


logger = logging.getLogger("dinov2")
warnings.filterwarnings("ignore", "xFormers is available")


class Trainer:
    def __init__(self, rank, args, world_size):
        self.rank = rank
        self.args = args
        self.world_size = world_size

    def __call__(self):
        from dinov2.train.train import do_train

        setup_gpu(self.rank, self.world_size)

        cfg = get_cfg_from_args(self.args)
        model = SSLMetaArch(cfg).to(torch.device(self.rank))
        model.prepare_for_distributed_training()
        logger.info("Model:\n{}".format(model))
        do_train(cfg, model)
        self._cleanup()

    def _cleanup():
        dist.destroy_process_group()


def train(rank, args, world_size):
    trainer = Trainer(rank, args, world_size)
    trainer()


def main():
    setup_logging()

    description = "Local launcher for DINOV2 training"
    train_args_parser = get_train_args_parser(add_help=False)
    args_parser = get_args_parser(description=description, parents=[train_args_parser])
    args = args_parser.parse_args()
    assert os.path.exists(args.config_file), "Configuration file does not exist!"

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(train, args=(args, WORLD_SIZE), nprocs=WORLD_SIZE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
