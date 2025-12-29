import os
import random
import torch
import numpy as np
import argparse
from omegaconf import OmegaConf

from datasets import DataLoaderFactory
from trainers import TrainerFactory

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    # Create log dir and copy the config file
    basedir = cfg.basedir
    expname = cfg.expname
    logdir = os.path.join(basedir, expname)

    dataloader = DataLoaderFactory.get_loader(cfg.dataset)
    train_dataset = dataloader(root_dir=cfg.datadir, split='train')
    val_dataset = dataloader(root_dir=cfg.datadir, split='val')

    trainer = TrainerFactory.get_trainer(cfg.trainer)(cfg, logdir)
    trainer.fit(train_dataset, val_dataset)
