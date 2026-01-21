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

    dataloader = DataLoaderFactory.get_loader(cfg.dataset.name)
    train_dataset = dataloader(split='train', **cfg.dataset)
    val_dataset = dataloader(split='val', **cfg.dataset)

    # this will only be used by nerfw
    if cfg.trainer in ["nerfw", "nsff"]:
        cfg.N_vocab = train_dataset.__len__()  # set vocab size
        # note that here appearance and transient embeddings are per-image
        # so N_vocab = number of images in the training set and since it is
        # only applied during training, val images do not have embeddings
    trainer = TrainerFactory.get_trainer(cfg.trainer)(cfg, logdir)
    trainer.fit(train_dataset, val_dataset)
