import os
import argparse
from omegaconf import OmegaConf

from datasets import DataLoaderFactory
from trainers import TrainerFactory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path to the trained weight",
    )
    args = parser.parse_args()

    config_path = os.path.join(args.log_path, 'config.yaml')
    cfg = OmegaConf.load(config_path)

    dataloader = DataLoaderFactory.get_loader(cfg.dataset.name)
    val_dataset = dataloader(split='val', **cfg.dataset)

    # this will only be used by nerfw
    if cfg.trainer in ["nerfw", "nsff"]:
        cfg.N_vocab = val_dataset.__len__()  # set vocab size
        # note that here appearance and transient embeddings are per-image
        # so N_vocab = number of images in the training set and since it is
        # only applied during training, val images do not have embeddings

    weight_path = os.path.join(args.log_path, 'best.pth')

    trainer = TrainerFactory.get_trainer(cfg.trainer)(
        cfg, log_dir=args.log_path, create_log_folder=False)
    trainer.load_model(weight_path)
    trainer.inference(val_dataset)
