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
    parser.add_argument(
        "--mode",
        type=str,
        default="spiral",
        choices=["spiral", "fixtime", "fixview", "all"],
        help="NSFF only: 'spiral' = moving camera + time (default demo); "
             "'fixtime' = bullet time (freeze time, orbit camera); "
             "'fixview' = fixed camera, advancing time; 'all' = run all three",
    )
    parser.add_argument(
        "--t_fixed",
        type=int,
        default=None,
        help="fixtime mode: which time index to freeze (default: middle)",
    )
    parser.add_argument(
        "--view_idx",
        type=int,
        default=None,
        help="fixview mode: which camera pose to hold (default: middle)",
    )
    args = parser.parse_args()

    config_path = os.path.join(args.log_path, 'config.yaml')
    cfg = OmegaConf.load(config_path)

    dataloader = DataLoaderFactory.get_loader(cfg.dataset.name)
    train_dataset = dataloader(split='train', **cfg.dataset)
    val_dataset = dataloader(split='val', **cfg.dataset)

    # this will only be used by nerfw
    if cfg.trainer in ["nerfw", "nsff"]:
        cfg.N_vocab = train_dataset.__len__()  # set vocab size
        # note that here appearance and transient embeddings are per-image
        # so N_vocab = number of images in the training set and since it is
        # only applied during training, val images do not have embeddings

    weight_path = os.path.join(args.log_path, 'best.pth')

    trainer = TrainerFactory.get_trainer(cfg.trainer)(
        cfg, log_dir=args.log_path, create_log_folder=False)
    trainer.load_model(weight_path)

    if cfg.trainer == "nsff" and args.mode != "spiral":
        if args.mode in ("fixtime", "all"):
            trainer.inference_fixed_time(val_dataset, t_fixed=args.t_fixed)
        if args.mode in ("fixview", "all"):
            trainer.inference_fixed_view(val_dataset, view_idx=args.view_idx)
        if args.mode == "all":
            trainer.inference(val_dataset)
    else:
        trainer.inference(val_dataset)
