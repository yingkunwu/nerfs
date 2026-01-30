import os
import re
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod


def _reserve_version_dir(root: str):
    """
    Create a new subdir root/version_{n} where n is the next integer.
    Handles concurrent launches by retrying if the chosen dir was taken.
    """
    os.makedirs(root, exist_ok=True)
    while True:
        # scan existing version_* dirs
        max_v = -1
        for d in os.listdir(root):
            p = os.path.join(root, d)
            if os.path.isdir(p):
                m = re.fullmatch(r"version_(\d+)", d)
                if m:
                    max_v = max(max_v, int(m.group(1)))

        version = max_v + 1
        path = os.path.join(root, f"version_{version}")
        try:
            os.makedirs(path)
            return path
        except FileExistsError:
            # another process grabbed it; loop to pick the next
            continue


class BaseTrainer(ABC):
    def __init__(self, cfg, log_dir, create_log_folder=True):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embeddings, self.models = self.create_nerf(cfg)

        parameters = []
        for name, embed in self.embeddings.items():
            self.embeddings[name] = embed.to(self.device)
            if isinstance(embed, torch.nn.Embedding):
                parameters += list(self.embeddings[name].parameters())
        for name, model in self.models.items():
            self.models[name] = model.to(self.device)
            parameters += list(self.models[name].parameters())
        self.optimizer = torch.optim.Adam(
            parameters, lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=cfg.scheduler.decay_step,
            gamma=cfg.scheduler.decay_gamma
        )

        # Logging & output dirs
        if create_log_folder:
            self.log_dir = _reserve_version_dir(log_dir)
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)
            self.save_vis_path = os.path.join(self.log_dir, "vis")
            os.makedirs(self.save_vis_path, exist_ok=True)
            f = os.path.join(self.log_dir, 'config.yaml')
            OmegaConf.save(config=cfg, f=f)
        else:
            self.log_dir = log_dir

    @abstractmethod
    def create_nerf(self, cfg):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def fit(self, train_dataset, val_dataset):
        pass

    def save_model(self):
        """
        Save models and embeddings into a single torch checkpoint.
        The checkpoint dict contains:
            - 'models': {name: state_dict}
            - 'embeddings': {name: state_dict}

        Example load:
            ckpt = torch.load(path)
            for name, sd in ckpt['models'].items():
                model = your_model_dict[name]
                model.load_state_dict(sd)
            for name, sd in ckpt['embeddings'].items():
                embedding = your_embedding_dict[name]
                embedding.load_state_dict(sd)
        """
        # Build checkpoint dict
        ckpt = {
            'models': {
                name: model.state_dict()
                for name, model in self.models.items()
            },
            'embeddings': {
                name: embed.state_dict()
                for name, embed in self.embeddings.items()
                if isinstance(embed, torch.nn.Module)
            }
        }

        # Filename
        path = os.path.join(self.log_dir, "best.pth")
        # Save
        torch.save(ckpt, path)

    def load_model(self, weight_path):
        ckpt = torch.load(weight_path, map_location=self.device)
        for name, model in self.models.items():
            if name in ckpt['models']:
                try:
                    missing, unexpected = model.load_state_dict(
                        ckpt['models'][name], strict=True)
                except RuntimeError as e:
                    print("Warning: Ignore loading model "
                          f"{name} due to size mismatch."
                          f"\nActual Error message: {e}")
                    continue
            else:
                raise KeyError(f"Model '{name}' not found in checkpoint.")
            if missing:
                print(f"Model '{name}' missing keys: {missing}")
            if unexpected:
                print(f"Model '{name}' unexpected keys: {unexpected}")

        for name, embed in self.embeddings.items():
            if not isinstance(embed, torch.nn.Module):
                continue
            if name in ckpt['embeddings']:
                try:
                    missing, unexpected = embed.load_state_dict(
                        ckpt['embeddings'][name], strict=True)
                except RuntimeError as e:
                    print("Warning: Ignore loading embedding "
                          f"{name} due to size mismatch."
                          f"\nActual Error message: {e}")
                    continue
            else:
                raise KeyError(f"Embedding '{name}' not found in checkpoint.")
            if missing:
                print(f"Embedding '{name}' missing keys: {missing}")
            if unexpected:
                print(f"Embedding '{name}' unexpected keys: {unexpected}")

        print("Model weight loaded successfully!")


class TrainerFactory:
    @staticmethod
    def get_trainer(trainer_name: str) -> BaseTrainer:
        """Factory method to create trainers based on input."""
        trainer_name = trainer_name.lower()
        if trainer_name == "nerf":
            from trainers.nerf_trainer import NeRFTrainer
            return NeRFTrainer
        if trainer_name == "nerfplusplus":
            from trainers.nerfplus_trainer import NeRFPlusPlusTrainer
            return NeRFPlusPlusTrainer
        if trainer_name == "nerfw":
            from trainers.nerfw_trainer import NeRFWTrainer
            return NeRFWTrainer
        if trainer_name == "nsff":
            from trainers.nsff_trainer import NSFFTrainer
            return NSFFTrainer
        else:
            raise ValueError(f"Unknown trainer type: {trainer_name}")
