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
    def __init__(self, cfg, log_dir):
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
        self.log_dir = _reserve_version_dir(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        self.save_vis_path = os.path.join(self.log_dir, "vis")
        os.makedirs(self.save_vis_path, exist_ok=True)

        f = os.path.join(self.log_dir, 'config.yaml')
        OmegaConf.save(config=cfg, f=f)

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
        Save models into a single torch checkpoint.
        The checkpoint dict contains:
            - 'models': {name: state_dict}

        Example load:
            ckpt = torch.load(path)
            for name, sd in ckpt['models'].items():
                model = your_model_dict[name]
                model.load_state_dict(sd)
        """
        # Build checkpoint dict
        ckpt = {
            'models': {
                name: model.state_dict()
                for name, model in self.models.items()
            }
        }

        # Filename
        path = os.path.join(self.log_dir, "best.pth")
        # Save
        torch.save(ckpt, path)


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
        else:
            raise ValueError(f"Unknown dataset type: {trainer_name}")
