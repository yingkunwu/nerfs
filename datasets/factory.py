from abc import abstractmethod
from torchvision import transforms as T


class DataLoader:
    """Abstract base class for data loaders."""

    def __init__(self):
        self.transform = T.ToTensor()

    @abstractmethod
    def load_data(self):
        pass


class DataLoaderFactory:
    @staticmethod
    def get_loader(dataset_name: str) -> DataLoader:
        """Factory method to create data loaders based on input."""
        dataset_name = dataset_name.lower()
        if dataset_name == "blender":
            from datasets.blender import BlenderDataLoader
            return BlenderDataLoader
        if dataset_name == "tanks_and_temples":
            from datasets.tanks_and_temples import TNTDataLoader
            return TNTDataLoader
        if dataset_name == "phototourism":
            from datasets.phototourism import PhototourismDataLoader
            return PhototourismDataLoader
        else:
            raise ValueError(f"Unknown dataset type: {dataset_name}")
