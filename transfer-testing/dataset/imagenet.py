import lightning.pytorch as pl
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

from pathlib import Path


class ImagenetDataModule(pl.LightningDataModule):
    def __init__(
        self, dataset: str = "tiny-imagenet-200", data_dir: str = "data"
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset

        self.traindir = Path(self.data_folder, self.dataset, "train")
        self.valdir = Path(self.data_folder, self.dataset, "val")
        self.valdir = Path(self.data_folder, self.dataset, "val")

        self.transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def prepare_data(self):
        # TODO: move code from utils here
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.imagenet_train = torchvision.datasets.ImageFolder(
                self.data_dir, transform=self.transform
            )
            self.imagenet_val = torchvision.datasets.ImageFolder(
                self.data_dir, transform=self.transform
            )

        if stage == "test":
            self.imagenet_val = torchvision.datasets.ImageFolder(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.imagenet_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.imagenet_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.imagenet_test, batch_size=32)
