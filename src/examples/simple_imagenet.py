import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from pathlib import Path


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument("--lr", default=0.001, type=float, help="learning rate")

parser.add_argument("--momentum", default=0.9, type=float, help="momentum")

parser.add_argument("--epochs", default=10, type=int, help="number of epochs")

parser.add_argument("--batch_size", default=32, type=int, help="batch size")

parser.add_argument(
    "--num_workers", default=1, type=int, help="number of workers"
)

parser.add_argument(
    "--dataset", default="tiny-imagenet-200", type=str, help="dataset name"
)


class LitMobileNet(pl.LightningModule):
    def __init__(
        self,
        num_labels: int,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(1280, self.hparams.num_labels)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
        )
        return optimizer


if __name__ == "__main__":
    args = parser.parse_args()

    transformation = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.ToTensor(),
        ]
    )

    data_folder = Path(os.getenv("DATA_FOLDER", "data"))

    traindir = Path(data_folder, args.dataset, "train")
    valdir = Path(data_folder, args.dataset, "val")

    normalize = transforms.Normalize(
        ean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = LitMobileNet(
        num_labels=len(train_dataset.classes),
        learning_rate=args.lr,
        momentum=args.momentum,
    )

    trainer = pl.Trainer()
    trainer.fit(model, train_dataloader)
