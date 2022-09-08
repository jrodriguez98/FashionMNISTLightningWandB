import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize
import torch.utils.data as data
import torchmetrics
from pytorch_lightning.loggers import WandbLogger

from models.model import SimpleNN


class LitNN(pl.LightningModule):
    def __init__(
            self,
            num_classes: int = 10,
            lr: float = 1e-3,
            wd: float = 1e-6,
            optimizer='adam',
            scheduler='cosine'
    ):
        """ Here we define computations """
        super(LitNN, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters in WandB

        self.classifier = SimpleNN(num_classes)  # Create nn.Module

        self.lr = lr
        self.wd = wd
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Prepare metrics
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        """ Should define the practical use of the module/model """
        logits = self.classifier(x)
        return logits

    def training_step(self, batch, batch_idx):
        """ Should define the training loop operations """
        x, y = batch
        logits = self.classifier(x)

        # Compute metrics
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)

        # Log metrics
        self.log('train/loss', loss)
        self.log('train/acc', self.train_acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.classifier(x)

        # Compute metrics
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)

        # Log metrics
        self.log('val/loss', loss)
        self.log('val/acc', self.val_acc, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.classifier(x)

        # Compute metrics
        loss = F.cross_entropy(logits, y)
        self.test_acc(logits, y)

        # Log metrics
        self.log('test/loss', loss)
        self.log('test/acc', self.test_acc)
        return loss

    def configure_optimizers(self):
        """ Define optimizers and LR schedulers """
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.lr, weight_decay=self.wd)

        if self.scheduler == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        if self.scheduler == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


class FashionMNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', batch_size: int = 64, train_split: float = 0.8):
        super(FashionMNISTDataModule, self).__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_split = train_split
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self) -> None:
        """ Used for one time preparation processes like downloading, tokenizing... """
        datasets.FashionMNIST(self.data_dir, train=True, download=True)
        datasets.FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None) -> None:
        """ Used for preparation processes you might to perform in every node """
        if stage in ('fit', 'validate', None):
            full_set = datasets.FashionMNIST(self.data_dir, train=True, transform=self.transform)
            train_size = int(self.train_split * len(full_set))
            val_size = len(full_set) - train_size
            self.train_set, self.val_set = data.random_split(full_set, [train_size, val_size])

        if stage in ('test', None):
            self.test_set = datasets.FashionMNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, 128)

    def test_dataloader(self):
        return DataLoader(self.test_set, 128)


if __name__ == '__main__':
    data_module = FashionMNISTDataModule(data_dir='./data', batch_size=64, train_split=0.8)

    model = LitNN()  # Create LigthningModule

    wandb_logger = WandbLogger(project='fMNIST-lightning')  # Create WandBLogger
    wandb_logger.watch(model)  # log gradients and model topology

    trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)  # Create trainer

    trainer.fit(model, datamodule=data_module)  # Train the model

    trainer.test(model, datamodule=data_module)  # Test the model
