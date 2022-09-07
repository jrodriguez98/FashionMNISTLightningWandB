import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchvision import datasets
from torchvision.transforms import ToTensor
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define transformations
    training_trans = transforms.Compose([
        transforms.ToTensor()
    ])

    test_trans = transforms.Compose([
        transforms.ToTensor()
    ])

    # Define datasets
    train_set = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=training_trans
    )

    test_set = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=test_trans
    )

    # split the train set into two
    train_size = int(len(train_set) * 0.8)
    val_size = len(train_set) - train_size
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(train_set, [train_size, val_size], generator=seed)

    # Create DataLoaders
    train_loader = DataLoader(train_set, 32, False, num_workers=2)
    val_loader = DataLoader(valid_set, 128, False, num_workers=2)
    test_loader = DataLoader(test_set, 128, False, num_workers=2)

    model = LitNN()  # Create LigthningModule

    wandb_logger = WandbLogger(project='cifar-lightning')  # Create WandBLogger
    wandb_logger.watch(model)  # log gradients and model topology

    trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)  # Create trainer

    trainer.fit(model, train_loader, val_loader)  # Train the model

    # Test the model
    trainer.test(model, test_loader)
