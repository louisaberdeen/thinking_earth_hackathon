import torch
import lightning as L
from torchmetrics import F1Score
import torch.nn as nn


class SimpleClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(45, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 7),
        )

    def forward(self, x):
        return self.net(x)


class ResNetClassifier(L.LightningModule):
    def __init__(self, load_backbone=True, class_weights=None):
        super().__init__()

        self.model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50")

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 7),
        )

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
            self.criterion = torch.nn.CrossEntropyLoss(
                weight=self.class_weights, label_smoothing=0.1
            )
        else:
            print("No class weights")
            self.criterion = torch.nn.CrossEntropyLoss()

        self.val_f1_macro = F1Score(task="multiclass", num_classes=7, average="macro")
        self.val_f1_weighted = F1Score(
            task="multiclass", num_classes=7, average="weighted"
        )
        self.val_f1_per_class = F1Score(task="multiclass", num_classes=7, average=None)

    def forward(self, x):
        features = self.model(x)
        if len(features.shape) == 3:
            features = features.mean(dim=1)
        return features

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Calculate accuracy
        acc = (preds == y).float().mean()

        # Update F1 score metrics
        f1_macro = self.val_f1_macro(preds, y)
        f1_weighted = self.val_f1_weighted(preds, y)
        f1_per_class = self.val_f1_per_class(preds, y)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1_macro", f1_macro, prog_bar=True)
        self.log("val_f1_weighted", f1_weighted, prog_bar=True)

        # Log per-class F1 scores
        class_names = [
            "High",
            "Low",
            "Moderate",
            "Non-burnable",
            "Very_High",
            "Very_Low",
            "Water",
        ]
        for i, class_name in enumerate(class_names):
            self.log(f"val_f1_{class_name}", f1_per_class[i])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 5,  # Match check_val_every_n_epoch
            },
        }
