import torch
import lightning as L
from torchmetrics import F1Score
import torch.nn as nn


class SelfAttentionClassifier(nn.Module):
    """
    Self-attention classifier that treats input features as a sequence.
    """

    def __init__(
        self, input_dim=45, hidden_dim=128, num_heads=8, num_classes=7, dropout=0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Project input to hidden dimension
        self.input_projection = nn.Linear(1, hidden_dim)

        # Positional encoding for the 45 feature positions
        self.pos_encoding = nn.Parameter(torch.randn(input_dim, hidden_dim))

        # Multi-head self-attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer normalization and feedforward
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Reshape: (batch_size, 45) -> (batch_size, 45, 1)
        x = x.unsqueeze(-1)

        # Project to hidden dimension: (batch_size, 45, hidden_dim)
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.pos_encoding.unsqueeze(0)

        # Self-attention with residual connection
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.norm1(x + attn_output)

        # Feedforward with residual connection
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)

        # Global average pooling across sequence dimension
        x = x.mean(dim=1)  # (batch_size, hidden_dim)

        # Classification
        return self.classifier(x)


class SimpleClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(45, 7))

    def forward(self, x):
        return self.net(x)


class DOFAClassifier(L.LightningModule):
    def __init__(self, load_backbone=True, class_weights=None):
        super().__init__()
        self.wavelengths = [0.665, 0.56, 0.49]

        if load_backbone:
            try:
                from dofa_v1 import vit_base_patch16

                checkpoint = torch.load(
                    "models/DOFA_ViT_base_e100.pth", map_location="cpu"
                )
                self.backbone = vit_base_patch16()
                self.backbone.load_state_dict(checkpoint, strict=False)
                for param in self.backbone.parameters():
                    param.requires_grad = False
            except:
                print("Warning: Could not load DOFA backbone, using dummy")
                raise
        else:
            self.backbone = torch.nn.Identity()

        self.classifier = SelfAttentionClassifier()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
            self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            print("No class weights")
            self.criterion = torch.nn.CrossEntropyLoss()

        self.val_f1_macro = F1Score(task="multiclass", num_classes=7, average="macro")
        self.val_f1_weighted = F1Score(
            task="multiclass", num_classes=7, average="weighted"
        )
        self.val_f1_per_class = F1Score(task="multiclass", num_classes=7, average=None)

    def forward(self, x):
        features = self.backbone(x, self.wavelengths)
        if len(features.shape) == 3:
            features = features.mean(dim=1)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
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
