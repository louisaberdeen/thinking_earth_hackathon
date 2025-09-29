import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import lightning as L
from torchmetrics import F1Score
from data_utils import FixedDataModule
from DOFAClassifier import DOFAClassifier
from sklearn.utils.class_weight import compute_class_weight
from ResNet import ResNetClassifier


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    # Create DataModule with fixed dataset
    dm = FixedDataModule(num_workers=8, batch_size=64)

    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    model = ResNetClassifier.load_from_checkpoint(
        f"lightning_logs\\version_2\\checkpoints\\epoch=4-step=21980.ckpt",
        class_weights=class_weights,
    )
    model.eval()

    trainer = L.Trainer(
        devices=1,
        accelerator="gpu",
        logger=False,
        enable_checkpointing=False,  # No need for checkpointing during validation
        enable_progress_bar=True,  # Show progress
    )

    val_results = trainer.validate(model, datamodule=dm)

    # Print results
    print("Validation results:", val_results)
