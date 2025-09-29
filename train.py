import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from data_utils import FixedDataModule
from DOFAClassifier import DOFAClassifier
import lightning as L
from sklearn.utils.class_weight import compute_class_weight
from ResNet import ResNetClassifier


if __name__ == "__main__":
    import argparse
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train DOFA Classifier")
    parser.add_argument(
        "-epochs",
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 10)",
    )
    args = parser.parse_args()

    # Create DataModule with fixed dataset
    dm = FixedDataModule(num_workers=4, batch_size=128, train_subset_fraction=0.1)
    dm.setup()

    # Test batch loading
    # print("Loading test batch")
    # batch = next(iter(dm.train_dataloader()))

    # import matplotlib.pyplot as plt

    # print(batch)
    # plt.imshow(batch["image"][0].permute(1, 2, 0).numpy())
    # 1 / 0

    # print(f"Batch loaded successfully: {batch['image'].shape}")

    # labels_list = [batch["label"] for batch in dm.train_dataloader()]

    # # Concatenate all batches
    # labels = torch.cat(labels_list)

    # class_weights = compute_class_weight(
    #     "balanced", classes=np.arange(7), y=labels.numpy()  # Extract from your dataset
    # )
    # class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = torch.tensor(
        [1.5958, 0.9386, 1.1660, 0.5595, 3.0744, 0.4618, 5.8110]
    )
    print(class_weights)
    # Create model
    # model = DOFAClassifier(load_backbone=True, class_weights=class_weights)
    try:
        model = ResNetClassifier.load_from_checkpoint(
            f"checkpoints\\best-model-epoch=04-val_loss=1.48.ckpt",
            class_weights=class_weights,
        )
        print("loaded last checkpoint")
    except:
        model = ResNetClassifier(class_weights=class_weights)
        print("bruh")

    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    # Create callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # metric to monitor
        min_delta=0.00,  # minimum change to qualify as improvement
        patience=10,  # number of checks with no improvement after which training stops
        verbose=True,  # print messages
        mode="min",  # "min" for loss, "max" for accuracy
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="best-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    # Train
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        check_val_every_n_epoch=5,
        callbacks=[early_stop_callback, checkpoint_callback],
        gradient_clip_val=1.0,
    )

    trainer.fit(model, dm)
    print("Training complete!")
