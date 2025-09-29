import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
import numpy as np


class FireRiskFixed(Dataset):
    """Custom FireRisk dataset that works with multiprocessing."""

    def __init__(self, root="data", split="train", transform=None):
        self.root = Path(root) / "FireRisk" / split
        self.transform = transform

        # Class mapping
        self.classes = [
            "High",
            "Low",
            "Moderate",
            "Non-burnable",
            "Very_High",
            "Very_Low",
            "Water",
        ]

        # Collect all image paths and labels
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.root / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((str(img_path), class_idx))

        if len(self.samples) == 0:
            # Download the dataset using torchgeo
            from torchgeo.datasets import FireRisk

            print(f"Downloading FireRisk dataset to {root}...")
            _ = FireRisk(root=root, split=split, download=True)

            # Re-collect samples after download
            for class_idx, class_name in enumerate(self.classes):
                class_dir = self.root / class_name
                if class_dir.exists():
                    for img_path in class_dir.glob("*.png"):
                        self.samples.append((str(img_path), class_idx))

        print(f"Found {len(self.samples)} samples in {split} split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load image and label
        img_path, label = self.samples[idx]

        # Load image with PIL
        img = Image.open(img_path).convert("RGB")

        # Convert to tensor
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW

        # Create sample dict
        sample = {"image": img_tensor, "label": torch.tensor(label, dtype=torch.long)}

        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)

        return sample


# Transform function
def transform_fn(sample):
    image = sample["image"]

    # Resize
    if image.shape[-2:] != (224, 224):
        image = F.interpolate(
            image.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)

    # Normalize to [0, 1]
    if image.max() > 1:
        image = image / 255.0

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image - mean) / std

    sample["image"] = image
    return sample


def worker_init_fn(worker_id):
    """Initialize each worker process."""
    import numpy as np
    import random
    import torch

    # Set seeds
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed() % 2**32)

    # Force single-threaded execution in workers
    torch.set_num_threads(1)


import lightning as L


class FixedDataModule(L.LightningDataModule):
    def __init__(self, num_workers=4, batch_size=64, train_subset_fraction=0.1):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_subset_fraction = train_subset_fraction

    def setup(self, stage=None):
        self.train_ds = FireRiskFixed(
            root="data", split="train", transform=transform_fn
        )
        self.val_ds = FireRiskFixed(root="data", split="val", transform=transform_fn)

        if self.train_subset_fraction < 1.0:
            subset_size = int(len(self.train_ds) * self.train_subset_fraction)
            indices = torch.randperm(len(self.train_ds))[:subset_size]
            self.train_ds = torch.utils.data.Subset(self.train_ds, indices)
            print(
                f"Using {len(self.train_ds)} training samples ({self.train_subset_fraction*100}%)"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True if torch.cuda.is_available() else False,
        )
