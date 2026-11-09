"""
Dataset classes for GAN training and YOLO training.
"""
import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class NEUDataset(Dataset):
    """
    NEU Surface Defect Dataset.

    Contains 6 classes of steel surface defects:
    0: crazing (Cr)
    1: inclusion (In)
    2: patches (Pa)
    3: pitted_surface (PS)
    4: rolled-in_scale (RS)
    5: scratches (Sc)

    Each class has 300 grayscale images (200x200 pixels).
    """

    # Class name to index mapping
    CLASS_NAMES = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches"
    ]

    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    def __init__(self, root_dir: str, image_size: int = 64,
                 channels: int = 1, split: str = "train",
                 train_ratio: float = 0.8, seed: int = 42,
                 transform=None):
        """
        Args:
            root_dir: Path to NEU-CLS directory
            image_size: Target image size
            channels: Number of channels (1 for grayscale, 3 for RGB)
            split: "train", "val", or "all"
            train_ratio: Ratio of training data
            seed: Random seed for reproducibility
            transform: Optional transform to apply
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.channels = channels
        self.split = split

        # Gather all image paths and labels
        self.image_paths = []
        self.labels = []

        for class_name in self.CLASS_NAMES:
            pattern = os.path.join(root_dir, f"{class_name}_*.*")
            files = sorted(glob.glob(pattern))
            class_idx = self.CLASS_TO_IDX[class_name]

            for f in files:
                self.image_paths.append(f)
                self.labels.append(class_idx)

        # Split dataset
        np.random.seed(seed)
        indices = np.random.permutation(len(self.image_paths))
        split_idx = int(len(indices) * train_ratio)

        if split == "train":
            selected_indices = indices[:split_idx]
        elif split == "val":
            selected_indices = indices[split_idx:]
        else:  # all
            selected_indices = indices

        self.image_paths = [self.image_paths[i] for i in selected_indices]
        self.labels = [self.labels[i] for i in selected_indices]

        # Default transform
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform

    def _get_default_transform(self):
        """Get default image transform."""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]

        if self.channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))

        # Normalize to [-1, 1] for GAN training
        transform_list.append(transforms.Normalize(
            mean=[0.5] * self.channels,
            std=[0.5] * self.channels
        ))

        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path)
        image = self.transform(image)

        return image, label

    def get_class_counts(self):
        """Get number of samples per class."""
        counts = np.zeros(len(self.CLASS_NAMES), dtype=int)
        for label in self.labels:
            counts[label] += 1
        return counts


class GANImageDataset(Dataset):
    """
    Dataset for GAN training - returns images and class labels.
    Supports conditional GAN training.
    """

    def __init__(self, root_dir: str, image_size: int = 64,
                 channels: int = 1, class_names: list = None,
                 transform=None):
        """
        Args:
            root_dir: Directory containing images organized by class subdirectories
                      or flat with class prefixes
            image_size: Target image size
            channels: Number of channels
            class_names: List of class names (if None, auto-detect)
            transform: Optional transform
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.channels = channels

        # Detect class structure
        self.image_paths = []
        self.labels = []

        # Check if classes are in subdirectories
        subdirs = [d for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d))]

        if subdirs:
            # Class subdirectory structure
            if class_names is None:
                self.class_names = sorted(subdirs)
            else:
                self.class_names = class_names

            for class_idx, class_name in enumerate(self.class_names):
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                for ext in ['*.jpg', '*.png', '*.bmp', '*.jpeg']:
                    files = sorted(glob.glob(os.path.join(class_dir, ext)))
                    for f in files:
                        self.image_paths.append(f)
                        self.labels.append(class_idx)
        else:
            # Flat structure with class prefixes (like NEU)
            if class_names is None:
                # Auto-detect from filenames
                prefixes = set()
                for f in os.listdir(root_dir):
                    if f.endswith(('.jpg', '.png', '.bmp', '.jpeg')):
                        # Extract prefix (class name)
                        parts = os.path.splitext(f)[0].rsplit('_', 1)
                        if len(parts) == 2:
                            prefixes.add(parts[0])
                self.class_names = sorted(prefixes)
            else:
                self.class_names = class_names

            class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

            for class_name in self.class_names:
                pattern = os.path.join(root_dir, f"{class_name}_*.*")
                files = sorted(glob.glob(pattern))
                class_idx = class_to_idx[class_name]
                for f in files:
                    self.image_paths.append(f)
                    self.labels.append(class_idx)

        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform

    def _get_default_transform(self):
        """Get default transform for GAN training."""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]

        if self.channels == 1:
            transform_list.insert(1, transforms.Grayscale(num_output_channels=1))

        # Normalize to [-1, 1]
        transform_list.append(transforms.Normalize(
            mean=[0.5] * self.channels,
            std=[0.5] * self.channels
        ))

        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path)
        image = self.transform(image)

        return image, label

    @property
    def num_classes(self):
        return len(self.class_names)


def get_gan_dataloader(data_dir: str, image_size: int = 64, channels: int = 1,
                       batch_size: int = 64, num_workers: int = 4,
                       class_names: list = None, shuffle: bool = True
                       ) -> DataLoader:
    """
    Create dataloader for GAN training.

    Args:
        data_dir: Path to dataset directory
        image_size: Image size
        channels: Number of channels
        batch_size: Batch size
        num_workers: Number of data loading workers
        class_names: List of class names
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    dataset = GANImageDataset(
        root_dir=data_dir,
        image_size=image_size,
        channels=channels,
        class_names=class_names
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader
