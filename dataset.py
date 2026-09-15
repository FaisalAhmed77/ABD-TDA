"""
Data loading utilities: image folders -> arrays, CSV -> TDA/HOG features,
and the PyTorch Dataset used by the fusion model.
"""

import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch

import config


def load_images_from_folder(folder_path, image_size=config.IMAGE_SIZE):
    """Load all .jpg images in a folder, sorted by filename, as RGB arrays."""
    data = []
    image_files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")
    )
    for img_file in image_files:
        image_path = os.path.join(folder_path, img_file)
        img = Image.open(image_path).convert("RGB")
        img = img.resize(image_size)
        data.append(np.array(img))
    return np.array(data)


def load_image_dataset(class_dirs):
    """
    Load images from a list of per-class folders and build matching labels.

    class_dirs[i] is assigned label i, so class_dirs must be given in the
    same order as config.CLASS_NAMES.
    """
    images_per_class = [load_images_from_folder(d) for d in class_dirs]
    labels_per_class = [
        np.full(len(imgs), label, dtype=np.int64)
        for label, imgs in enumerate(images_per_class)
    ]

    images = np.concatenate(images_per_class, axis=0).astype(np.float32) / 255.0
    labels = np.concatenate(labels_per_class, axis=0)
    return images, labels


def load_side_features(csv_path, start_col=config.FEATURE_START_COL,
                        end_col=config.FEATURE_END_COL):
    """Load TDA/HOG features from a headerless CSV, dropping the filename column."""
    df = pd.read_csv(csv_path, header=None)
    df = df.iloc[:, start_col:end_col]
    return df.values.astype(np.float32)


def scale_side_features(train_features, test_features):
    """Fit a StandardScaler on train only, apply to both splits (no leakage)."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features).astype(np.float32)
    test_scaled = scaler.transform(test_features).astype(np.float32)
    return train_scaled, test_scaled, scaler


class FusionDataset(Dataset):
    """Pairs each image with its corresponding side-branch feature vector and label."""

    def __init__(self, images, side_features, labels):
        assert len(images) == len(side_features) == len(labels), (
            "images, side_features, and labels must all be the same length"
        )
        self.images = images
        self.side_features = side_features
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1)
        side = torch.tensor(self.side_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, side, label
