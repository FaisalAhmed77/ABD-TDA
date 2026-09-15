"""
TDA feature extraction: Betti-0 and Betti-1 curves computed on the
R, G, B, H, S, V channels of each image.

Per channel: homology_dimensions=[0, 1], n_bins=100 -> 200 features/channel
6 channels (R, G, B, H, S, V) x 200 features = 1200 features per image

Output: one CSV per split (train/test), header=None, where column 0 is a
"<class_folder_name>/<filename>" identifier and columns 1-1200 are the
1200 Betti features, in the exact order the fusion pipeline's
dataset.load_image_dataset() consumes images (folders in CLASS order,
files sorted alphabetically within each folder) so that feature rows
line up with image rows and labels.
"""

import os

import numpy as np
import pandas as pd
from PIL import Image
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve

# ------------------------------------------------------------------
# Config — edit per dataset. CLASS_DIRS order MUST match the order used
# in dataset.py / config.py (TRAIN_DIRS / TEST_DIRS), since that order
# determines the label each row belongs to.
# ------------------------------------------------------------------

TRAIN_CLASS_DIRS = [
    "/path/to/dataset/train/non-tumor",
    "/path/to/dataset/train/non-viable-tumor",
    "/path/to/dataset/train/viable",
]

TEST_CLASS_DIRS = [
    "/path/to/dataset/validation/non-tumor",
    "/path/to/dataset/validation/non-viable-tumor",
    "/path/to/dataset/validation/viable",
]

IMAGE_EXTENSION = ".jpg"

TRAIN_OUTPUT_CSV = "data_train.csv"
TEST_OUTPUT_CSV = "data_val.csv"

N_BINS = 100
HOMOLOGY_DIMENSIONS = [0, 1]  # Betti-0 and Betti-1

# ------------------------------------------------------------------
# TDA transformers (shared across all channels/images)
# ------------------------------------------------------------------

CP = CubicalPersistence(
    homology_dimensions=HOMOLOGY_DIMENSIONS,
    coeff=3,
    n_jobs=1,
)
BC = BettiCurve(n_bins=N_BINS)


def betti_features_for_channel(channel_array):
    """Betti-0 + Betti-1 curves for a single 2D channel -> (200,) vector."""
    diagram = CP.fit_transform(channel_array[None, :, :].astype(np.float64))
    betti_curves = BC.fit_transform(diagram)  # shape: (1, n_homology_dims, n_bins)
    return betti_curves.reshape(-1)  # (len(HOMOLOGY_DIMENSIONS) * N_BINS,)


def extract_features_for_image(image_path):
    """RGB (3 channels) + HSV (3 channels) Betti features, concatenated -> (1200,)."""
    rgb_image = Image.open(image_path).convert("RGB")
    hsv_image = rgb_image.convert("HSV")

    r, g, b = rgb_image.split()
    h, s, v = hsv_image.split()

    channel_arrays = [np.array(c) for c in (r, g, b, h, s, v)]
    channel_features = [betti_features_for_channel(c) for c in channel_arrays]

    return np.concatenate(channel_features)  # (6 * 200,) = (1200,)


def extract_features_for_dataset(class_dirs):
    """
    Process each class folder in order, files sorted within each folder,
    matching the image-loading order used elsewhere in the pipeline.
    Returns a DataFrame: column 0 = identifier, columns 1..1200 = features.
    """
    rows = []
    for class_dir in class_dirs:
        class_name = os.path.basename(os.path.normpath(class_dir))
        image_files = sorted(
            f for f in os.listdir(class_dir) if f.lower().endswith(IMAGE_EXTENSION)
        )
        for img_file in image_files:
            image_path = os.path.join(class_dir, img_file)
            features = extract_features_for_image(image_path)
            identifier = f"{class_name}/{img_file}"
            rows.append([identifier] + features.tolist())
            print(f"Processed {identifier}")

    return pd.DataFrame(rows)


def main():
    print("Extracting TDA features for training set...")
    train_df = extract_features_for_dataset(TRAIN_CLASS_DIRS)
    train_df.to_csv(TRAIN_OUTPUT_CSV, header=False, index=False)
    print(f"Saved {train_df.shape[0]} rows x {train_df.shape[1] - 1} features to {TRAIN_OUTPUT_CSV}")

    print("\nExtracting TDA features for test/validation set...")
    test_df = extract_features_for_dataset(TEST_CLASS_DIRS)
    test_df.to_csv(TEST_OUTPUT_CSV, header=False, index=False)
    print(f"Saved {test_df.shape[0]} rows x {test_df.shape[1] - 1} features to {TEST_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
