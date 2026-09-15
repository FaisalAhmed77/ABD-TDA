"""
Main entry point for training and evaluating the ViT + TDA fusion model.

Usage:
    python train.py

All dataset paths, hyperparameters, and output filenames are set in
config.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from dataset import FusionDataset, load_image_dataset, load_side_features, scale_side_features
from engine import evaluate, fit
from metrics_utils import compute_metrics, plot_confusion_matrix, plot_roc_curves, print_metrics
from models import ViTTDAFusion


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    X_train_img, y_train = load_image_dataset(config.TRAIN_DIRS)
    X_test_img, y_test = load_image_dataset(config.TEST_DIRS)
    print("Train Images Shape:", X_train_img.shape)
    print("Test Images Shape:", X_test_img.shape)

    X_train_tda = load_side_features(config.TRAIN_CSV)
    X_test_tda = load_side_features(config.TEST_CSV)
    print("Train TDA feature shape:", X_train_tda.shape)
    print("Test TDA feature shape:", X_test_tda.shape)

    assert X_train_tda.shape[0] == y_train.shape[0], "Train feature rows and label count don't match"
    assert X_test_tda.shape[0] == y_test.shape[0], "Test feature rows and label count don't match"

    X_train_tda, X_test_tda, _ = scale_side_features(X_train_tda, X_test_tda)

    train_dataset = FusionDataset(X_train_img, X_train_tda, y_train)
    test_dataset = FusionDataset(X_test_img, X_test_tda, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------
    # Model / loss / optimizer
    # ------------------------------------------------------------------
    model = ViTTDAFusion(side_dim=X_train_tda.shape[1], num_classes=config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    fit(
        model, train_loader, test_loader, optimizer, criterion, device,
        epochs=config.EPOCHS, patience=config.PATIENCE,
        checkpoint_path=config.CHECKPOINT_PATH,
    )

    # ------------------------------------------------------------------
    # Final evaluation with the best checkpoint
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
    model.eval()
    print("\nLoaded Best Model.")

    _, _, y_true, y_pred, y_probs = evaluate(model, test_loader, criterion, device)

    metrics = compute_metrics(y_true, y_pred, y_probs, config.CLASS_NAMES)
    print_metrics(metrics)

    plot_confusion_matrix(y_true, y_pred, config.CLASS_NAMES, config.CONFUSION_MATRIX_PATH)
    plot_roc_curves(metrics["y_true_bin"], y_probs, config.CLASS_NAMES, config.ROC_CURVE_PATH)


if __name__ == "__main__":
    main()
