"""
Multi-class metrics and plots: accuracy / macro precision & recall / macro
one-vs-rest AUC, confusion matrix heatmap, and per-class ROC curves.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true, y_pred, y_probs, class_names):
    """Returns overall metrics plus a list of (class_name, precision, recall)."""
    num_classes = len(class_names)
    class_ids = list(range(num_classes))

    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    y_true_bin = label_binarize(y_true, classes=class_ids)
    auc_macro = roc_auc_score(y_true_bin, y_probs, average="macro", multi_class="ovr")

    per_class = []
    for i, name in enumerate(class_names):
        p = precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        r = recall_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        per_class.append((name, p, r))

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "auc_macro": auc_macro,
        "per_class": per_class,
        "y_true_bin": y_true_bin,
    }


def print_metrics(metrics):
    print("\nFINAL METRICS")
    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Precision (macro)  : {metrics['precision_macro']:.4f}")
    print(f"Recall (macro)     : {metrics['recall_macro']:.4f}")
    print(f"AUC (macro, OVR)   : {metrics['auc_macro']:.4f}")
    print("\nPer-class report:")
    for name, p, r in metrics["per_class"]:
        print(f"  {name:20s} precision={p:.4f}  recall={r:.4f}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    class_ids = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_roc_curves(y_true_bin, y_probs, class_names, save_path,
                     title="ROC Curves (One-vs-Rest)"):
    plt.figure(figsize=(7, 7))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
