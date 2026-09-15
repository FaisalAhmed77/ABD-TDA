# ABD-TDA: ViT + Topological Data Analysis Fusion for Histopathological Classification

This repo implements a fusion model combining a pretrained Vision Transformer
(ViT) with a topological data analysis (TDA) side-branch for multi-class
histopathological image classification.

## Structure

| File | Purpose |
|---|---|
| `config.py` | Dataset paths, class names, and hyperparameters. Edit this per dataset. |
| `dataset.py` | Image loading, CSV side-feature loading, scaling, and the `FusionDataset` class. |
| `models.py` | `TDANetwork` (side-branch MLP) and `ViTTDAFusion` (fusion model). |
| `engine.py` | `train_epoch`, `evaluate`, and the `fit` loop with early stopping. |
| `metrics_utils.py` | Multi-class metrics (accuracy, macro precision/recall, macro OVR AUC), confusion matrix, and ROC curve plots. |
| `train.py` | Main entry point — wires everything together. |
| `requirements.txt` | Python dependencies. |

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Edit `config.py`:
   - Set `CLASS_NAMES` to your class labels, in the same order as your folders.
   - Set `TRAIN_DIRS` / `TEST_DIRS` to the per-class image folders (same order as `CLASS_NAMES`).
   - Set `TRAIN_CSV` / `TEST_CSV` to your TDA/HOG feature files.
3. Run training and evaluation:
   ```bash
   python train.py
   ```

This produces a saved checkpoint (`best_model.pth` by default), a confusion
matrix image, and an ROC curve image (per-class, one-vs-rest), plus printed
accuracy / macro precision / macro recall / macro AUC.

## Notes

- Side-branch CSVs are expected with `header=None`, where column 0 is a
  filename identifier (dropped) and the next 1200 columns are features
  (Betti-0/Betti-1 descriptors from RGB and HSV color spaces in our setup).
  Adjust `FEATURE_START_COL` / `FEATURE_END_COL` in `config.py` if your
  feature layout differs.
- `NUM_CLASSES` is inferred automatically from `len(CLASS_NAMES)`, so the
  same code supports both the ICIAR2018/BACH (4-class) and UT-Osteosarcoma
  (3-class) datasets.
