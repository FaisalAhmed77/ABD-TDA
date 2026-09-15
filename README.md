# ABD-TDA: ViT + Topological Data Analysis Fusion for Histopathological Classification

This repo implements a fusion model combining a pretrained Vision Transformer
(ViT) with a topological data analysis (TDA) side-branch for multi-class
histopathological image classification, plus the TDA feature extraction
pipeline used to generate the side-branch inputs.

## Structure

| File | Purpose |
|---|---|
| `extract_tda_features.py` | Extracts 1200 Betti-0/Betti-1 features (RGB + HSV channels) per image and writes them to CSV. |
| `config.py` | Dataset paths, class names, and hyperparameters for the fusion model. Edit this per dataset. |
| `dataset.py` | Image loading, CSV side-feature loading, scaling, and the `FusionDataset` class. |
| `models.py` | `TDANetwork` (side-branch MLP) and `ViTTDAFusion` (fusion model). |
| `engine.py` | `train_epoch`, `evaluate`, and the `fit` loop with early stopping. |
| `metrics_utils.py` | Multi-class metrics (accuracy, macro precision/recall, macro OVR AUC), confusion matrix, and ROC curve plots. |
| `train.py` | Main training entry point — wires everything together. |
| `requirements.txt` | Python dependencies. |

## Pipeline overview

1. **Extract TDA features** from your image folders (`extract_tda_features.py`).
2. **Train and evaluate the fusion model** on the images + extracted features (`train.py`).

## 1. TDA feature extraction

`extract_tda_features.py` computes topological (Betti) features per image:

- For each image, 6 channels are used: **R, G, B** (from the RGB image) and
  **H, S, V** (from the HSV conversion of the same image).
- For each channel, `CubicalPersistence` is computed with
  `homology_dimensions=[0, 1]` (Betti-0 and Betti-1), then converted into a
  100-bin `BettiCurve` per homology dimension — **200 features per channel**.
- Concatenating all 6 channels gives **1200 features per image**
  (6 channels × 2 homology dimensions × 100 bins).

### Usage

Edit the config block at the top of `extract_tda_features.py`:

- `TRAIN_CLASS_DIRS` / `TEST_CLASS_DIRS` — per-class image folders, **in the
  same order** you'll use for `TRAIN_DIRS` / `TEST_DIRS` in `config.py`, since
  that order determines which label each row belongs to.
- `IMAGE_EXTENSION` — e.g. `.jpg` or `.jpeg`.
- `TRAIN_OUTPUT_CSV` / `TEST_OUTPUT_CSV` — output CSV paths.

Then run:

```bash
python extract_tda_features.py
```

Each output CSV has no header row: column 0 is a `<class_folder>/<filename>`
identifier and columns 1–1200 are the Betti features, with rows in the same
order (class folder order, then alphabetical filename order within each
folder) that `dataset.load_image_dataset()` uses to load images — this
keeps feature rows aligned with image rows and labels.

## 2. Fusion model training

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Edit `config.py`:
   - Set `CLASS_NAMES` to your class labels, in the same order as your folders.
   - Set `TRAIN_DIRS` / `TEST_DIRS` to the per-class image folders (same order as `CLASS_NAMES`, and the same order used in `extract_tda_features.py`).
   - Set `TRAIN_CSV` / `TEST_CSV` to the CSVs produced in step 1 (`TRAIN_OUTPUT_CSV` / `TEST_OUTPUT_CSV`).
3. Run training and evaluation:
   ```bash
   python train.py
   ```

This produces a saved checkpoint (`best_model.pth` by default), a confusion
matrix image, and an ROC curve image (per-class, one-vs-rest), plus printed
accuracy / macro precision / macro recall / macro AUC.

## Notes

- `NUM_CLASSES` is inferred automatically from `len(CLASS_NAMES)` in
  `config.py`, so the same fusion code supports both the ICIAR2018/BACH
  (4-class) and UT-Osteosarcoma (3-class) datasets.
- If you use a different number of channels, homology dimensions, or bins
  for feature extraction, update `FEATURE_START_COL` / `FEATURE_END_COL` in
  `config.py` to match the resulting CSV's feature column count.
