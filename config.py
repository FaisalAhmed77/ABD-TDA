"""
Central configuration for the ViT + TDA fusion pipeline.

Edit CLASS_NAMES / TRAIN_DIRS / TEST_DIRS / *_CSV to point at whichever
dataset you're running (e.g. ICIAR2018/BACH, UT-Osteosarcoma). The number
of classes is inferred automatically from len(CLASS_NAMES), and TRAIN_DIRS
/ TEST_DIRS must be given in the SAME order as CLASS_NAMES so that folder i
is labeled with class i.
"""

# ------------------------------------------------------------------
# Dataset: class names and folder paths (same order = same label index)
# ------------------------------------------------------------------

CLASS_NAMES = ["Non-tumor", "Non-viable Tumor", "Viable Tumor"]

TRAIN_DIRS = [
    "/path/to/dataset/train/non-tumor",
    "/path/to/dataset/train/non-viable-tumor",
    "/path/to/dataset/train/viable",
]

TEST_DIRS = [
    "/path/to/dataset/validation/non-tumor",
    "/path/to/dataset/validation/non-viable-tumor",
    "/path/to/dataset/validation/viable",
]

NUM_CLASSES = len(CLASS_NAMES)

# ------------------------------------------------------------------
# Side-branch (TDA/HOG) feature CSVs
# CSVs are read with header=None; column 0 is assumed to be a filename
# and is dropped; the next 1200 columns are used as features.
# ------------------------------------------------------------------

TRAIN_CSV = "data_300_train.csv"
TEST_CSV = "data_300_val.csv"
FEATURE_START_COL = 1
FEATURE_END_COL = 1201  # exclusive

# ------------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------------

IMAGE_SIZE = (224, 224)  # matches ViT-base patch16-224 input size

# ------------------------------------------------------------------
# Model / training hyperparameters
# ------------------------------------------------------------------

VIT_MODEL_NAME = "google/vit-base-patch16-224"
VIT_FEATURE_DIM = 768
SIDE_HIDDEN_DIM = 64  # output dim of the TDA side-branch network

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 25  # early stopping patience

CHECKPOINT_PATH = "best_model.pth"
CONFUSION_MATRIX_PATH = "confusion_matrix.png"
ROC_CURVE_PATH = "roc_curves.png"
