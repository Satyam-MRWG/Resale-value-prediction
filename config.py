# ──────────────────────────────────────────────
#  config.py  –  Central configuration file
# ──────────────────────────────────────────────

import os

# ── Paths ──────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "car_resale_prices.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "car_price_model.pkl")

# ── Train / Test Split ─────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Cities ─────────────────────────────────────
TIER_1_CITIES = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad"]

# ── Insurance Mapping ──────────────────────────
INSURANCE_MAP = {
    "Third Party": 1,
    "Comprehensive": 2,
    "Zero Dep": 3,
}

# ── Owner Mapping ──────────────────────────────
OWNER_MAP = {
    "First Owner":  1,
    "Second Owner": 2,
    "Third Owner":  3,
}

# ── Transmission Mapping ───────────────────────
TRANSMISSION_MAP = {
    "Manual":    0,
    "Automatic": 1,
}

# ── Ensemble Weights ───────────────────────────
RF_WEIGHT  = 0.8
XGB_WEIGHT = 0.2

# ── Hyper-parameter Search ─────────────────────
RF_PARAM_GRID = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":    ["sqrt", "log2"],
}

XGB_PARAM_GRID = {
    "n_estimators":    [100, 200, 300],
    "learning_rate":   [0.01, 0.05, 0.1],
    "max_depth":       [3, 5, 7, 9],
    "subsample":       [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "gamma":           [0, 0.1, 0.3, 1],
    "min_child_weight": [1, 3, 5],
}

RANDOM_SEARCH_ITER = 100
CV_FOLDS           = 3
