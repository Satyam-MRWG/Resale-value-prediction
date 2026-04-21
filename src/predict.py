import argparse
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import MODEL_PATH

# ── Logging setup ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_pipeline(model_path: str = MODEL_PATH) -> dict:
    """Load saved pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'.\n"
            "Please run  python src/train.py  first."
        )
    logger.info(f"Loading pipeline from: {model_path}")
    return joblib.load(model_path)


# ── Input validation ───────────────────────────

def validate_input(df: pd.DataFrame) -> None:
    """Raise ValueError for obviously wrong inputs."""
    checks = {
        "engine_capacity_cc": (50, 10_000),
        "car_age_in_year":    (0,  50),
        "kms_driven":         (0,  1_000_000),
        "max_power":          (20, 1_000),
        "mileage":            (1,  100),
        "seats":              (2,  10),
    }
    for col, (lo, hi) in checks.items():
        if col in df.columns:
            bad = df[(df[col] < lo) | (df[col] > hi)]
            if not bad.empty:
                raise ValueError(
                    f"Column '{col}' has values outside expected range [{lo}, {hi}]: "
                    f"{bad[col].tolist()}"
                )


# ── Core prediction function ───────────────────

def predict_price(input_df: pd.DataFrame, pipeline: dict = None) -> np.ndarray:
    
    if pipeline is None:
        pipeline = load_pipeline()

    rf_model   = pipeline["rf_model"]
    xgb_model  = pipeline["xgb_model"]
    scaler     = pipeline["scaler"]
    imputer    = pipeline["imputer"]
    columns    = pipeline["columns"]
    num_cols   = pipeline["num_cols"]
    rf_w, xgb_w = pipeline["weights"]
    
    validate_input(input_df)

    # Align columns (add missing dummies as 0)
    df = input_df.reindex(columns=columns, fill_value=0)

    df[num_cols] = imputer.transform(df[num_cols])
    df[num_cols] = scaler.transform(df[num_cols])

    rf_pred   = rf_model.predict(df)
    xgb_pred  = xgb_model.predict(df)
    final     = (rf_w * rf_pred) + (xgb_w * xgb_pred)

    return final


def main():
    sample = pd.DataFrame([{
   'engine_capacity_cc': 1197,
    'car_age_in_year': 5,
    'kms_driven': 40000,
    'kms_per_year': 8000,
    'max_power': 83.1,
    'mileage': 21.4,
    'seats': 5,
    'owner_type': 1,
    'transmission_type': 0,
    'insurance': 1,
    'brand_Maruti':1,
    'brand_Tata': 0,
    'fuel_type_Petrol': 1,
    'fuel_type_Diesel': 0,
    'body_type_Hatchback':1,
    'body_type_Sedan': 0,
    'city_Agra': 1
}])

    price = predict_price(sample)
    logger.info(f"Predicted resale price: ₹ {price[0]:,.0f}")
    print(f"\n  Estimated Resale Price: ₹ {price[0]:,.0f}\n")


if __name__ == "__main__":
    main()
