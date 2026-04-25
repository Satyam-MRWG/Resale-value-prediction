import os
import sys
import logging

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_PATH as PIPELINE_PATH

# ── Logging ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Load pipeline once ─────────────────────────
if not os.path.exists(PIPELINE_PATH):
    raise FileNotFoundError(
        f"Pipeline not found at '{PIPELINE_PATH}'.\n"
        "Please run  python src/train.py  first."
    )
logger.info(f"Loading pipeline from: {PIPELINE_PATH}")
pipeline = joblib.load(PIPELINE_PATH)


# ── Input validation ───────────────────────────
def validate_input(df: pd.DataFrame) -> None:
    """Raise ValueError for obviously wrong inputs."""
    checks = {
        "engine_capacity_cc": (50,  10_000),
        "car_age_in_year":    (0,   50),
        "kms_driven":         (0,   1_000_000),
        "max_power":          (20,  1_000),
        "mileage":            (1,   100),
        "seats":              (2,   10),
    }
    for col, (lo, hi) in checks.items():
        if col in df.columns:
            bad = df[(df[col] < lo) | (df[col] > hi)]
            if not bad.empty:
                raise ValueError(
                    f"Column '{col}' has values outside expected range [{lo}, {hi}]: "
                    f"{bad[col].tolist()}"
                )


# ── RF + XGB ensemble prediction ──────────────
def predict_price(input_df: pd.DataFrame) -> float:
    """
    Run RF + XGBoost ensemble and return predicted base price.

    Args:
        input_df: DataFrame with the 17 tabular feature columns.

    Returns:
        float: predicted base resale price in INR.
    """
    rf_model     = pipeline["rf_model"]
    xgb_model    = pipeline["xgb_model"]
    scaler       = pipeline["scaler"]
    imputer      = pipeline["imputer"]
    columns      = pipeline["columns"]
    num_cols     = pipeline["num_cols"]
    rf_w, xgb_w  = pipeline["weights"]

    validate_input(input_df)

    # Align columns (add missing dummies as 0)
    df = input_df.reindex(columns=columns, fill_value=0)
    df[num_cols] = imputer.transform(df[num_cols])
    df[num_cols] = scaler.transform(df[num_cols])

    rf_pred  = rf_model.predict(df)
    xgb_pred = xgb_model.predict(df)
    price    = float((rf_w * rf_pred + xgb_w * xgb_pred)[0])

    logger.info(f"Predicted base resale price: ₹{price:,.0f}")
    return price


# ── Damage penalty for resale price ───────────
def calculate_damage_penalty(damage_counts: dict, damage_details: list) -> float:
    """
    Returns a price multiplier (0.60–1.0) based on detected damage.

    0 damage   → 1.00  (no reduction)
    Light      → ~0.90 (10% reduction)
    Moderate   → ~0.80 (20% reduction)
    Heavy      → ~0.70 (30% reduction)
    Critical   → ~0.60 (40% reduction max)
    """
    SEVERITY = {
        "dent":          "Medium",
        "scratch":       "Low",
        "crack":         "High",
        "glass shatter": "Critical",
        "lamp broken":   "High",
        "tire flat":     "Critical",
        "Car-Damage":    "High",
    }

    total = sum(damage_counts.values())
    if total == 0:
        return 1.0

    severity_weight = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    weighted_sum = sum(
        severity_weight.get(SEVERITY.get(dmg, "Medium"), 2) * count
        for dmg, count in damage_counts.items()
    )
    avg_conf = (
        sum(d["confidence"] for d in damage_details) / len(damage_details)
        if damage_details else 0
    )
    raw_penalty = (weighted_sum * 0.05) + ((avg_conf / 100) * 0.10)
    return round(1.0 - min(raw_penalty, 0.40), 4)


# ── CLI (for quick local testing) ─────────────
if __name__ == "__main__":
    sample = pd.DataFrame([{
        "engine_capacity_cc":  1197,
        "car_age_in_year":     5,
        "kms_driven":          40000,
        "kms_per_year":        8000,
        "max_power":           83.1,
        "mileage":             21.4,
        "seats":               5,
        "owner_type":          1,
        "transmission_type":   0,
        "insurance":           1,
        "brand_Maruti":        1,
        "brand_Tata":          0,
        "fuel_type_Petrol":    1,
        "fuel_type_Diesel":    0,
        "body_type_Hatchback": 1,
        "body_type_Sedan":     0,
        "city_Agra":           1,
    }])

    price = predict_price(sample)
    print(f"\n  Estimated Resale Price: ₹ {price:,.0f}\n")