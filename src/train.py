import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import (
    CV_FOLDS,
    DATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_SEARCH_ITER,
    RANDOM_STATE,
    RF_PARAM_GRID,
    RF_WEIGHT,
    TEST_SIZE,
    XGB_PARAM_GRID,
    XGB_WEIGHT,
)
from src.preprocess import preprocess

# ── Logging setup ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Evaluation helper ──────────────────────────

def _evaluate(name: str, y_true, y_pred, split: str) -> dict:
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    logger.info(f"[{name}] {split:5s} → R²: {r2:.4f}  MAE: {mae:,.0f}  RMSE: {rmse:,.0f}")
    return {"model": name, "split": split, "r2": r2, "mae": mae, "rmse": rmse}


def run_baseline_models(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Train several models and log metrics. Returns a summary DataFrame."""
    logger.info("=" * 60)
    logger.info("Running baseline model comparison …")
    logger.info("=" * 60)

    models = {
        "Linear Regression":  LinearRegression(),
        "Ridge":              Ridge(),
        "Lasso":              Lasso(),
        "Decision Tree":      DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest":      RandomForestRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting":  GradientBoostingRegressor(random_state=RANDOM_STATE),
        "KNN":                KNeighborsRegressor(),
        "SVR":                SVR(),
        "XGBoost":            XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(_evaluate(name, y_train, model.predict(X_train), "train"))
        results.append(_evaluate(name, y_test,  model.predict(X_test),  "test"))

    return pd.DataFrame(results)


def tune_models(X_train, y_train) -> dict:
    logger.info("=" * 60)
    logger.info("Tuning RF and XGBoost with RandomizedSearchCV …")
    logger.info("=" * 60)

    search_configs = [
        ("RF",  RandomForestRegressor(random_state=RANDOM_STATE), RF_PARAM_GRID),
        ("XGB", XGBRegressor(random_state=RANDOM_STATE, verbosity=0), XGB_PARAM_GRID),
    ]

    best_params = {}
    for name, model, param_grid in search_configs:
        logger.info(f"Searching best params for {name} …")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=RANDOM_SEARCH_ITER,
            cv=CV_FOLDS,
            verbose=1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        search.fit(X_train, y_train)
        best_params[name] = search.best_params_
        logger.info(f"Best params for {name}: {search.best_params_}")

    return best_params


def train_final_models(X_train, X_test, y_train, y_test, best_params: dict):
    logger.info("=" * 60)
    logger.info("Training final tuned models …")
    logger.info("=" * 60)

    rf_best = RandomForestRegressor(**best_params["RF"], random_state=RANDOM_STATE)
    xgb_best = XGBRegressor(**best_params["XGB"], random_state=RANDOM_STATE, verbosity=0)

    rf_best.fit(X_train, y_train)
    xgb_best.fit(X_train, y_train)

    rf_pred   = rf_best.predict(X_test)
    xgb_pred  = xgb_best.predict(X_test)
    final_pred = (RF_WEIGHT * rf_pred) + (XGB_WEIGHT * xgb_pred)

    logger.info("=" * 60)
    logger.info("FINAL ENSEMBLE PERFORMANCE (test set)")
    _evaluate("Ensemble (RF+XGB)", y_test, final_pred, "test")
    logger.info("=" * 60)

    return rf_best, xgb_best


def save_pipeline(rf_model, xgb_model, scaler, num_imputer, feature_columns, num_cols):
    os.makedirs(MODEL_DIR, exist_ok=True)

    pipeline = {
        "rf_model":    rf_model,
        "xgb_model":   xgb_model,
        "scaler":      scaler,
        "imputer":     num_imputer,
        "columns":     feature_columns,
        "num_cols":    num_cols,
        "weights":     (RF_WEIGHT, XGB_WEIGHT),
    }
    joblib.dump(pipeline, MODEL_PATH)
    logger.info(f"Pipeline saved → {MODEL_PATH}")


# ── Entry point ────────────────────────────────

def main():
    # 1. Load & preprocess
    df = preprocess(DATA_PATH)

    X = df.drop("resale_price_in_inr", axis=1)
    y = df["resale_price_in_inr"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 2. Identify column types
    num_cols = X_train.select_dtypes(include=["int", "float"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()

    # 3. Impute
    num_imputer = SimpleImputer(strategy="median")
    num_imputer.fit(X_train[num_cols])
    X_train[num_cols] = num_imputer.transform(X_train[num_cols])
    X_test[num_cols]  = num_imputer.transform(X_test[num_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        cat_imputer.fit(X_train[cat_cols])
        X_train[cat_cols] = cat_imputer.transform(X_train[cat_cols])
        X_test[cat_cols]  = cat_imputer.transform(X_test[cat_cols])

    # 4. Scale
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    # 5. Baselines
    run_baseline_models(X_train, X_test, y_train, y_test)

    # 6. Tune
    best_params = tune_models(X_train, y_train)

    # 7. Final models
    rf_best, xgb_best = train_final_models(X_train, X_test, y_train, y_test, best_params)

    # 8. Save
    save_pipeline(rf_best, xgb_best, scaler, num_imputer, X_train.columns, num_cols)

    logger.info("Training complete ✓")


if __name__ == "__main__":
    main()
