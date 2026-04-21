import logging
import numpy as np
import pandas as pd
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import (
    TIER_1_CITIES,
    INSURANCE_MAP,
    OWNER_MAP,
    TRANSMISSION_MAP,
)

logger = logging.getLogger(__name__)

def _kms_driven_converter(value):
    if isinstance(value, str):
        value = value.split()[0].replace(",", "")
        return float(value)
    return value


def _car_age(registered_year):
    current_year = datetime.now().year
    return current_year - registered_year if pd.notnull(registered_year) else registered_year

def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    if "Unnamed: 0" in df.columns:
        df.drop(columns="Unnamed: 0", inplace=True)
    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Filling missing values with mode …")
    null_cols = df.columns[df.isnull().any()].tolist()
    for col in null_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def parse_full_name(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Parsing full_name → registered_year, brand, model …")
    df["registered_year"] = df["full_name"].str.split().apply(lambda x: int(x[0]))
    df["full_name"] = df["full_name"].str.split().str[1:].str.join(" ")
    df["brand"] = df["full_name"].str.split().str[0]
    df["model"] = df["full_name"].str.split().str[1:].str.join(" ")
    df.drop(columns=["full_name"], inplace=True)

    # Move brand and model to front
    for col_name in ["model", "brand"]:
        col = df.pop(col_name)
        df.insert(0, col_name, col)
    return df


def parse_resale_price(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Parsing resale_price → resale_price_in_inr …")
    df["resale_price_in_inr"] = (
        df["resale_price"]
        .str.split().str[1]
        .str.replace(",", ".")
        .astype(float) * 100_000
    )
    df.drop(columns=["resale_price"], inplace=True)
    col = df.pop("resale_price_in_inr")
    df.insert(2, "resale_price_in_inr", col)
    return df


def parse_engine_capacity(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Parsing engine_capacity …")
    df["engine_capacity_cc"] = df["engine_capacity"].str.split().apply(
        lambda x: int(x[0])
    )
    col = df.pop("engine_capacity_cc")
    df.insert(3, "engine_capacity_cc", col)
    df.drop(columns=["engine_capacity"], inplace=True)
    return df


def clean_insurance(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning insurance column …")
    replacements = {
        "Third Party insurance": "Third Party",
        "2": "Second",
        "1": "First",
        "Not Availabel": np.nan,
    }
    df["insurance"] = df["insurance"].replace(replacements)
    df["insurance"] = df["insurance"].fillna(df["insurance"].mode()[0])
    return df


def parse_kms_driven(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Parsing kms_driven …")
    df["kms_driven"] = df["kms_driven"].apply(_kms_driven_converter)
    return df


def parse_max_power(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Parsing max_power …")
    df["max_power"] = df["max_power"].str.extract(r"(\d+\.?\d*)").astype(float)
    return df


def parse_mileage(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Parsing mileage …")
    df["mileage"] = df["mileage"].str.split().apply(lambda x: float(x[0]))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering features …")

    # Car age
    df["car_age_in_year"] = df["registered_year"].apply(_car_age)
    df.drop(columns=["registered_year"], inplace=True)
    col = df.pop("car_age_in_year")
    df.insert(4, "car_age_in_year", col)

    # KMs per year 
    df["kms_per_year"] = np.where(
        df["car_age_in_year"] > 0,
        np.round(df["kms_driven"] / df["car_age_in_year"], 0),
        0,
    )
    col = df.pop("kms_per_year")
    df.insert(8, "kms_per_year", col)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Encoding features …")

    # Ordinal
    df["insurance"]        = df["insurance"].map(INSURANCE_MAP)
    df["owner_type"]       = df["owner_type"].map(OWNER_MAP)
    df["transmission_type"] = df["transmission_type"].map(TRANSMISSION_MAP)

    # One-hot
    df = pd.get_dummies(df, columns=["brand", "fuel_type", "body_type"], drop_first=True)

    # City tier
    df["city_tier"] = df["city"].apply(lambda x: 1 if x in TIER_1_CITIES else 2)

    # Drop non-predictive columns
    df.drop(columns=["model", "city"], inplace=True, errors="ignore")

    return df


def preprocess(path: str) -> pd.DataFrame:
    df = load_data(path)
    df = fill_missing_values(df)
    df = parse_full_name(df)
    df = parse_resale_price(df)
    df = parse_engine_capacity(df)
    df = clean_insurance(df)
    df["seats"] = df["seats"].astype("int")
    df = parse_kms_driven(df)
    df = parse_max_power(df)
    df = parse_mileage(df)
    df = engineer_features(df)
    df = encode_features(df)
    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df
