"""
Data Preprocessing and Exploratory Data Analysis module for Textile Digital Twin.
Handles robust loading, cleaning, feature parsing, and summary statistics.
"""

import os
import re
import pandas as pd
import numpy as np

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PRIMARY_DATASET = "weaving_dataset_full.csv"
PREPROCESSING_VERSION = "2026-09-19-v2"

def get_available_datasets():
    """List available CSV files in the data directory."""
    if not os.path.exists(DEFAULT_DATA_DIR):
        return []
    files = [f for f in os.listdir(DEFAULT_DATA_DIR) if f.endswith(".csv")]
    # Keep the full weaving dataset first because it is the analytical source of truth.
    files.sort(key=lambda x: (x != PRIMARY_DATASET, x.lower()))
    return files

def load_data(file_path_or_buffer=None):
    """
    Safely load textile dataset from a file path, file buffer, or default directory.
    Returns:
        pd.DataFrame: Loaded raw or cleaned dataframe.
    """
    if file_path_or_buffer is None:
        file_path = os.path.join(DEFAULT_DATA_DIR, PRIMARY_DATASET)
        if not os.path.exists(file_path):
            available = get_available_datasets()
            if available:
                file_path = os.path.join(DEFAULT_DATA_DIR, available[0])
            else:
                raise FileNotFoundError("No CSV dataset found in data/ directory.")
    else:
        file_path = file_path_or_buffer

    df = pd.read_csv(file_path, low_memory=False)
    return df

def clean_and_preprocess(df):
    """
    Cleans and preprocesses the dataset:
    - Drops duplicates
    - Parses numeric count fields if formatted as strings
    - Imputes missing values safely
    - Creates derived textile engineering features (e.g. Total Cover Factor, Warp/Weft Ratio)
    """
    df_clean = df.copy()
    original_rows = len(df_clean)
    summary = {
        "original_rows": original_rows,
        "index_columns_removed": 0,
        "aggregate_rows_removed": 0,
        "invalid_rows_removed": 0,
        "missing_values_normalized": 0,
        "duplicates_removed": 0,
        "final_rows": original_rows,
        "preprocessing_version": PREPROCESSING_VERSION,
    }

    # Remove exact source duplicates before dropping an export index column. Otherwise,
    # distinct records with identical business fields would be collapsed incorrectly.
    before_duplicates = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    summary["duplicates_removed"] = before_duplicates - len(df_clean)

    # Drop Unnamed columns if present
    unnamed_cols = [c for c in df_clean.columns if "unnamed" in c.lower()]
    if unnamed_cols:
        df_clean = df_clean.drop(columns=unnamed_cols)
        summary["index_columns_removed"] = len(unnamed_cols)

    # Normalize common text sentinels before type conversion. They are missing values,
    # except for TOTAL, which denotes an aggregate row in this source export.
    text_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        normalized = df_clean[col].astype("string").str.strip()
        summary["missing_values_normalized"] += int(
            normalized.str.lower().isin({"na", "n/a", "null", "none", "nan", ""}).sum()
        )
        df_clean[col] = normalized.mask(
            normalized.str.lower().isin({"na", "n/a", "null", "none", "nan", ""})
        )

    # In the full export, Previous_pdn=TOTAL identifies aggregate rows. The
    # secondary rejection dataset uses that same text as a valid field value,
    # so only apply this rule when the full-export schema is present.
    aggregate_mask = pd.Series(False, index=df_clean.index)
    if {"Rej_and_cut_Piece", "Total_pdn_m/c"}.issubset(df_clean.columns):
        aggregate_mask = df_clean["Previous_pdn"].astype("string").str.upper().eq("TOTAL").fillna(False) if "Previous_pdn" in df_clean.columns else aggregate_mask
    summary["aggregate_rows_removed"] = int(aggregate_mask.sum())
    if aggregate_mask.any():
        df_clean = df_clean.loc[~aggregate_mask].copy()

    # Ensure numeric columns are converted safely
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            # Try to convert to numeric if column looks mostly numeric
            try:
                # remove any trailing non-numeric symbols if clean
                cleaned_series = df_clean[col].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
                if cleaned_series.notna().mean() > 0.8:
                    df_clean[col + "_num"] = pd.to_numeric(cleaned_series, errors="coerce")
            except Exception:
                pass

    # Handle standard columns in the textile dataset
    if "warp_count" in df_clean.columns and df_clean["warp_count"].dtype == object:
        df_clean["warp_count_num"] = df_clean["warp_count"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    
    if "weft_count" in df_clean.columns:
        df_clean["weft_count"] = pd.to_numeric(df_clean["weft_count"], errors="coerce")

    if "epi" in df_clean.columns:
        df_clean["epi"] = pd.to_numeric(df_clean["epi"], errors="coerce")

    if "ppi" in df_clean.columns:
        df_clean["ppi"] = pd.to_numeric(df_clean["ppi"], errors="coerce")

    # Create binary rejection target if 'Rejection' column exists
    if "Rejection" in df_clean.columns:
        df_clean["Rejection_Qty"] = pd.to_numeric(df_clean["Rejection"], errors="coerce").fillna(0)
        df_clean["Has_Rejection"] = (df_clean["Rejection_Qty"] > 0).astype(int)
    elif "Rej_and_cut_Piece" in df_clean.columns:
        df_clean["Rejection_Qty"] = pd.to_numeric(df_clean["Rej_and_cut_Piece"], errors="coerce").fillna(0)
        df_clean["Has_Rejection"] = (df_clean["Rejection_Qty"] > 0).astype(int)

    # Impute numeric NaNs with median
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df_clean[c].isna().any():
            df_clean[c] = df_clean[c].fillna(df_clean[c].median())

    # Impute categorical NaNs with mode or 'Unknown'
    cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    for c in cat_cols:
        if df_clean[c].isna().any():
            mode_val = df_clean[c].mode()[0] if not df_clean[c].mode().empty else "Unknown"
            df_clean[c] = df_clean[c].fillna(mode_val)

    # Feature Engineering for Textile Manufacturing
    # Warp Cover Factor = EPI / sqrt(Warp_Count) (approx)
    if "warp_count_num" in df_clean.columns:
        w_count = df_clean["warp_count_num"]
    elif "warp_count" in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean["warp_count"]):
        w_count = df_clean["warp_count"]
    else:
        w_count = pd.Series(40.0, index=df_clean.index)

    if "weft_count" in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean["weft_count"]):
        weft_count = df_clean["weft_count"]
    else:
        weft_count = pd.Series(40.0, index=df_clean.index)
    
    if "epi" in df_clean.columns and "ppi" in df_clean.columns:
        w_cnt_safe = np.maximum(w_count.fillna(40.0), 1.0)
        weft_cnt_safe = np.maximum(weft_count.fillna(40.0), 1.0)
        df_clean["warp_cover_factor"] = df_clean["epi"] / np.sqrt(w_cnt_safe)
        df_clean["weft_cover_factor"] = df_clean["ppi"] / np.sqrt(weft_cnt_safe)
        df_clean["total_fabric_density"] = df_clean["epi"] + df_clean["ppi"]

    summary["final_rows"] = len(df_clean)
    df_clean.attrs["preprocessing_summary"] = summary
    return df_clean

def get_dataset_summary(df):
    """
    Returns high-level dataset metrics and information for the dashboard.
    """
    total_rows, total_cols = df.shape
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    # Calculate target statistics if present
    rejection_rate = None
    if "Has_Rejection" in df.columns:
        rejection_rate = float(df["Has_Rejection"].mean() * 100)
    elif "Rejection" in df.columns and pd.api.types.is_numeric_dtype(df["Rejection"]):
        rejection_rate = float((df["Rejection"] > 0).mean() * 100)

    total_production_yds = None
    if "Total_Pdn(yds)" in df.columns:
        total_production_yds = float(df["Total_Pdn(yds)"].sum())
    elif "Total_pdn_per_order" in df.columns:
        total_production_yds = float(df["Total_pdn_per_order"].sum())

    preprocessing_summary = dict(df.attrs.get("preprocessing_summary", {}))
    return {
        "rows": total_rows,
        "cols": total_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "rejection_rate": rejection_rate,
        "total_production_yds": total_production_yds,
        "preprocessing_summary": preprocessing_summary,
    }
