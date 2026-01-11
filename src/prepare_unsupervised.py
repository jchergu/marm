# src/prepare_unsupervised.py

import pandas as pd

def prepare_unsupervised_data(df: pd.DataFrame, target_col="track_genre"):
    """
    Prepares data for unsupervised learning by:
      - extracting the target column (genre) BEFORE preprocessing
      - returning the original dataframe untouched
      - returning the target aligned by index

    This avoids any leakage into ARM or clustering.
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    # keep a clean copy of the target
    y = df[target_col].copy()

    # keep original df for preprocessing
    X_raw = df.copy()

    return {
        "X_raw": X_raw,
        "y": y,
    }

