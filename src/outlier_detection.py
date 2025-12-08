import pandas as pd
from typing import Iterable


def detect_outliers_iqr(df: pd.DataFrame, cols: Iterable[str], factor: float = 1.5) -> pd.Series:
    """
    Return a boolean mask (True = row has any outlier) using IQR rule for the given columns.
    """
    mask = pd.Series(False, index=df.index)
    for c in cols:
        col = df[c].dropna()
        if col.empty:
            continue
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = mask | (~df[c].between(lower, upper))
    return mask


def winsorize_df(df: pd.DataFrame, cols: Iterable[str], factor: float = 1.5) -> pd.DataFrame:
    """
    Return a copy of df where numeric columns in cols are clipped to [Q1 - factor*IQR, Q3 + factor*IQR].
    Similar to winsorization but uses clipping (keeps rows).
    """
    out = df.copy()
    for c in cols:
        col = out[c].dropna()
        if col.empty:
            continue
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        out[c] = out[c].clip(lower=lower, upper=upper)
    return out