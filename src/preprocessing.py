import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    df = df.copy()

    # remove first column (it's an index)
    df = df.iloc[:, 1:]

    # Handle missing values
    df = df.dropna()

    # Convert types
    df = df.convert_dtypes()

    # Columns to normalize:
    # duration_ms, loudness, tempo, time_signature(?), key(?)

    return df
