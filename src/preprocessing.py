import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    df = df.copy()

    # remove first column (it's an index)
    df = df.iloc[:, 1:]

    # columns to remove
    drop_cols = {"track_id", "track_name", "album_name", "artists", "popularity", "explicit", "track_genre"}
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Handle missing values
    df = df.dropna()

    # Convert types
    df = df.convert_dtypes()

    # Columns to normalize:
    # duration_ms, loudness, tempo, time_signature(?), key(?)

    return df
