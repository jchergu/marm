from .dataset_loader import load_dataset
import pandas as pd

df = load_dataset()

def _fmt_num(x):
    try:
        if pd.isna(x):
            return "NaN"
        fx = float(x)
        if fx.is_integer():
            return str(int(fx))
        return f"{fx:.3f}".rstrip('0').rstrip('.')
    except Exception:
        return str(x)

def check_consistency(df):
    print("\n[consistency check] starting...")

    print(df.head())
    df.info()

    # columns to exclude from analysis
    exclude = {df.columns[0], "track_id", "artists", "album_name", "track_name", "explicit", "track_genre", "popularity"}

    for col in df.columns:
        if col in exclude:
            continue

        series = df[col].dropna()
        print(f"\nColumn: {col}")
        if pd.api.types.is_numeric_dtype(series):
            print(f"  min:  {_fmt_num(series.min())}")
            print(f"  max:  {_fmt_num(series.max())}")
            print(f"  mean: {_fmt_num(series.mean())}")
            print(f"  std:  {_fmt_num(series.std())}")
        else:
            numeric = pd.to_numeric(series, errors='coerce')
            if numeric.notna().any():
                print("  (converted to numeric)")
                print(f"  min:  {_fmt_num(numeric.min())}")
                print(f"  max:  {_fmt_num(numeric.max())}")
                print(f"  mean: {_fmt_num(numeric.mean())}")
                print(f"  std:  {_fmt_num(numeric.std())}")
            else:
                try:
                    print(f"  min:  {series.min()}")
                    print(f"  max:  {series.max()}")
                except Exception:
                    print("  min/max not available for this column")

    print("\n[consistency check] done")
