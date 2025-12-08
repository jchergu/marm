import pandas as pd
from pathlib import Path

def _make_bin_labels(n):
    if n == 2:
        return ["low", "high"]
    if n == 3:
        return ["low", "medium", "high"]
    if n == 4:
        return ["very_low", "low", "high", "very_high"]
    if n == 5:
        return ["very_low", "low", "medium", "high", "very_high"]
    return [f"bin_{i+1}" for i in range(n)]


def create_arm_dataset(df, *, numeric_bins=2, output_dir=None,
                       keep_high_only=False,
                       min_item_count=None,
                       min_item_freq=None):
    """
    Create ARM datasets.
    - numeric_bins: number of bins to compute (quantile). For binary split set 2.
    - keep_high_only: if True, for numeric columns only produce a single "<col>_high" indicator
      (value 1 for the top bin), reducing columns.
    - min_item_count / min_item_freq: prune one-hot columns that occur less than the given
      absolute count or relative frequency (applied after one-hot creation).
    Returns: (onehot_df, transactions_list, paths_dict)
    """
    df = df.copy()
    output_dir = Path(output_dir) if output_dir is not None else None

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    parts = []

    for c in num_cols:
        ser = df[c]
        if ser.nunique(dropna=True) <= 1:
            # constant column -> skip or keep as is (skip here)
            continue

        if keep_high_only:
            # single binary indicator: 1 if value in top half (median/quantile)
            try:
                # use median split
                thresh = ser.quantile(0.5)
                high = (ser > thresh).astype(int)
            except Exception:
                high = (ser > ser.mean()).astype(int)
            colname = f"{c}_high"
            parts.append(high.rename(colname).to_frame())
        else:
            # regular binning into numeric_bins with readable labels
            if ser.nunique(dropna=True) <= numeric_bins:
                binned = ser.astype("Int64").astype(str)
            else:
                labels = _make_bin_labels(numeric_bins)
                try:
                    binned = pd.qcut(ser, q=numeric_bins, labels=labels, duplicates="raise")
                except Exception:
                    try:
                        binned = pd.qcut(ser, q=numeric_bins, duplicates="drop")
                        cats = binned.cat.categories
                        new_labels = _make_bin_labels(len(cats))
                        binned = binned.cat.rename_categories(new_labels)
                    except Exception:
                        new_labels = _make_bin_labels(numeric_bins)
                        binned = pd.cut(ser, bins=numeric_bins, labels=new_labels)
                binned = binned.astype(str)
            parts.append(pd.get_dummies(binned, prefix=c))

    # categorical columns
    for c in cat_cols:
        ser = df[c].astype(str)
        parts.append(pd.get_dummies(ser, prefix=c))

    if parts:
        onehot = pd.concat(parts, axis=1)
    else:
        onehot = pd.DataFrame(index=df.index)

    # Ensure 0/1 ints
    onehot = onehot.fillna(0).astype(int)

    # Prune rare items if requested
    if min_item_freq is not None:
        min_item_count = max(1, int(min_item_freq * len(df)))
    if min_item_count is not None:
        counts = onehot.sum(axis=0)
        keep = counts[counts >= min_item_count].index
        removed = set(onehot.columns) - set(keep)
        if removed:
            print(f"[create_arm_dataset] Pruning {len(removed)} rare item columns (min_count={min_item_count})")
            onehot = onehot[keep]

    # transactions
    transactions = []
    for _, row in onehot.iterrows():
        items = list(row[row == 1].index)
        transactions.append(items)

    paths = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        onehot_path = output_dir / "arm_onehot.csv"
        trans_path = output_dir / "arm_transactions.txt"

        onehot.to_csv(onehot_path, index=False)
        with trans_path.open("w", encoding="utf-8") as fh:
            for items in transactions:
                fh.write(",".join(items) + "\n")

        paths["onehot_csv"] = str(onehot_path)
        paths["transactions_txt"] = str(trans_path)

    return onehot, transactions, paths