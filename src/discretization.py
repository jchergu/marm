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
    # fallback: readable generic labels
    return [f"bin_{i+1}" for i in range(n)]


def create_arm_dataset(df, *, numeric_bins=4, output_dir=None):
    """
    Create datasets ready for association rule mining:
      - discretize numeric columns using quantile bins (pd.qcut)
      - one-hot encode binned numeric + categorical columns
      - write one-hot CSV and transactions TXT to output_dir (if provided)
    Returns: (onehot_df, transactions_list, paths_dict)
    """

    print("\n[create arm dataset] starting...")

    df = df.copy()
    output_dir = Path(output_dir) if output_dir is not None else None

    print("[create arm dataset] discretizing and one-hot encoding...")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    parts = []

    print("[create arm dataset] processing numeric columns...")
    for c in num_cols:
        ser = df[c]
        # if column has very few unique values, keep as categorical string
        if ser.nunique(dropna=True) <= numeric_bins:
            binned = ser.astype("Int64").astype(str)
        else:
            labels = _make_bin_labels(numeric_bins)
            try:
                # try to create quantile bins using the target number of bins and labels
                binned = pd.qcut(ser, q=numeric_bins, labels=labels, duplicates="raise")
            except Exception:
                # fallback: qcut that may drop duplicate bins, then rename categories to readable labels
                try:
                    binned = pd.qcut(ser, q=numeric_bins, duplicates="drop")
                    cats = binned.cat.categories
                    new_labels = _make_bin_labels(len(cats))
                    binned = binned.cat.rename_categories(new_labels)
                except Exception:
                    # last-resort: use cut with generated labels
                    new_labels = _make_bin_labels(numeric_bins)
                    binned = pd.cut(ser, bins=numeric_bins, labels=new_labels)

            # convert categories to string labels
            binned = binned.astype(str)
        parts.append(pd.get_dummies(binned, prefix=c))

    print("[create arm dataset] processing categorical columns...")
    for c in cat_cols:
        ser = df[c].astype(str)
        parts.append(pd.get_dummies(ser, prefix=c))

    if parts:
        onehot = pd.concat(parts, axis=1)
    else:
        onehot = pd.DataFrame(index=df.index)

    # Ensure boolean / binary values 0/1
    onehot = onehot.astype(int)

    # transactions: list of item strings per row
    transactions = []
    for _, row in onehot.iterrows():
        items = list(row[row == 1].index)
        transactions.append(items)

    print("[create arm dataset] writing output files...")
    paths = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        onehot_path = output_dir / "arm_onehot.csv"
        trans_path = output_dir / "arm_transactions.txt"

        onehot.to_csv(onehot_path, index=False)
        # write transactions as comma-separated item lists (one line per transaction)
        with trans_path.open("w", encoding="utf-8") as fh:
            for items in transactions:
                fh.write(",".join(items) + "\n")

        paths["onehot_csv"] = str(onehot_path)
        paths["transactions_txt"] = str(trans_path)

    print("[create arm dataset] done\n")

    return onehot, transactions, paths