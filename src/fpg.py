from pathlib import Path
import pandas as pd
import json
import time

try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import fpgrowth, association_rules
except Exception as e:
    raise ImportError("mlxtend is required. Install with: pip install mlxtend") from e


def _ensure_bool_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame to proper boolean type for mlxtend"""
    df = df.copy()
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_bool_dtype(ser):
            pass
        elif pd.api.types.is_numeric_dtype(ser):
            df[col] = (ser != 0).astype(bool)  
        else:
            df[col] = ser.fillna("").astype(str).str.len().gt(0).astype(bool) 
    return df


def apply_fpgrowth(
    transactions=None,
    transactions_path=None,
    onehot_path=None,
    *,
    min_support=0.02,
    min_confidence=0.5,
    output_dir=None,
):
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_bool = None

    if onehot_path is None:
        candidate = output_dir / "arm_onehot.csv"
        if candidate.exists():
            onehot_path = candidate

    if onehot_path is not None:
        onehot_path = Path(onehot_path)
        if not onehot_path.exists():
            raise FileNotFoundError(f"onehot file not found: {onehot_path}")
        df = pd.read_csv(onehot_path)
        df_bool = _ensure_bool_df(df)
        print(f"[fpg] Loaded one-hot CSV: {onehot_path} shape={df_bool.shape}")

    if df_bool is None:
        if transactions is None:
            if transactions_path is None:
                transactions_path = output_dir / "arm_transactions.txt"
            transactions_path = Path(transactions_path)
            if not transactions_path.exists():
                raise FileNotFoundError(f"transactions file not found: {transactions_path}")
            transactions = []
            with transactions_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    items = [it.strip() for it in line.strip().split(",") if it.strip()]
                    transactions.append(items)
            print(f"[fpg] Loaded transactions from: {transactions_path} (n_tx={len(transactions)})")

        if not transactions:
            raise ValueError("no transactions to process")

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_bool = pd.DataFrame(te_ary, columns=te.columns_).astype(int)
        print(f"[fpg] Built boolean DataFrame from transactions: shape={df_bool.shape}")

    n_tx, n_items = df_bool.shape
    est_cells = n_tx * n_items
    print(f"[fpg] n_tx={n_tx}, n_items={n_items}, est_cells={est_cells}")
    if est_cells > 10_000_000:
        print("[fpg] WARNING: boolean matrix large — consider increasing min_support or pruning rare items")

    start = time.time()
    freq = fpgrowth(df_bool, min_support=min_support, use_colnames=True).copy()
    if not freq.empty:
        freq["support_count"] = freq["itemsets"].apply(lambda s: int(df_bool[list(s)].all(axis=1).sum()))
        freq = freq.sort_values("support", ascending=False).reset_index(drop=True)

    rules = pd.DataFrame()
    if not freq.empty:
        try:
            rules = association_rules(freq, metric="confidence", min_threshold=min_confidence).copy()
            rules["antecedents_str"] = rules["antecedents"].apply(lambda s: "|".join(sorted(list(s))))
            rules["consequents_str"] = rules["consequents"].apply(lambda s: "|".join(sorted(list(s))))
        except Exception as e:
            print(f"[fpg] association_rules generation failed: {e}")

    duration = time.time() - start
    ts = int(time.time())

    freq_path = output_dir / f"arm_frequent_itemsets_mlxtend_{int(min_support*1000)}_{ts}.csv"
    rules_path = output_dir / f"arm_association_rules_mlxtend_{int(min_confidence*1000)}_{ts}.csv"
    meta_path = output_dir / f"arm_fpg_meta_mlxtend_{ts}.json"

    if not freq.empty:
        freq_out = freq.copy()
        freq_out["itemset_str"] = freq_out["itemsets"].apply(lambda s: "|".join(sorted(list(s))) if s is not None else "")
        freq_out.to_csv(freq_path, index=False)
    else:
        pd.DataFrame(columns=["itemset_str", "support", "support_count"]).to_csv(freq_path, index=False)

    if not rules.empty:
        rules_out = rules.copy()
        rules_out.to_csv(rules_path, index=False)
    else:
        pd.DataFrame(columns=["antecedents_str", "consequents_str", "confidence", "support"]).to_csv(rules_path, index=False)

    meta = {
        "n_transactions": n_tx,
        "n_items": n_items,
        "min_support": min_support,
        "min_confidence": min_confidence,
        "n_frequent_itemsets": int(len(freq)),
        "n_rules": int(len(rules)),
        "freq_path": str(freq_path),
        "rules_path": str(rules_path),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start)),
        "duration_seconds": duration,
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[fpg] Found {len(freq)} frequent itemsets and {len(rules)} rules in {duration:.2f}s")
    print(f"[fpg] Results written: {freq_path}, {rules_path}, meta: {meta_path}")

    return {"frequent_itemsets": str(freq_path), "rules": str(rules_path), "meta": str(meta_path)}


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    onehot = data_dir / "arm_onehot.csv"
    tx = data_dir / "arm_transactions.txt"
    if onehot.exists():
        print(f"[fpg __main__] Detected one-hot file: {onehot}")
    if tx.exists():
        n = sum(1 for _ in tx.open("r", encoding="utf-8"))
        print(f"[fpg __main__] Detected transactions file: {tx} (n_tx={n})")

    out = apply_fpgrowth()
    print(out)  