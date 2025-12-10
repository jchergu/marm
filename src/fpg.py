from pathlib import Path
import pandas as pd
import json
import time
from typing import List, Optional, Dict, Tuple

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


def _resolve_output_directory(output_dir: Optional[Path]) -> Path:
    """Resolve and create output directory if needed"""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_onehot_dataframe(onehot_path: Optional[Path], output_dir: Path) -> Optional[pd.DataFrame]:
    """Load and convert one-hot encoded CSV to boolean DataFrame"""
    if onehot_path is None:
        candidate = output_dir / "arm_onehot.csv"
        if candidate.exists():
            onehot_path = candidate
        else:
            return None
    
    onehot_path = Path(onehot_path)
    if not onehot_path.exists():
        raise FileNotFoundError(f"onehot file not found: {onehot_path}")
    
    df = pd.read_csv(onehot_path)
    df_bool = _ensure_bool_df(df)
    print(f"[fpg] Loaded one-hot CSV: {onehot_path} shape={df_bool.shape}")
    return df_bool


def _load_transactions(transactions_path: Optional[Path], output_dir: Path) -> List[List[str]]:
    """Load transactions from text file"""
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
    return transactions


def _transactions_to_boolean_df(transactions: List[List[str]]) -> pd.DataFrame:
    """Convert transaction list to boolean DataFrame using TransactionEncoder"""
    if not transactions:
        raise ValueError("no transactions to process")
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_bool = pd.DataFrame(te_ary, columns=te.columns_).astype(int)
    print(f"[fpg] Built boolean DataFrame from transactions: shape={df_bool.shape}")
    return df_bool


def _prepare_boolean_dataframe(
    transactions: Optional[List[List[str]]],
    transactions_path: Optional[Path],
    onehot_path: Optional[Path],
    output_dir: Path
) -> pd.DataFrame:
    """Prepare boolean DataFrame from either one-hot CSV or transactions"""
    # Try loading from one-hot first
    df_bool = _load_onehot_dataframe(onehot_path, output_dir)
    
    if df_bool is not None:
        return df_bool
    
    # Fall back to transactions
    if transactions is None:
        transactions = _load_transactions(transactions_path, output_dir)
    
    return _transactions_to_boolean_df(transactions)


def _check_matrix_size(df_bool: pd.DataFrame) -> None:
    """Check and warn if boolean matrix is too large"""
    n_tx, n_items = df_bool.shape
    est_cells = n_tx * n_items
    print(f"[fpg] n_tx={n_tx}, n_items={n_items}, est_cells={est_cells}")
    
    if est_cells > 10_000_000:
        print("[fpg] WARNING: boolean matrix large — consider increasing min_support or pruning rare items")


def _mine_frequent_itemsets(df_bool: pd.DataFrame, min_support: float) -> pd.DataFrame:
    """Mine frequent itemsets using FP-Growth"""
    freq = fpgrowth(df_bool, min_support=min_support, use_colnames=True).copy()
    
    if not freq.empty:
        freq["support_count"] = freq["itemsets"].apply(
            lambda s: int(df_bool[list(s)].all(axis=1).sum())
        )
        freq = freq.sort_values("support", ascending=False).reset_index(drop=True)
    
    return freq


def _generate_association_rules(freq: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    """Generate association rules from frequent itemsets"""
    if freq.empty:
        return pd.DataFrame()
    
    try:
        rules = association_rules(freq, metric="confidence", min_threshold=min_confidence).copy()
        rules["antecedents_str"] = rules["antecedents"].apply(
            lambda s: "|".join(sorted(list(s)))
        )
        rules["consequents_str"] = rules["consequents"].apply(
            lambda s: "|".join(sorted(list(s)))
        )
        return rules
    except Exception as e:
        print(f"[fpg] association_rules generation failed: {e}")
        return pd.DataFrame()


def _prepare_output_paths(
    output_dir: Path,
    min_support: float,
    min_confidence: float,
    timestamp: int
) -> Dict[str, Path]:
    """Generate output file paths"""
    return {
        "frequent": output_dir / f"arm_frequent_itemsets_mlxtend_{int(min_support*1000)}_{timestamp}.csv",
        "rules": output_dir / f"arm_association_rules_mlxtend_{int(min_confidence*1000)}_{timestamp}.csv",
        "meta": output_dir / f"arm_fpg_meta_mlxtend_{timestamp}.json"
    }


def _save_frequent_itemsets(freq: pd.DataFrame, output_path: Path) -> None:
    """Save frequent itemsets to CSV"""
    if not freq.empty:
        freq_out = freq.copy()
        freq_out["itemset_str"] = freq_out["itemsets"].apply(
            lambda s: "|".join(sorted(list(s))) if s is not None else ""
        )
        freq_out.to_csv(output_path, index=False)
    else:
        pd.DataFrame(columns=["itemset_str", "support", "support_count"]).to_csv(
            output_path, index=False
        )


def _save_association_rules(rules: pd.DataFrame, output_path: Path) -> None:
    """Save association rules to CSV"""
    if not rules.empty:
        rules.to_csv(output_path, index=False)
    else:
        pd.DataFrame(
            columns=["antecedents_str", "consequents_str", "confidence", "support"]
        ).to_csv(output_path, index=False)


def _save_metadata(
    meta_path: Path,
    n_transactions: int,
    n_items: int,
    min_support: float,
    min_confidence: float,
    n_frequent: int,
    n_rules: int,
    paths: Dict[str, Path],
    start_time: float,
    duration: float
) -> None:
    """Save mining metadata to JSON"""
    meta = {
        "n_transactions": n_transactions,
        "n_items": n_items,
        "min_support": min_support,
        "min_confidence": min_confidence,
        "n_frequent_itemsets": n_frequent,
        "n_rules": n_rules,
        "freq_path": str(paths["frequent"]),
        "rules_path": str(paths["rules"]),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "duration_seconds": duration,
    }
    
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def apply_fpgrowth(
    transactions: Optional[List[List[str]]] = None,
    transactions_path: Optional[Path] = None,
    onehot_path: Optional[Path] = None,
    *,
    min_support: float = 0.02,
    min_confidence: float = 0.5,
    output_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Apply FP-Growth algorithm to mine frequent itemsets and association rules.
    
    Args:
        transactions: List of transactions (list of item lists)
        transactions_path: Path to transactions text file
        onehot_path: Path to one-hot encoded CSV
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold for rules
        output_dir: Directory for output files
        
    Returns:
        Dictionary with paths to generated files
    """
    start_time = time.time()
    
    # Setup output directory
    output_dir = _resolve_output_directory(output_dir)
    
    # Prepare boolean DataFrame
    df_bool = _prepare_boolean_dataframe(transactions, transactions_path, onehot_path, output_dir)
    
    # Check matrix size
    _check_matrix_size(df_bool)
    
    # Mine frequent itemsets
    freq = _mine_frequent_itemsets(df_bool, min_support)
    
    # Generate association rules
    rules = _generate_association_rules(freq, min_confidence)
    
    # Prepare output paths
    timestamp = int(time.time())
    paths = _prepare_output_paths(output_dir, min_support, min_confidence, timestamp)
    
    # Save results
    _save_frequent_itemsets(freq, paths["frequent"])
    _save_association_rules(rules, paths["rules"])
    
    # Save metadata
    duration = time.time() - start_time
    n_tx, n_items = df_bool.shape
    _save_metadata(
        paths["meta"], n_tx, n_items, min_support, min_confidence,
        len(freq), len(rules), paths, start_time, duration
    )
    
    # Print summary
    print(f"[fpg] Found {len(freq)} frequent itemsets and {len(rules)} rules in {duration:.2f}s")
    print(f"[fpg] Results written: {paths['frequent']}, {paths['rules']}, meta: {paths['meta']}")
    
    return {
        "frequent_itemsets": str(paths["frequent"]),
        "rules": str(paths["rules"]),
        "meta": str(paths["meta"])
    }


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