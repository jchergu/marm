from src.dataset_loader import load_dataset
from src.consistency_check import check_consistency
from src.preprocessing import preprocess
from src.header_print import print_header
from src.discretization import create_arm_dataset
from src.fpg import apply_fpgrowth
from pathlib import Path

def main():

    print_header()

    df = load_dataset()

    check_consistency(df)

    res = preprocess(
        df,
        remove_duplicates=True,
        winsorize_outliers=True,
        outlier_factor=1.5,
        pca_components=None,
        test_size=0.2,
        val_size=None,
        random_state=42,
    )

    X = res["X"]
    X_train = res["X_train"]
    X_test = res["X_test"]
    X_val = res["X_val"]

    print(f"[main] Transformed full X: {X.shape}")
    print(f"[main] X_train: {X_train.shape}, X_test: {X_test.shape}, X_val: {None if X_val is None else X_val.shape}")
    print(f"[main] Feature names ({len(res['feature_names'])}): {res['feature_names'][:10]}{'...' if len(res['feature_names'])>10 else ''}")

    # create ARM-ready datasets from the processed_df (before scaling)
    data_dir = Path(__file__).resolve().parent / "data"
    onehot_df, transactions, paths = create_arm_dataset(res["processed_df"], numeric_bins=4, output_dir=data_dir)

    print(f"[main] ARM one-hot shape: {onehot_df.shape}")
    if paths:
        print(f"[main] Wrote one-hot CSV: {paths.get('onehot_csv')}")
        print(f"[main] Wrote transactions TXT: {paths.get('transactions_txt')}")

    # run FP-growth on the generated transactions and persist results
    try:
        fpg_paths = apply_fpgrowth(transactions=transactions, output_dir=data_dir, min_support=0.02, min_confidence=0.5)
        print(f"[main] FP-growth results written:")
        print(f"  frequent itemsets: {fpg_paths.get('frequent_itemsets')}")
        print(f"  rules: {fpg_paths.get('rules')}")
        print(f"  meta: {fpg_paths.get('meta')}")
    except Exception as e:
        print(f"[main] FP-growth failed: {e}")    

if __name__ == "__main__":
    main()