from src.dataset_loader import load_dataset
from src.consistency_check import check_consistency
from src.preprocessing import preprocess
from src.header_print import print_header
from src.discretization import create_arm_dataset
from src.fpg import apply_fpgrowth
from src.arm_summary import summarize_arm_results
from src.clustering import run_clustering
from pathlib import Path

from src.config import MIN_SUPPORT, MIN_CONFIDENCE, OUT_PATH, OUT_ARM, OUT_CLUS

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

    # create ARM-ready datasets from the processed_df (before scaling)
    print("[main] creating arm dataset...")
    onehot_df, transactions, paths = create_arm_dataset(res["processed_df"], numeric_bins=4, output_dir=OUT_ARM)

    print(f"[main] ARM one-hot shape: {onehot_df.shape}")
    if paths:
        print(f"[main] Wrote one-hot CSV: {paths.get('onehot_csv')}")
        print(f"[main] Wrote transactions TXT: {paths.get('transactions_txt')}\n")

    # run FP-growth on the generated transactions and persist results
    try:
        fpg_paths = apply_fpgrowth(transactions=transactions, output_dir=OUT_ARM, min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE)
        print(f"\n[main] FP-growth results written:")
        print(f"  frequent itemsets: {fpg_paths.get('frequent_itemsets')}")
        print(f"  rules: {fpg_paths.get('rules')}")
        print(f"  meta: {fpg_paths.get('meta')}")
    except Exception as e:
        print(f"[main] FP-growth failed: {e}")   

    img_dir = OUT_ARM / "images"
    
    # plots and txt rules summary
    summarize_arm_results(
        itemsets_csv=fpg_paths["frequent_itemsets"],
        rules_csv=fpg_paths["rules"],
        output_dir=img_dir,
    )

    run_clustering(res["X"])

if __name__ == "__main__":
    main()
