from src.dataset_loader import load_dataset
from src.consistency_check import check_consistency
from src.preprocessing import preprocess
from src.header_print import print_header
from src.discretization import create_arm_dataset
from src.fpg import apply_fpgrowth
from src.arm_summary import summarize_arm_results
from src.clustering import run_clustering
from pathlib import Path
from src.prepare_unsupervised import prepare_unsupervised_data
from src.randomforest import run_random_forest


from src.config import MIN_SUPPORT, MIN_CONFIDENCE, OUT_PATH, OUT_ARM

def main():

    print_header()

    df = load_dataset()

    check_consistency(df)

    # saves dataset for random forest with genre column (dropped by preprocessing)
    prep_unsup = prepare_unsupervised_data(df, target_col="track_genre")
    df_raw = prep_unsup["X_raw"]
    y_genre = prep_unsup["y"]

    # preprocessing
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

    # association rule mining

    # create ARM-ready datasets from the processed_df (before scaling)
    onehot_df, transactions, paths = create_arm_dataset(res["processed_df"], numeric_bins=4, output_dir=OUT_ARM)
    # run FP-growth on the generated transactions and persist results
    fpg_paths = apply_fpgrowth(transactions=transactions, output_dir=OUT_ARM, min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE)
    # plots and txt rules summary
    summarize_arm_results(itemsets_csv=fpg_paths["frequent_itemsets"],rules_csv=fpg_paths["rules"],output_dir=OUT_ARM / "images")

    # clustering

    run_clustering(res["X"])

    # random forest (classification)
    
    y_genre = y_genre.loc[res["X"].index]
    run_random_forest(X=res["X"], y=y_genre)


if __name__ == "__main__":
    main()
