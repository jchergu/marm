from src.dataset_loader import load_dataset
from src.consistency_check import check_consistency
from src.preprocessing import preprocess
from pathlib import Path

def main():

    header_path = Path(__file__).resolve().parent / "data" / "header.txt"
    try:
        with header_path.open("r", encoding="utf-8") as fh:
            print(fh.read())
    except FileNotFoundError:
        print(f"header file not found: {header_path}")

    print("[load dataset] starting...")
    df = load_dataset()
    print(f"[load dataset] done: {df.shape}")

    print("\n[consistency check] starting...")
    check_consistency(df)
    print("\n[consistency check] done")

    print("\n[preprocessing] starting...")
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
    print("\n[preprocessing] done")

    X = res["X"]
    X_train = res["X_train"]
    X_test = res["X_test"]
    X_val = res["X_val"]

    print(f"Transformed full X: {X.shape}")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, X_val: {None if X_val is None else X_val.shape}")
    print(f"Feature names ({len(res['feature_names'])}): {res['feature_names'][:10]}{'...' if len(res['feature_names'])>10 else ''}")

if __name__ == "__main__":
    main()