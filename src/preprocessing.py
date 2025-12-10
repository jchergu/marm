import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from .outlier_detection import winsorize_df

DROP_COLS = {"track_id", "track_name", "album_name", "artists", "popularity", "explicit", "track_genre", "time_signature"}


def _infer_numeric_columns(df):
    # try to coerce columns to numeric if majority parseable
    nums = []
    for c in df.columns:
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().sum() >= (0.5 * len(df)):  # at least 50% numeric -> treat as numeric
            df[c] = coerced
            nums.append(c)
    return df, nums


def _phase_remove_columns(df):
    """Phase 1: Remove first index column and configured DROP_COLS."""
    print("[preprocessing] phase 1: removing columns")
    
    # remove first column (index column present before track_id)
    if df.shape[1] > 0:
        df = df.iloc[:, 1:]
    print(f"[preprocessing]   - removing: {DROP_COLS}")

    # drop the specified metadata columns if present
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    
    return df


def _phase_remove_duplicates(df, remove_duplicates=True):
    """Phase 2: Remove duplicate rows."""
    print("[preprocessing] phase 2: duplicate removal")
    
    if remove_duplicates:
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        print(f"[preprocessing]   - removed {removed} duplicate rows")
    else:
        print("[preprocessing]   - skipped (disabled)")
    
    return df


def _phase_infer_dtypes(df):
    """Phase 3: Infer numeric dtypes and convert."""
    print("[preprocessing] phase 3: dtype inference")
    
    df = df.convert_dtypes()
    df, inferred_numeric = _infer_numeric_columns(df)
    print(f"[preprocessing]   - inferred {len(inferred_numeric)} numeric columns")
    
    return df


def _phase_identify_columns(df):
    """Phase 4: Identify numeric and categorical columns."""
    print("[preprocessing] phase 4: column identification")
    
    duration_col = "duration_ms" if "duration_ms" in df.columns else None
    # numeric columns after inference
    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns.tolist()]
    if duration_col:
        numeric_cols = [c for c in numeric_cols if c != duration_col]
    # categorical columns are the rest
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != duration_col]
    
    print(f"[preprocessing]   - numeric: {len(numeric_cols)}, categorical: {len(categorical_cols)}, duration: {1 if duration_col else 0}")
    
    return duration_col, numeric_cols, categorical_cols


def _phase_handle_outliers(df, numeric_cols, winsorize_outliers=True, outlier_factor=1.5):
    """Phase 5: Handle outliers via winsorization."""
    print("[preprocessing] phase 5: outlier handling")
    
    if winsorize_outliers and numeric_cols:
        df = winsorize_df(df, numeric_cols, factor=outlier_factor)
        print(f"[preprocessing]   - winsorized {len(numeric_cols)} numeric columns (factor={outlier_factor})")
    else:
        print("[preprocessing]   - skipped (disabled or no numeric cols)")
    
    return df


def _phase_build_transformers(duration_col, numeric_cols, categorical_cols):
    """Phase 6: Build sklearn transformers for numeric and categorical data."""
    print("[preprocessing] phase 6: building transformers")
    
    transformers = []

    if duration_col:
        duration_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("log1p", FunctionTransformer(np.log1p, validate=False)),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("duration", duration_pipe, [duration_col]))

    if numeric_cols:
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline(
            [("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"))]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))

    ct = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    print(f"[preprocessing]   - created {len(transformers)} transformer(s)")
    
    return ct


def _phase_transform_data(df, ct):
    """Phase 7: Fit and transform data using ColumnTransformer."""
    print("[preprocessing] phase 7: transforming data")
    
    X = ct.fit_transform(df)
    
    # feature names
    try:
        feature_names = ct.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    X_df = pd.DataFrame(X, columns=feature_names, index=df.index)
    print(f"[preprocessing]   - transformed to shape {X_df.shape}")
    
    return X_df, feature_names


def _phase_apply_pca(X_df, pca_components, random_state=42):
    """Phase 8: Apply PCA for dimensionality reduction (optional)."""
    print("[preprocessing] phase 8: PCA dimensionality reduction")
    
    pca = None
    feature_names = None
    
    if pca_components:
        pca = PCA(n_components=pca_components, random_state=random_state)
        X_reduced = pca.fit_transform(X_df)
        # name components
        comp_names = [f"PC{i+1}" for i in range(X_reduced.shape[1])]
        X_df = pd.DataFrame(X_reduced, columns=comp_names, index=X_df.index)
        feature_names = comp_names
        variance = sum(pca.explained_variance_ratio_)
        print(f"[preprocessing]   - applied PCA ({pca_components} components, {variance:.2%} variance)")
    else:
        print("[preprocessing]   - skipped (disabled)")
    
    return X_df, pca, feature_names


def _phase_train_test_split(X_df, test_size=0.2, val_size=None, random_state=42):
    """Phase 9: Split data into train/test(/val) sets."""
    print("[preprocessing] phase 9: train/test/val split")
    
    if val_size:
        # split into train+val and test, then split train into train/val
        X_temp, X_test = train_test_split(X_df, test_size=test_size, random_state=random_state)
        val_ratio = val_size / (1 - test_size)
        X_train, X_val = train_test_split(X_temp, test_size=val_ratio, random_state=random_state)
        print(f"[preprocessing]   - train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    else:
        X_train, X_test = train_test_split(X_df, test_size=test_size, random_state=random_state)
        X_val = None
        print(f"[preprocessing]   - train: {len(X_train)}, test: {len(X_test)}")
    
    return X_train, X_test, X_val




def preprocess(
    df,
    *,
    remove_duplicates=True,
    winsorize_outliers=True,
    outlier_factor=1.5,
    pca_components=None,  # int or None
    test_size=0.2,
    val_size=None,
    random_state=42,
):
    """
    Full preprocessing pipeline:
     - remove first index column and configured DROP_COLS
     - remove duplicates (optional)
     - infer numeric dtypes
     - winsorize numeric outliers (optional)
     - impute/mask missing values and scale numerics
     - log1p-transform duration_ms (if present)
     - one-hot encode categorical cols
     - optional PCA for dimensionality reduction
     - train/test(/val) split of transformed data
    Returns a dict with keys: transformer, pca (or None), X (full transformed DF),
    X_train, X_test, X_val (if requested), feature_names
    """

    print("\n[preprocessing] starting...")

    df = df.copy()

    # Phase 1: Remove columns
    df = _phase_remove_columns(df)
    
    # Phase 2: Remove duplicates
    df = _phase_remove_duplicates(df, remove_duplicates=remove_duplicates)
    
    # Phase 3: Infer dtypes
    df = _phase_infer_dtypes(df)
    
    # Phase 4: Identify columns
    duration_col, numeric_cols, categorical_cols = _phase_identify_columns(df)
    
    # Phase 5: Handle outliers
    df = _phase_handle_outliers(df, numeric_cols, winsorize_outliers=winsorize_outliers, outlier_factor=outlier_factor)
    
    # Phase 6: Build transformers
    ct = _phase_build_transformers(duration_col, numeric_cols, categorical_cols)
    
    # Phase 7: Transform data
    X_df, feature_names = _phase_transform_data(df, ct)
    
    # Phase 8: Apply PCA (optional)
    X_df, pca, pca_feature_names = _phase_apply_pca(X_df, pca_components, random_state=random_state)
    if pca_feature_names:
        feature_names = pca_feature_names
    
    # Phase 9: Train/test/val split
    X_train, X_test, X_val = _phase_train_test_split(X_df, test_size=test_size, val_size=val_size, random_state=random_state)

    print("[preprocessing] done\n")

    return {
        "transformer": ct,
        "pca": pca,
        "X": X_df,
        "X_train": X_train,
        "X_test": X_test,
        "X_val": X_val,
        "feature_names": feature_names.tolist() if hasattr(feature_names, "tolist") else list(feature_names),
        "processed_df": df,
    }
