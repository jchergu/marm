import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DROP_COLS = {"track_id", "track_name", "album_name", "artists", "popularity", "explicit", "track_genre"}


def _infer_numeric_columns(df):
    # try to coerce columns to numeric if majority parseable
    nums = []
    for c in df.columns:
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().sum() >= (0.5 * len(df)):  # at least 50% numeric -> treat as numeric
            df[c] = coerced
            nums.append(c)
    return df, nums


def _winsorize(df, numeric_cols, factor=1.5):
    # clip values to [Q1 - factor*IQR, Q3 + factor*IQR] per column
    for c in numeric_cols:
        col = df[c].dropna()
        if col.empty:
            continue
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        df[c] = df[c].clip(lower=lower, upper=upper)
    return df


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

    # remove first column (index column present before track_id)
    if df.shape[1] > 0:
        df = df.iloc[:, 1:]
    print("[preprocessing] removed first index column")

    # drop the specified metadata columns if present
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    if remove_duplicates:
        df = df.drop_duplicates()

    # try to infer numeric columns by coercion
    df = df.convert_dtypes()
    df, inferred_numeric = _infer_numeric_columns(df)
    print("[preprocessing] infer numeric column")

    # identify columns
    duration_col = "duration_ms" if "duration_ms" in df.columns else None
    # numeric columns after inference
    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns.tolist()]
    if duration_col:
        numeric_cols = [c for c in numeric_cols if c != duration_col]
    # categorical columns are the rest
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != duration_col]

    # handle outliers by winsorization (keeps rows)
    if winsorize_outliers and numeric_cols:
        df = _winsorize(df, numeric_cols, factor=outlier_factor)
    print("[preprocessing] outlier handling")

    # define transformers
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

    X = ct.fit_transform(df)

    print("[preprocessing] normalization")

    # feature names
    try:
        feature_names = ct.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    X_df = pd.DataFrame(X, columns=feature_names, index=df.index)

    pca = None
    if pca_components:
        pca = PCA(n_components=pca_components, random_state=random_state)
        X_reduced = pca.fit_transform(X_df)
        # name components
        comp_names = [f"PC{i+1}" for i in range(X_reduced.shape[1])]
        X_df = pd.DataFrame(X_reduced, columns=comp_names, index=df.index)
        feature_names = comp_names

    print("[preprocessing] train/test/val splitting...")

    # train/test(/val) split
    if val_size:
        # split into train+val and test, then split train into train/val
        X_temp, X_test = train_test_split(X_df, test_size=test_size, random_state=random_state)
        val_ratio = val_size / (1 - test_size)
        X_train, X_val = train_test_split(X_temp, test_size=val_ratio, random_state=random_state)
    else:
        X_train, X_test = train_test_split(X_df, test_size=test_size, random_state=random_state)
        X_val = None

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
