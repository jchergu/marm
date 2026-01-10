import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .config import (
    OUT_CLUS,
    KMEANS_K,
    KMEANS_INIT,
    KMEANS_N_INIT,
    KMEANS_MAX_ITER,
    KMEANS_RANDOM_STATE,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    DBSCAN_METRIC,
    PCA_PLOT_COMPONENTS,
)


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _plot_clusters(X_2d, labels, title, out_path):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        s=10,
        alpha=0.7,
    )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_kmeans(X_df: pd.DataFrame):
    print("[clustering] running K-Means")

    pca = PCA(n_components=PCA_PLOT_COMPONENTS, random_state=42)
    X_2d = pca.fit_transform(X_df)

    for k in KMEANS_K:
        run_dir = OUT_CLUS / f"kmeans_k={k}"
        _ensure_dir(run_dir)

        model = KMeans(
            n_clusters=k,
            init=KMEANS_INIT,
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
            random_state=KMEANS_RANDOM_STATE,
        )

        labels = model.fit_predict(X_df)

        # --- metrics ---
        inertia = model.inertia_
        sil = silhouette_score(X_df, labels)

        # --- save outputs ---
        pd.DataFrame({"cluster": labels}).to_csv(run_dir / "labels.csv", index=False)

        with open(run_dir / "metrics.txt", "w") as f:
            f.write(f"inertia: {inertia}\n")
            f.write(f"silhouette: {sil}\n")

        _plot_clusters(
            X_2d,
            labels,
            title=f"K-Means (k={k})",
            out_path=run_dir / "scatter.png",
        )

        print(f"[clustering]   - K-Means k={k} done")


def run_dbscan(X_df: pd.DataFrame):
    print("[clustering] running DBSCAN")

    pca = PCA(n_components=PCA_PLOT_COMPONENTS, random_state=42)
    X_2d = pca.fit_transform(X_df)

    for eps in DBSCAN_EPS:
        for min_samples in DBSCAN_MIN_SAMPLES:
            run_dir = OUT_CLUS / f"dbscan_eps={eps}_min={min_samples}"
            _ensure_dir(run_dir)

            model = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=DBSCAN_METRIC,
            )

            labels = model.fit_predict(X_df)

            # noise label = -1
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            # silhouette only if meaningful
            sil = None
            if n_clusters > 1:
                sil = silhouette_score(X_df, labels)

            # --- save outputs ---
            pd.DataFrame({"cluster": labels}).to_csv(run_dir / "labels.csv", index=False)

            with open(run_dir / "stats.txt", "w") as f:
                f.write(f"clusters: {n_clusters}\n")
                f.write(f"noise points: {n_noise}\n")
                if sil is not None:
                    f.write(f"silhouette: {sil}\n")

            _plot_clusters(
                X_2d,
                labels,
                title=f"DBSCAN (eps={eps}, min={min_samples})",
                out_path=run_dir / "scatter.png",
            )

            print(f"[clustering]   - DBSCAN eps={eps}, min={min_samples} done")


def run_clustering(X_df: pd.DataFrame):
    print("[clustering] starting comparison")
    _ensure_dir(OUT_CLUS)

    run_kmeans(X_df)
    run_dbscan(X_df)

    print("[clustering] done")

