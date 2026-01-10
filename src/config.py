# fp-growth parameters

MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.6
LIFT = 1.2
LEVERAGE = 0.0

# output path
# every time parameters change, a new folder is created
from pathlib import Path
OUT_PATH = Path.cwd() / "data" / "processed"
OUT_ARM = OUT_PATH / "arm" / f"{MIN_SUPPORT}_{MIN_CONFIDENCE}"

# clustering parameters
OUT_CLUS = OUT_PATH / "clustering"
# kmeans
KMEANS_K = [2]              # number of clusters
KMEANS_INIT = "k-means++"
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300
KMEANS_RANDOM_STATE = 42

# dbscan
DBSCAN_EPS = [0.15, 0.9]      # neighborhood radius
DBSCAN_MIN_SAMPLES = [5, 10]
DBSCAN_METRIC = "euclidean"

# plotting
PCA_PLOT_COMPONENTS = 2           # for plotting clusters
