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
KMEANS_K = [2, 3, 5, 8]              # number of clusters
KMEANS_INIT = "k-means++"
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300
KMEANS_RANDOM_STATE = 42

# dbscan
DBSCAN_EPS = [0.15, 0.3, 0.5, 0.8, 0.9]      # neighborhood radius
DBSCAN_MIN_SAMPLES = [5, 10]
DBSCAN_METRIC = "euclidean"

# plotting
PCA_PLOT_COMPONENTS = 2           # for plotting clusters

# random forest
OUT_RND = OUT_PATH / "classification"

RF_N_ESTIMATORS = 300
RF_N_ESTIMATORS_CV = 100
RF_MAX_DEPTH = None
RF_MAX_FEATURES = "sqrt"
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF = 1
RF_CLASS_WEIGHT = "balanced"
RF_BOOTSTRAP = True
RF_RANDOM_STATE = 42

# evaluation
RF_TEST_SIZE = 0.2
RF_USE_CV = True
RF_N_FOLDS = 3

