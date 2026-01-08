# fp-growth parameters

MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.6
LIFT = 1.2

# output path
# every time parameters change, a new folder is created
from pathlib import Path
OUT_PATH = Path.cwd() / "data" / "processed" / f"{MIN_SUPPORT}_{MIN_CONFIDENCE}"
