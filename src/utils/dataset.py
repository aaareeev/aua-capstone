from pathlib import Path
import os

DATASET_ROOT = Path(os.environ["DATASET_ROOT"])
DATASET_PATH = DATASET_ROOT / "hf" / "merrec"

FILES = list(DATASET_PATH.rglob("*.parquet"))
