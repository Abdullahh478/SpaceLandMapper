from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
DOCS_DIR = BASE_DIR / "docs"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

BASELINE_DIR = OUTPUTS_DIR / "baseline"
CNN_DIR = OUTPUTS_DIR / "cnn"