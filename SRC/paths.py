from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
OUTPUTS_DIR = BASE_DIR / "Outputs"
DOCS_DIR = BASE_DIR / "Docs"
EXPERIMENTS_DIR = BASE_DIR / "Experiments"

BASELINE_DIR = OUTPUTS_DIR / "baseline"
CNN_DIR = OUTPUTS_DIR / "cnn"