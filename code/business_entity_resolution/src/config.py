import os
import sys
from pathlib import Path

# Repo root is 3 levels up from src/config.py (code/business_entity_resolution/src/config.py -> ML_AMAZON)
DEFAULT_BASE_DIR = Path(__file__).resolve().parents[3]
BASE_DIR = Path(os.environ.get("AMAZON_ML_BASE_DIR", str(DEFAULT_BASE_DIR)))

def get_dataset_dir(custom_path=None):
    """
    Discovers the dataset directory across local, Colab, or custom environments.
    """
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
            
    # Check environment variable
    if "AMAZON_ML_DATA_DIR" in os.environ:
        p = Path(os.environ["AMAZON_ML_DATA_DIR"])
        if p.exists():
            return p
            
    # Candidate search paths
    candidates = [
        BASE_DIR / "student_resource" / "dataset",
        Path("/content/student_resource/dataset"),
        Path("/content/dataset"),
        Path("/content/drive/MyDrive/ML_AMAZON/student_resource/dataset"),
        Path("/content/drive/MyDrive/dataset"),
    ]
    
    for cand in candidates:
        if cand.exists() and (cand / "train" / "train_source1.tsv").exists():
            return cand
            
    # Fallback default
    return BASE_DIR / "student_resource" / "dataset"

DATA_DIR = get_dataset_dir()
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

EXPERIMENTS_DIR = BASE_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

