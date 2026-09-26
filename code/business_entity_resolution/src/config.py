import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Repo root is 3 levels up from src/config.py (code/business_entity_resolution/src/config.py -> ML_AMAZON)
DEFAULT_BASE_DIR = Path(__file__).resolve().parents[3]
BASE_DIR = Path(os.environ.get("AMAZON_ML_BASE_DIR", str(DEFAULT_BASE_DIR)))


def get_dataset_dir(custom_path: Optional[str] = None) -> Path:
    """
    Discovers the root dataset directory with strict precedence:
    1. Explicit custom_path CLI argument
    2. AMAZON_ML_DATA_DIR environment variable
    3. Confirmed Google Drive dataset paths
    4. Local fallback paths
    """
    # 1. Explicit CLI argument (Highest Priority)
    if custom_path:
        clean_path = str(custom_path).strip().strip('"').strip("'")
        p = Path(clean_path).expanduser().resolve()
        if p.exists():
            return p
        else:
            # If explicit path doesn't exist, log warning but still check if subpath/parent exists
            if (p / "dataset").exists():
                return (p / "dataset").resolve()
            return p

    # 2. Environment Variable
    if "AMAZON_ML_DATA_DIR" in os.environ:
        clean_env = os.environ["AMAZON_ML_DATA_DIR"].strip().strip('"').strip("'")
        p = Path(clean_env).expanduser().resolve()
        if p.exists():
            return p

    # 3. Candidate Search Paths (Google Drive & Local Defaults)
    candidates = [
        Path("/content/drive/MyDrive/DATASET_ML-AMAZON/dataset"),
        Path("/content/drive/MyDrive/DATASET_ML-AMAZON"),
        Path("/content/drive/MyDrive/ML_AMAZON/student_resource/dataset"),
        Path("/content/drive/MyDrive/ML_AMAZON"),
        Path("/content/student_resource/dataset"),
        Path("/content/dataset"),
        BASE_DIR / "student_resource" / "dataset",
        BASE_DIR / "dataset",
    ]

    for cand in candidates:
        if cand.exists():
            # Check if it has test/train directories or files
            if (cand / "test").exists() or (cand / "train").exists():
                return cand.resolve()
            if any(cand.glob("*source1*")):
                return cand.resolve()

    # Default fallback
    return (BASE_DIR / "student_resource" / "dataset").resolve()


def _find_file_candidates(root_dir: Path, subdirs: List[str], file_names: List[str]) -> Optional[Path]:
    """Searches for any matching filename across specified subdirectories or root."""
    # 1. Search in subdirectories
    for sub in subdirs:
        sub_path = root_dir / sub
        if sub_path.exists():
            for fname in file_names:
                candidate = sub_path / fname
                if candidate.exists() and candidate.is_file():
                    return candidate.resolve()
                    
    # 2. Search in root_dir directly
    for fname in file_names:
        candidate = root_dir / fname
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
            
    # 3. Case-insensitive pattern match fallback
    for sub in subdirs:
        sub_path = root_dir / sub
        if sub_path.exists():
            for p in sub_path.iterdir():
                for fname in file_names:
                    if p.name.lower() == fname.lower() and p.is_file():
                        return p.resolve()
                        
    for p in root_dir.iterdir() if root_dir.exists() else []:
        for fname in file_names:
            if p.name.lower() == fname.lower() and p.is_file():
                return p.resolve()

    return None


def resolve_dataset_paths(custom_dir: Optional[str] = None) -> Dict[str, Optional[Path]]:
    """
    Resolves exact file paths for all 7 train and test dataset files with flexible naming.
    """
    data_dir = get_dataset_dir(custom_dir)
    
    # Train files candidates
    s1_train_names = ["train_source1.tsv", "source1.tsv", "train_source1.csv", "source1.csv"]
    s2_train_names = ["train_source2.tsv", "source2.tsv", "train_source2.csv", "source2.csv"]
    s3_train_names = ["train_source3.tsv", "source3.tsv", "train_source3.csv", "source3.csv"]
    gt_train_names = ["train_ground_truth.tsv", "ground_truth.tsv", "train_ground_truth.csv", "ground_truth.csv"]
    
    # Test files candidates
    s1_test_names = ["test_source1.tsv", "source1.tsv", "test_source1.csv", "source1.csv"]
    s2_test_names = ["test_source2.tsv", "source2.tsv", "test_source2.csv", "source2.csv"]
    s3_test_names = ["test_source3.tsv", "source3.tsv", "test_source3.csv", "source3.csv"]

    paths = {
        "data_dir": data_dir,
        "train_source1": _find_file_candidates(data_dir, ["train", "TRAIN"], s1_train_names),
        "train_source2": _find_file_candidates(data_dir, ["train", "TRAIN"], s2_train_names),
        "train_source3": _find_file_candidates(data_dir, ["train", "TRAIN"], s3_train_names),
        "train_ground_truth": _find_file_candidates(data_dir, ["train", "TRAIN"], gt_train_names),
        "test_source1": _find_file_candidates(data_dir, ["test", "TEST"], s1_test_names),
        "test_source2": _find_file_candidates(data_dir, ["test", "TEST"], s2_test_names),
        "test_source3": _find_file_candidates(data_dir, ["test", "TEST"], s3_test_names),
    }
    return paths


def print_dataset_diagnostics(resolved_paths: Dict[str, Optional[Path]]):
    """Prints clear startup diagnostics of all resolved dataset files."""
    print("=" * 80)
    print("DATASET PATH RESOLUTION DIAGNOSTIC:")
    data_dir = resolved_paths.get("data_dir")
    print(f"  DATA DIRECTORY:     {data_dir}")
    print(f"  EXISTS:             {data_dir.exists() if data_dir else False}")
    print("-" * 80)
    
    for key, path in resolved_paths.items():
        if key == "data_dir":
            continue
        status = "EXISTS" if (path and path.exists()) else "MISSING"
        size_str = f"({path.stat().st_size / (1024**2):.2f} MB)" if (path and path.exists()) else ""
        print(f"  {key.upper():<20}: [{status:7s}] {str(path)} {size_str}")
    print("=" * 80)


def stage_test_files_locally(resolved_paths: Dict[str, Optional[Path]], target_dir: Optional[Path] = None) -> Dict[str, Optional[Path]]:
    """
    Optionally stages test TSV files from slow mounts (e.g. Google Drive /content/drive/...)
    to fast local disk (/content/amazon_ml_dataset/ or target_dir) to avoid FUSE latency and connection drops.
    Returns updated resolved_paths pointing to local files.
    """
    import shutil
    import time
    
    if target_dir is None:
        if Path("/content").exists():
            target_dir = Path("/content/amazon_ml_dataset")
        else:
            target_dir = BASE_DIR / ".local_dataset_cache"
            
    target_dir.mkdir(parents=True, exist_ok=True)
    staged_paths = dict(resolved_paths)
    
    test_keys = ["test_source1", "test_source2", "test_source3"]
    print(f"\n[LOCAL STAGING] Staging test files to local high-speed disk: {target_dir} ...")
    
    for k in test_keys:
        src = resolved_paths.get(k)
        if src and src.exists():
            dst = target_dir / src.name
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                print(f"  - {k}: already staged: {dst} ({dst.stat().st_size / (1024**2):.2f} MB)")
            else:
                t0 = time.time()
                print(f"  - Copying {src.name} ({src.stat().st_size / (1024**2):.2f} MB) -> {dst} ...", end="", flush=True)
                shutil.copyfile(src, dst)
                print(f" done in {time.time() - t0:.2f}s")
            staged_paths[k] = dst
            
    staged_paths["data_dir"] = target_dir
    return staged_paths


DATA_DIR = get_dataset_dir()
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

EXPERIMENTS_DIR = BASE_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


