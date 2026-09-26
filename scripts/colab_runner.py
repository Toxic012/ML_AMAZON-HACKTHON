#!/usr/bin/env python3
"""
Colab Runner & Environment Verification for Amazon ML Challenge 2026
Business Entity Resolution

Usage:
  python scripts/colab_runner.py --mode check
  python scripts/colab_runner.py --mode check --data-dir /path/to/dataset
  python scripts/colab_runner.py --mode exp --config experiments/EXP-0001/config.json
"""

import os
import sys
import json
import time
import random
import platform
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add repository root and code directories to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.config import get_dataset_dir, resolve_dataset_paths, BASE_DIR, EXPERIMENTS_DIR, SEED
    from src.normalization import normalize_record, normalize_text, tokenize
    from src.features import compute_features
except ImportError:
    from code.business_entity_resolution.src.config import get_dataset_dir, resolve_dataset_paths, BASE_DIR, EXPERIMENTS_DIR, SEED
    from code.business_entity_resolution.src.normalization import normalize_record, normalize_text, tokenize
    from code.business_entity_resolution.src.features import compute_features


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_git_info():
    info = {"commit": "unknown", "branch": "unknown", "is_dirty": False}
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        info["commit"] = res.stdout.strip()
    except Exception:
        pass
    try:
        res = subprocess.run(["git", "branch", "--show-current"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        info["branch"] = res.stdout.strip()
    except Exception:
        pass
    try:
        res = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        info["is_dirty"] = bool(res.stdout.strip())
    except Exception:
        pass
    return info


def get_hardware_info():
    hw = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_count_logical": os.cpu_count(),
        "total_ram_gb": None,
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_devices": []
    }
    
    # Check RAM
    try:
        import psutil
        hw["total_ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        hw["available_ram_gb"] = round(psutil.virtual_memory().available / (1024 ** 3), 2)
    except ImportError:
        # Fallback for Linux/Colab
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        hw["total_ram_gb"] = round(kb / (1024 ** 2), 2)
                        break
        except Exception:
            pass

    # Check GPU via PyTorch
    try:
        import torch
        hw["torch_version"] = torch.__version__
        hw["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hw["gpu_available"] = True
            hw["gpu_count"] = torch.cuda.device_count()
            hw["cuda_version"] = torch.version.cuda
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                hw["gpu_devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    "multi_processor_count": props.multi_processor_count,
                    "major_capability": props.major,
                    "minor_capability": props.minor
                })
    except ImportError:
        hw["torch_version"] = "not_installed"
        hw["cuda_available"] = False

    return hw


def get_installed_packages():
    key_packages = [
        "torch", "torchvision", "torchaudio",
        "lightgbm", "xgboost", "catboost",
        "polars", "pandas", "numpy", "scipy",
        "scikit-learn", "transformers", "sentence-transformers",
        "faiss-cpu", "faiss-gpu", "rapidfuzz", "joblib"
    ]
    installed = {}
    for pkg in key_packages:
        try:
            import importlib.metadata
            ver = importlib.metadata.version(pkg)
            installed[pkg] = ver
        except Exception:
            try:
                mod_name = pkg.replace("-", "_")
                mod = __import__(mod_name)
                installed[pkg] = getattr(mod, "__version__", "installed")
            except Exception:
                installed[pkg] = "not_installed"
    return installed


def count_lines_fast(file_path):
    """Memory-safe line counter using chunked buffered reads."""
    count = 0
    with open(file_path, "rb") as f:
        buffer_size = 1024 * 1024
        while chunk := f.read(buffer_size):
            count += chunk.count(b"\n")
    return count


def run_tiny_gpu_benchmark(sample_texts=None):
    """Runs a tiny embedding inference benchmark on GPU if CUDA is available."""
    gpu_bench = {
        "cuda_available": False,
        "gpu_model": None,
        "gpu_vram_gb": None,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 16,
        "sample_count": 16,
        "runtime_ms": None,
        "status": "not_run",
        "error": None
    }
    try:
        import torch
        if torch.cuda.is_available():
            gpu_bench["cuda_available"] = True
            gpu_bench["gpu_model"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_bench["gpu_vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
            
            # Use small dummy batch or sample texts
            if not sample_texts:
                sample_texts = [
                    "amazon web services inc seattle wa",
                    "google llc mountain view ca",
                    "microsoft corporation redmond wa",
                    "apple inc cupertino ca"
                ] * 4  # 16 items
            
            gpu_bench["sample_count"] = len(sample_texts)
            
            # Try sentence-transformers if available, else PyTorch embedding layer
            try:
                from sentence_transformers import SentenceTransformer
                t0 = time.time()
                model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda:0")
                t_load = time.time() - t0
                
                t1 = time.time()
                embeddings = model.encode(sample_texts, batch_size=16, show_progress_bar=False, device="cuda:0")
                torch.cuda.synchronize()
                t_infer = time.time() - t1
                
                gpu_bench["runtime_ms"] = round(t_infer * 1000, 2)
                gpu_bench["model_load_sec"] = round(t_load, 2)
                gpu_bench["status"] = "success"
                gpu_bench["embedding_dim"] = int(embeddings.shape[1])
            except Exception as e_st:
                # Fallback to pure PyTorch CUDA tensor matmul/embedding operation
                t0 = time.time()
                dev = torch.device("cuda:0")
                emb = torch.nn.Embedding(1000, 384).to(dev)
                inp = torch.randint(0, 1000, (16, 32), device=dev)
                out = emb(inp).mean(dim=1)
                torch.cuda.synchronize()
                t_infer = time.time() - t0
                gpu_bench["model_name"] = "pytorch_cuda_embedding_fallback"
                gpu_bench["runtime_ms"] = round(t_infer * 1000, 2)
                gpu_bench["status"] = "success_pytorch_fallback"
                gpu_bench["embedding_dim"] = 384
                gpu_bench["note"] = f"sentence-transformers skipped: {e_st}"
        else:
            gpu_bench["status"] = "skipped_cuda_not_available"
            gpu_bench["error"] = "CUDA is not available on this runtime."
    except Exception as e:
        gpu_bench["status"] = "failed"
        gpu_bench["error"] = str(e)
        
    return gpu_bench


def run_tiny_benchmark(data_dir):
    """Runs a 5-stage CPU micro-benchmark on a small controlled sample."""
    bench = {
        "status": "pending",
        "sample_size": 100,
        "stage_1_normalization": {"runtime_ms": 0.0, "records_processed": 0},
        "stage_2_tokenization": {"runtime_ms": 0.0, "total_tokens": 0},
        "stage_3_inverted_index": {"runtime_ms": 0.0, "unique_keys": 0},
        "stage_4_candidate_generation": {"runtime_ms": 0.0, "candidate_pairs_found": 0},
        "stage_5_feature_extraction": {"runtime_ms": 0.0, "pairs_evaluated": 0},
        "total_cpu_time_ms": 0.0,
        "memory_info": {}
    }
    
    train_s1 = Path(data_dir) / "train" / "train_source1.tsv"
    if not train_s1.exists():
        bench["status"] = "skipped_no_data"
        return bench

    import csv
    csv.field_size_limit(sys.maxsize)
    
    # 1. Load raw sample
    raw_rows = []
    with open(train_s1, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= 100:
                break
            raw_rows.append(row)
            
    # Stage 1: Unicode Normalization
    t0 = time.perf_counter()
    normalized_names = []
    for r in raw_rows:
        norm_n = normalize_text(r.get("business_name", ""))
        norm_a = normalize_text(r.get("business_address", ""))
        r["norm_name"] = norm_n
        r["norm_addr"] = norm_a
        normalized_names.append((r["entity_id"], norm_n, norm_a, r.get("country", "")))
    t1 = time.perf_counter()
    bench["stage_1_normalization"]["runtime_ms"] = round((t1 - t0) * 1000, 3)
    bench["stage_1_normalization"]["records_processed"] = len(raw_rows)
    
    # Stage 2: Tokenization
    t2 = time.perf_counter()
    total_tokens = 0
    records = []
    for r in raw_rows:
        n_tok = tokenize(r["norm_name"])
        a_tok = tokenize(r["norm_addr"])
        r["name_tokens"] = n_tok
        r["addr_tokens"] = a_tok
        total_tokens += len(n_tok) + len(a_tok)
        records.append(r)
    t3 = time.perf_counter()
    bench["stage_2_tokenization"]["runtime_ms"] = round((t3 - t2) * 1000, 3)
    bench["stage_2_tokenization"]["total_tokens"] = total_tokens

    # Stage 3: Inverted Index Construction
    t4 = time.perf_counter()
    token_index = {}
    for r in records:
        eid = r["entity_id"]
        for tok in r["name_tokens"]:
            if len(tok) >= 3:
                token_index.setdefault(tok, []).append(eid)
    t5 = time.perf_counter()
    bench["stage_3_inverted_index"]["runtime_ms"] = round((t5 - t4) * 1000, 3)
    bench["stage_3_inverted_index"]["unique_keys"] = len(token_index)

    # Stage 4: Candidate Generation (Token Overlap Blocking)
    t6 = time.perf_counter()
    candidate_pairs = set()
    rec_by_id = {r["entity_id"]: r for r in records}
    for tok, eids in token_index.items():
        if len(eids) > 1 and len(eids) < 50:  # Skip high-frequency stops
            for i in range(len(eids)):
                for j in range(i + 1, len(eids)):
                    candidate_pairs.add(tuple(sorted([eids[i], eids[j]])))
    t7 = time.perf_counter()
    bench["stage_4_candidate_generation"]["runtime_ms"] = round((t7 - t6) * 1000, 3)
    bench["stage_4_candidate_generation"]["candidate_pairs_found"] = len(candidate_pairs)

    # Stage 5: Feature Extraction
    t8 = time.perf_counter()
    pairs_list = list(candidate_pairs)[:50] if candidate_pairs else []
    # If no overlapping pairs in 100 rows, generate pairwise test slice
    if not pairs_list:
        pairs_list = [(records[i]["entity_id"], records[j]["entity_id"]) for i in range(min(5, len(records))) for j in range(i + 1, min(5, len(records)))]
    
    for e1, e2 in pairs_list:
        compute_features(rec_by_id.get(e1, {}), rec_by_id.get(e2, {}))
    t9 = time.perf_counter()
    bench["stage_5_feature_extraction"]["runtime_ms"] = round((t9 - t8) * 1000, 3)
    bench["stage_5_feature_extraction"]["pairs_evaluated"] = len(pairs_list)
    
    bench["total_cpu_time_ms"] = round(((t1 - t0) + (t3 - t2) + (t5 - t4) + (t7 - t6) + (t9 - t8)) * 1000, 3)
    bench["status"] = "success"
    
    # Process memory
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        bench["memory_info"]["rss_mb"] = round(proc.memory_info().rss / (1024 ** 2), 2)
    except Exception:
        pass
        
    return bench



def verify_dataset(data_dir=None):
    resolved = resolve_dataset_paths(data_dir)
    data_path = resolved.get("data_dir")
    path_exists = data_path.exists() if data_path else False
    
    results = {
        "data_dir": str(data_path) if data_path else "",
        "dataset_path_status": "DATASET_PATH_FOUND" if path_exists else "DATASET_PATH_NOT_FOUND",
        "exists": path_exists,
        "files": {},
        "all_required_present": False,
        "total_size_mb": 0.0
    }
    
    file_keys = [
        "train_source1", "train_source2", "train_source3", "train_ground_truth",
        "test_source1", "test_source2", "test_source3"
    ]
    
    all_present = True
    total_bytes = 0
    
    for key in file_keys:
        path = resolved.get(key)
        exists = path is not None and path.exists()
        size_bytes = path.stat().st_size if exists else 0
        total_bytes += size_bytes
        if not exists:
            all_present = False
            
        file_info = {
            "path": str(path) if path else "NOT_RESOLVED",
            "exists": exists,
            "size_mb": round(size_bytes / (1024 ** 2), 2),
            "header": None,
            "approx_rows": None
        }
        
        # Read header and count lines safely without loading entire file in memory
        if exists:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    header_line = f.readline().strip()
                    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
                    file_info["header"] = header_line.split(delimiter)
                lines = count_lines_fast(path)
                file_info["line_count"] = lines
                file_info["approx_rows"] = max(0, lines - 1)  # minus header
            except Exception as e:
                file_info["read_error"] = str(e)
                
        results["files"][key] = file_info

    results["all_required_present"] = (all_present and path_exists)
    results["total_size_mb"] = round(total_bytes / (1024 ** 2), 2)
    return results


def run_environment_check(data_dir=None, save_json=True):
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026 — COLAB ENVIRONMENT VERIFICATION")
    print("=" * 70)
    
    set_seed(SEED)
    git_info = get_git_info()
    hw_info = get_hardware_info()
    pkg_info = get_installed_packages()
    
    resolved_data_dir = get_dataset_dir(data_dir)
    ds_info = verify_dataset(resolved_data_dir)
    
    print(f"\n[1] Git State:")
    print(f"    Commit:          {git_info['commit']}")
    print(f"    Branch:          {git_info['branch']}")
    print(f"    Dirty Status:    {git_info['is_dirty']}")

    
    print(f"\n[2] Hardware & Runtime:")
    print(f"    Platform:       {hw_info['platform']}")
    print(f"    Python:         {hw_info['python_version']}")
    print(f"    CPU Logical:    {hw_info['cpu_count_logical']} cores")
    print(f"    Total RAM:      {hw_info['total_ram_gb']} GB")
    print(f"    CUDA Available: {hw_info.get('cuda_available', False)}")
    if hw_info.get("gpu_available"):
        for dev in hw_info["gpu_devices"]:
            print(f"    GPU [{dev['index']}]:      {dev['name']} ({dev['total_memory_gb']} GB VRAM)")
    else:
        print("    GPU:            No CUDA GPU detected (CPU mode active)")
        
    print(f"\n[3] Key Packages:")
    for pkg in ["torch", "lightgbm", "polars", "pandas", "numpy", "transformers", "faiss-cpu", "faiss-gpu"]:
        print(f"    {pkg:20s}: {pkg_info.get(pkg, 'not_installed')}")
        
    print(f"\n[4] Dataset Status:")
    print(f"    Search Path:    {resolved_data_dir}")
    print(f"    Path Status:    {ds_info['dataset_path_status']}")
    print(f"    All Present:    {'YES' if ds_info['all_required_present'] else 'NO (Missing files or path not found)'}")
    print(f"    Total Size:     {ds_info['total_size_mb']} MB")
    for k, v in ds_info["files"].items():
        status_sym = "OK" if v["exists"] else "MISSING"
        rows_str = f"({v['approx_rows']:,} rows)" if v.get("approx_rows") is not None else ""
        header_str = f" | cols: {len(v['header'])}" if v.get("header") else ""
        print(f"    - {k:20s}: [{status_sym:7s}] {v['size_mb']:8.2f} MB  {rows_str}{header_str}")

    print(f"\n[5] GPU Inference Benchmark:")
    gpu_bench = run_tiny_gpu_benchmark()
    print(f"    CUDA Available: {gpu_bench['cuda_available']}")
    if gpu_bench["cuda_available"]:
        print(f"    GPU Model:      {gpu_bench['gpu_model']} ({gpu_bench['gpu_vram_gb']} GB VRAM)")
        print(f"    Model Name:     {gpu_bench['model_name']}")
        print(f"    Batch / Samples:{gpu_bench['batch_size']} / {gpu_bench['sample_count']}")
        print(f"    Inference Time: {gpu_bench['runtime_ms']} ms")
        print(f"    Status:         {gpu_bench['status']}")
    else:
        print(f"    Status:         Skipped ({gpu_bench.get('error')})")

    print(f"\n[6] CPU 5-Stage Micro-Benchmark (Controlled Sample N=100):")
    bench_info = run_tiny_benchmark(resolved_data_dir)
    print(f"    Status:                  {bench_info['status']}")
    print(f"    Stage 1 - Normalization: {bench_info['stage_1_normalization']['runtime_ms']} ms ({bench_info['stage_1_normalization']['records_processed']} records)")
    print(f"    Stage 2 - Tokenization:  {bench_info['stage_2_tokenization']['runtime_ms']} ms ({bench_info['stage_2_tokenization']['total_tokens']} tokens)")
    print(f"    Stage 3 - Inverted Index:{bench_info['stage_3_inverted_index']['runtime_ms']} ms ({bench_info['stage_3_inverted_index']['unique_keys']} unique keys)")
    print(f"    Stage 4 - Candidate Gen: {bench_info['stage_4_candidate_generation']['runtime_ms']} ms ({bench_info['stage_4_candidate_generation']['candidate_pairs_found']} pairs found)")
    print(f"    Stage 5 - Feature Extr:  {bench_info['stage_5_feature_extraction']['runtime_ms']} ms ({bench_info['stage_5_feature_extraction']['pairs_evaluated']} pairs evaluated)")
    print(f"    Total CPU Pipeline Time: {bench_info['total_cpu_time_ms']} ms")
    if bench_info.get("memory_info"):
        print(f"    Process RSS Memory:      {bench_info['memory_info'].get('rss_mb')} MB")

    print(f"\n[7] Compute Execution Policy:")
    print("    - CPU Operations: TSV streaming/parsing, unicode normalization, token index construction, deterministic token-blocking, disk I/O, final submission TSV generation.")
    print("    - GPU Operations: Sentence transformer / embedding inference, cross-encoder neural reranking, GPU-accelerated GBDT / FAISS (where benchmarked and advantageous).")

    report = {
        "experiment_id": "EXP-0001_colab_env_check",
        "timestamp": datetime.now().isoformat(),
        "git": git_info,
        "hardware": hw_info,
        "packages": pkg_info,
        "dataset": ds_info,
        "gpu_benchmark": gpu_bench,
        "cpu_benchmark": bench_info,
        "compute_policy": {
            "cpu_operations": ["tsv_streaming", "unicode_normalization", "token_index", "token_blocking", "submission_formatting"],
            "gpu_operations": ["embeddings", "neural_reranking", "gpu_gbdt_if_benchmarked"]
        }
    }
    
    if save_json:
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        exp_0001_dir = EXPERIMENTS_DIR / "EXP-0001_colab_env_check"
        exp_0001_dir.mkdir(parents=True, exist_ok=True)
        
        with open(exp_0001_dir / "environment_check.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        with open(exp_0001_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        print(f"\n[OK] Environment check report saved to: {exp_0001_dir / 'environment_check.json'}")
            
    print("=" * 70)
    return report



def main():
    parser = argparse.ArgumentParser(description="Colab Runner & Full Production Pipeline")
    parser.add_argument("--mode", choices=["check", "blocking", "matching", "full_pipeline", "exp"], default="check", help="Execution mode")
    parser.add_argument("--data-dir", type=str, default=None, help="Custom dataset directory path")
    parser.add_argument("--sample-size", type=int, default=1000, help="S1 sample size for evaluation")
    parser.add_argument("--target-sample-size", type=int, default=50000, help="Target S2/S3 sample size per source")
    parser.add_argument("--top-k", type=int, default=50, help="Candidate top-K limit per query")
    parser.add_argument("--threshold", type=float, default=0.83, help="Decision threshold")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for S1 processing")
    parser.add_argument("--max-token-freq", type=int, default=5000, help="Max token frequency limit")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment config YAML/JSON")
    parser.add_argument("--experiment-id", type=str, default=None, help="Unique experiment ID")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume checkpoint")
    
    args = parser.parse_args()
    
    if args.mode == "check":
        run_environment_check(data_dir=args.data_dir, save_json=True)
    elif args.mode == "blocking":
        from scripts.run_blocking_eval import run_blocking_experiment
        exp_id = args.experiment_id if args.experiment_id else "EXP-0003_blocking_char_address"
        run_blocking_experiment(
            data_dir=args.data_dir,
            s1_sample_size=args.sample_size,
            target_sample_size=args.target_sample_size,
            top_k=args.top_k,
            max_token_freq=args.max_token_freq,
            experiment_id=exp_id,
            seed=args.seed,
            save_json=True
        )
    elif args.mode == "matching":
        from scripts.train_and_evaluate_matching import run_phase_3_pipeline
        exp_id = args.experiment_id if args.experiment_id else "PHASE_3_pairwise_matching"
        run_phase_3_pipeline(
            data_dir=args.data_dir,
            s1_sample_size=args.sample_size,
            target_sample_size=args.target_sample_size,
            top_k=args.top_k,
            seed=args.seed,
            experiment_id=exp_id
        )
    elif args.mode == "full_pipeline":
        from scripts.run_production_pipeline import run_production_pipeline
        run_production_pipeline(
            data_dir=args.data_dir,
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.batch_size,
            resume=not args.no_resume
        )
    else:
        print(f"Experiment execution mode requested: config={args.config}")




if __name__ == "__main__":
    main()
