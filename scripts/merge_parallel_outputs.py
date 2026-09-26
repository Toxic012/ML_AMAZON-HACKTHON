#!/usr/bin/env python3
"""
Parallel Worker Output Merger & Validator
Amazon ML Challenge 2026 — Business Entity Resolution

Combines partitioned TSV outputs from multiple independent Colab workers into a single,
validated final submission conforming strictly to competition specifications.

Verification & Safety Gates:
1. Verifies that all expected worker directories and TSVs exist and are non-empty.
2. Checks headers of matching_results.tsv and candidate_pairs.tsv.
3. Stream-merges records in strict worker order (worker 0 -> 1 -> ... -> W-1).
4. Verifies line-by-line entity ID alignment between matching and candidate files.
5. Detects and fails loudly on duplicate S1 entity IDs across workers.
6. Verifies that the total S1 record count matches expected count (e.g. 1,732,544).
7. Optionally checks alignment against test_source1.tsv.
8. Automatically executes official student_resource submission validator.
9. Writes merged_production_report.json and merged_production_report.md.
10. Backs up final merged files to Google Drive final_submission directory if available.
"""

import os
import sys
import csv
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

csv.field_size_limit(sys.maxsize)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids\n"
CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids\n"


def merge_parallel_outputs(
    worker_dirs: Optional[List[str]] = None,
    parent_dir: Optional[str] = None,
    num_workers: int = 3,
    output_dir: Optional[str] = None,
    expected_total_s1: int = 1732544,
    s1_file: Optional[str] = None,
    test_dir: Optional[str] = None,
    skip_validator: bool = False
) -> Dict[str, Any]:
    print("=" * 95)
    print("AMAZON ML CHALLENGE 2026 — PARALLEL WORKER SUBMISSION MERGER & VALIDATOR")
    print("=" * 95)
    t0 = time.time()
    
    # 1. Resolve Worker Directories
    resolved_worker_dirs: List[Path] = []
    if worker_dirs and len(worker_dirs) > 0:
        resolved_worker_dirs = [Path(d).resolve() for d in worker_dirs]
    elif parent_dir:
        p_dir = Path(parent_dir).resolve()
        resolved_worker_dirs = [p_dir / f"worker_{i}" for i in range(num_workers)]
    else:
        # Default local output folder
        default_p = REPO_ROOT / "output"
        resolved_worker_dirs = [default_p / f"worker_{i}" for i in range(num_workers)]
        
    num_w = len(resolved_worker_dirs)
    print(f"\n[1] Verifying {num_w} worker output directories:")
    for i, w_dir in enumerate(resolved_worker_dirs):
        print(f"    Worker {i}: {w_dir}")
        if not w_dir.exists() or not w_dir.is_dir():
            raise FileNotFoundError(f"[FATAL] Missing directory for worker {i}: {w_dir}")
            
        m_file = w_dir / "matching_results.tsv"
        c_file = w_dir / "candidate_pairs.tsv"
        if not m_file.exists() or m_file.stat().st_size == 0:
            raise FileNotFoundError(f"[FATAL] Missing or empty matching_results.tsv for worker {i}: {m_file}")
        if not c_file.exists() or c_file.stat().st_size == 0:
            raise FileNotFoundError(f"[FATAL] Missing or empty candidate_pairs.tsv for worker {i}: {c_file}")
            
    print("    [OK] All worker output directories and TSVs exist and are non-empty.")

    # 2. Setup Output Directory
    if output_dir:
        out_path = Path(output_dir).resolve()
    elif parent_dir:
        out_path = Path(parent_dir).resolve()
    else:
        out_path = (REPO_ROOT / "output").resolve()
        
    out_path.mkdir(parents=True, exist_ok=True)
    merged_matching_file = out_path / "matching_results.tsv"
    merged_candidate_file = out_path / "candidate_pairs.tsv"
    
    print(f"\n[2] Target merged output files:")
    print(f"    Matching:  {merged_matching_file}")
    print(f"    Candidate: {merged_candidate_file}")

    # 3. Stream Merge Matching Results and Collect S1 Order
    print(f"\n[3] Stream-merging matching_results.tsv across {num_w} workers...")
    seen_s1_ids = set()
    s1_id_order = []
    worker_row_counts = []
    total_matches = 0
    singletons = 0
    multi_matches = 0
    zero_matches = 0
    
    with open(merged_matching_file, "w", encoding="utf-8", newline="") as f_out_match:
        f_out_match.write(MATCHING_HEADER)
        
        for w_idx, w_dir in enumerate(resolved_worker_dirs):
            m_path = w_dir / "matching_results.tsv"
            w_rows = 0
            with open(m_path, "r", encoding="utf-8") as f_in:
                r = csv.reader(f_in, delimiter="\t")
                header = next(r, None)
                if not header or header[0].strip() != "source1_entity_id" or header[1].strip() != "matched_entity_ids":
                    raise ValueError(f"[FATAL] Invalid header in {m_path}: {header}")
                    
                for row in r:
                    if not row or not row[0].strip():
                        continue
                    s1_id = row[0].strip()
                    if s1_id in seen_s1_ids:
                        raise ValueError(f"[FATAL] Duplicate S1 entity ID '{s1_id}' detected! Already seen in previous worker; duplicate in worker {w_idx} ({m_path}).")
                        
                    seen_s1_ids.add(s1_id)
                    s1_id_order.append(s1_id)
                    w_rows += 1
                    
                    matched_str = row[1].strip() if len(row) > 1 else ""
                    if not matched_str:
                        zero_matches += 1
                    else:
                        m_list = matched_str.split(",")
                        total_matches += len(m_list)
                        if len(m_list) == 1:
                            singletons += 1
                        else:
                            multi_matches += 1
                            
                    f_out_match.write(f"{s1_id}\t{matched_str}\n")
                    
            print(f"    Worker {w_idx}: {w_rows:,} matching rows merged.")
            worker_row_counts.append(w_rows)

    total_merged_s1 = len(s1_id_order)
    print(f"    [OK] Total matching rows merged: {total_merged_s1:,}")
    
    if expected_total_s1 is not None and total_merged_s1 != expected_total_s1:
        raise ValueError(
            f"[FATAL] Total merged S1 row count ({total_merged_s1:,}) does not match expected total ({expected_total_s1:,})! "
            f"Worker partitions are incomplete or have missing rows."
        )

    # 4. Stream Merge Candidate Pairs with Exact Order Verification
    print(f"\n[4] Stream-merging candidate_pairs.tsv across {num_w} workers...")
    total_candidates = 0
    cand_row_idx = 0
    
    with open(merged_candidate_file, "w", encoding="utf-8", newline="") as f_out_cand:
        f_out_cand.write(CANDIDATE_HEADER)
        
        for w_idx, w_dir in enumerate(resolved_worker_dirs):
            c_path = w_dir / "candidate_pairs.tsv"
            w_cands = 0
            with open(c_path, "r", encoding="utf-8") as f_in:
                r = csv.reader(f_in, delimiter="\t")
                header = next(r, None)
                if not header or header[0].strip() != "source1_entity_id" or header[1].strip() != "candidate_entity_ids":
                    raise ValueError(f"[FATAL] Invalid header in {c_path}: {header}")
                    
                for row in r:
                    if not row or not row[0].strip():
                        continue
                    s1_id = row[0].strip()
                    if cand_row_idx >= total_merged_s1:
                        raise ValueError(f"[FATAL] candidate_pairs.tsv contains more rows than matching_results.tsv at row {cand_row_idx} in worker {w_idx}!")
                        
                    expected_s1 = s1_id_order[cand_row_idx]
                    if s1_id != expected_s1:
                        raise ValueError(
                            f"[FATAL] S1 ID order mismatch at row {cand_row_idx}! "
                            f"matching_results had '{expected_s1}', but candidate_pairs has '{s1_id}' in worker {w_idx}."
                        )
                    cand_row_idx += 1
                    
                    cand_str = row[1].strip() if len(row) > 1 else ""
                    if cand_str:
                        c_list = cand_str.split(",")
                        w_cands += len(c_list)
                        total_candidates += len(c_list)
                        
                    f_out_cand.write(f"{s1_id}\t{cand_str}\n")
                    
            print(f"    Worker {w_idx}: candidates merged (total candidates from worker: {w_cands:,}).")

    if cand_row_idx != total_merged_s1:
        raise ValueError(
            f"[FATAL] candidate_pairs.tsv row count ({cand_row_idx:,}) does not match matching_results.tsv ({total_merged_s1:,})!"
        )
    print(f"    [OK] Total candidate rows merged: {cand_row_idx:,} (Total Candidate Pairs: {total_candidates:,})")

    # 5. Optional S1 Order Validation against test_source1.tsv
    if s1_file is None:
        # Check standard locations
        for cand in [
            Path("/content/drive/MyDrive/DATASET_ML-AMAZON/dataset/test/test_source1.tsv"),
            REPO_ROOT / "student_resource" / "dataset" / "test" / "test_source1.tsv"
        ]:
            if cand.exists():
                s1_file = str(cand)
                break
                
    if s1_file and Path(s1_file).exists():
        print(f"\n[5] Validating exact entity ID sequence against source: {s1_file} ...")
        mismatch_count = 0
        total_checked = 0
        with open(s1_file, "r", encoding="utf-8") as f_s1:
            r = csv.DictReader(f_s1, delimiter="\t")
            for idx, row in enumerate(r):
                if idx >= total_merged_s1:
                    break
                s1_id = row.get("entity_id")
                if not s1_id:
                    continue
                if s1_id != s1_id_order[idx]:
                    mismatch_count += 1
                    if mismatch_count <= 5:
                        print(f"    [ERROR] Row {idx}: expected '{s1_id}', found '{s1_id_order[idx]}'")
                total_checked += 1
        if mismatch_count > 0:
            raise ValueError(f"[FATAL] Sequence validation failed: {mismatch_count} entity ID mismatches against {s1_file}!")
        print(f"    [OK] Verified {total_checked:,} S1 entity IDs in exact sequence (0 mismatches, 0 gaps).")
    else:
        print("\n[5] Sequence validation against test_source1.tsv skipped (source file not found locally).")

    # 6. Official Submission Validator
    validator_passed = False
    validator_output = ""
    if not skip_validator:
        print("\n[6] Running Official Submission Validator on merged outputs...")
        val_script = REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"
        
        # Resolve test_dir
        if test_dir is None:
            for cand_td in [
                Path("/content/drive/MyDrive/DATASET_ML-AMAZON/dataset/test"),
                REPO_ROOT / "student_resource" / "dataset" / "test"
            ]:
                if cand_td.exists():
                    test_dir = str(cand_td)
                    break
                    
        if val_script.exists() and test_dir and Path(test_dir).exists():
            val_cmd = [
                sys.executable, str(val_script),
                "--matching", str(merged_matching_file),
                "--candidate", str(merged_candidate_file),
                "--test-dir", str(test_dir)
            ]
            res = subprocess.run(val_cmd, capture_output=True, text=True)
            validator_output = res.stdout.strip()
            print(f"    Validator Return Code: {res.returncode}")
            print("\n" + "=" * 50 + " OFFICIAL VALIDATOR REPORT " + "=" * 50)
            print(validator_output)
            print("=" * 125)
            
            validator_passed = (res.returncode == 0)
            if not validator_passed:
                raise RuntimeError("[FATAL] Official submission validator reported errors on merged files! Fix the listed issues.")
            print("\n>>> [SUCCESS] OFFICIAL SUBMISSION VALIDATOR PASSED (100% COMPLIANT) <<<")
        else:
            print("    [WARNING] Validator skipped: validate_submission.py or test directory not found.")
    else:
        print("\n[6] Validator skipped as requested via --skip-validator.")

    # 7. Write Merged Production Reports
    total_elapsed = round(time.time() - t0, 2)
    matching_size_mb = round(merged_matching_file.stat().st_size / (1024 ** 2), 2)
    candidate_size_mb = round(merged_candidate_file.stat().st_size / (1024 ** 2), 2)
    
    report_json = out_path / "merged_production_report.json"
    report_md = out_path / "merged_production_report.md"
    
    report_dict = {
        "status": "SUCCESS" if (validator_passed or skip_validator) else "COMPLETED_UNVERIFIED",
        "num_workers": num_w,
        "worker_directories": [str(d) for d in resolved_worker_dirs],
        "worker_row_counts": worker_row_counts,
        "total_merged_s1": total_merged_s1,
        "total_candidates": total_candidates,
        "total_matches": total_matches,
        "singletons": singletons,
        "multi_matches": multi_matches,
        "zero_matches": zero_matches,
        "matching_size_mb": matching_size_mb,
        "candidate_size_mb": candidate_size_mb,
        "matching_file": str(merged_matching_file),
        "candidate_file": str(merged_candidate_file),
        "validator_passed": validator_passed,
        "merge_runtime_seconds": total_elapsed,
        "completed_at": datetime.now().isoformat()
    }
    with open(report_json, "w", encoding="utf-8") as f_rj:
        json.dump(report_dict, f_rj, indent=2)
        
    md_content = f"""# Amazon ML Challenge 2026 — Merged Production Submission Report

- **Status:** {'SUCCESS — VALIDATOR PASSED' if validator_passed else 'COMPLETED'}
- **Merged Workers:** {num_w} workers
- **Completed At:** {datetime.now().isoformat()}
- **Merge Wall-Clock Time:** {total_elapsed:.2f} seconds

## Worker Contribution Breakdown
"""
    for i, (w_dir, rc) in enumerate(zip(resolved_worker_dirs, worker_row_counts)):
        md_content += f"- **Worker {i}:** {rc:,} S1 rows (`{w_dir.name}`)\n"

    md_content += f"""
## Final Submission Statistics
- **Total S1 Queries:** {total_merged_s1:,} records
- **Total Candidate Pairs:** {total_candidates:,} pairs
- **Total Predicted Matches:** {total_matches:,} matches
- **Singleton Matches:** {singletons:,}
- **Multi-Match Queries:** {multi_matches:,}
- **Zero-Match Queries:** {zero_matches:,}

## Output Artifacts
- `matching_results.tsv`: {matching_size_mb:.2f} MB ({total_merged_s1:,} rows)
- `candidate_pairs.tsv`: {candidate_size_mb:.2f} MB ({total_merged_s1:,} rows)
- **Official Submission Validator:** {'PASSED' if validator_passed else ('SKIPPED' if skip_validator else 'FAILED')}
"""
    with open(report_md, "w", encoding="utf-8") as f_rm:
        f_rm.write(md_content)

    # 8. Copy Merged Files to Google Drive Final Submission folder if available
    drive_backup_paths = []
    drive_final_dir = Path("/content/drive/MyDrive/DATASET_ML-AMAZON/final_submission")
    if drive_final_dir.parent.exists() and out_path != drive_final_dir:
        try:
            drive_final_dir.mkdir(parents=True, exist_ok=True)
            dm = drive_final_dir / "matching_results.tsv"
            dc = drive_final_dir / "candidate_pairs.tsv"
            drj = drive_final_dir / "merged_production_report.json"
            drm = drive_final_dir / "merged_production_report.md"
            print(f"\n[8] Backing up merged files to Google Drive root ({drive_final_dir}) ...")
            shutil.copy2(merged_matching_file, dm)
            shutil.copy2(merged_candidate_file, dc)
            shutil.copy2(report_json, drj)
            shutil.copy2(report_md, drm)
            drive_backup_paths = [str(dm), str(dc), str(drj), str(drm)]
            print("    [OK] Backed up to Google Drive successfully.")
        except Exception as e_drive:
            print(f"    [WARNING] Drive backup skipped: {e_drive}")

    print("\n" + "=" * 95)
    print("PARALLEL MERGE & VALIDATION COMPLETE")
    print("=" * 95)
    print(f"Total S1 merged:        {total_merged_s1:,}")
    print(f"Total candidate pairs:  {total_candidates:,}")
    print(f"Total matches:          {total_matches:,}")
    print(f"Validator Status:       {'PASSED (100% COMPLIANT)' if validator_passed else ('SKIPPED' if skip_validator else 'FAILED')}")
    print(f"Matching Results TSV:   {merged_matching_file} ({matching_size_mb:.2f} MB)")
    print(f"Candidate Pairs TSV:    {merged_candidate_file} ({candidate_size_mb:.2f} MB)")
    print(f"Report JSON:            {report_json}")
    print(f"Report Markdown:        {report_md}")
    print("=" * 95 + "\n")

    return report_dict


def main():
    parser = argparse.ArgumentParser(description="Parallel Worker Output Merger & Validator")
    parser.add_argument("--worker-dirs", nargs="+", default=None, help="List of worker output directories")
    parser.add_argument("--parent-dir", type=str, default=None, help="Parent directory containing worker_0, worker_1, ...")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of worker directories to expect (default 3)")
    parser.add_argument("--output-dir", type=str, default=None, help="Target directory for merged TSV files")
    parser.add_argument("--expected-total-s1", type=int, default=1732544, help="Expected total S1 entity rows")
    parser.add_argument("--s1-file", type=str, default=None, help="Path to test_source1.tsv for sequence verification")
    parser.add_argument("--test-dir", type=str, default=None, help="Path to dataset test directory for validator")
    parser.add_argument("--skip-validator", action="store_true", help="Skip running validator")
    
    args = parser.parse_args()
    
    merge_parallel_outputs(
        worker_dirs=args.worker_dirs,
        parent_dir=args.parent_dir,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        expected_total_s1=args.expected_total_s1,
        s1_file=args.s1_file,
        test_dir=args.test_dir,
        skip_validator=args.skip_validator
    )


if __name__ == "__main__":
    main()
