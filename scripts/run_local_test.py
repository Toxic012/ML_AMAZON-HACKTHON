import sys
from pathlib import Path

# Add business_entity_resolution to sys.path
project_root = Path(__file__).resolve().parent.parent
ber_dir = project_root / "code" / "business_entity_resolution"
sys.path.append(str(ber_dir))

from src.data_loading import load_data_list
from src.blocking.token_index import TokenIndex
from src.blocking.union_blocking import generate_candidates

def run_local_test():
    sample_dir = project_root / "student_resource" / "sample_dataset" / "train"
    
    print("Loading S2...")
    s2_records = load_data_list(sample_dir / "train_source2.tsv")
    s2_idx = TokenIndex()
    for r in s2_records:
        s2_idx.add_record(r)
        
    print("Loading S3...")
    s3_records = load_data_list(sample_dir / "train_source3.tsv")
    s3_idx = TokenIndex()
    for r in s3_records:
        s3_idx.add_record(r)
        
    print("Loading S1...")
    s1_records = load_data_list(sample_dir / "train_source1.tsv")
    
    print("Generating candidates for S1 sample...")
    total_candidates = 0
    empty_candidates = 0
    for s1 in s1_records:
        cands = generate_candidates(s1, s2_idx, s3_idx, top_k=50)
        total_candidates += len(cands)
        if len(cands) == 0:
            empty_candidates += 1
            
    print(f"Processed {len(s1_records)} S1 entities.")
    print(f"Total candidates generated: {total_candidates}")
    print(f"Mean candidates per S1: {total_candidates / len(s1_records):.2f}")
    print(f"S1 with zero candidates: {empty_candidates}")
    
    print("Local test passed!")

if __name__ == "__main__":
    run_local_test()
