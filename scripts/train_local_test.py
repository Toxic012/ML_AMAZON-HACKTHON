import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
ber_dir = project_root / "code" / "business_entity_resolution"
sys.path.append(str(ber_dir))

from src.data_loading import load_data_list
from src.blocking.token_index import TokenIndex
from src.blocking.union_blocking import generate_candidates
from src.features import compute_features
from src.labels import load_ground_truth
from sklearn.linear_model import LogisticRegression

def train_and_eval_local():
    sample_dir = project_root / "student_resource" / "sample_dataset" / "train"
    
    print("Loading data...")
    s2_records = load_data_list(sample_dir / "train_source2.tsv")
    s3_records = load_data_list(sample_dir / "train_source3.tsv")
    s1_records = load_data_list(sample_dir / "train_source1.tsv")
    gt = load_ground_truth(sample_dir / "train_ground_truth.tsv")
    
    s2_idx = TokenIndex()
    for r in s2_records: s2_idx.add_record(r)
        
    s3_idx = TokenIndex()
    for r in s3_records: s3_idx.add_record(r)

    print("Building features & labels...")
    X = []
    y = []
    candidate_pairs = []
    
    for s1 in s1_records:
        s1_id = s1["entity_id"]
        cands = generate_candidates(s1, s2_idx, s3_idx, top_k=20)
        true_matches = gt.get(s1_id, set())
        
        for cand_id in cands:
            is_match = 1 if cand_id in true_matches else 0
            # Retrieve cand record
            cand_record = s2_idx.entity_store.get(cand_id) or s3_idx.entity_store.get(cand_id)
            feats = compute_features(s1, cand_record)
            X.append(feats)
            y.append(is_match)
            candidate_pairs.append((s1_id, cand_id, is_match))
            
    print(f"Dataset size: {len(X)} candidate pairs.")
    pos_count = sum(y)
    print(f"Positive labels: {pos_count}, Negative labels: {len(y) - pos_count}")
    
    if pos_count == 0:
        print("No positive labels in this small subset! Test complete.")
        return
        
    print("Training Logistic Regression...")
    clf = LogisticRegression(class_weight='balanced', random_state=42)
    clf.fit(X, y)
    
    print("Evaluating...")
    preds = clf.predict(X)
    from sklearn.metrics import classification_report
    print(classification_report(y, preds))

if __name__ == "__main__":
    train_and_eval_local()
