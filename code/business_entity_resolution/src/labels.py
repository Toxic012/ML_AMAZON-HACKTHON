import csv

def load_ground_truth(file_path):
    """Returns a dict mapping source1_entity_id -> set of matched entity IDs."""
    gt = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            s1_id = row["source1_entity_id"]
            matched = row["matched_entity_ids"]
            if matched:
                gt[s1_id] = set(matched.split(","))
            else:
                gt[s1_id] = set()
    return gt
