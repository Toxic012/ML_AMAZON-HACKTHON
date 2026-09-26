import csv
import sys
import os
from collections import Counter

csv.field_size_limit(sys.maxsize)

def profile_tsv(file_path):
    print(f"Profiling {file_path} ...")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    row_count = 0
    null_counts = Counter()
    country_dist = Counter()
    id_set = set()
    duplicate_ids = 0
    
    total_name_len = 0
    total_address_len = 0
    name_count = 0
    address_count = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row_count += 1
            
            # Check ID
            if "entity_id" in row:
                entity_id = row.get("entity_id", "")
                if entity_id in id_set:
                    duplicate_ids += 1
                else:
                    id_set.add(entity_id)
                
            # Check country
            if "country" in row:
                country = row.get("country", "")
                if country:
                    country_dist[country] += 1
                else:
                    null_counts["country"] += 1
                
            # Stats for name
            if "business_name" in row:
                name = row.get("business_name", "")
                if name:
                    total_name_len += len(name)
                    name_count += 1
                else:
                    null_counts["business_name"] += 1
                
            # Stats for address
            if "business_address" in row:
                address = row.get("business_address", "")
                if address:
                    total_address_len += len(address)
                    address_count += 1
                else:
                    null_counts["business_address"] += 1

    print(f"Row count: {row_count}")
    print(f"Duplicate IDs: {duplicate_ids}")
    print("Null counts:", dict(null_counts))
    if len(country_dist) > 0:
        print("Country distribution:", dict(country_dist))
    
    if name_count > 0:
        print(f"Mean name length: {total_name_len / name_count:.2f}")
    if address_count > 0:
        print(f"Mean address length: {total_address_len / address_count:.2f}")
    print("-" * 40)

if __name__ == "__main__":
    base_dir = r"c:\Users\HP\Desktop\ML_AMAZON\student_resource\dataset"
    files = [
        r"train\train_source1.tsv",
        r"train\train_source2.tsv",
        r"train\train_source3.tsv",
        r"train\train_ground_truth.tsv",
        r"test\test_source1.tsv",
        r"test\test_source2.tsv",
        r"test\test_source3.tsv",
    ]
    for f in files:
        profile_tsv(os.path.join(base_dir, f))
