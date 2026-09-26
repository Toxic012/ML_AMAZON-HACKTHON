import csv
import sys
from pathlib import Path
try:
    from src.normalization import normalize_record
except ImportError:
    try:
        from code.business_entity_resolution.src.normalization import normalize_record
    except ImportError:
        from normalization import normalize_record

csv.field_size_limit(sys.maxsize)

def load_data_generator(file_path):
    """Generator that yields normalized dict rows."""
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            yield normalize_record(row)

def load_data_list(file_path, limit=None):
    """Loads records into a list."""
    records = []
    for i, row in enumerate(load_data_generator(file_path)):
        if limit and i >= limit:
            break
        records.append(row)
    return records
