import unicodedata
import re

def normalize_text(text):
    if not text or not isinstance(text, str):
        return ""
    # NFC normalization
    text = unicodedata.normalize('NFC', text)
    # Casefold
    text = text.casefold()
    # Punctuation to space
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(normalized_text):
    if not normalized_text:
        return set()
    return set(normalized_text.split())

def normalize_record(row):
    """Takes a dict row and adds normalized fields."""
    b_name = row.get("business_name", "")
    b_addr = row.get("business_address", "")
    
    norm_name = normalize_text(b_name)
    norm_addr = normalize_text(b_addr)
    
    row["norm_name"] = norm_name
    row["norm_addr"] = norm_addr
    row["name_tokens"] = tokenize(norm_name)
    row["addr_tokens"] = tokenize(norm_addr)
    return row
