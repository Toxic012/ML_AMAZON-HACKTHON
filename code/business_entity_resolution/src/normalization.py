import unicodedata
import re
from typing import Set, Dict, Any

def normalize_text(text: Any) -> str:
    """
    Unicode-safe string normalization.
    Performs NFC canonical decomposition/composition, case-folding,
    punctuation replacement with space, and whitespace collapse.
    Preserves all non-Latin Unicode characters (Cyrillic, Arabic, CJK, etc.).
    """
    if not text or not isinstance(text, str):
        return ""
    # NFC normalization (preserves Unicode characters across languages)
    text = unicodedata.normalize('NFC', text)
    # Full Unicode casefold
    text = text.casefold()
    # Punctuation to space (keep all word characters \w across Unicode)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(normalized_text: str) -> Set[str]:
    """Tokenizes normalized text into word tokens."""
    if not normalized_text:
        return set()
    return set(normalized_text.split())

def char_ngrams(text: str, n: int = 3) -> Set[str]:
    """
    Extracts Unicode-safe character n-grams from text.
    Operates on space-collapsed text without ASCII-stripping.
    If string is shorter than n, returns the whole non-empty string.
    """
    if not text:
        return set()
    # Remove internal spaces or join tokens to capture contiguous character n-grams
    clean = "".join(text.split())
    if not clean:
        return set()
    if len(clean) <= n:
        return {clean}
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}

def extract_postal_tokens(address_text: str) -> Set[str]:
    """
    Extracts postal codes, zip codes, and building/house numbers from raw/normalized address.
    Matches numeric codes (3-6 digits), alphanumeric postal codes (e.g. UK/CA style),
    and building number identifiers.
    """
    if not address_text or not isinstance(address_text, str):
        return set()
    
    tokens = set()
    # Match standard numeric postal codes and building numbers (3 to 6 digits)
    num_matches = re.findall(r'\b\d{3,6}\b', address_text)
    for m in num_matches:
        tokens.add(m)
        
    # Match UK/Canadian/international alphanumeric postal patterns (e.g. SW1A 1AA, 75008, M5V 2T6)
    alpha_num_matches = re.findall(r'\b[A-Za-z]{1,2}\d{1,2}[A-Za-z]?\s*\d[A-Za-z]{2}\b', address_text)
    for m in alpha_num_matches:
        cleaned_m = re.sub(r'\s+', '', m).casefold()
        if cleaned_m:
            tokens.add(cleaned_m)
            
    return tokens

def normalize_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Takes a dict row and enriches it with normalized fields, word tokens, character n-grams, and postal tokens."""
    b_name = row.get("business_name", "")
    b_addr = row.get("business_address", "")
    
    norm_name = normalize_text(b_name)
    norm_addr = normalize_text(b_addr)
    
    row["norm_name"] = norm_name
    row["norm_addr"] = norm_addr
    row["name_tokens"] = tokenize(norm_name)
    row["addr_tokens"] = tokenize(norm_addr)
    row["char_3grams"] = char_ngrams(norm_name, n=3)
    row["char_4grams"] = char_ngrams(norm_name, n=4)
    row["postal_tokens"] = extract_postal_tokens(b_addr) if b_addr else set()
    row["comb_tokens"] = row["name_tokens"] | row["addr_tokens"]
    
    return row

