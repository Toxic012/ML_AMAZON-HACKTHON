import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.normalization import normalize_text, tokenize, char_ngrams, extract_postal_tokens, normalize_record
    from src.blocking.token_index import InvertedTokenIndex
    from src.blocking.strategies import (
        block_exact_name,
        block_rare_tokens,
        block_char_3gram,
        block_char_4gram,
        block_address,
        block_disjunctive_union,
        block_hybrid_A_char3,
        block_hybrid_A_char4,
        block_hybrid_A_char3_char4,
        block_hybrid_full_union
    )
except ImportError:
    from code.business_entity_resolution.src.normalization import normalize_text, tokenize, char_ngrams, extract_postal_tokens, normalize_record
    from code.business_entity_resolution.src.blocking.token_index import InvertedTokenIndex
    from code.business_entity_resolution.src.blocking.strategies import (
        block_exact_name,
        block_rare_tokens,
        block_char_3gram,
        block_char_4gram,
        block_address,
        block_disjunctive_union,
        block_hybrid_A_char3,
        block_hybrid_A_char4,
        block_hybrid_A_char3_char4,
        block_hybrid_full_union
    )


class TestNormalization:
    def test_unicode_normalization(self):
        """Test Unicode preservation (German, Japanese, Cyrillic, Arabic, Accents)."""
        german = "Müller & Co. GmbH"
        assert "müller" in normalize_text(german)
        
        japanese = "株式会社ソニー"
        assert "株式会社ソニー" in normalize_text(japanese)
        
        cyrillic = "ООО Яндекс Маркет"
        assert "яндекс" in normalize_text(cyrillic)
        
        arabic = "شركة النصر للتجارة"
        assert "النصر" in normalize_text(arabic)
        
        french = "Café de Paris S.A.R.L."
        assert "café" in normalize_text(french)

    def test_punctuation_and_case_normalization(self):
        text = "  A.B.C. --- International, Inc. (USA)  "
        normalized = normalize_text(text)
        assert normalized == "a b c international inc usa"
        tokens = tokenize(normalized)
        assert tokens == {"a", "b", "c", "international", "inc", "usa"}

    def test_character_ngrams_unicode_safe(self):
        name = "müller"
        ngrams_3 = char_ngrams(name, n=3)
        assert "mül" in ngrams_3
        assert "üll" in ngrams_3
        assert "lle" in ngrams_3
        assert "ler" in ngrams_3
        
        short_name = "ab"
        assert char_ngrams(short_name, n=3) == {"ab"}
        
        empty = ""
        assert char_ngrams(empty, n=3) == set()

    def test_postal_code_extraction(self):
        addr_us = "1600 Amphitheatre Pkwy, Mountain View, CA 94043"
        tokens_us = extract_postal_tokens(addr_us)
        assert "94043" in tokens_us
        assert "1600" in tokens_us
        
        addr_uk = "10 Downing St, London SW1A 2AA, United Kingdom"
        tokens_uk = extract_postal_tokens(addr_uk)
        assert "sw1a2aa" in tokens_uk

    def test_empty_and_null_fields(self):
        row = {"entity_id": "S1-001", "business_name": None, "business_address": None, "country": None}
        norm = normalize_record(row)
        assert norm["norm_name"] == ""
        assert norm["norm_addr"] == ""
        assert norm["name_tokens"] == set()
        assert norm["addr_tokens"] == set()
        assert norm["char_3grams"] == set()
        assert norm["char_4grams"] == set()
        assert norm["postal_tokens"] == set()

    def test_duplicate_tokens(self):
        row = {"entity_id": "S1-002", "business_name": "Amazon Amazon Amazon LLC", "business_address": "Main St Main St"}
        norm = normalize_record(row)
        assert norm["name_tokens"] == {"amazon", "llc"}


class TestBlockingRetrieval:
    @pytest.fixture
    def setup_indices(self):
        s2_idx = InvertedTokenIndex()
        s3_idx = InvertedTokenIndex()
        
        # Populate Target S2 records
        records_s2 = [
            {"entity_id": "S2-101", "business_name": "Starbucks Coffee Company", "business_address": "2401 Utah Ave S, Seattle WA 98134", "country": "US"},
            {"entity_id": "S2-102", "business_name": "Müller Drogerie Markt", "business_address": "Albstraße 92, 89081 Ulm", "country": "DE"},
            {"entity_id": "S2-103", "business_name": "Microsoft Corporation", "business_address": "One Microsoft Way, Redmond, WA 98052", "country": "US"},
            {"entity_id": "S2-104", "business_name": "Acme Industrial Supplies", "business_address": "123 Industrial Rd, Chicago, IL 60601", "country": "US"}
        ]
        for r in records_s2:
            s2_idx.add_record(normalize_record(r))
            
        # Populate Target S3 records
        records_s3 = [
            {"entity_id": "S3-201", "business_name": "Starbucks Seattle Store", "business_address": "2401 Utah Avenue South 98134", "country": "US"},
            {"entity_id": "S3-202", "business_name": "Mueller Drogeriemarkt GmbH", "business_address": "Albstrasse 92, Ulm 89081", "country": "DE"},
            {"entity_id": "S3-203", "business_name": "Microsft Corp", "business_address": "1 Microsoft Way 98052", "country": "US"} # Typo in name
        ]
        for r in records_s3:
            s3_idx.add_record(normalize_record(r))
            
        return s2_idx, s3_idx

    def test_exact_name_blocking(self, setup_indices):
        s2_idx, s3_idx = setup_indices
        s1_query = normalize_record({
            "entity_id": "S1-001",
            "business_name": "Starbucks Coffee Company",
            "country": "US"
        })
        cands = block_exact_name(s1_query, s2_idx, s3_idx, top_k=10)
        assert "S2-101" in cands

    def test_spelling_corruption_retrieval_via_char_ngrams(self, setup_indices):
        s2_idx, s3_idx = setup_indices
        # Typo: "Microsft" vs "Microsoft"
        s1_query = normalize_record({
            "entity_id": "S1-002",
            "business_name": "Microsoft Corporation",
            "business_address": "One Microsoft Way, Redmond 98052",
            "country": "US"
        })
        # Exact name misses S3-203 (has typo)
        cands_exact = block_exact_name(s1_query, s2_idx, s3_idx, top_k=10)
        assert "S3-203" not in cands_exact
        
        # Char 3-gram retrieval recovers S3-203 despite typo
        cands_char3 = block_char_3gram(s1_query, s2_idx, s3_idx, top_k=10)
        assert "S3-203" in cands_char3

    def test_address_and_postal_retrieval(self, setup_indices):
        s2_idx, s3_idx = setup_indices
        # Different name variation but matching address and postal code
        s1_query = normalize_record({
            "entity_id": "S1-003",
            "business_name": "Coffee Shop #442",
            "business_address": "2401 Utah Ave, Seattle, WA 98134",
            "country": "US"
        })
        cands_addr = block_address(s1_query, s2_idx, s3_idx, top_k=10)
        assert "S2-101" in cands_addr or "S3-201" in cands_addr

    def test_hybrid_full_union_coverage(self, setup_indices):
        s2_idx, s3_idx = setup_indices
        # German business with missing address
        s1_query_de = normalize_record({
            "entity_id": "S1-004",
            "business_name": "Müller Drogerie",
            "business_address": "",
            "country": "DE"
        })
        cands_full = block_hybrid_full_union(s1_query_de, s2_idx, s3_idx, top_k=10)
        assert "S2-102" in cands_full
