# Amazon ML Challenge 2026 - System Map & Project Brain

## A. Repository Structure
```
C:\Users\HP\Desktop\ML_AMAZON\
 ├── student_resource/                   # Raw challenge data
 ├── code/
 │   └── business_entity_resolution/
 │       ├── scripts/
 │       │   └── profile_data.py         # Created & executed
 │       └── src/                        # To be implemented
 └── PROJECT_CONTEXT.md                  # Initial audit
```

## B. Dataset Locations & Sizes
Data profiled successfully. Total rows processed: ~26.4 Million.
- **train_source1**: 2,206,821 rows
- **train_source2**: 5,034,616 rows
- **train_source3**: 5,285,603 rows
- **train_ground_truth**: 2,206,821 rows
- **test_source1**: 1,732,544 rows
- **test_source2**: 4,887,273 rows
- **test_source3**: 5,082,316 rows

**Data Quality Notes:**
- Duplicate IDs: 0 across all files.
- Missing values: S1 (0), S2/S3 (business_address is missing in ~3% of rows).
- Country Distribution: Train contains `US`, `India`. Test introduces `France`.

## C. Input/Ground-Truth Schemas
- **Source Files**: `entity_id`, `business_name`, `business_address`, `country` (Tab-separated).
- **Ground Truth**: `source1_entity_id`, `matched_entity_ids` (comma-separated list of S2/S3 IDs, or empty).

## D. Planned Pipeline Flow (Baseline)
1. **Normalization (`normalization.py`)**: NFC Unicode normalization, casefolding, tokenization.
2. **Blocking (`blocking/`)**: Exact token/country matching to filter the 10M+ S2/S3 candidates per S1 entity.
3. **Candidate Generation**: Output unioned pairs for train/validation.
4. **Features (`features.py`)**: Jaccard similarity on tokens, exact string matches.
5. **Model (`models/train_baseline.py`)**: Logistic Regression on a small subset to validate end-to-end plumbing.
6. **Evaluation (`evaluate.py`)**: Macro F0.5 per S1 entity, including singleton (empty match) evaluation.
7. **Output (`output.py`)**: Generate `matching_results.tsv` and `candidate_pairs.tsv`.

## E. Current Bottlenecks & Risks
- **Data Volume**: S2+S3 is over 10 million rows per split. A Cartesian join against 2.2M S1 rows is impossible. Country-partitioned blocking is mandatory.
- **Missing Addresses**: 3% of candidates have no address; fallback to name-only matching is required.
- **Unseen Countries**: France appears in the test set. Hardcoding country rules will fail.

## F. Next Phase
**PHASE 2 & 3 (Normalization & Blocking)**: Implement the foundational data loaders and the baseline token-blocking strategy to generate a candidate set with measurable recall.
