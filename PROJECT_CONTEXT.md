# PROJECT_CONTEXT.md

## PART 1 — PROJECT OVERVIEW

- **Project name:** Amazon ML Challenge — Business Entity Resolution
- **Challenge name:** ML Challenge 2026
- **Problem statement:** Determine which records across 3 independent data sources (with noisy, inconsistent fields and no common identifiers) refer to the same real-world business entity. 
- **Objective:** Find all matching records from Source 2 (S2) and Source 3 (S3) for each Source 1 (S1) reference entity.
- **Expected input:** `.tsv` files representing fragments of business identity data from 3 sources.
- **Expected output:** Two tab-separated files: `matching_results.tsv` (final matches for leaderboard) and `candidate_pairs.tsv` (candidate set from blocking).
- **Overall architecture:** NOT IMPLEMENTED. The codebase is currently empty except for the initial dataset, challenge instructions, and an output validation script.
- **Current implementation status:** NOT IMPLEMENTED.

## PART 2 — REPOSITORY STRUCTURE

```
C:\Users\HP\Desktop\ML_AMAZON\
 ├── 6ab10eb3b23ba_student_resource.zip          # Initial challenge zip
 ├── 6ab674645103d_emails_comms_amazon_ml_challenge_2026.pdf # Challenge communications
 └── student_resource/                           # Extracted challenge resources
      ├── README.md                              # Main challenge instructions
      ├── Documentation_template.md              # Template for final methodology write-up
      ├── dataset/                               # Directory containing all datasets
      │    ├── train/
      │    │    ├── train_source1.tsv            # S1 reference source (2,206,823 rows)
      │    │    ├── train_source2.tsv            # S2 records (5,034,618 rows)
      │    │    ├── train_source3.tsv            # S3 records
      │    │    └── train_ground_truth.tsv       # S1 matches to S2/S3
      │    └── test/
      │         ├── test_source1.tsv             # S1 test reference source (1,732,546 rows)
      │         ├── test_source2.tsv             # S2 test records
      │         └── test_source3.tsv             # S3 test records
      └── utils/
           └── validate_submission.py            # Official script to check output formats
```

**Note:** There is currently NO code implementation directory (e.g., `src/`, `preprocessing/`, `model/`). It must be created.

## PART 3 — DATASET

- **Dataset Structure:** Tab-separated files (`.tsv`), requiring `sep="\t"` when loading.
- **Columns:**
  - `entity_id`: Unique identifier containing the source prefix (`S1-`, `S2-`, or `S3-`).
  - `business_name`: Name of business. Contains typos, transliterations, and abbreviations.
  - `business_address`: Address string. Contains format variations and missing components.
  - `country`: Country label (`US` and `India` in train; `US`, `India`, and `France` in test).
- **ID format:** Prefix-based string (e.g., `S1-925783039`).
- **Data Types:** All columns are strings.
- **Known noise patterns:** Abbreviations (Corp/Corporation), missing components, word-order transpositions, translations/transliterations (e.g. Hindi characters like "राम मार्केटिंग प्राइवेट लिमिटेड").

**Actual observed examples:**
*Source 1 (train):*
- `S1-925783039` | `Orelee's Barbershop` | `1795 Westchester Drive, High Point, NC` | `US`
- `S1-755362802` | `Prabhav Business Center` | `797, Lake Town Block A, Kolkata, Howrah, West Bengal` | `India`

*Source 2 (train):*
- `S2-166376419` | `राम मार्केटिंग प्राइवेट लिमिटेड` | `KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi` | `India`

*Source 1 (test - showing unseen country France):*
- `S1-156285671` | `<< Team Ecole` | `175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine` | `France`

## PART 4 — DATA LOADING

NOT IMPLEMENTED.
*Note: README explicitly specifies that reading `.tsv` without `sep="\t"` will silently fail due to commas in addresses.*

## PART 5 — PREPROCESSING / NORMALIZATION

NOT IMPLEMENTED.
*Note: Business names and addresses will need aggressive normalization to handle transliteration (e.g., Devanagari text), missing pins, and abbreviations (Rd vs Road).*

## PART 6 — CANDIDATE GENERATION / BLOCKING

NOT IMPLEMENTED.
*Note: Must output `candidate_pairs.tsv` to document exactly what candidates were considered prior to inference.*

## PART 7 — FEATURE ENGINEERING

NOT IMPLEMENTED.

## PART 8 — MACHINE LEARNING MODEL

NOT IMPLEMENTED.

## PART 9 — TRAINING DATA / LABEL GENERATION

NOT IMPLEMENTED.
*Note: Ground truth structure consists of `source1_entity_id` and a comma-separated list of `matched_entity_ids` (S2/S3 ids, e.g., `S2-681193310,S2-743505751,S3-775321672`). Empty list represents a singleton with no matches.*

## PART 10 — VALIDATION / EVALUATION

NOT IMPLEMENTED.
*Note: Official metric is F_0.5 macro-averaged per Source 1 entity. Needs local validation split since test labels are unavailable.*

## PART 11 — MATCH DECISION LOGIC

NOT IMPLEMENTED.

## PART 12 — OUTPUT GENERATION

NOT IMPLEMENTED.

## PART 13 — OFFICIAL VALIDATOR

- **Validator Command:** `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`
- **Location:** `student_resource/utils/validate_submission.py`
- **Current Status:** NOT VERIFIED. Output files do not exist yet.

## PART 14 — CURRENT PERFORMANCE

NOT IMPLEMENTED.

## PART 15 — CURRENT PROBLEMS

- **CRITICAL:** The entire ML pipeline (preprocessing, blocking, candidate generation, modeling, and output generation) is currently missing.
- **CRITICAL:** There is no existing codebase structure (e.g. `src/` directory).

## PART 16 — CHALLENGE COMPLIANCE

| Requirement | Current Status | Evidence/File | Risk |
|---|---|---|---|
| Read inputs as `.tsv` with tabs | NOT IMPLEMENTED | `README.md` | HIGH |
| Produce `matching_results.tsv` and `candidate_pairs.tsv` | NOT IMPLEMENTED | `README.md` | HIGH |
| Do NOT use external API/data (strictly prohibited) | COMPLIANT | (No code exists) | LOW |
| Match S1 test entities to S2/S3 (include singletons) | NOT IMPLEMENTED | `README.md` | HIGH |
| Open set for `country` (must handle France) | NOT IMPLEMENTED | `README.md` | HIGH |
| Final Model constraints (MIT/Apache 2.0, < 8B params) | NOT IMPLEMENTED | `README.md` | HIGH |

## PART 17 — CURRENT ARCHITECTURE DIAGRAM

NOT IMPLEMENTED.
(The directory currently contains only the raw challenge data and validation script.)

## PART 18 — END-TO-END EXECUTION FLOW

NOT IMPLEMENTED.

## PART 19 — ENVIRONMENT

NOT IMPLEMENTED. No `requirements.txt` or environment variables exist.

## PART 20 — REPRODUCIBILITY

NOT IMPLEMENTED. No code exists to reproduce.

## PART 21 — KNOWN LIMITATIONS

NOT IMPLEMENTED. (Cannot evaluate limitations of a system that hasn't been built.)

## PART 22 — FUTURE IMPROVEMENT CANDIDATES

Since the baseline does not exist yet, everything constitutes a future improvement. Key recommendations for the initial pipeline:
- **Preprocessing:** Implement robust character normalization, transliteration (handling Hindi Devnagari script seen in S2), and abbreviation standardization for names/addresses.
- **Blocking/Candidate Generation:** Implement highly recalled blocking keys (e.g., TF-IDF on n-grams, BM25) to narrow down the 5M+ S2 rows and S3 rows for each S1 entity.
- **Model:** Train an initial XGBoost/LightGBM model using pairwise similarity features (Levenshtein, Jaccard, embeddings) between candidate pairs.
- **Post-processing/Thresholding:** optimize threshold specifically for F_0.5 to balance precision (heavily weighted) and recall.

## PART 23 — QUESTIONS / UNKNOWN INFORMATION

- NEEDS VERIFICATION: The exact size/row-count of `train_source3.tsv`, `test_source2.tsv`, and `test_source3.tsv` (not loaded into memory during audit).
- NEEDS VERIFICATION: Preferred framework for ML model and data loading (Pandas vs Polars vs Spark) given the large dataset size (>200MB per file).
