# Implementation Architecture Plan — Amazon Business Entity Resolution (Scale: Millions of Rows)

## 0. Grounding in Actual Project State

Per `PROJECT_CONTEXT.md`: **no implementation exists yet.** This is a from-scratch build, not an audit. Confirmed real scale from the repo:

| File | Rows (known) |
|---|---|
| train_source1 | 2,206,823 |
| train_source2 | 5,034,618 |
| train_source3 | unverified — must count before building |
| test_source1 | 1,732,546 |
| test_source2 / test_source3 | unverified — must count before building |

**First implementation step, before any modeling code:** a `scripts/profile_data.py` that loads each file with `sep="\t"`, prints row counts, null rates, duplicate IDs, country distribution, and name/address length stats. Do not proceed past this step on assumed numbers — S3 and test S2/S3 sizes are currently unknown and materially affect every downstream cost estimate below.

At this scale, a naive S1×S2/S3 cross join is on the order of **10¹³ comparisons** — computationally impossible. Every design decision below exists to avoid that.

---

## 1. Directory Structure

```
code/business_entity_resolution/
├── src/
│   ├── config.py                 # all constants, seeds, paths, thresholds
│   ├── data_loading.py           # tsv loaders, schema validation
│   ├── profiling.py              # EDA / data profiling
│   ├── normalization.py          # unicode-safe name/address normalization
│   ├── blocking/
│   │   ├── token_index.py        # inverted token index
│   │   ├── ngram_lsh.py          # MinHash LSH on character n-grams
│   │   ├── ann_retrieval.py      # FAISS approximate nearest neighbor
│   │   └── union_blocking.py     # combines + dedupes candidate sources
│   ├── features.py               # pairwise feature engineering
│   ├── labels.py                 # ground-truth -> pair label construction
│   ├── models/
│   │   ├── train_baseline.py     # logistic regression baseline
│   │   ├── train_boosters.py     # LightGBM / XGBoost / CatBoost
│   │   ├── stack.py              # meta-learner stacking
│   │   └── calibration.py        # isotonic calibration
│   ├── threshold.py              # per-entity F0.5 threshold search
│   ├── decision.py               # entity-level decision logic
│   ├── output.py                 # matching_results.tsv / candidate_pairs.tsv writers
│   └── evaluate.py                # local F0.5 macro evaluator (mirrors official metric)
├── README.md
└── requirements.txt
output/
Documentation_template.md
```

---

## 2. Data Loading Strategy

**Decision: Polars (primary) with a pandas fallback path, not Spark.**

Reasoning:
- Spark is unjustified here — this is single-machine, batch, non-distributed work; Spark's overhead (cluster/session setup, shuffle costs) is not worth it below ~50M rows on one box, and adds a large reproducibility/deployment burden for a competition submission.
- Pandas alone risks memory pressure at 5M+ rows with wide feature sets — a full pairwise feature matrix for even a modest candidate set can balloon.
- **Polars** gives lazy evaluation, multithreaded CSV/TSV parsing, and much lower memory overhead, while keeping the API simple enough to stay reproducible and auditable. Read with `pl.read_csv(path, separator="\t")` (or `scan_csv` for lazy/chunked access).
- Where a specific library step only supports pandas (e.g., certain sklearn calibration calls), convert small slices (candidate-pair feature tables, not raw source tables) to pandas at that point — never the full raw source files.

Fallback: if Polars is unavailable or introduces bugs during implementation, use `pandas` with explicit `dtype=str`, `chunksize` for source loading, and `pyarrow` engine, but expect materially higher memory use — verify against the container's actual memory budget in `config.py`.

All loads must use `sep="\t"` (or Polars' `separator="\t"`) — never default comma parsing. Verify column names and row counts against the profiling step's earlier output on every load, not just once.

---

## 3. Normalization Strategy

Produce **multiple parallel representations** per record, not one destructive rewrite:

1. `name_raw` — original, untouched (needed for exact-match / debug)
2. `name_normalized` — Unicode NFKC normalized, casefolded, whitespace-collapsed, punctuation-standardized, legal-suffix-stripped (Inc/LLC/Ltd/Pvt/Pte/Corp mapped to a canonical token, not deleted — keep a `had_legal_suffix` flag)
3. `name_tokens` — token list from `name_normalized`, script-agnostic tokenization (do not assume Latin whitespace tokenization works uniformly across Devanagari/Tamil/Latin mixed text — use Unicode word-boundary segmentation, e.g. `regex` module's `\w+` with Unicode flag, not the plain `str.split()`)
4. `name_ngrams` — character 3–5-gram set from `name_normalized`, used for LSH/TF-IDF retrieval (this is the most robust cross-script signal and should be treated as a first-class representation, not an afterthought)

Same four-representation pattern applied to `business_address`, plus address-specific extraction:
5. `address_numeric_tokens` — extracted digit sequences (postal codes, house/plot numbers) — kept separate from generic numeric noise since postal-code equality is a strong signal while a stray "1" is weak
6. `address_locality_tokens` — heuristically extracted trailing/city-like tokens (best-effort, validated empirically per country rather than hardcoded per-country rules)

**Country handling:** never hardcode `{US, India}`. Read the distinct `country` values present in the actual loaded data at runtime and treat `country` purely as a categorical blocking/feature signal. France (or any unseen country) must flow through the same code path with zero special-casing.

Do not strip non-ASCII text at any stage — Devanagari, Tamil, and mixed-script content must survive normalization.

---

## 4. Candidate Generation / Blocking — the Central Problem at This Scale

With S1≈2.2M and S2≈5M+ (train), the blocking design determines whether the whole pipeline is even runnable. Design as a **union of cheap, complementary retrieval strategies**, evaluated individually for recall/cost before combining.

### 4.1 Strategy comparison (evaluate all, keep what earns its cost)

| Strategy | What it catches | Approx. cost at this scale | Notes |
|---|---|---|---|
| Country + first-token exact/inverted index | clean, low-noise matches | very low (hash join) | high precision blocking, but misses corrupted names |
| Country + rare-token inverted index | partial name matches, reordering | low | rare tokens (low document frequency) are far more discriminative than common ones — weight accordingly |
| Character n-gram MinHash LSH | typos, corruption, minor transliteration noise | moderate — but sub-quadratic via banding, tractable at millions of rows | primary defense against `Fanni's Dmaigesostcis`-style corruption |
| TF-IDF / BM25 sparse retrieval (per country partition) | lexical similarity beyond exact tokens | moderate — must be run in country-partitioned batches, not globally, to keep sparse matrix sizes manageable | BM25 generally outperforms raw TF-IDF cosine for this kind of short-text retrieval; benchmark both |
| FAISS ANN over char n-gram / TF-IDF vectors | severe corruption, reordering, cross-script near matches | moderate-high build cost, cheap query cost; use `IndexIVFFlat` or HNSW (not flat) at this row count | only worth it if it demonstrably improves recall beyond LSH+BM25 on validation — do not assume |
| Address token / postal-code / numeric blocking | disambiguates common-name collisions | low | especially valuable combined with a name-based candidate set, not as a standalone generator (addresses are too often missing) |

**Critical scale rule: partition every retrieval strategy by `country` first.** This alone can cut comparison space by an order of magnitude or more since cross-country matches are rare, while still allowing controlled cross-country leakage checks per Section 4.2.

### 4.2 Country-crossing candidates
Before hard-partitioning by country, measure on training data: what fraction of true matches (from ground truth) have differing `country` values between S1 and its matched S2/S3 record? If effectively zero, partition strictly by country (large cost saving). If non-trivial, keep a small supplementary cross-country retrieval pass restricted to high-confidence name matches only (e.g., exact normalized name match) rather than a full cross-country blocking pass.

### 4.3 Candidate union & final list
- Generate candidates per strategy per S1 entity, capped at a top-K per strategy (e.g., top-50) to bound output size.
- Union and deduplicate by `(S1_id, candidate_id)`.
- **Measure candidate recall against `train_ground_truth.tsv` before building any features** — this is the single most important number in the whole project. Target ≥97–98%; if a strategy combination can't reach that, no downstream model can recover the lost recall.
- Report candidate count distribution per S1 (mean/median/p95) — this drives the feature-computation cost estimate for the next stage.
- **The final unioned, deduplicated, top-K-capped list is what gets written to `candidate_pairs.tsv` — literally the same object passed into feature engineering and scoring, not a superset of it.** Enforce this with a programmatic assertion in `output.py`: every scored pair must have been present in the persisted candidate table.

---

## 5. Feature Engineering at Scale

Computing all listed similarity features (Levenshtein, Jaro-Winkler, TF-IDF cosine, char n-gram cosine, etc.) over potentially hundreds of millions of candidate pairs (S1 count × mean candidates per S1) requires:

- **Vectorized batch computation**, not per-row Python loops — use `polars`/`numpy` vectorized string ops where possible, and batched calls into optimized libraries (`rapidfuzz` for edit-distance-family metrics — it's implemented in C++ and dramatically faster than pure-Python `python-Levenshtein` alternatives at this volume).
- **Sparse matrix reuse**: the TF-IDF/char-ngram vectors built during blocking (Section 4) should be reused for feature computation, not rebuilt — cosine similarity of already-vectorized pairs is just a sparse dot product.
- **Chunked/batched processing**: process candidate pairs in fixed-size batches (e.g., 500K pairs) to bound peak memory, writing intermediate parquet files per batch rather than holding the full feature matrix in memory at once.
- Skip pairs early using cheap features first (e.g., if country mismatch and no name-token overlap at all, many expensive string-distance computations can be short-circuited) — validate that this doesn't silently drop true matches before relying on it.

Feature set: follow Sections 10–12 of the master prompt (name features, address features, cross-field/interaction features) — those are cost-appropriate here since they operate only on the already-reduced candidate set, not the full cross join.

---

## 6. Label Construction

1. **Verify `train_ground_truth.tsv`'s actual schema by inspection, not by filename assumption** — load it and print columns/dtypes/sample rows before writing any parsing code.
2. Positive pairs: every `(source1_entity_id, matched_id)` pair from a non-empty `matched_entity_ids` list.
3. Negative pairs: **do not sample negatives randomly from the full space** — at this scale that produces mostly "obviously different" easy negatives that teach the model little. Instead:
   - Primary negative source: non-matching candidates that survived blocking for a given S1 (these are the pairs the model will actually have to discriminate at inference time — this is the correct negative distribution).
   - **Hard-negative mining**: explicitly include near-miss negatives — same locality + similar name but wrong entity, common-word name collisions, abbreviation collisions — sampled from high-blocking-score non-matches. Validate whether hard-negative mining measurably improves validation F0.5 before making it a permanent pipeline stage (it adds training-set construction complexity).
4. Keep the positive:negative ratio realistic to the blocking output distribution, not artificially rebalanced to 50/50 — apply `scale_pos_weight`/`class_weight` at the model level instead of oversampling, since oversampling at this row count multiplies training cost.

---

## 7. Model Benchmarking Strategy

Per the "do not preselect the model" instruction: build the baseline first, then benchmark alternatives with real numbers, at real scale.

### 7.1 Baseline (build first, before anything fancier)
Normalized-exact + token-overlap + edit-similarity features → **Logistic Regression**. This establishes a floor and sanity-checks the whole pipeline (blocking → features → labels → eval) before investing in heavier models. Do not skip this step even though it won't win.

### 7.2 Candidates to benchmark against the baseline
- LightGBM (fast, low memory, handles this row count well — likely the primary workhorse given data volume)
- XGBoost (`hist` tree method for speed at this scale)
- CatBoost (strong with fewer hyperparameter tuning passes, but historically slower to train at multi-million-row scale — measure actual training time here specifically, since at this row count training-time cost is a real constraint, not just a footnote)
- HistGradientBoostingClassifier (dependency-light fallback if LightGBM/XGBoost/CatBoost installation is constrained)
- A stacked ensemble of the above (Section 14 of the master prompt) — **only if training-time budget allows retraining 3 base models + meta-learner at this row count**; explicitly weigh this against the "simplest architecture that works" principle from the project context, since at 5M+ candidate rows, ensembling cost is no longer negligible the way it was assumed to be at smaller scale.

Record for each: F0.5, precision, recall, candidate recall (inherited from blocking), training time, inference time, peak memory. **At this scale, training time and memory are first-class selection criteria, not footnotes** — a marginal F0.5 gain from a slower/heavier model may not be worth it if it makes the pipeline impractical to reproduce within reasonable time/resources.

---

## 8. Validation Strategy

- Entity-level (S1-level) split for train/validation — never split by pair, to avoid leaking near-duplicate records of the same entity across the split.
- Given 2.2M S1 entities, a single held-out validation split (e.g., 85/15) is likely sufficient rather than full k-fold — k-fold at this row count multiplies training cost k times for a robustness benefit that may not be worth it. Justify whichever choice is made with a documented cost/benefit note, and use a second re-check split only for final threshold confirmation (Section 9), not full retraining.
- Implement the **exact official metric** (`evaluate.py`): F0.5 computed per S1 entity (precision/recall from predicted vs. true `matched_entity_ids` set, singleton correctly-empty = 1.0), then macro-averaged across all S1 entities. This must match the challenge's formula exactly, including singleton handling.

---

## 9. Threshold & Entity-Level Decision

- Sweep threshold on validation-split calibrated probabilities, optimizing the *per-entity macro F0.5* from `evaluate.py` — not global pairwise F0.5, which can select a different, suboptimal threshold.
- Test whether a single global threshold suffices, or whether per-country thresholds measurably help (given France's absence from training, any per-country logic must generalize to an unseen country without a trained threshold for it — a global fallback threshold is required regardless).
- Explicit no-match / rejection path: if the best candidate's calibrated probability is below threshold, emit an empty match list — never force a match.
- One-to-many: apply the threshold independently per candidate, optionally with a margin-based tiebreak (Section 21 of the master prompt), validated on the held-out split before adoption.

---

## 10. Output Generation & Validator Integration

- `candidate_pairs.tsv` and `matching_results.tsv` writers share one underlying entity-id-indexed table to guarantee the subset invariant by construction, not by post-hoc filtering.
- Deterministic ordering: sort matched/candidate IDs lexicographically before writing.
- Every test S1 entity present exactly once, including ones with zero candidates after blocking (must still emit an empty row — do not silently drop them).
- Run `utils/validate_submission.py` as a final automated step in the pipeline (not a manual afterthought) — pipeline should fail loudly if the validator fails.

---

## 11. Memory & Runtime Budget Notes

- Raw source files: S1 (~2.2M rows) and S2 (~5M rows) at 4 short string columns each are individually manageable in memory (likely low hundreds of MB each as Polars frames); the risk is in the **candidate-pair feature table**, which can be 10–50× larger depending on mean candidates-per-S1 — measure this concretely after Section 4.3's recall/count report and budget from there, rather than assuming.
- Sparse matrices (TF-IDF/n-gram) for blocking should be built once per country partition and cached to disk (parquet/npz), not recomputed for train vs. test separately if reusable.
- Write intermediate artifacts (candidate tables, feature tables) to disk in chunks/parquet rather than holding everything in memory across all pipeline stages simultaneously — treat this as a multi-stage batch pipeline, not a single in-memory script.

---

## 12. Experiment Plan (Incremental, Cost-Aware)

```text
1. Baseline: token/exact blocking + basic string features + Logistic Regression
2. + character n-gram LSH blocking → measure candidate recall delta
3. + BM25/TF-IDF retrieval blocking → measure candidate recall delta vs. cost
4. + richer pairwise features (edit similarity, cross-field interactions)
5. + LightGBM / XGBoost / CatBoost benchmark (Section 7.2 table)
6. + hard-negative mining → measure F0.5 delta
7. + calibration + per-entity threshold optimization
8. (only if budget allows) + FAISS ANN retrieval, + stacked ensemble
```

Keep only steps with measured validation F0.5 improvement that justifies their added compute cost at this row count — cost-awareness is a first-class constraint here, not a secondary concern.

---

## 13. Error Analysis & Documentation Plan

Same structure as the master prompt (Sections 29–30/28 there): false positive/negative inspection by category, SHAP or feature-importance-based explainability on a sample (not the full multi-million-row set — sampling is required here for tractability), and a filled `Documentation_template.md` covering blocking strategy, candidate recall numbers, model comparison table, and final F0.5.

---

## 14. Immediate Next Steps

1. Run data profiling (`scripts/profile_data.py`) to get real row counts for S3/test files and confirm the estimates above.
2. Verify `train_ground_truth.tsv` schema by direct inspection.
3. Build the Section 7.1 baseline end-to-end (even if weak) to validate the full pipeline plumbing before investing in blocking/model sophistication.
4. Report candidate recall from initial blocking before writing a single feature function.
