# MASTER PROMPT: Amazon Business Entity Resolution Challenge

## Role

Act as a senior Machine Learning Engineer, Entity Resolution / Record Linkage specialist, and production-grade data science architect.

You are reviewing and improving an implementation for the **Amazon Business Entity Resolution Challenge**. Your task is not merely to produce a generic entity-matching solution. You must inspect the existing project, understand the challenge constraints, validate the current approach, identify weaknesses, and improve the solution wherever technically justified.

The final implementation must be accurate, reproducible, explainable, computationally practical, and compliant with every competition rule.

---

# 1. Challenge Objective

The challenge is to resolve business entities across three independent sources containing noisy and inconsistent business records.

There are:

- **Source 1 (S1):** deduplicated reference entities.
- **Source 2 (S2):** noisy business records.
- **Source 3 (S3):** noisy business records.

For every S1 entity, identify **all matching S2 and S3 records**.

A Source 1 entity may have:

- zero matches,
- one match,
- or multiple matches.

The final system must therefore support both one-to-one-looking cases and one-to-many relationships.

---

# 2. Mandatory Input Schema

Each source contains:

```text
entity_id
business_name
business_address
country
```

Entity ID prefixes indicate the source:

```text
S1-...
S2-...
S3-...
```

The training countries include US and India, while the test data also contains France.

Therefore:

**Do NOT hard-code the country list to US and India.**

Country handling must be open-set and automatically support all countries present in the data.

---

# 3. Data Loading Requirements

The source files are TSV files.

They MUST be read using:

```python
pd.read_csv(path, sep="\t")
```

Do not assume commas are delimiters because addresses and ID lists can contain commas.

Expected training files:

```text
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
```

Expected test files:

```text
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

Before implementing the pipeline, inspect:

- row counts,
- column names,
- null rates,
- duplicate IDs,
- duplicate business records,
- country distribution,
- name length distributions,
- address length distributions,
- multilingual content,
- corrupted/malformed values,
- repeated entities,
- exact duplicates,
- near duplicates.

Do not assume the sample rows represent the complete dataset distribution.

---

# 4. Ground Truth

The training ground truth is expected to contain:

```text
source1_entity_id
matched_entity_ids
```

`matched_entity_ids` contains comma-separated S2/S3 entity IDs.

An empty value means the S1 entity has no matching S2/S3 record.

The implementation MUST verify the actual file schema before proceeding. If the provided file appears to have the wrong schema, do not silently reinterpret it.

Create training pair labels from the ground truth:

```text
S1 + S2/S3 candidate pair -> MATCH / NON-MATCH
```

Do not train using guessed labels.

---

# 5. Observed Data Characteristics

The provided samples show substantial real-world noise.

Observed name noise includes:

- capitalization differences,
- punctuation differences,
- extra spaces,
- legal suffixes,
- abbreviations,
- duplicated words,
- spelling errors,
- character corruption,
- inserted/deleted characters,
- DBA/trade names,
- multilingual names,
- mixed-language names,
- transliterated or differently represented business names.

Examples include patterns such as:

```text
Fanni's Dmaigesostcis
```

and:

```text
Animal Hanisch Hospirlg
```

and multilingual/mixed-language records such as:

```text
मॉडर्न फाइनेंस
Sun पावर Provision
ஈஸ்டர்ன் கன்சல்டன்சி பிரைவேட் லிமिटेड
```

Therefore, the system MUST NOT assume English-only text.

Do NOT simply remove all non-ASCII characters.

Unicode-aware processing is required.

---

# 6. Address Noise

Observed address noise includes:

- different component ordering,
- abbreviations,
- missing components,
- duplicated components,
- punctuation differences,
- inconsistent casing,
- transliteration,
- municipal/locality variations,
- landmarks,
- apartment/unit variations,
- house/plot numbers,
- postal/ZIP information,
- state/city variations,
- malformed values,
- null/empty addresses.

Some addresses are missing entirely.

Therefore:

**Address matching must be robust but must not be mandatory for every record.**

The system must gracefully fall back to name-heavy matching when address information is missing.

---

# 7. Core Pipeline

Build or improve the pipeline around:

```text
Raw Data
   ↓
Data Validation / Profiling
   ↓
Unicode-Aware Normalization
   ↓
Blocking / Candidate Generation
   ↓
Candidate Pair Dataset
   ↓
Pairwise Feature Engineering
   ↓
Supervised Match Model
   ↓
Probability Calibration / Confidence
   ↓
Threshold / Decision Optimization for F0.5
   ↓
Final Match Selection
   ↓
matching_results.tsv
candidate_pairs.tsv
   ↓
Submission Validator
```

Do not skip candidate generation.

---

# 8. Candidate Generation / Blocking

Blocking is one of the most important parts of this challenge.

The system should dramatically reduce the number of impossible S1-S2/S3 comparisons while maintaining high candidate recall.

Use multiple complementary blocking strategies instead of relying on a single block.

Potential blocking signals include:

### Country blocking

Prefer candidates from the same country.

However, do not blindly discard cross-country candidates if exploratory analysis proves that noisy data can legitimately produce country inconsistencies.

The final decision should be evidence-driven.

### Name-based blocking

Potential approaches:

- normalized name prefixes,
- first token,
- rare tokens,
- character n-gram inverted index,
- TF-IDF retrieval,
- token signatures,
- phonetic representations where appropriate,
- fuzzy retrieval,
- multilingual-compatible retrieval.

### Address-based blocking

Potential approaches:

- city/locality tokens,
- state/province tokens,
- postal codes,
- house/plot numbers,
- distinctive address tokens,
- character n-grams,
- TF-IDF retrieval.

### Hybrid blocking

Combine signals:

```text
country + name token
country + address token
country + character n-gram
name retrieval + address retrieval
```

Use a union of candidate generators.

The final candidate set must prioritize **candidate recall**.

If a true match is removed during blocking, the downstream ML model cannot recover it.

---

# 9. Candidate Set Requirement

The competition requires:

`candidate_pairs.tsv` to represent the **final candidate list actually fed into the final ML matcher**.

If the pipeline internally does:

```text
1000 initial candidates
        ↓
200 refined candidates
        ↓
ML model
```

then:

```text
candidate_pairs.tsv
```

must contain the 200 candidates, not the initial 1000.

Final predicted matches MUST be a subset of the candidate pairs.

---

# 10. Feature Engineering

Build rich pairwise features for every candidate pair.

## Business Name Features

Consider:

- normalized exact match,
- token overlap,
- token-set similarity,
- Jaccard similarity,
- character n-gram cosine similarity,
- TF-IDF cosine similarity,
- edit similarity,
- Levenshtein similarity,
- Jaro-Winkler,
- longest common subsequence ratio,
- length difference,
- common token count,
- rare-token overlap,
- legal-suffix-aware similarity,
- character-level similarity,
- multilingual-safe similarity.

Do not blindly use every feature.

Use validation and feature importance/ablation testing to determine which features actually help.

---

# 11. Address Features

Consider:

- normalized exact match,
- token overlap,
- token-set similarity,
- Jaccard similarity,
- TF-IDF cosine,
- character n-gram cosine,
- edit similarity,
- city match,
- state/province match,
- postal code match,
- house/plot number match,
- apartment/unit number match,
- numeric token overlap,
- landmark token overlap,
- locality overlap,
- address length difference,
- missing-address indicators.

Numbers should be treated carefully because they can be highly discriminative in addresses.

Do not assume every number has the same meaning.

---

# 12. Cross-Field Features

Create features such as:

```text
country_exact_match
name_address_joint_similarity
name_length_ratio
address_length_ratio
number_overlap
city_overlap
state_overlap
token_count_difference
missing_name_flag
missing_address_flag
```

Also consider interactions such as:

```text
high_name_similarity + high_address_similarity
high_name_similarity + low_address_similarity
low_name_similarity + high_address_similarity
```

These interactions may be highly informative for distinguishing true matches from false positives.

---

# 13. Multilingual Processing

The dataset contains English, Hindi, Tamil, and French examples.

The implementation must:

- preserve Unicode,
- normalize Unicode safely,
- avoid destructive ASCII-only preprocessing,
- support multilingual tokens,
- consider transliteration only where it improves validation performance,
- avoid assuming English word boundaries are always sufficient.

If multilingual embeddings are considered, test them empirically.

Do not add a large language model or embedding model simply because it sounds advanced.

Every additional model must demonstrate validation benefit relative to computational cost.

---

# 14. Recommended ML Model

Start with a **gradient-boosted tree model for pairwise classification**.

Preferred candidates to benchmark:

1. **LightGBM**
2. **XGBoost**
3. **CatBoost**
4. HistGradientBoostingClassifier if dependency simplicity is important.

The primary recommendation is:

**LightGBM or XGBoost on engineered pairwise similarity features.**

Why:

- entity resolution produces heterogeneous numerical features,
- non-linear interactions matter,
- missing values are common,
- tree ensembles handle mixed feature scales well,
- they are considerably cheaper than large language models,
- they are easier to validate and reproduce,
- they can provide feature importance,
- they are suitable for precision-focused threshold optimization.

Do NOT assume the first recommendation is automatically optimal.

Claude must benchmark reasonable alternatives using the validation protocol and select the approach based on measured F0.5, precision, recall, runtime, memory, and reproducibility.

---

# 15. Consider a Retrieval + Reranking Architecture

If the dataset is large, use:

```text
Blocking / Retrieval
        ↓
Top-K candidate generation
        ↓
Feature-based ML reranker
        ↓
Final match decision
```

Do not compare every S1 against every S2/S3 record if that creates an unnecessary O(N²) problem.

The retrieval stage should maximize recall.

The reranker should maximize precision.

---

# 16. Optional Advanced Models

Claude should evaluate whether the following can improve the solution:

### TF-IDF retrieval

Useful for fast sparse lexical retrieval.

### Character n-gram retrieval

Very useful for spelling errors, abbreviations, and noisy names.

### Sentence embeddings

Consider multilingual sentence embeddings such as a suitable multilingual Sentence-BERT model if validation demonstrates improvement.

However, embeddings must be evaluated against classical character/token similarity features.

### Cross-encoder / LLM

Only consider a cross-encoder or local LLM for a small ambiguous subset if:

- it materially improves validation F0.5,
- it satisfies the competition's model/license restrictions,
- it fits the runtime/memory constraints,
- it does not rely on external data,
- it remains reproducible.

Do NOT make an LLM the default matcher without evidence.

---

# 17. Model Selection Must Be Evidence-Based

Do not simply state:

"XGBoost is best."

Actually compare appropriate alternatives.

For example:

```text
Model A:
Logistic Regression

Model B:
LightGBM

Model C:
XGBoost

Model D:
CatBoost
```

Evaluate using the challenge metric:

```text
F0.5
Precision
Recall
```

Also record:

```text
candidate recall
training time
inference time
memory usage
```

Choose the model based on validation evidence.

If a more advanced method is clearly better, use it.

If a simpler method performs equally well or better, prefer the simpler method.

---

# 18. Validation Strategy

Do NOT tune thresholds on the test set.

Create a proper validation strategy from the training data.

Prefer an entity-level split that avoids leakage.

Potentially use:

```text
train entities → model training
validation entities → threshold/model selection
```

If the data generation process supports it, use multiple validation folds or repeated validation.

Avoid putting near-identical records from the same underlying entity into both training and validation if that causes leakage.

---

# 19. F0.5 Optimization

The official evaluation metric is:

```text
F0.5
```

It weights precision more heavily than recall.

Therefore:

**False positive entity merges are particularly harmful.**

Do not simply use:

```python
threshold = 0.5
```

Instead evaluate a threshold range on validation data, for example:

```text
0.10 → 0.95
```

and select the threshold that maximizes the correct macro-averaged F0.5 evaluation procedure.

Be careful about the challenge's per-S1 macro averaging.

A threshold that produces excellent global pairwise F0.5 may not maximize the official entity-level metric.

Implement the official evaluation logic as closely as possible.

---

# 20. Singleton Handling

A Source 1 entity may have no match.

This is extremely important.

Do NOT force every S1 entity to match at least one S2/S3 entity.

If evidence is insufficient:

```text
matched_entity_ids = empty
```

should be allowed.

A false match on a true singleton can severely hurt the score.

---

# 21. One-to-Many Matching

Do NOT automatically select only:

```text
top 1 candidate
```

The challenge allows multiple matches.

The model must determine which candidates independently satisfy the match criteria.

However, because F0.5 is precision-heavy, do not output weak additional matches just because they are plausible.

Consider:

```text
candidate score
threshold
margin from next-best candidate
evidence strength
```

where justified by validation.

---

# 22. Potential Two-Stage Decision

Evaluate whether this is beneficial:

### Stage 1

Candidate pair probability:

```text
P(match | S1, candidate)
```

### Stage 2

Entity-level decision:

- threshold,
- score margin,
- number of strong fields,
- duplicate/consistency constraints.

Do not add complex post-processing unless validation proves that it improves official F0.5.

---

# 23. Leakage Prevention

Be extremely careful about leakage.

Do not use:

- test ground truth,
- external business databases,
- external geocoding,
- internet lookups,
- commercial entity-resolution APIs,
- government registries,
- external business directories.

The challenge explicitly prohibits external data lookup and augmentation.

All information must come from the provided dataset.

---

# 24. Competition Constraints

The final implementation MUST respect:

- exact output format,
- only S2/S3 IDs may be matched,
- every S1 test entity must appear exactly once,
- no duplicate matched IDs,
- singleton entities must be represented correctly,
- final matches must be candidates,
- model must satisfy the challenge's license/parameter constraints,
- maximum allowed model size is 8B parameters,
- no prohibited external data lookup.

---

# 25. Required Output

Generate:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

## matching_results.tsv

Every test S1 entity must have exactly one row.

Expected structure:

```text
source1_entity_id    matched_entity_ids
```

For no match:

```text
S1-12345
```

with an empty `matched_entity_ids`.

For multiple matches:

```text
S1-12345    S2-111,S2-222,S3-333
```

Ensure deterministic ordering.

---

# 26. candidate_pairs.tsv

This must represent the final candidate list passed to the final ML matcher.

It must include candidate relationships between S1 and S2/S3.

Final predictions must be a subset of these candidates.

Ensure deterministic output.

---

# 27. Submission Validation

Run:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

The implementation should not be considered complete until the validator passes.

---

# 28. Reproducibility

The final project must be reproducible.

Include:

```text
code/business_entity_resolution/src/
README.md
requirements.txt
Documentation_template.md
output/
```

Use:

- deterministic random seeds,
- configuration files/constants,
- documented preprocessing,
- documented feature generation,
- documented model parameters,
- documented threshold,
- documented candidate-generation parameters.

Avoid hard-coded machine-specific paths.

---

# 29. Computational Efficiency

The solution must be practical for the actual dataset size.

Before implementing expensive algorithms, inspect:

- row counts,
- number of candidate pairs,
- memory usage,
- runtime.

Use:

- vectorization,
- inverted indexes,
- sparse matrices,
- efficient retrieval,
- caching,
- batching,
- multiprocessing only when beneficial.

Do not generate an enormous Cartesian product unnecessarily.

---

# 30. Explainability

The system should make it possible to understand why a candidate matched.

For important candidates, retain/debug information such as:

```text
name_similarity
address_similarity
country_match
numeric_overlap
city_match
state_match
model_probability
decision_threshold
```

This is useful for debugging false positives and false negatives.

---

# 31. Error Analysis

After validation, explicitly inspect:

### False positives

Especially:

- similar names but different addresses,
- same city but different businesses,
- common words,
- duplicated company names,
- legal suffix-only similarity.

### False negatives

Especially:

- severe spelling corruption,
- multilingual names,
- missing addresses,
- DBA names,
- transliteration,
- reordered addresses,
- abbreviations.

Use these findings to improve blocking and features.

---

# 32. Ablation Testing

Perform controlled experiments such as:

```text
Baseline:
name + address exact features

Experiment 1:
+ edit similarity

Experiment 2:
+ character n-grams

Experiment 3:
+ TF-IDF

Experiment 4:
+ address numeric features

Experiment 5:
+ multilingual embeddings

Experiment 6:
+ advanced reranking
```

Measure actual validation impact.

Do not add complexity without measurable benefit.

---

# 33. Important Instruction to Claude

You are explicitly authorized and expected to improve this specification.

If you identify a:

- better blocking algorithm,
- better similarity function,
- better feature,
- better ML model,
- better retrieval architecture,
- better multilingual strategy,
- better validation design,
- better thresholding method,
- better calibration technique,
- better entity-level decision strategy,
- better computational optimization,

then **add it to the final implementation if it is justified by the data and validation results.**

Do not blindly follow the proposed LightGBM/XGBoost recommendation if another method is demonstrably better.

The goal is:

**maximize official validation F0.5 while preserving candidate recall, reproducibility, computational feasibility, and competition compliance.**

---

# 34. Critical Review Requirement

Before changing code, inspect the existing implementation.

Identify:

1. What is currently correct?
2. What is currently incorrect?
3. What is inefficient?
4. What can cause false positives?
5. What can cause false negatives?
6. What violates the challenge specification?
7. What causes data leakage?
8. What prevents reproducibility?
9. What prevents good F0.5?
10. What should be replaced rather than patched?

Do not preserve a bad architecture merely because it already exists.

---

# 35. Do Not Guess Missing Information

If a file, schema, dataset property, or ground-truth format has not been verified:

- inspect the actual file,
- report the uncertainty,
- do not invent values,
- do not fabricate matches.

In particular, verify that `train_ground_truth.tsv` actually has:

```text
source1_entity_id
matched_entity_ids
```

before building labels.

---

# 36. Claude's Deliverables

After reviewing the existing project, provide:

### A. Technical Assessment

Explain the current architecture and its weaknesses.

### B. Recommended Architecture

Provide the final improved architecture and justify every major component.

### C. Model Selection

Compare suitable ML approaches and explain the final choice using validation results.

### D. Blocking Strategy

Document every blocking method and candidate-recall reasoning.

### E. Feature Engineering

Document the final feature set.

### F. Validation

Document:

- split strategy,
- leakage prevention,
- F0.5 calculation,
- threshold tuning,
- model comparison.

### G. Implementation

Implement the improved pipeline.

### H. Outputs

Generate:

```text
matching_results.tsv
candidate_pairs.tsv
```

### I. Documentation

Update the project's README and methodology documentation.

### J. Reproducibility

Provide exact commands to:

1. install dependencies,
2. train,
3. validate,
4. generate test predictions,
5. validate submission.

---

# 37. Final Quality Gate

Before declaring the implementation complete, verify:

- [ ] Dataset loaded with TSV delimiter.
- [ ] Actual schemas verified.
- [ ] Ground truth verified.
- [ ] No external data used.
- [ ] Unicode preserved.
- [ ] Country handling is open-set.
- [ ] Missing addresses handled.
- [ ] Blocking implemented.
- [ ] Candidate recall measured.
- [ ] candidate_pairs.tsv represents final ML candidates.
- [ ] ML matcher trained only from legitimate training labels.
- [ ] Validation split prevents leakage.
- [ ] F0.5 optimized.
- [ ] Singleton handling implemented.
- [ ] One-to-many matching supported.
- [ ] Final matches are a subset of candidates.
- [ ] Every test S1 entity appears exactly once.
- [ ] No duplicate matched IDs.
- [ ] Only valid S2/S3 IDs are output.
- [ ] Validator passes.
- [ ] Runtime and memory are reasonable.
- [ ] Results are deterministic.
- [ ] README/documentation is complete.

---

# 38. Final Instruction

Treat this as a serious competitive Entity Resolution / Record Linkage system, not a toy fuzzy-matching script.

Do not optimize for code simplicity at the expense of official F0.5.

Do not optimize for recall at the expense of precision.

Do not optimize for model sophistication at the expense of reproducibility.

The final objective is a robust:

**Blocking → Candidate Retrieval → Pairwise Feature Engineering → ML Reranking → F0.5-Optimized Decision → Validated Submission**

system.

If your analysis of the actual dataset shows that a different architecture is superior, modify this plan accordingly and clearly explain why.

Always prioritize measured validation performance and competition compliance over assumptions.
