# Business Entity Resolution - Amazon ML Challenge 2026

Team: [YOUR TEAM NAME]

## Pipeline Overview

1. **Blocking** (`src/blocking.py`) — generates candidate matches per Source1 entity
   using TF-IDF character n-gram similarity over normalized business names.
2. **Matching** (`src/matching.py`) — engineers pairwise similarity features
   (name/address fuzzy match scores, country match) and trains an XGBoost
   classifier. Threshold is tuned on a held-out split to maximize F_0.5.

## Environment Setup

```bash
pip install -r requirements.txt
```

## Reproducing Results End-to-End

Expected data layout:
```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### Step 1 — EDA (optional, informational)
```bash
python src/load_data.py --data-dir dataset/train
```

### Step 2 — Generate candidate pairs (train set, for model training)
```bash
python src/blocking.py --data-dir dataset/train --split train --top-k 20 \
    --out output/candidate_pairs_train.tsv
```

### Step 3 — Train the matching model
```bash
python src/matching.py train \
    --data-dir dataset/train \
    --candidates output/candidate_pairs_train.tsv \
    --model model.json
```
This prints per-threshold F_0.5 scores on a held-out validation split and
reports the best threshold. Note that value for Step 5.

### Step 4 — Generate candidate pairs (test set, for final submission)
```bash
python src/blocking.py --data-dir dataset/test --split test --top-k 20 \
    --out output/candidate_pairs.tsv
```

### Step 5 — Run inference on the test set
```bash
python src/matching.py predict \
    --data-dir dataset/test \
    --candidates output/candidate_pairs.tsv \
    --model model.json \
    --threshold <BEST_THRESHOLD_FROM_STEP_3> \
    --out output/matching_results.tsv
```

### Step 6 — Validate before submitting
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Methodology Summary

See `Documentation_template.md` for the full write-up. Briefly:

- **Blocking:** normalized business names (lowercased, punctuation stripped,
  legal-suffix abbreviations standardized — e.g. Corporation→Corp, Ltd→Limited)
  then TF-IDF character n-gram (2-4) cosine similarity via nearest-neighbor
  search, top-20 candidates per Source1 entity.
- **Matching:** RapidFuzz-based string similarity features (ratio, token sort
  ratio, partial ratio) on name and address, plus exact country match.
  XGBoost binary classifier, threshold tuned to maximize F_0.5 (precision
  weighted 2x over recall, per challenge spec) on a held-out validation split.

## Constraints Compliance

- No external data lookups used (fair play rule).
- Model: XGBoost (open-source, no license restriction, well under 8B parameters).
