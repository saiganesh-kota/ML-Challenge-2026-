"""
matching.py
Business Entity Resolution Challenge - Amazon ML Challenge 2026

Takes candidate pairs from blocking.py, engineers similarity features,
trains a precision-tuned XGBoost classifier (F_0.5 rewards precision 2x),
and produces the final matching_results.tsv.

Usage (train + evaluate on held-out split):
    python src/matching.py train --data-dir dataset/train --candidates output/candidate_pairs.tsv

Usage (run inference on test set with a saved model):
    python src/matching.py predict --data-dir dataset/test --candidates output/candidate_pairs.tsv --model model.json
"""

import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from rapidfuzz import fuzz
from sklearn.model_selection import train_test_split

from load_data import load_sources, load_ground_truth
from blocking import normalize_name


# ---------- Feature engineering ----------

def build_pair_features(s1_row, other_row):
    """Compute similarity features between a Source1 record and a candidate record."""
    name1 = normalize_name(s1_row["business_name"])
    name2 = normalize_name(other_row["business_name"])
    addr1 = str(s1_row["business_address"]).lower()
    addr2 = str(other_row["business_address"]).lower()

    return {
        "name_ratio": fuzz.ratio(name1, name2) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(name1, name2) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(name1, name2) / 100.0,
        "addr_ratio": fuzz.ratio(addr1, addr2) / 100.0,
        "addr_token_sort_ratio": fuzz.token_sort_ratio(addr1, addr2) / 100.0,
        "country_match": int(
            str(s1_row.get("country", "")).lower() == str(other_row.get("country", "")).lower()
        ),
        "name_len_diff": abs(len(name1) - len(name2)),
    }


def build_feature_table(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                         candidates: pd.DataFrame):
    """
    Expand candidate_pairs into one row per (source1_id, candidate_id) with features.
    Returns a DataFrame: source1_entity_id, candidate_entity_id, <feature columns>
    """
    s1_idx = s1.set_index("entity_id")
    other_idx = pd.concat([s2, s3]).set_index("entity_id")

    rows = []
    for _, row in candidates.iterrows():
        s1_id = row["source1_entity_id"]
        cand_str = row["candidate_entity_ids"]
        if not isinstance(cand_str, str) or cand_str == "":
            continue
        s1_row = s1_idx.loc[s1_id]
        for cand_id in cand_str.split(","):
            if cand_id not in other_idx.index:
                continue
            other_row = other_idx.loc[cand_id]
            feats = build_pair_features(s1_row, other_row)
            feats["source1_entity_id"] = s1_id
            feats["candidate_entity_id"] = cand_id
            rows.append(feats)

    return pd.DataFrame(rows)


def label_pairs(feature_table: pd.DataFrame, gt: pd.DataFrame):
    """Attach a 0/1 'label' column: 1 if (s1_id, candidate_id) is a true match."""
    true_pairs = set()
    for _, row in gt.iterrows():
        if row["matched_entity_ids"]:
            for m in row["matched_entity_ids"].split(","):
                true_pairs.add((row["source1_entity_id"], m))

    feature_table["label"] = feature_table.apply(
        lambda r: int((r["source1_entity_id"], r["candidate_entity_id"]) in true_pairs),
        axis=1,
    )
    return feature_table


# ---------- F_0.5 evaluation ----------

def f_beta_macro(predictions: pd.DataFrame, gt: pd.DataFrame, beta: float = 0.5):
    """
    predictions: DataFrame with source1_entity_id, matched_entity_ids (comma str)
    gt: DataFrame with source1_entity_id, matched_entity_ids (comma str)
    Computes per-entity F_beta, macro-averaged across all Source1 entities in gt.
    """
    pred_map = dict(zip(predictions["source1_entity_id"], predictions["matched_entity_ids"]))
    scores = []
    for _, row in gt.iterrows():
        true_set = set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
        pred_str = pred_map.get(row["source1_entity_id"], "")
        pred_set = set(pred_str.split(",")) if pred_str else set()

        if not true_set and not pred_set:
            scores.append(1.0)
            continue
        if not pred_set:
            scores.append(0.0)
            continue

        tp = len(true_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(true_set) if true_set else 0.0
        if precision == 0 and recall == 0:
            scores.append(0.0)
            continue
        f = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
        scores.append(f)

    return float(np.mean(scores))


# ---------- Train / predict ----------

FEATURE_COLS = [
    "name_ratio", "name_token_sort_ratio", "name_partial_ratio",
    "addr_ratio", "addr_token_sort_ratio", "country_match", "name_len_diff",
]


def train(args):
    s1, s2, s3 = load_sources(args.data_dir, split="train")
    gt = load_ground_truth(args.data_dir)
    candidates = pd.read_csv(args.candidates, sep="\t").fillna("")

    print("Building feature table (this can take a while on large data)...")
    feats = build_feature_table(s1, s2, s3, candidates)
    feats = label_pairs(feats, gt)
    print(f"Feature table: {feats.shape}, positive rate: {feats['label'].mean():.4f}")

    train_ids, val_ids = train_test_split(
        s1["entity_id"].unique(), test_size=0.2, random_state=42
    )
    train_feats = feats[feats["source1_entity_id"].isin(train_ids)]
    val_feats = feats[feats["source1_entity_id"].isin(val_ids)]

    dtrain = xgb.DMatrix(train_feats[FEATURE_COLS], label=train_feats["label"])
    dval = xgb.DMatrix(val_feats[FEATURE_COLS], label=val_feats["label"])

    params = {
        "max_depth": 5,
        "eta": 0.1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    model = xgb.train(
        params, dtrain, num_boost_round=200,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=20, verbose_eval=20,
    )
    model.save_model(args.model)
    print(f"Model saved to {args.model}")

    # Sweep thresholds on validation set to find best F_0.5 (precision-heavy metric)
    val_feats = val_feats.copy()
    val_feats["score"] = model.predict(dval)

    gt_val = gt[gt["source1_entity_id"].isin(val_ids)]
    best_threshold, best_f05 = 0.5, -1
    for threshold in np.arange(0.3, 0.95, 0.05):
        preds = make_predictions(val_feats, threshold)
        score = f_beta_macro(preds, gt_val, beta=0.5)
        print(f"threshold={threshold:.2f}  F_0.5={score:.4f}")
        if score > best_f05:
            best_f05, best_threshold = score, threshold

    print(f"\nBest threshold: {best_threshold:.2f}  Best F_0.5: {best_f05:.4f}")


def make_predictions(feats_with_scores: pd.DataFrame, threshold: float):
    """Collapse pair-level scores into per-entity matched_entity_ids at a given threshold."""
    matched = feats_with_scores[feats_with_scores["score"] >= threshold]
    grouped = matched.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda ids: ",".join(sorted(set(ids)))
    )
    all_ids = feats_with_scores["source1_entity_id"].unique()
    result = pd.DataFrame({"source1_entity_id": all_ids})
    result["matched_entity_ids"] = result["source1_entity_id"].map(grouped).fillna("")
    return result


def predict(args):
    s1, s2, s3 = load_sources(args.data_dir, split="test")
    candidates = pd.read_csv(args.candidates, sep="\t").fillna("")

    feats = build_feature_table(s1, s2, s3, candidates)
    model = xgb.Booster()
    model.load_model(args.model)

    dmat = xgb.DMatrix(feats[FEATURE_COLS])
    feats["score"] = model.predict(dmat)

    preds = make_predictions(feats, threshold=args.threshold)

    # Ensure every Source1 test entity has exactly one row, even with no candidates
    all_s1 = pd.DataFrame({"source1_entity_id": s1["entity_id"]})
    preds = all_s1.merge(preds, on="source1_entity_id", how="left")
    preds["matched_entity_ids"] = preds["matched_entity_ids"].fillna("")

    preds.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(preds)} rows to {args.out}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data-dir", default="dataset/train")
    p_train.add_argument("--candidates", default="output/candidate_pairs.tsv")
    p_train.add_argument("--model", default="model.json")
    p_train.set_defaults(func=train)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--data-dir", default="dataset/test")
    p_pred.add_argument("--candidates", default="output/candidate_pairs.tsv")
    p_pred.add_argument("--model", default="model.json")
    p_pred.add_argument("--threshold", type=float, default=0.6)
    p_pred.add_argument("--out", default="output/matching_results.tsv")
    p_pred.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
