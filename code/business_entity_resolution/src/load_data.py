"""
load_data.py
Business Entity Resolution Challenge - Amazon ML Challenge 2026

Loads the 3 source TSVs + ground truth, and runs the EDA that shapes
blocking/matching strategy (singleton rate, country distribution, noise samples).

Usage:
    python src/load_data.py --data-dir dataset/train
"""

import argparse
import pandas as pd


def load_sources(data_dir: str, split: str = "train"):
    """Load source1/source2/source3 TSVs for a given split ('train' or 'test')."""
    s1 = pd.read_csv(f"{data_dir}/{split}_source1.tsv", sep="\t")
    s2 = pd.read_csv(f"{data_dir}/{split}_source2.tsv", sep="\t")
    s3 = pd.read_csv(f"{data_dir}/{split}_source3.tsv", sep="\t")
    return s1, s2, s3


def load_ground_truth(data_dir: str):
    """Load train_ground_truth.tsv (only exists for train split)."""
    gt = pd.read_csv(f"{data_dir}/train_ground_truth.tsv", sep="\t")
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    gt["n_matches"] = gt["matched_entity_ids"].apply(
        lambda x: 0 if x == "" else len(x.split(","))
    )
    return gt


def run_eda(s1, s2, s3, gt):
    """Print the diagnostics that inform blocking + matching strategy."""
    print("=" * 60)
    print("SHAPE CHECK")
    print("=" * 60)
    print(f"Source1: {s1.shape}  Source2: {s2.shape}  Source3: {s3.shape}")
    print(f"Ground truth: {gt.shape}")

    print("\n" + "=" * 60)
    print("SINGLETON RATE (critical for F_0.5 - empty match = 1.0 if correct)")
    print("=" * 60)
    dist = gt["n_matches"].value_counts().sort_index()
    print(dist)
    singleton_pct = (gt["n_matches"] == 0).mean() * 100
    print(f"\n{singleton_pct:.1f}% of Source1 entities have ZERO true matches")

    print("\n" + "=" * 60)
    print("COUNTRY DISTRIBUTION (open set - don't hardcode to train values)")
    print("=" * 60)
    for name, df in [("Source1", s1), ("Source2", s2), ("Source3", s3)]:
        print(f"\n{name}:")
        print(df["country"].value_counts())

    print("\n" + "=" * 60)
    print("SAMPLE NAME/ADDRESS NOISE (eyeball this for normalization rules)")
    print("=" * 60)
    print(s1[["business_name", "business_address", "country"]].head(15).to_string())

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    for name, df in [("Source1", s1), ("Source2", s2), ("Source3", s3)]:
        print(f"{name}:\n{df.isnull().sum()}\n")

    return {
        "singleton_pct": singleton_pct,
        "n_matches_dist": dist,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/train")
    args = parser.parse_args()

    s1, s2, s3 = load_sources(args.data_dir, split="train")
    gt = load_ground_truth(args.data_dir)
    run_eda(s1, s2, s3, gt)


if __name__ == "__main__":
    main()
