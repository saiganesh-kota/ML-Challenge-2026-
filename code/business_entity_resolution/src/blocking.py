"""
blocking.py
Business Entity Resolution Challenge - Amazon ML Challenge 2026

Candidate generation stage: for each Source1 entity, find plausible
candidate matches from Source2 and Source3 using normalized-name
similarity search. This sets the recall ceiling for the whole pipeline,
so it deliberately over-generates (favor recall here; precision is the
matching model's job).

Usage:
    python src/blocking.py --data-dir dataset/train --split train --top-k 20
"""

import argparse
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from load_data import load_sources


LEGAL_SUFFIX_MAP = {
    "corporation": "corp",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "company": "co",
    " and ": " & ",
}


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, normalize common legal-suffix abbreviations."""
    name = str(name).lower().strip()
    name = re.sub(r"[^\w\s&]", "", name)
    for full, abbr in LEGAL_SUFFIX_MAP.items():
        name = name.replace(full, abbr)
    return re.sub(r"\s+", " ", name).strip()


def build_blocking_index(candidate_df: pd.DataFrame, top_k: int):
    """
    Fit a TF-IDF + NearestNeighbors index on candidate_df's normalized names.
    Returns the fitted vectorizer and NN model for querying.
    """
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(candidate_df["name_norm"])

    n_neighbors = min(top_k, len(candidate_df))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(tfidf_matrix)

    return vectorizer, nn


def generate_candidates(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, top_k: int = 20):
    """
    For each Source1 entity, return top_k candidate entity_ids from
    Source2 + Source3 combined, ranked by normalized-name similarity.

    Returns a DataFrame with columns: source1_entity_id, candidate_entity_ids (comma-joined str)
    """
    for df in (s1, s2, s3):
        df["name_norm"] = df["business_name"].apply(normalize_name)

    # Combine Source2 + Source3 into one candidate pool
    pool = pd.concat(
        [s2[["entity_id", "name_norm"]], s3[["entity_id", "name_norm"]]],
        ignore_index=True,
    )

    vectorizer, nn = build_blocking_index(pool, top_k)

    s1_vectors = vectorizer.transform(s1["name_norm"])
    distances, indices = nn.kneighbors(s1_vectors)

    results = []
    for i, s1_id in enumerate(s1["entity_id"]):
        candidate_ids = pool.iloc[indices[i]]["entity_id"].tolist()
        # dedupe while preserving order
        seen = set()
        deduped = [c for c in candidate_ids if not (c in seen or seen.add(c))]
        results.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(deduped),
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/train")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", default="output/candidate_pairs.tsv")
    args = parser.parse_args()

    s1, s2, s3 = load_sources(args.data_dir, split=args.split)
    candidates = generate_candidates(s1, s2, s3, top_k=args.top_k)

    candidates.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(candidates)} rows to {args.out}")
    print(f"Avg candidates per entity: {candidates['candidate_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0).mean():.1f}")


if __name__ == "__main__":
    main()
