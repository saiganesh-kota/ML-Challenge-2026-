"""
blocking.py (v4 - chunked streaming, low-memory)
Business Entity Resolution Challenge - Amazon ML Challenge 2026

v3 still loaded full Source2/Source3 files into pandas DataFrames before
indexing - with 5M+ rows each and 3 text columns, that alone can exceed
several GB of RAM (pandas stores strings as Python objects with real
overhead). On a memory-constrained instance (e.g. t3.medium, ~3.7GB, no
swap), this gets OOM-killed before blocking logic even runs.

v4 never holds a full source file in memory. It streams Source2/Source3
in chunks to build the (already-capped) bucket index, then streams
Source1 in chunks to look up candidates and writes results incrementally
to disk. Peak memory is bounded by chunk size + index size, not by total
row count.

Usage:
    python src/blocking.py --data-dir dataset/train --split train --out output/candidate_pairs_train.tsv
"""

import argparse
import re
from collections import defaultdict

import pandas as pd

LEGAL_SUFFIX_MAP = {
    "corporation": "corp",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "company": "co",
    " and ": " & ",
}

MAX_BUCKET_SIZE = 50   # cap candidates stored per blocking key
CHUNK_SIZE = 200_000   # rows read per chunk


def normalize_name(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"[^\w\s&]", "", name)
    for full, abbr in LEGAL_SUFFIX_MAP.items():
        name = name.replace(full, abbr)
    return re.sub(r"\s+", " ", name).strip()


def make_keys(name_norm: str, country: str):
    country_norm = str(country).lower().strip()
    prefix_key = f"{country_norm}|{name_norm[:4]}"
    first_token = name_norm.split()[0] if name_norm.split() else ""
    token_key = f"{country_norm}|{first_token}"
    return prefix_key, token_key


def build_index_from_file(path: str):
    """Stream a source TSV in chunks, building capped bucket indexes without
    ever holding the full file in memory."""
    prefix_index = defaultdict(list)
    token_index = defaultdict(list)

    rows_seen = 0
    for chunk in pd.read_csv(
        path, sep="\t", chunksize=CHUNK_SIZE,
        usecols=["entity_id", "business_name", "country"],
        dtype=str,
    ):
        for entity_id, name, country in zip(
            chunk["entity_id"], chunk["business_name"], chunk["country"]
        ):
            name_norm = normalize_name(name)
            prefix_key, token_key = make_keys(name_norm, country)
            if len(prefix_index[prefix_key]) < MAX_BUCKET_SIZE:
                prefix_index[prefix_key].append(entity_id)
            if len(token_index[token_key]) < MAX_BUCKET_SIZE:
                token_index[token_key].append(entity_id)
        rows_seen += len(chunk)
        print(f"  indexed {rows_seen} rows from {path}...")

    return prefix_index, token_index


def merge_index(dst_prefix, dst_token, src_prefix, src_token):
    """Merge src bucket dicts into dst, respecting the size cap."""
    for k, ids in src_prefix.items():
        bucket = dst_prefix[k]
        for eid in ids:
            if len(bucket) >= MAX_BUCKET_SIZE:
                break
            bucket.append(eid)
    for k, ids in src_token.items():
        bucket = dst_token[k]
        for eid in ids:
            if len(bucket) >= MAX_BUCKET_SIZE:
                break
            bucket.append(eid)


def run_blocking(data_dir: str, split: str, out_path: str, max_candidates: int):
    s2_path = f"{data_dir}/{split}_source2.tsv"
    s3_path = f"{data_dir}/{split}_source3.tsv"
    s1_path = f"{data_dir}/{split}_source1.tsv"

    print(f"Building index from {s2_path} ...")
    prefix_index, token_index = build_index_from_file(s2_path)

    print(f"Building index from {s3_path} ...")
    p3, t3 = build_index_from_file(s3_path)
    merge_index(prefix_index, token_index, p3, t3)
    del p3, t3

    print(f"Index built. Prefix buckets: {len(prefix_index)}  Token buckets: {len(token_index)}")

    # Stream Source1 and write results incrementally - never hold full output in memory
    print(f"Scoring candidates for {s1_path} ...")
    first_write = True
    total_rows = 0
    total_with_candidates = 0

    for chunk in pd.read_csv(
        s1_path, sep="\t", chunksize=CHUNK_SIZE,
        usecols=["entity_id", "business_name", "country"],
        dtype=str,
    ):
        out_rows = []
        for entity_id, name, country in zip(
            chunk["entity_id"], chunk["business_name"], chunk["country"]
        ):
            name_norm = normalize_name(name)
            prefix_key, token_key = make_keys(name_norm, country)
            candidates = set(prefix_index.get(prefix_key, [])) | set(token_index.get(token_key, []))
            candidates = list(candidates)[:max_candidates]
            if candidates:
                total_with_candidates += 1
            out_rows.append({
                "source1_entity_id": entity_id,
                "candidate_entity_ids": ",".join(candidates),
            })

        out_df = pd.DataFrame(out_rows)
        out_df.to_csv(
            out_path, sep="\t", index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )
        first_write = False
        total_rows += len(chunk)
        print(f"  scored {total_rows} Source1 entities...")

    print(f"\nDone. Wrote {total_rows} rows to {out_path}")
    print(f"Entities with >=1 candidate: {total_with_candidates} ({total_with_candidates/total_rows*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/train")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--out", default="output/candidate_pairs.tsv")
    args = parser.parse_args()

    run_blocking(args.data_dir, args.split, args.out, args.max_candidates)


if __name__ == "__main__":
    main()
