"""
Prepare FEBRL datasets 1–4 for the ContinuousMedicalFilteringBenchmark.

FEBRL (Freely Extensible Biomedical Record Linkage) datasets contain synthetic
patient records (name, DOB, address) with known ground-truth duplicate pairs.

Dataset summary:
  FEBRL 1 — 1 000 records  (500 originals + 500 duplicates)  — Dirty ER
  FEBRL 2 — 5 000 records  (4 000 originals + 1 000 dups)    — Dirty ER
  FEBRL 3 — 5 000 records  (2 000 originals + 3 000 dups)    — Dirty ER
  FEBRL 4 — 10 000 records split into 4a (5 000) / 4b (5 000) — Clean-Clean ER

For FEBRL 1–3 (Dirty ER / deduplication):
  - The full dataset is written as both A and B (same file, deduplication task).
  - Ground truth links are all pairs (rec_id_i, rec_id_j) where j > i and the
    pair is a known match (lower-triangular of the match matrix).

For FEBRL 4 (Clean-Clean ER):
  - dfA = dataset4a.csv  (originals)
  - dfB = dataset4b.csv  (duplicates)
  - Ground truth: each record in A has exactly one matching record in B;
    the link index encodes this as (rec-X_org, rec-X_dup) pairs.

Output per dataset (written to rawdata/):
  febrl1.csv / febrl1_groundtruth.csv
  febrl2.csv / febrl2_groundtruth.csv
  febrl3.csv / febrl3_groundtruth.csv
  febrlA.csv / febrlB.csv / febrl4_groundtruth.csv

Run from the medicalDatasets/prepare/ directory:
  python prepare_febrl.py
"""

import os
import pandas as pd
from recordlinkage.datasets import load_febrl1, load_febrl2, load_febrl3, load_febrl4

OUTPUT_DIR = "blockingWorkflows/data/medical/rawdata"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_groundtruth(links: pd.MultiIndex, path: str) -> int:
    """Convert a MultiIndex of match pairs to a two-column CSV (id1, id2)."""
    gt = links.to_frame(index=False)
    gt.columns = ["id1", "id2"]
    gt.to_csv(path, index=False)
    return len(gt)

# FEBRL 1–3: Dirty ER (single dataset, deduplication)
for num, loader in [(1, load_febrl1), (2, load_febrl2), (3, load_febrl3)]:
    df, links = loader(return_links=True)
    df = df.reset_index()  # promote rec_id from index to a regular column

    out_csv = os.path.join(OUTPUT_DIR, f"febrl{num}.csv")
    out_gt  = os.path.join(OUTPUT_DIR, f"febrl{num}_groundtruth.csv")

    df.to_csv(out_csv, index=False)
    n_matches = save_groundtruth(links, out_gt)

    print(f"FEBRL {num}: {len(df)} records, {n_matches} duplicate pairs "
          f"→ {out_csv}, {out_gt}")

# FEBRL 4: Clean-Clean ER (two separate datasets)
dfA, dfB, links = load_febrl4(return_links=True)
dfA = dfA.reset_index()  # column: rec_id
dfB = dfB.reset_index()

out_A  = os.path.join(OUTPUT_DIR, "febrlA.csv")
out_B  = os.path.join(OUTPUT_DIR, "febrlB.csv")
out_gt = os.path.join(OUTPUT_DIR, "febrl4_groundtruth.csv")

dfA.to_csv(out_A, index=False)
dfB.to_csv(out_B, index=False)
n_matches = save_groundtruth(links, out_gt)

print(f"FEBRL 4: A={len(dfA)} records, B={len(dfB)} records, "
      f"{n_matches} matching pairs → {out_A}, {out_B}, {out_gt}")