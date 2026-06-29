"""
Prepare the MedMentions dataset for the ContinuousMedicalFilteringBenchmark.

Task: Clean-Clean Entity Resolution across biomedical entity mentions.
  Two different textual mentions (in different PubMed abstracts) that refer to
  the same UMLS concept (same CUI) are a true match.

  Collection A — first mention of each CUI in the corpus
  Collection B — second mention of each CUI in the corpus
  Ground truth — (mention_id_A, mention_id_B) pairs sharing the same CUI

Dataset: 4,392 PubMed abstracts annotated with UMLS CUIs (full corpus).
Source:  https://github.com/chanzuckerberg/MedMentions

Prerequisites — ONE command (no account needed):
  git clone https://github.com/chanzuckerberg/MedMentions.git \\
      rawdata/MedMentions

  That gives you:  rawdata/MedMentions/full/data/corpus_pubtator.txt

Run from the medicalDatasets/prepare/ directory:
  python prepare_medmentions.py

Output (written to rawdata/):
  medmentionsA.csv            — first mentions per CUI
  medmentionsB.csv            — second mentions per CUI
  medmentions_groundtruth.csv — matching pairs (id1, id2)
"""

import os
import sys
import pandas as pd

# Resolve all paths relative to this script's own location so the script works
# regardless of which directory you invoke it from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Candidate locations for corpus_pubtator.txt — checked in order
CORPUS_CANDIDATES = [
    # 1. Canonical location in the blocking data tree (where your file currently lives)
    os.path.join(REPO_ROOT, "blockingWorkflows", "data", "medical", "rawdata",
                 "MedMentions", "full", "data", "corpus_pubtator.txt"),
    # 2. Next to the other rawdata files, co-located with this script
    os.path.join(SCRIPT_DIR, "rawdata", "MedMentions", "full", "data", "corpus_pubtator.txt"),
]

CORPUS_PATH = next((p for p in CORPUS_CANDIDATES if os.path.isfile(p)), None)

# Output always goes to medicalDatasets/prepare/rawdata/ (alongside other datasets)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "rawdata")

# Preflight check
if CORPUS_PATH is None:
    print("ERROR: corpus_pubtator.txt not found in any of these locations:")
    for p in CORPUS_CANDIDATES:
        print(f"  {p}")
    print()
    print("Run this one command to get the data (no account needed):")
    print()
    print("  git clone https://github.com/chanzuckerberg/MedMentions.git \\")
    print(f"      {os.path.join(REPO_ROOT, 'blockingWorkflows', 'data', 'medical', 'rawdata', 'MedMentions')}")
    sys.exit(1)

print(f"Using corpus: {CORPUS_PATH}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parse PubTator format
# The file has two line types:
#   PMID|t|title text          ← title line  (contains '|')
#   PMID|a|abstract text       ← abstract line (contains '|')
#   PMID\tstart\tend\ttext\ttype\tCUI  ← annotation line (tab-separated, no '|')
# Blank lines separate documents.

print(f"Parsing {CORPUS_PATH} ...")
records = []
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or '|' in line:
            continue                         # skip blank lines and title/abstract lines
        parts = line.split('\t')
        if len(parts) != 6:
            continue                         # skip malformed lines
        pmid, start, end, text, etype, cui = parts
        if not cui.startswith('C'):          # skip non-UMLS annotations (e.g. '-1')
            continue
        records.append({
            'id':          f"{pmid}_{start}_{end}",
            'pmid':        pmid,
            'start':       int(start),
            'end':         int(end),
            'mention_text': text,
            'entity_type': etype,
            'cui':         cui,
        })

df = pd.DataFrame(records).drop_duplicates(subset='id')
print(f"  Total valid annotations: {len(df):,}")
print(f"  Unique CUIs:             {df['cui'].nunique():,}")

# Build two collections
# Keep only CUIs that appear at least twice (needed for a pair).
# Take the 1st occurrence per CUI as collection A, 2nd as collection B.
# Using nth(0) / nth(1) after groupby gives reproducible, deterministic picks.

cui_groups = df.groupby('cui', sort=False).filter(lambda x: len(x) >= 2)

dfA = cui_groups.groupby('cui', sort=False).nth(0).reset_index(drop=True)
dfB = cui_groups.groupby('cui', sort=False).nth(1).reset_index(drop=True)

# Align on the CUI key — every CUI in A has exactly one counterpart in B
assert set(dfA['cui']) == set(dfB['cui']), "CUI sets should match after nth()"

# Build ground truth
gt = (dfA[['id', 'cui']]
      .merge(dfB[['id', 'cui']], on='cui', suffixes=('_a', '_b'))
      .rename(columns={'id_a': 'id1', 'id_b': 'id2'})[['id1', 'id2']]
      .drop_duplicates())

# Save
dfA.to_csv(os.path.join(OUTPUT_DIR, "medmentionsA.csv"), index=False)
dfB.to_csv(os.path.join(OUTPUT_DIR, "medmentionsB.csv"), index=False)
gt.to_csv(os.path.join(OUTPUT_DIR, "medmentions_groundtruth.csv"), index=False)

print(f"\nDone.")
print(f"  Collection A: {len(dfA):,} mentions  → {os.path.join(OUTPUT_DIR, 'medmentionsA.csv')}")
print(f"  Collection B: {len(dfB):,} mentions  → {os.path.join(OUTPUT_DIR, 'medmentionsB.csv')}")
print(f"  Match pairs:  {len(gt):,}            → {os.path.join(OUTPUT_DIR, 'medmentions_groundtruth.csv')}")
