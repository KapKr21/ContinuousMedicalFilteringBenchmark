"""
Prepare RxNorm for the ContinuousMedicalFilteringBenchmark.

Task: Clean-Clean Entity Resolution across drug name term types.
  RxNorm assigns one CUI (Concept Unique Identifier) to each clinical drug
  concept, but stores multiple term types for the same concept:
    - IN  = ingredient name       (e.g. "acetaminophen")
    - BN  = brand name            (e.g. "Tylenol")
    - SY  = synonym
    - PIN = precise ingredient
  Two atoms sharing a CUI are the same real-world drug → true match.

  Collection A — ingredient names  (TTY = IN)
  Collection B — brand names       (TTY = BN)
  Ground truth — shared RxCUI

Source: RxNorm "Current Prescribable Content" (CPC) subset.
  - No UMLS license required, no account needed.
  - Released monthly; ~100 MB zipped.
  - URL: https://download.nlm.nih.gov/rxnorm/RxNorm_current_prescribable_content.zip

Download (~50 MB, no login required):
  curl -O "https://download.nlm.nih.gov/rxnorm/RxNorm_full_prescribe_06012026.zip"
  unzip RxNorm_full_prescribe_06012026.zip -d /tmp/rxnorm_cpc
  # The RRF files are inside the rrf/ subdirectory
  cp /tmp/rxnorm_cpc/rrf/RXNCONSO.RRF \\
    <repo>/blockingWorkflows/data/medical/rawdata/RXNCONSO.RRF

  Check for newer releases at:
    https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html
  → "Current Prescribable Content" row (marked "no license required")

Run from any directory:
  python medicalDatasets/prepare/prepare_rxnorm.py

Output (written to blockingWorkflows/data/medical/rawdata/):
  rxnormA.csv            — ingredient name atoms  (TTY=IN)
  rxnormB.csv            — brand name atoms       (TTY=BN)
  rxnorm_groundtruth.csv — matching pairs by shared RxCUI
"""

import os
import sys
import pandas as pd

# Path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RAW_DIR    = os.path.join(REPO_ROOT, "blockingWorkflows", "data", "medical", "rawdata")
OUTPUT_DIR = RAW_DIR

# The CPC subset's RXNCONSO.RRF — same format as full RxNorm but much smaller
RXNCONSO_PATH = os.path.join(RAW_DIR, "RXNCONSO.RRF")

# Preflight check
if not os.path.isfile(RXNCONSO_PATH):
    print(f"ERROR: {RXNCONSO_PATH} not found.")
    print()
    print("Download with these commands (no login required):")
    print()
    print('  curl -O "https://download.nlm.nih.gov/rxnorm/RxNorm_full_prescribe_06012026.zip"')
    print('  unzip RxNorm_full_prescribe_06012026.zip -d /tmp/rxnorm_cpc')
    print(f'  cp /tmp/rxnorm_cpc/rrf/RXNCONSO.RRF "{RXNCONSO_PATH}"')
    print()
    print("Check for a newer release at:")
    print("  https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html")
    print('  → look for the "Current Prescribable Content" row (no license required)')
    print(f"  → extract rrf/RXNCONSO.RRF → place at: {RXNCONSO_PATH}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load RXNCONSO.RRF
# RxNorm RRF columns (pipe-delimited, no header):
# RXCUI | LAT | TS | LUI | STT | SUI | ISPREF | RXAUI | SAUI | SCUI |
# SDUI  | SAB | TTY | CODE | STR | SRL | SUPPRESS | CVF
COLS = [
    'RXCUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF',
    'RXAUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE',
    'STR', 'SRL', 'SUPPRESS', 'CVF',
]

print(f"Loading {RXNCONSO_PATH} ...")
df = pd.read_csv(
    RXNCONSO_PATH,
    sep='|',
    names=COLS,
    index_col=False,
    dtype=str,
    low_memory=False,
)

# Keep English, non-suppressed atoms only
df = df[(df['LAT'] == 'ENG') & (df['SUPPRESS'] == 'N')]
print(f"  English non-suppressed atoms: {len(df):,}")
print(f"  Unique RxCUIs: {df['RXCUI'].nunique():,}")
print(f"  TTY distribution (top 10):\n{df['TTY'].value_counts().head(10).to_string()}")

# Build two collections
# Collection A: ingredient names (TTY = IN)
# Collection B: brand names      (TTY = BN)
# Using RXAUI as the atom-level unique id.

ingr = (
    df[df['TTY'] == 'IN'][['RXAUI', 'RXCUI', 'STR', 'TTY']]
    .copy()
    .rename(columns={'RXAUI': 'id', 'STR': 'name'})
    .drop_duplicates(subset='id')
)

brand = (
    df[df['TTY'] == 'BN'][['RXAUI', 'RXCUI', 'STR', 'TTY']]
    .copy()
    .rename(columns={'RXAUI': 'id', 'STR': 'name'})
    .drop_duplicates(subset='id')
)

print(f"  Ingredient atoms (TTY=IN): {len(ingr):,}")
print(f"  Brand name atoms (TTY=BN): {len(brand):,}")

# Build ground truth
# Every (ingredient atom, brand atom) pair that shares a RxCUI = true match.
# Many-to-many is fine and reflects real-world ER difficulty.
gt = (
    ingr[['id', 'RXCUI']]
    .merge(brand[['id', 'RXCUI']], on='RXCUI', suffixes=('_ingr', '_brand'))
    .rename(columns={'id_ingr': 'id1', 'id_brand': 'id2'})[['id1', 'id2']]
    .drop_duplicates()
)

# Filter collections to matched atoms only
ingr_matched  = ingr[ingr['id'].isin(gt['id1'])].copy()
brand_matched = brand[brand['id'].isin(gt['id2'])].copy()

print(f"  Matched ingredient atoms: {len(ingr_matched):,}")
print(f"  Matched brand atoms:      {len(brand_matched):,}")
print(f"  Match pairs:              {len(gt):,}")

if len(gt) == 0:
    print()
    print("WARNING: No matches found between IN and BN term types.")
    print("  The Prescribable Content subset may not include BN terms.")
    print("  Falling back to IN vs SY (synonym) pairs instead...")

    syn = (
        df[df['TTY'] == 'SY'][['RXAUI', 'RXCUI', 'STR', 'TTY']]
        .copy()
        .rename(columns={'RXAUI': 'id', 'STR': 'name'})
        .drop_duplicates(subset='id')
    )
    print(f"  Synonym atoms (TTY=SY): {len(syn):,}")

    gt = (
        ingr[['id', 'RXCUI']]
        .merge(syn[['id', 'RXCUI']], on='RXCUI', suffixes=('_ingr', '_syn'))
        .rename(columns={'id_ingr': 'id1', 'id_syn': 'id2'})[['id1', 'id2']]
        .drop_duplicates()
    )
    ingr_matched  = ingr[ingr['id'].isin(gt['id1'])].copy()
    brand_matched = syn[syn['id'].isin(gt['id2'])].copy()
    print(f"  Matched ingredient atoms: {len(ingr_matched):,}")
    print(f"  Matched synonym atoms:    {len(brand_matched):,}")
    print(f"  Match pairs (IN vs SY):   {len(gt):,}")

# Save
ingr_matched.to_csv(os.path.join(OUTPUT_DIR, "rxnormA.csv"), index=False)
brand_matched.to_csv(os.path.join(OUTPUT_DIR, "rxnormB.csv"), index=False)
gt.to_csv(os.path.join(OUTPUT_DIR, "rxnorm_groundtruth.csv"), index=False)

print(f"\nDone.")
print(f"  Collection A (ingredients): {len(ingr_matched):,} atoms  → {os.path.join(OUTPUT_DIR, 'rxnormA.csv')}")
print(f"  Collection B (brands):      {len(brand_matched):,} atoms  → {os.path.join(OUTPUT_DIR, 'rxnormB.csv')}")
print(f"  Match pairs:                {len(gt):,}           → {os.path.join(OUTPUT_DIR, 'rxnorm_groundtruth.csv')}")
