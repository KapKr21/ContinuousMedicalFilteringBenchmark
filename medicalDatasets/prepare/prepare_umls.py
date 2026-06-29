"""
Prepare the UMLS Metathesaurus dataset for the ContinuousMedicalFilteringBenchmark.

Task: Clean-Clean Entity Resolution across two biomedical vocabularies.
  Collection A — SNOMED CT (SNOMEDCT_US): clinical concepts with preferred terms
  Collection B — MeSH (MSH): Medical Subject Headings terms

Ground truth: two entries sharing the same CUI (Concept Unique Identifier)
refer to the same real-world biomedical concept, so they are a matching pair.

Because the UMLS mapping is many-to-many (one SNOMED concept can map to multiple
MeSH terms and vice versa), the ground truth preserves all valid pairs.

Prerequisites:
  1. A UMLS license — sign up free at https://uts.nlm.nih.gov/uts/signup-login
  2. Download the UMLS Metathesaurus release from:
       https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html
     You only need META/MRCONSO.RRF from the archive (~1.5 GB file).
  3. Place it at:  medicalDatasets/prepare/META/MRCONSO.RRF

Run from the medicalDatasets/prepare/ directory:
  python prepare_umls.py

Output (written to rawdata/):
  umlsA.csv            — SNOMED CT terms that have at least one MeSH match
  umlsB.csv            — MeSH terms that have at least one SNOMED match
  umls_groundtruth.csv — matching pairs (id1=SNOMED AUI, id2=MeSH AUI)
"""

import os
import sys
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RAW_DIR      = os.path.join(REPO_ROOT, "blockingWorkflows", "data", "medical", "rawdata")

# MRCONSO.RRF candidate locations — checked in order
MRCONSO_CANDIDATES = [
    # 1. Standard location alongside the other datasets
    os.path.join(RAW_DIR, "MRCONSO.RRF"),
    # 2. Legacy location used in earlier versions of this script
    os.path.join(SCRIPT_DIR, "META", "MRCONSO.RRF"),
]

MRCONSO_PATH = next((p for p in MRCONSO_CANDIDATES if os.path.isfile(p)), None)
OUTPUT_DIR   = RAW_DIR

# Preflight check
if MRCONSO_PATH is None:
    print("ERROR: MRCONSO.RRF not found in any of these locations:")
    for p in MRCONSO_CANDIDATES:
        print(f"  {p}")
    print()
    print("Place MRCONSO.RRF at:")
    print(f"  {MRCONSO_CANDIDATES[0]}")
    print()
    print("Download: https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html")
    sys.exit(1)

print(f"Using: {MRCONSO_PATH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load MRCONSO.RRF
# MRCONSO has 19 pipe-separated columns with no header row.
# The 19th column is always empty (trailing pipe), named '_' to absorb it.
cols = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI',
        'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF', '_']

print("Loading MRCONSO.RRF (this may take a minute for the full release)...")
df = pd.read_csv(
    MRCONSO_PATH,
    sep='|',
    names=cols,
    index_col=False,
    low_memory=False,
    dtype=str,          # read everything as str to avoid mixed-type warnings
)

# Keep only English, non-suppressed rows
df = df[(df['LAT'] == 'ENG') & (df['SUPPRESS'] == 'N')]
print(f"  English non-suppressed rows: {len(df):,}")

# Build the two collections
# AUI  = Atom Unique Identifier  → used as the entity id (unique per row)
# CUI  = Concept Unique Identifier → shared key for ground truth
# STR  = string / preferred term
# SAB  = source vocabulary abbreviation
# TTY  = term type

snomedct = (df[df['SAB'] == 'SNOMEDCT_US'][['AUI', 'CUI', 'STR', 'SAB', 'TTY']]
            .copy()
            .rename(columns={'AUI': 'id', 'STR': 'preferred_term'})
            .drop_duplicates(subset='id'))

mesh = (df[df['SAB'] == 'MSH'][['AUI', 'CUI', 'STR', 'SAB', 'TTY']]
        .copy()
        .rename(columns={'AUI': 'id', 'STR': 'preferred_term'})
        .drop_duplicates(subset='id'))

print(f"  SNOMED CT atoms: {len(snomedct):,}")
print(f"  MeSH atoms:      {len(mesh):,}")

# Build ground truth
# Cross-join on CUI: every (SNOMED AUI, MeSH AUI) pair that shares a CUI is a
# true match.  Many-to-many is intentional and realistic.
merged = snomedct[['id', 'CUI']].merge(
    mesh[['id', 'CUI']],
    on='CUI',
    suffixes=('_snomed', '_mesh')
)
gt = (merged[['id_snomed', 'id_mesh']]
      .rename(columns={'id_snomed': 'id1', 'id_mesh': 'id2'})
      .drop_duplicates())   # safety dedup in case of upstream duplicates

# Filter collections to only matched entities
# This keeps the datasets focused: every entity has at least one true match.
snomedct_matched = snomedct[snomedct['id'].isin(gt['id1'])].copy()
mesh_matched     = mesh[mesh['id'].isin(gt['id2'])].copy()

# Save
snomedct_matched.to_csv(os.path.join(OUTPUT_DIR, "umlsA.csv"), index=False)
mesh_matched.to_csv(os.path.join(OUTPUT_DIR, "umlsB.csv"), index=False)
gt.to_csv(os.path.join(OUTPUT_DIR, "umls_groundtruth.csv"), index=False)

print(f"\nDone.")
print(f"  SNOMED CT (A): {len(snomedct_matched):,} matched atoms  → rawdata/umlsA.csv")
print(f"  MeSH      (B): {len(mesh_matched):,} matched atoms  → rawdata/umlsB.csv")
print(f"  Match pairs:   {len(gt):,}               → rawdata/umls_groundtruth.csv")
