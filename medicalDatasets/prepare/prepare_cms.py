"""
Prepare the CMS Open Data dataset for the ContinuousMedicalFilteringBenchmark.

Task: Clean-Clean Entity Resolution across two CMS healthcare provider datasets.
  The same healthcare provider (identified by NPI) appears in both sources but
  with different attribute sets — making this a realistic cross-source ER task.

  Collection A — NPPES National Provider Registry
                 (name, address, taxonomy/specialty code)
  Collection B — Medicare Provider Utilization & Payment Data
                 (name, address, provider_type description)
  Ground truth — NPI appears in both → same real-world provider → true match

Both datasets are FREE, publicly available, no account required.

Prerequisites — download two files:

  File 1: NPPES NPI Registry (~8 GB zipped, but we only read the first 500k rows)
    URL: https://download.cms.gov/nppes/NPI_Files.html
    → Click "NPPES Data Dissemination" → download the full replacement file
    → Unzip → find the file named like: npidata_pfile_20050523-20XXXXXX.csv
    → Place at: <repo>/blockingWorkflows/data/medical/rawdata/npidata_pfile.csv

  File 2: Medicare Physician & Other Practitioners (~500 MB)
    URL: https://data.cms.gov/provider-summary-by-type-of-service/
         medicare-physician-other-practitioners/
         medicare-physician-other-practitioners-by-provider
    → Click "Export" → CSV → download
    → Place at: <repo>/blockingWorkflows/data/medical/rawdata/medicare_providers.csv

Run from any directory:
  python medicalDatasets/prepare/prepare_cms.py

Output (written to blockingWorkflows/data/medical/rawdata/):
  cmsA.csv            — NPPES provider records that have a Medicare match
  cmsB.csv            — Medicare records for those same providers
  cms_groundtruth.csv — matching pairs (id1=NPI in A, id2=NPI in B)
"""

import os
import sys
import pandas as pd

# Path resolution — works regardless of working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# All raw data lives here — consistent with how every other dataset is stored
RAW_DIR = os.path.join(REPO_ROOT, "blockingWorkflows", "data", "medical", "rawdata")
OUTPUT_DIR    = RAW_DIR
NPPES_PATH    = os.path.join(RAW_DIR, "npidata_pfile.csv")
MEDICARE_PATH = os.path.join(RAW_DIR, "medicare_providers.csv")

NROWS = 500_000   # read first 500k rows from each file

# Helper
def find_col(df: pd.DataFrame, candidates: list) -> str:
    """Return the first column name in df that matches any candidate (case-insensitive).
    Raises a clear KeyError listing available columns if nothing matches."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise KeyError(
        f"None of {candidates} found.\n"
        f"Available columns: {list(df.columns)}"
    )

# Preflight checks
missing = [p for p in [NPPES_PATH, MEDICARE_PATH] if not os.path.isfile(p)]
if missing:
    for p in missing:
        print(f"ERROR: {p} not found.")
    print()
    print("Download instructions:")
    print("  NPPES:    https://download.cms.gov/nppes/NPI_Files.html")
    print("  Medicare: https://data.cms.gov/provider-summary-by-type-of-service/")
    print("            medicare-physician-other-practitioners/")
    print("            medicare-physician-other-practitioners-by-provider")
    print()
    print("Place files as:")
    print(f"  {NPPES_PATH}")
    print(f"  {MEDICARE_PATH}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load NPPES (Collection A)
print(f"Loading NPPES (first {NROWS:,} rows) ...")
npi = pd.read_csv(NPPES_PATH, low_memory=False, nrows=NROWS, dtype=str)

# Column names vary slightly across annual NPPES releases — match flexibly.
npi_col_map = {
    'id':         find_col(npi, ['NPI']),
    'first_name': find_col(npi, ['Provider First Name']),
    'last_name':  find_col(npi, ['Provider Last Name (Legal Name)',
                                  'Provider Organization Name (Legal Business Name)']),
    'city':       find_col(npi, ['Provider Business Practice Location Address City Name']),
    'state':      find_col(npi, ['Provider Business Practice Location Address State Name']),
    'taxonomy':   find_col(npi, ['Healthcare Provider Taxonomy Code_1']),
}
npi = npi[[v for v in npi_col_map.values()]].copy()
npi.columns = list(npi_col_map.keys())
npi = npi.dropna(subset=['id']).drop_duplicates(subset='id')
npi['id'] = npi['id'].str.strip()
print(f"  NPPES rows loaded: {len(npi):,}")

# Load Medicare (Collection B)
print(f"Loading Medicare (first {NROWS:,} rows) ...")
med = pd.read_csv(MEDICARE_PATH, low_memory=False, nrows=NROWS, dtype=str)

# CMS renamed several columns in the 2022+ release (Rndrng_* prefix).
med_col_map = {
    'id':            find_col(med, ['NPI', 'Rndrng_NPI']),
    'first_name':    find_col(med, ['nppes_provider_first_name',  'Rndrng_Prvdr_First_Name']),
    'last_name':     find_col(med, ['nppes_provider_last_org_name', 'Rndrng_Prvdr_Last_Org_Name']),
    'city':          find_col(med, ['nppes_provider_city',  'Rndrng_Prvdr_City']),
    'state':         find_col(med, ['nppes_provider_state', 'Rndrng_Prvdr_State_Abrvtn']),
    'provider_type': find_col(med, ['provider_type', 'Rndrng_Prvdr_Type']),
}
med = med[[v for v in med_col_map.values()]].copy()
med.columns = list(med_col_map.keys())
med = med.dropna(subset=['id']).drop_duplicates(subset='id')
med['id'] = med['id'].str.strip()
print(f"  Medicare rows loaded: {len(med):,}")

# Build ground truth on shared NPI
# NPIs are 10-digit strings; the same NPI in both files = same real provider.
# Ground truth: id1 refers to the row in cmsA, id2 to the row in cmsB.
common_npi = set(npi['id']) & set(med['id'])
print(f"  Providers appearing in both sources: {len(common_npi):,}")

if len(common_npi) == 0:
    print("WARNING: No common NPIs found.")
    print("  Check that both CSV files contain a 10-digit NPI column.")
    print("  Tip: inspect the first few column names with:")
    print("    python -c \"import pandas as pd; print(pd.read_csv('<file>', nrows=2).columns.tolist())\"")
    sys.exit(1)

cmsA = npi[npi['id'].isin(common_npi)].copy().reset_index(drop=True)
cmsB = med[med['id'].isin(common_npi)].copy().reset_index(drop=True)

# Build ground truth by joining on the shared NPI value.
gt = (cmsA[['id']].rename(columns={'id': 'id1'})
      .merge(cmsB[['id']].rename(columns={'id': 'id2'}), left_on='id1', right_on='id2')
      [['id1', 'id2']]
      .drop_duplicates())

# Save
cmsA.to_csv(os.path.join(OUTPUT_DIR, "cmsA.csv"), index=False)
cmsB.to_csv(os.path.join(OUTPUT_DIR, "cmsB.csv"), index=False)
gt.to_csv(os.path.join(OUTPUT_DIR, "cms_groundtruth.csv"), index=False)

print(f"\nDone.")
print(f"  Collection A (NPPES):    {len(cmsA):,} providers → {os.path.join(OUTPUT_DIR, 'cmsA.csv')}")
print(f"  Collection B (Medicare): {len(cmsB):,} providers → {os.path.join(OUTPUT_DIR, 'cmsB.csv')}")
print(f"  Match pairs:             {len(gt):,}             → {os.path.join(OUTPUT_DIR, 'cms_groundtruth.csv')}")
