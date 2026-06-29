# Medical Datasets for Entity Resolution

This directory contains preparation scripts and conversion tools for 7 medical datasets used to extend the ContinuousFilteringBenchmark to the healthcare domain.

## Datasets Overview

| Dataset | Type | Source | D1 | D2 | GT | ER Task |
|---|---|---|---|---|---|---|
| FEBRL-1 | Synthetic patient records | Febrl Python library | 1,000 | 1,000 | 500 | Dirty ER |
| FEBRL-2 | Synthetic patient records | Febrl Python library | 5,000 | 5,000 | 1,934 | Dirty ER |
| FEBRL-3 | Synthetic patient records | Febrl Python library | 5,000 | 5,000 | 6,538 | Dirty ER |
| FEBRL-4 | Synthetic patient records | Febrl Python library | 5,000 | 5,000 | 5,000 | Clean-Clean ER |
| Synthea | Synthetic EHR patients | Synthea generator | 5,660 | 6,228 | 500 | Dirty ER |
| MedMentions | PubMed entity mentions | MedMentions corpus | 22,248 | 22,248 | 22,248 | Clean-Clean ER |
| CMS | Medicare physician records | CMS Open Payments | 64,177 | 64,177 | 64,177 | Clean-Clean ER |
| UMLS | Medical ontology concepts | UMLS MRCONSO.RRF | 134,992 | 156,932 | 643,166 | Clean-Clean ER |
| RxNorm | Drug terminology (IN vs BN) | NLM RxNorm CPC | 808 | 868 | 868 | Clean-Clean ER |

## Directory Structure

```
medicalDatasets/
├── README.md                  # This file
├── prepare/                   # Python scripts to download & prepare CSV files
│   ├── prepare_febrl.py
│   ├── prepare_synthea.py
│   ├── prepare_medmentions.py
│   ├── prepare_cms.py
│   ├── prepare_umls.py
│   ├── prepare_rxnorm.py
│   └── prepare_drugbank_open.py
└── convert/                   # Java converter: CSV -> JedAI serialized format
    └── MedicalDataConverter.java
```

## Pipeline

### Step 1: Prepare CSV files

Each prepare script reads raw data from `blockingWorkflows/data/medical/rawdata/` and outputs CSV files (collectionA, collectionB, groundtruth) to the same directory.

```bash
# Run from any directory (scripts use __file__-based path resolution)
python medicalDatasets/prepare/prepare_febrl.py
python medicalDatasets/prepare/prepare_medmentions.py
python medicalDatasets/prepare/prepare_cms.py
python medicalDatasets/prepare/prepare_umls.py
python medicalDatasets/prepare/prepare_rxnorm.py
```

### Step 2: Convert to JedAI serialized format

The Java converter reads the CSVs and produces serialized `EntityProfile` and `IdDuplicates` objects in `blockingWorkflows/data/medical/`.

```bash
javac -cp "blockingWorkflows/lib/*" -d out medicalDatasets/convert/MedicalDataConverter.java
java -cp "out:blockingWorkflows/lib/*" MedicalDataConverter
```

### Step 3: Run experiments

#### Parameter-Free Blocking Workflow
```bash
javac -cp "blockingWorkflows/lib/*" -d out MedicalBFRunner.java
java -Xmx12g -cp "out:blockingWorkflows/lib/*" MedicalBFRunner
```

#### Parameter-Tuned Blocking Workflow (grid search)
```bash
javac -cp "blockingWorkflows/lib/*" -d out MedicalBFRunnerFineTuned.java
java -Xmx12g -cp "out:blockingWorkflows/lib/*" MedicalBFRunnerFineTuned

# Run a single dataset by index (0-8):
java -Xmx12g -cp "out:blockingWorkflows/lib/*" MedicalBFRunnerFineTuned 3
```

#### Epsilon-Join
```bash
javac -cp "joins/lib/*:blockingWorkflows/lib/*" -d out joins/src/utilities/*.java MedicalEpsilonJoinsRunner.java
java -Xmx12g -cp "out:joins/lib/*:blockingWorkflows/lib/*" MedicalEpsilonJoinsRunner
```

## Raw Data Requirements

Place raw data files in `blockingWorkflows/data/medical/rawdata/`:

| Dataset | Required File(s) | How to Obtain |
|---|---|---|
| FEBRL 1-4 | Auto-generated | `pip install recordlinkage` (script downloads automatically) |
| Synthea | `syntheaA.csv`, `syntheaB_with_dups.csv` | Run Synthea generator |
| MedMentions | `MedMentions/full/data/corpus_pubtator.txt` | `git clone` MedMentions repo |
| CMS | `npidata_pfile.csv`, `medicare_providers.csv` | Download from CMS.gov |
| UMLS | `MRCONSO.RRF` | UMLS license required (NIH/NLM) |
| RxNorm | `RXNCONSO.RRF` | Free download from NLM (no license needed) |

## Notes

- All prepare scripts use `__file__`-based path resolution and work from any working directory.
- UMLS and RxNorm use Q-Grams blocking (q=6) instead of Standard Blocking due to short ontology terms.
- Synthea is treated as Dirty ER using a single merged collection with injected duplicates.
- Run with `-Xmx12g` for UMLS/CMS due to memory requirements.
