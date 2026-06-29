# Filtering Techniques for Entity Resolution in Medical Data

This repository applies state-of-the-art filtering techniques for Entity Resolution (ER) to medical datasets. It is a fork of [ContinuousFilteringBenchmark](https://github.com/gpapadis/ContinuousFilteringBenchmark) by Papadakis et al. (ICDE 2023).

**Supervisor:** Franziska Neuhof  
**Team:** Adam Ben Rejeb, Kapil Kumar Khatri, Melvin Vincent  
**Institution:** Leibniz University Hannover

## Overview

Entity Resolution is the task of identifying records across one or more datasets that refer to the same real-world entity, without a shared unique key. The naive pairwise approach has quadratic complexity, so filtering techniques reduce the candidate space before verification.

Three families of filtering methods are benchmarked:

| Family | Approach |
|---|---|
| Blocking Workflows | Token/Q-gram blocks + block purging + meta-blocking (JedAI) |
| String Similarity Joins | Inverted-index similarity join with cosine/Jaccard threshold |
| NN Methods | Neural embedding-based candidate generation |

## Medical Datasets

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

## Experiment Runners

| Runner | Method | Description |
|---|---|---|
| `MedicalBFRunner.java` | Parameter-Free Blocking Workflow | Standard/Q-Grams + Block Purging + Comparison Propagation |
| `MedicalBFRunnerFineTuned.java` | Parameter-Tuned Blocking Workflow | Grid search over block building, filtering, and meta-blocking |
| `MedicalEpsilonJoinsRunner.java` | Epsilon-Join | Inverted-index similarity join with cosine/Jaccard threshold |

## How to Run

### Step 1: Prepare datasets

```bash
python medicalDatasets/prepare/prepare_febrl.py
python medicalDatasets/prepare/prepare_medmentions.py
python medicalDatasets/prepare/prepare_cms.py
python medicalDatasets/prepare/prepare_umls.py
python medicalDatasets/prepare/prepare_rxnorm.py
```

### Step 2: Convert to JedAI serialized format

```bash
javac -cp "blockingWorkflows/lib/*" -d out medicalDatasets/convert/MedicalDataConverter.java
java -cp "out:blockingWorkflows/lib/*" MedicalDataConverter
```

### Step 3: Run experiments

```bash
# Parameter-Free Blocking Workflow
javac -cp "blockingWorkflows/lib/*" -d out MedicalBFRunner.java
java -Xmx12g -cp "out:blockingWorkflows/lib/*" MedicalBFRunner

# Parameter-Tuned Blocking Workflow (grid search)
javac -cp "blockingWorkflows/lib/*" -d out MedicalBFRunnerFineTuned.java
java -Xmx12g -cp "out:blockingWorkflows/lib/*" MedicalBFRunnerFineTuned

# Run a single dataset by index (0-8):
java -Xmx12g -cp "out:blockingWorkflows/lib/*" MedicalBFRunnerFineTuned 3

# Epsilon-Join
javac -cp "joins/lib/*:blockingWorkflows/lib/*" -d out joins/src/utilities/*.java MedicalEpsilonJoinsRunner.java
java -Xmx12g -cp "out:joins/lib/*:blockingWorkflows/lib/*" MedicalEpsilonJoinsRunner
```

## Directory Structure

```
.
├── MedicalBFRunner.java              # Parameter-free blocking workflow
├── MedicalBFRunnerFineTuned.java     # Fine-tuned blocking workflow (grid search)
├── MedicalEpsilonJoinsRunner.java    # Epsilon-join experiments
├── medicalDatasets/
│   ├── prepare/                      # Python scripts to prepare CSV files
│   ├── convert/                      # Java converter: CSV -> JedAI serialized format
│   └── README.md                     # Detailed dataset preparation instructions
├── blockingWorkflows/                # Original blocking workflow code and data
│   ├── lib/                          # JedAI library JARs
│   └── data/medical/                 # Serialized medical dataset profiles
├── joins/                            # Original joins code
│   ├── lib/                          # Join library JARs
│   └── src/                          # Join source code (utilities, TopK, etc.)
└── nnMethods/                        # Neural network methods (future work)
```

## Notes

- UMLS and RxNorm use Q-Grams blocking (q=6) instead of Standard Blocking due to short ontology terms.
- Synthea is treated as Dirty ER using a single merged collection with injected duplicates.
- Run with `-Xmx12g` for UMLS and CMS due to memory requirements.
- All prepare scripts use `__file__`-based path resolution and work from any working directory.

## Base Paper

*George Papadakis, Marco Fisichella, Franziska Schoger, George Mandilaras, Nikolaus Augsten, Wolfgang Nejdl: "How to reduce the search space of Entity Resolution: with Blocking or Nearest Neighbor search?"* ([pdf](https://arxiv.org/abs/2202.12521))

Original repository: [gpapadis/ContinuousFilteringBenchmark](https://github.com/gpapadis/ContinuousFilteringBenchmark)
