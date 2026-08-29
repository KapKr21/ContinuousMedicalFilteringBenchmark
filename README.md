# Continuous Medical Filtering Benchmark

An extension of the [Continuous Filtering Benchmark](https://github.com/gpapadis/ContinuousFilteringBenchmark) to the medical domain, evaluating blocking workflows, sparse nearest-neighbor methods (ε-Join, kNN-Join), and dense nearest-neighbor methods (FAISS, ScaNN, DeepBlocker) across nine medical and biomedical ER datasets.

**Supervisor:** Franziska Neuhof  
**Team:** Adam Ben Rejeb, Kapil Kumar Khatri, Melvin Vincent  
**Institution:** Leibniz University Hannover  
**Course:** AI/ML in Healthcare (SS 2026)

## Publication

> *Adam Ben Rejeb, Kapil Kumar Khatri, and Vincent Melvin*. "Continuous Medical Filtering Benchmark: An Extension of Filtering Techniques for Entity Resolution in Medical Data." 2026.

**Paper (Overleaf):** [Read-only view](https://tex.cloud.uni-hannover.de/read/qrmbkxbhnkjh#490b69)  
**Paper (PDF):** [latex/paper.pdf](latex/paper.pdf)  
**LaTeX Source:** [latex/paper.tex](latex/paper.tex)  
**Code:** [github.com/KapKr21/ContinuousMedicalFilteringBenchmark](https://github.com/KapKr21/ContinuousMedicalFilteringBenchmark)

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
| Synthea | Synthetic EHR patients | Synthea generator | 5,660 | 6,228 | 500 | Clean-Clean ER |
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

### Dense Nearest-Neighbor Methods (nnmethods/)

Neural embedding-based filtering experiments using:
- **FAISS** (Flat/exact and HNSW/approximate indexes)
- **ScaNN** (Brute-force and Asymmetric Hashing)
- **DeepBlocker** (AutoEncoder, CTT, Hybrid tuple embeddings)

See `nnmethods/README.md` for setup and execution instructions.

## How to Run

### Step 1: Prepare datasets

```bash
python medicalDatasets/prepare/prepare_febrl.py
python medicalDatasets/prepare/prepare_synthea.py
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

#### Blocking Workflows

```bash
# Parameter-Free Blocking Workflow
javac -cp "blockingWorkflows/lib/*" -d out MedicalBFRunner.java
java -Xmx12g -cp "out:blockingWorkflows/lib/*" MedicalBFRunner

# Parameter-Tuned Blocking Workflow (grid search)
javac -cp "blockingWorkflows/lib/*" -d out MedicalBFRunnerFineTuned.java
java -Xmx12g -cp "out:blockingWorkflows/lib/*" MedicalBFRunnerFineTuned

# Run a single dataset by index (0-8):
java -Xmx12g -cp "out:blockingWorkflows/lib/*" MedicalBFRunnerFineTuned 3
```

#### Sparse Nearest-Neighbor Methods (ε-Join, kNN-Join)

```bash
# Epsilon-Join
javac -cp "joins/lib/*:blockingWorkflows/lib/*" -d out joins/src/utilities/*.java MedicalEpsilonJoinsRunner.java
java -Xmx12g -cp "out:joins/lib/*:blockingWorkflows/lib/*" MedicalEpsilonJoinsRunner

# kNN-Join (uses TopKSchemaAgnosticJoin from base benchmark)
# Configure datasets in TopKSchemaAgnosticJoin.java before running
javac -cp "joins/lib/*:blockingWorkflows/lib/*" -d out joins/src/utilities/*.java joins/src/joins/TopKSchemaAgnosticJoin.java
java -Xmx12g -cp "out:joins/lib/*:blockingWorkflows/lib/*" joins.TopKSchemaAgnosticJoin
```

#### Dense Nearest-Neighbor Methods (FAISS, ScaNN, DeepBlocker)

```bash
cd nnmethods/medical

# Install dependencies
pip install -r requirements.txt

# Prepare embeddings (using all-MiniLM-L6-v2)
python prepare_medical_nn_data.py

# Run FAISS (Flat and HNSW indexes)
python run_faiss_medical.py

# Run ScaNN (Brute-force and Asymmetric Hashing)
python run_scann_medical.py

# Run DeepBlocker (AutoEncoder, CTT, Hybrid)
python run_deepblocker_medical.py
```

**Note:** Dense NN methods were run on SLURM cluster. See `nnmethods/medical/slurm_*.sh` for batch job scripts.

## Directory Structure

```
.
├── MedicalBFParameterFreeRunner.java   # Parameter-free blocking workflow
├── MedicalBFRunnerFineTuned.java       # Fine-tuned blocking workflow (grid search)
├── MedicalBFRunnerFineTunedLite.java   # Fine-tuned blocking workflow (lite grid search)
├── MedicalEpsilonJoinsRunner.java      # Epsilon-join experiments
├── latex/                              # Paper LaTeX source and PDF
│   ├── paper.tex                       # Main LaTeX source
│   ├── paper.pdf                       # Compiled paper
│   ├── llncs.cls                       # Springer LNCS document class
│   └── splncs04.bst                    # Bibliography style
├── medicalDatasets/
│   ├── prepare/                        # Python scripts to prepare CSV files
│   │   ├── prepare_febrl.py
│   │   ├── prepare_synthea.py
│   │   ├── prepare_medmentions.py
│   │   ├── prepare_cms.py
│   │   ├── prepare_umls.py
│   │   └── prepare_rxnorm.py
│   ├── convert/                        # Java converter: CSV -> JedAI serialized format
│   └── README.md                       # Detailed dataset preparation instructions
├── blockingWorkflows/                  # Original blocking workflow code and data
│   ├── lib/                            # JedAI library JARs
│   └── data/medical/                   # Serialized medical dataset profiles
├── joins/                              # Original joins code (ε-Join, kNN-Join)
│   ├── lib/                            # Join library JARs
│   └── src/                            # Join source code (utilities, TopK, ε-Join, etc.)
├── nnmethods/                          # Neural network methods
│   ├── medical/                        # Medical dataset experiments
│   │   ├── prepare_medical_nn_data.py  # Embedding preparation
│   │   ├── run_faiss_medical.py        # FAISS experiments
│   │   ├── run_scann_medical.py        # ScaNN experiments
│   │   ├── run_deepblocker_medical.py  # DeepBlocker experiments
│   │   ├── requirements.txt            # Python dependencies
│   │   └── slurm_*.sh                  # SLURM batch scripts
│   ├── faiss/                          # FAISS implementation
│   ├── scann/                          # ScaNN implementation
│   └── deepblocker/                    # DeepBlocker implementation
└── README.md                           # This file
```

## Notes

- UMLS and RxNorm use Q-Grams blocking (q=6) instead of Standard Blocking due to short ontology terms.
- Synthea is treated as Dirty ER using a single merged collection with injected duplicates.
- Run with `-Xmx12g` for UMLS and CMS due to memory requirements.
- All prepare scripts use `__file__`-based path resolution and work from any working directory.

## AI Usage Declaration

During this project, we used AI assistance (Claude) for:

- **Brainstorming and ideation**: Dataset selection, method comparison design, experimental setup discussions
- **Code debugging**: Error analysis, implementation troubleshooting, dependency resolution
- **Documentation**: README structuring, code comments, result interpretation guidance
- **Data preprocessing**: Script generation for CSV parsing, data format conversion, ground truth validation

All core research contributions, experimental design decisions, result analysis, and paper writing were performed by the team. AI tools served as accelerators for implementation and documentation tasks, not as primary authors or decision-makers.

## References

**Base Benchmark:**  
George Papadakis, Marco Fisichella, Franziska Schoger, George Mandilaras, Nikolaus Augsten, Wolfgang Nejdl. "Benchmarking filtering techniques for entity resolution." *Proceedings of the 39th IEEE International Conference on Data Engineering (ICDE)*, 2023. ([pdf](https://arxiv.org/abs/2202.12521))

**Extended Journal Version:**  
Felix Neuhof, Marco Fisichella, George Papadakis, Kostas Nikoletos, Nikolaus Augsten, Wolfgang Nejdl, Manolis Koubarakis. "Open benchmark for filtering techniques in entity resolution." *The VLDB Journal* 33(5), 2024.

**Original Repository:**  
[gpapadis/ContinuousFilteringBenchmark](https://github.com/gpapadis/ContinuousFilteringBenchmark)

## License

This project extends work originally licensed under Apache License 2.0. See `LICENSE` file for details.
