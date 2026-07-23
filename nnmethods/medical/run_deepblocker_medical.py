"""
run_deepblocker_medical.py
===========================
Runs DeepBlocker on medical datasets using three tuple embedding models:
  - AutoEncoder: unsupervised, learns compressed representation
  - CTT (Cross-Tuple Training): self-supervised contrastive approach
  - Hybrid: combines AutoEncoder + CTT

DeepBlocker generates its OWN embeddings from raw text — it does NOT use
the pre-computed sentence-transformer embeddings. It trains a small model
on the input text and uses that for blocking.

Reads text-only CSVs from prepare_medical_nn_data.py (deepblocker/ subfolder).
Reports: Recall (PC), Precision (PQ), F1, candidates, runtime.

Prerequisites:
  git clone https://github.com/qcri/DeepBlocker.git ~/DeepBlocker
  cd ~/DeepBlocker && pip install -e .

Usage on Pascal:
  source ~/nnenv/bin/activate
  cd ~/ContinuousMedicalFilteringBenchmark

  # Run all datasets (skip UMLS — too large):
  nice -n 19 nohup python nnmethods/medical/run_deepblocker_medical.py \
      > nnmethods/medical/results/results_deepblocker.log 2>&1 &

  # Run specific datasets:
  nice -n 19 python nnmethods/medical/run_deepblocker_medical.py \
      --datasets febrl1 febrl4 rxnorm

  # Single model only:
  nice -n 19 python nnmethods/medical/run_deepblocker_medical.py \
      --models AutoEncoder --k_values 1 5 10

NOTE: UMLS (135K x 157K) and CMS (64K x 64K) are very large for DeepBlocker.
      Start with smaller datasets first. Use --datasets to control which ones run.
"""

import os
import sys
import time
import argparse
import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "deepblocker")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Add DeepBlocker to path — try common locations
DEEPBLOCKER_PATHS = [
    os.path.expanduser("~/DeepBlocker"),
    os.path.join(SCRIPT_DIR, "DeepBlocker"),
    os.path.join(SCRIPT_DIR, "..", "DeepBlocker"),
]

for db_path in DEEPBLOCKER_PATHS:
    if os.path.isdir(db_path):
        sys.path.insert(0, db_path)
        break

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# Exclude UMLS by default — too large for DeepBlocker's training phase
ALL_DATASETS = ['febrl1', 'febrl2', 'febrl3', 'febrl4', 'synthea',
                'medmentions', 'rxnorm', 'cms']
DEFAULT_K_VALUES = [1, 5, 10, 50]
DEFAULT_MODELS = ['AutoEncoder', 'CTT', 'Hybrid']


def get_model_class(model_name):
    """Lazily import and return the tuple embedding model class."""
    from tuple_embedding_models import (
        AutoEncoderTupleEmbedding,
        CTTTupleEmbedding,
        HybridTupleEmbedding
    )
    models = {
        'AutoEncoder': AutoEncoderTupleEmbedding,
        'CTT': CTTTupleEmbedding,
        'Hybrid': HybridTupleEmbedding,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(models.keys())}")
    return models[model_name]


def load_dataset(name, data_dir):
    """Load DeepBlocker-format CSVs for a dataset."""
    deli = '|'
    pathA = os.path.join(data_dir, f"{name}A.csv")
    pathB = os.path.join(data_dir, f"{name}B.csv")
    pathGT = os.path.join(data_dir, f"{name}_groundtruth.csv")

    for p in [pathA, pathB, pathGT]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {p}. Run prepare_medical_nn_data.py first.")

    left_df = pd.read_csv(pathA, sep=deli)
    right_df = pd.read_csv(pathB, sep=deli)
    golden_df = pd.read_csv(pathGT, sep=deli)

    return left_df, right_df, golden_df


def run_deepblocker(left_df, right_df, golden_df, model_name, k, cols_to_block):
    """Run DeepBlocker with specified model and K."""
    from deep_blocker import DeepBlocker
    from vector_pairing_models import ExactTopKVectorPairing
    import blocking_utils

    ModelClass = get_model_class(model_name)

    t0 = time.time()
    tuple_embedding_model = ModelClass()
    topK_vector_pairing_model = ExactTopKVectorPairing(K=k)
    db = DeepBlocker(tuple_embedding_model, topK_vector_pairing_model)
    candidate_set_df = db.block_datasets(left_df, right_df, cols_to_block)
    elapsed = time.time() - t0

    # Compute blocking statistics
    stats = blocking_utils.compute_blocking_statistics(
        candidate_set_df, golden_df, left_df, right_df)

    recall = stats.get('recall', 0.0)
    precision = stats.get('pq', 0.0)
    candidates = stats.get('candidates', 0)
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0

    return {
        'method': f'DeepBlocker_{model_name}',
        'k': k,
        'recall_PC': recall,
        'precision_PQ': precision,
        'f1': f1,
        'candidates': candidates,
        'gt_size': len(golden_df),
        'total_time_s': round(elapsed, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Run DeepBlocker on medical datasets")
    parser.add_argument('--data_dir', default=DATA_DIR,
                        help=f"Directory with DeepBlocker CSVs (default: {DATA_DIR})")
    parser.add_argument('--datasets', nargs='+', default=None,
                        help=f"Datasets to run (default: {ALL_DATASETS})")
    parser.add_argument('--k_values', nargs='+', type=int, default=DEFAULT_K_VALUES,
                        help=f"K values to test (default: {DEFAULT_K_VALUES})")
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help=f"Embedding models (default: {DEFAULT_MODELS})")
    parser.add_argument('--cols', nargs='+', default=['Aggregate Value'],
                        help="Columns to use for blocking (default: ['Aggregate Value'])")
    parser.add_argument('--output', default=None,
                        help="Output CSV path (default: results/results_deepblocker_medical.csv)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = args.output or os.path.join(RESULTS_DIR, "results_deepblocker_medical.csv")

    datasets = args.datasets or ALL_DATASETS
    all_results = []

    print("=" * 70)
    print("DeepBlocker Medical Benchmark")
    print(f"  Datasets: {datasets}")
    print(f"  K values: {args.k_values}")
    print(f"  Models: {args.models}")
    print(f"  Blocking columns: {args.cols}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output: {output_path}")
    print("=" * 70)

    # Verify DeepBlocker is importable
    try:
        from deep_blocker import DeepBlocker
        from tuple_embedding_models import AutoEncoderTupleEmbedding
        from vector_pairing_models import ExactTopKVectorPairing
        import blocking_utils
        print("  DeepBlocker imported successfully.")
    except ImportError as e:
        print(f"\n  ERROR: Cannot import DeepBlocker: {e}")
        print(f"  Make sure DeepBlocker is installed:")
        print(f"    git clone https://github.com/qcri/DeepBlocker.git ~/DeepBlocker")
        print(f"    cd ~/DeepBlocker && pip install -e .")
        print(f"  Or add it to PYTHONPATH:")
        print(f"    export PYTHONPATH=$PYTHONPATH:~/DeepBlocker")
        sys.exit(1)

    for dataset in datasets:
        print(f"\n{'─'*70}")
        print(f"  DATASET: {dataset}")
        print(f"{'─'*70}")

        try:
            left_df, right_df, golden_df = load_dataset(dataset, args.data_dir)
            print(f"  Left: {len(left_df)} records, Right: {len(right_df)} records, "
                  f"GT: {len(golden_df)} pairs")
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        # Warn if dataset is very large
        total_records = len(left_df) + len(right_df)
        if total_records > 50000:
            print(f"  WARNING: Large dataset ({total_records} total records). "
                  f"DeepBlocker training may be slow.")

        for model_name in args.models:
            for k in args.k_values:
                print(f"\n  [{model_name}] k={k} ...", end=" ", flush=True)
                try:
                    res = run_deepblocker(
                        left_df, right_df, golden_df, model_name, k, args.cols)
                    res['dataset'] = dataset
                    all_results.append(res)
                    print(f"PC={res['recall_PC']:.4f}  PQ={res['precision_PQ']:.6f}  "
                          f"F1={res['f1']:.4f}  time={res['total_time_s']:.1f}s")

                    # Save intermediate results after each run (in case of crash)
                    df_interim = pd.DataFrame(all_results)
                    df_interim.to_csv(output_path + ".partial", index=False)

                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    # Record the error
                    all_results.append({
                        'dataset': dataset,
                        'method': f'DeepBlocker_{model_name}',
                        'k': k,
                        'recall_PC': None,
                        'precision_PQ': None,
                        'f1': None,
                        'candidates': None,
                        'gt_size': len(golden_df),
                        'total_time_s': None,
                        'error': str(e),
                    })

    # Save final results
    if all_results:
        df = pd.DataFrame(all_results)
        cols = ['dataset', 'method', 'k', 'recall_PC', 'precision_PQ', 'f1',
                'candidates', 'gt_size', 'total_time_s']
        available_cols = [c for c in cols if c in df.columns]
        if 'error' in df.columns:
            available_cols.append('error')
        df = df[available_cols]
        df.to_csv(output_path, index=False)

        # Clean up partial file
        partial_path = output_path + ".partial"
        if os.path.exists(partial_path):
            os.remove(partial_path)

        print(f"\n\n{'='*70}")
        print(f"RESULTS SAVED: {output_path}")
        print(f"{'='*70}")
        # Print only successful results
        df_ok = df[df['recall_PC'].notna()]
        if not df_ok.empty:
            print(df_ok.to_string(index=False))
        else:
            print("No successful results.")
    else:
        print("\nNo results generated.")


if __name__ == '__main__':
    main()
