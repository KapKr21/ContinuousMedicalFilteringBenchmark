"""
run_scann_medical.py
=====================
Runs ScaNN (Scalable Nearest Neighbors by Google) on all medical datasets
for multiple K values.

Two index types:
  - ScaNN AH (Asymmetric Hashing): fast approximate search via quantization
  - ScaNN BF (Brute Force via ScaNN): exact search through ScaNN's interface

Reads pre-computed embeddings (.npy) from prepare_medical_nn_data.py.
Reports: Recall (PC), Precision (PQ), F1, candidates, runtime.

Usage on Pascal:
  source ~/nnenv/bin/activate
  cd ~/ContinuousMedicalFilteringBenchmark

  # Run all datasets:
  nice -n 19 nohup python nnmethods/medical/run_scann_medical.py \
      > nnmethods/medical/results/results_scann.log 2>&1 &

  # Run specific datasets:
  nice -n 19 python nnmethods/medical/run_scann_medical.py \
      --datasets febrl1 febrl4 rxnorm

  # Custom K values:
  nice -n 19 python nnmethods/medical/run_scann_medical.py \
      --k_values 1 5 10 50 100 200 1000
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import scann

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "embeddings")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ALL_DATASETS = ['febrl1', 'febrl2', 'febrl3', 'febrl4', 'synthea',
                'medmentions', 'cms', 'umls', 'rxnorm']
DEFAULT_K_VALUES = [1, 5, 10, 50, 100]


def load_dataset(name, data_dir):
    """Load pre-computed embeddings and ground truth for a dataset."""
    emb_A_path = os.path.join(data_dir, f"{name}_A.npy")
    emb_B_path = os.path.join(data_dir, f"{name}_B.npy")
    gt_path = os.path.join(data_dir, f"{name}_groundtruth.csv")
    ids_A_path = os.path.join(data_dir, f"{name}_idsA.csv")
    ids_B_path = os.path.join(data_dir, f"{name}_idsB.csv")

    for p in [emb_A_path, emb_B_path, gt_path, ids_A_path, ids_B_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}. Run prepare_medical_nn_data.py first.")

    embA = np.load(emb_A_path).astype(np.float32)
    embB = np.load(emb_B_path).astype(np.float32)

    gt_df = pd.read_csv(gt_path)
    idsA_df = pd.read_csv(ids_A_path)
    idsB_df = pd.read_csv(ids_B_path)

    # Build ID-to-position mapping
    id2posA = {}
    for _, row in idsA_df.iterrows():
        id2posA[row['original_id']] = int(row['pos'])

    id2posB = {}
    for _, row in idsB_df.iterrows():
        id2posB[row['original_id']] = int(row['pos'])

    # Convert ground truth to positional indices
    gt_positional = []
    for _, row in gt_df.iterrows():
        posA = id2posA.get(row['id1'])
        posB = id2posB.get(row['id2'])
        if posA is not None and posB is not None:
            gt_positional.append((posA, posB))

    return embA, embB, gt_positional


def compute_metrics(neighbors, gt_positional, n_query, k):
    """
    Compute Recall (PC), Precision (PQ), and F1.

    neighbors: numpy array of shape (n_query, k) — neighbor indices per query
    gt_positional: list of (posA, posB) tuples
    """
    matches = 0
    for (posA, posB) in gt_positional:
        if posB < len(neighbors):
            if posA in neighbors[posB]:
                matches += 1

    total_gt = len(gt_positional)
    candidates = n_query * k

    recall = matches / total_gt if total_gt > 0 else 0.0
    precision = matches / candidates if candidates > 0 else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0

    return recall, precision, f1, matches, candidates


def normalize(data):
    """L2-normalize vectors for dot product = cosine similarity."""
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero
    return data / norms


def compute_scann_params(n_records):
    """
    Compute appropriate ScaNN tree parameters based on dataset size.
    ScaNN requires num_leaves <= number of data points and
    training_sample_size >= num_leaves.
    """
    if n_records < 100:
        num_leaves = max(2, n_records // 2)
        num_leaves_to_search = num_leaves
        training_sample_size = n_records
    elif n_records < 1000:
        num_leaves = 50
        num_leaves_to_search = 50
        training_sample_size = min(n_records, 250)
    elif n_records < 10000:
        num_leaves = 100
        num_leaves_to_search = 100
        training_sample_size = min(n_records, 1000)
    else:
        num_leaves = int(np.sqrt(n_records))
        num_leaves_to_search = min(num_leaves, 300)
        training_sample_size = min(n_records, num_leaves * 10)

    return num_leaves, num_leaves_to_search, training_sample_size


def run_scann_ah(embA, embB, gt_positional, k):
    """Run ScaNN with Asymmetric Hashing (approximate)."""
    # Normalize for dot product similarity
    embA_norm = normalize(embA)
    embB_norm = normalize(embB)

    n_records = len(embA_norm)
    num_leaves, num_leaves_to_search, training_sample_size = compute_scann_params(n_records)

    # Build ScaNN AH index
    t0 = time.time()
    searcher = scann.scann_ops_pybind.builder(embA_norm, k, "dot_product").tree(
        num_leaves=num_leaves,
        num_leaves_to_search=num_leaves_to_search,
        training_sample_size=training_sample_size
    ).score_ah(
        dimensions_per_block=2
    ).build()
    index_time = time.time() - t0

    # Batch search
    t0 = time.time()
    neighbors, distances = searcher.search_batched(embB_norm)
    search_time = time.time() - t0

    recall, precision, f1, matches, candidates = compute_metrics(
        neighbors, gt_positional, len(embB), k)

    return {
        'method': 'ScaNN_AH',
        'k': k,
        'recall_PC': recall,
        'precision_PQ': precision,
        'f1': f1,
        'true_matches': matches,
        'candidates': candidates,
        'gt_size': len(gt_positional),
        'index_time_s': round(index_time, 3),
        'search_time_s': round(search_time, 3),
        'total_time_s': round(index_time + search_time, 3),
    }


def run_scann_bf(embA, embB, gt_positional, k):
    """Run ScaNN with Brute Force (exact, via ScaNN's interface)."""
    embA_norm = normalize(embA)
    embB_norm = normalize(embB)

    n_records = len(embA_norm)
    num_leaves, num_leaves_to_search, training_sample_size = compute_scann_params(n_records)

    # Build ScaNN BF index (exact but using ScaNN's tree partitioning)
    t0 = time.time()
    searcher = scann.scann_ops_pybind.builder(embA_norm, k, "dot_product").tree(
        num_leaves=num_leaves,
        num_leaves_to_search=num_leaves_to_search,
        training_sample_size=training_sample_size
    ).score_brute_force(
        quantize=False
    ).build()
    index_time = time.time() - t0

    # Batch search
    t0 = time.time()
    neighbors, distances = searcher.search_batched(embB_norm)
    search_time = time.time() - t0

    recall, precision, f1, matches, candidates = compute_metrics(
        neighbors, gt_positional, len(embB), k)

    return {
        'method': 'ScaNN_BF',
        'k': k,
        'recall_PC': recall,
        'precision_PQ': precision,
        'f1': f1,
        'true_matches': matches,
        'candidates': candidates,
        'gt_size': len(gt_positional),
        'index_time_s': round(index_time, 3),
        'search_time_s': round(search_time, 3),
        'total_time_s': round(index_time + search_time, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Run ScaNN on medical datasets")
    parser.add_argument('--data_dir', default=DATA_DIR,
                        help=f"Directory with .npy embeddings (default: {DATA_DIR})")
    parser.add_argument('--datasets', nargs='+', default=None,
                        help="Datasets to run (default: all)")
    parser.add_argument('--k_values', nargs='+', type=int, default=DEFAULT_K_VALUES,
                        help=f"K values to test (default: {DEFAULT_K_VALUES})")
    parser.add_argument('--skip_bf', action='store_true',
                        help="Skip Brute Force (only run AH/approximate)")
    parser.add_argument('--output', default=None,
                        help="Output CSV path (default: results/results_scann_medical.csv)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = args.output or os.path.join(RESULTS_DIR, "results_scann_medical.csv")

    datasets = args.datasets or ALL_DATASETS
    all_results = []

    print("=" * 70)
    print("ScaNN Medical Benchmark")
    print(f"  Datasets: {datasets}")
    print(f"  K values: {args.k_values}")
    print(f"  Methods: AH (approximate)" + ("" if args.skip_bf else " + BF (exact via ScaNN)"))
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output: {output_path}")
    print("=" * 70)

    for dataset in datasets:
        print(f"\n{'─'*70}")
        print(f"  DATASET: {dataset}")
        print(f"{'─'*70}")

        try:
            embA, embB, gt_positional = load_dataset(dataset, args.data_dir)
            print(f"  A: {embA.shape}, B: {embB.shape}, GT: {len(gt_positional)} pairs")
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        for k in args.k_values:
            # --- ScaNN AH ---
            print(f"\n  [AH] k={k} ...", end=" ", flush=True)
            try:
                res = run_scann_ah(embA.copy(), embB.copy(), gt_positional, k)
                res['dataset'] = dataset
                all_results.append(res)
                print(f"PC={res['recall_PC']:.4f}  PQ={res['precision_PQ']:.6f}  "
                      f"F1={res['f1']:.4f}  time={res['total_time_s']:.2f}s")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

            # --- ScaNN BF ---
            if not args.skip_bf:
                print(f"  [BF] k={k} ...", end=" ", flush=True)
                try:
                    res = run_scann_bf(embA.copy(), embB.copy(), gt_positional, k)
                    res['dataset'] = dataset
                    all_results.append(res)
                    print(f"PC={res['recall_PC']:.4f}  PQ={res['precision_PQ']:.6f}  "
                          f"F1={res['f1']:.4f}  time={res['total_time_s']:.2f}s")
                except Exception as e:
                    print(f"ERROR: {e}")

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        cols = ['dataset', 'method', 'k', 'recall_PC', 'precision_PQ', 'f1',
                'true_matches', 'candidates', 'gt_size',
                'index_time_s', 'search_time_s', 'total_time_s']
        df = df[[c for c in cols if c in df.columns]]
        df.to_csv(output_path, index=False)
        print(f"\n\n{'='*70}")
        print(f"RESULTS SAVED: {output_path}")
        print(f"{'='*70}")
        print(df.to_string(index=False))
    else:
        print("\nNo results generated.")


if __name__ == '__main__':
    main()
