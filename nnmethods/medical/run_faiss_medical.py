"""
run_faiss_medical.py
=====================
Runs FAISS (Flat = exact brute-force, HNSW = approximate graph-based) on all
medical datasets for multiple K values.

Reads pre-computed embeddings (.npy) from prepare_medical_nn_data.py.
Reports: Recall (PC), Precision (PQ), F1, candidates, runtime.

Usage on Pascal:
  source ~/nnenv/bin/activate
  cd ~/ContinuousMedicalFilteringBenchmark

  # Run all datasets:
  nice -n 19 nohup python nnmethods/medical/run_faiss_medical.py \
      > nnmethods/medical/results/results_faiss.log 2>&1 &

  # Run specific datasets:
  nice -n 19 python nnmethods/medical/run_faiss_medical.py \
      --datasets febrl1 febrl4 rxnorm

  # Custom K values:
  nice -n 19 python nnmethods/medical/run_faiss_medical.py \
      --k_values 1 5 10 50 100 200 1000
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import faiss

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


def compute_metrics(I, gt_positional, n_query, k):
    """
    Compute Recall (PC), Precision (PQ), and F1.

    I: numpy array of shape (n_query, k) — neighbor indices for each query
    gt_positional: list of (posA, posB) tuples — ground truth in positional form
    """
    # For each query (posB), check if its true match (posA) is in the results
    matches = 0
    for (posA, posB) in gt_positional:
        if posB < len(I):
            if posA in I[posB]:
                matches += 1

    total_gt = len(gt_positional)
    candidates = n_query * k

    recall = matches / total_gt if total_gt > 0 else 0.0
    precision = matches / candidates if candidates > 0 else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0

    return recall, precision, f1, matches, candidates


def run_faiss_flat(embA, embB, gt_positional, k):
    """Run FAISS IndexFlatIP (exact cosine search on normalized vectors)."""
    d = embA.shape[1]

    # Normalize (embeddings should already be normalized, but ensure it)
    faiss.normalize_L2(embA)
    faiss.normalize_L2(embB)

    # Build index on collection A
    index = faiss.IndexFlatIP(d)

    t0 = time.time()
    index.add(embA)
    index_time = time.time() - t0

    # Search with collection B as queries
    t0 = time.time()
    D, I = index.search(embB, k)
    search_time = time.time() - t0

    recall, precision, f1, matches, candidates = compute_metrics(
        I, gt_positional, len(embB), k)

    return {
        'method': 'FAISS_Flat',
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


def run_faiss_hnsw(embA, embB, gt_positional, k, M=32, ef_construction=200):
    """Run FAISS IndexHNSWFlat (approximate graph-based search)."""
    d = embA.shape[1]

    faiss.normalize_L2(embA)
    faiss.normalize_L2(embB)

    # Build HNSW index
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.metric_type = faiss.METRIC_INNER_PRODUCT

    t0 = time.time()
    index.add(embA)
    index_time = time.time() - t0

    # Set search parameters — higher efSearch = better recall but slower
    index.hnsw.efSearch = max(k * 4, 128)

    t0 = time.time()
    D, I = index.search(embB, k)
    search_time = time.time() - t0

    recall, precision, f1, matches, candidates = compute_metrics(
        I, gt_positional, len(embB), k)

    return {
        'method': f'FAISS_HNSW(M={M})',
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
    parser = argparse.ArgumentParser(description="Run FAISS on medical datasets")
    parser.add_argument('--data_dir', default=DATA_DIR,
                        help=f"Directory with .npy embeddings (default: {DATA_DIR})")
    parser.add_argument('--datasets', nargs='+', default=None,
                        help=f"Datasets to run (default: all)")
    parser.add_argument('--k_values', nargs='+', type=int, default=DEFAULT_K_VALUES,
                        help=f"K values to test (default: {DEFAULT_K_VALUES})")
    parser.add_argument('--skip_hnsw', action='store_true',
                        help="Skip HNSW (only run Flat/exact)")
    parser.add_argument('--output', default=None,
                        help="Output CSV path (default: results/results_faiss_medical.csv)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = args.output or os.path.join(RESULTS_DIR, "results_faiss_medical.csv")

    datasets = args.datasets or ALL_DATASETS
    all_results = []

    print("=" * 70)
    print("FAISS Medical Benchmark")
    print(f"  Datasets: {datasets}")
    print(f"  K values: {args.k_values}")
    print(f"  Methods: Flat (exact)" + ("" if args.skip_hnsw else " + HNSW (approximate)"))
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
            # --- FAISS Flat ---
            print(f"\n  [Flat] k={k} ...", end=" ", flush=True)
            try:
                # Make copies since normalize_L2 modifies in-place
                res = run_faiss_flat(embA.copy(), embB.copy(), gt_positional, k)
                res['dataset'] = dataset
                all_results.append(res)
                print(f"PC={res['recall_PC']:.4f}  PQ={res['precision_PQ']:.6f}  "
                      f"F1={res['f1']:.4f}  time={res['total_time_s']:.2f}s")
            except Exception as e:
                print(f"ERROR: {e}")

            # --- FAISS HNSW ---
            if not args.skip_hnsw:
                print(f"  [HNSW] k={k} ...", end=" ", flush=True)
                try:
                    res = run_faiss_hnsw(embA.copy(), embB.copy(), gt_positional, k)
                    res['dataset'] = dataset
                    all_results.append(res)
                    print(f"PC={res['recall_PC']:.4f}  PQ={res['precision_PQ']:.6f}  "
                          f"F1={res['f1']:.4f}  time={res['total_time_s']:.2f}s")
                except Exception as e:
                    print(f"ERROR: {e}")

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        # Reorder columns
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
