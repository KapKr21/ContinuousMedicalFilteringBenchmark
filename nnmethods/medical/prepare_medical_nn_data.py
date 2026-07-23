"""
prepare_medical_nn_data.py
===========================
Converts raw medical CSV datasets into the format expected by FAISS, ScaNN,
and DeepBlocker scripts.

For FAISS/ScaNN: generates 384-dim embeddings using sentence-transformers
                 (all-MiniLM-L6-v2) and saves as pipe-delimited CSVs.
For DeepBlocker: creates text-only CSVs with aggregate values (no embeddings
                 needed — DeepBlocker generates its own via AutoEncoder/CTT).

Usage on Pascal:
  source ~/nnenv/bin/activate
  cd ~/ContinuousMedicalFilteringBenchmark
  nice -n 19 python nnmethods/medical/prepare_medical_nn_data.py

  # Or specific datasets only:
  nice -n 19 python nnmethods/medical/prepare_medical_nn_data.py --datasets febrl1 febrl4 rxnorm

Output directory: nnmethods/medical/data/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RAW_DIR = os.path.join(REPO_ROOT, "blockingWorkflows", "data", "medical", "rawdata")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASETS = {
    'febrl1': {
        'type': 'dirty',
        'csv': 'febrl1.csv',
        'gt': 'febrl1_groundtruth.csv',
        'id_col': 'rec_id',
    },
    'febrl2': {
        'type': 'dirty',
        'csv': 'febrl2.csv',
        'gt': 'febrl2_groundtruth.csv',
        'id_col': 'rec_id',
    },
    'febrl3': {
        'type': 'dirty',
        'csv': 'febrl3.csv',
        'gt': 'febrl3_groundtruth.csv',
        'id_col': 'rec_id',
    },
    'febrl4': {
        'type': 'clean',
        'csvA': 'febrlA.csv',
        'csvB': 'febrlB.csv',
        'gt': 'febrl4_groundtruth.csv',
        'id_col': 'rec_id',
    },
    'synthea': {
        'type': 'dirty_synthea',
        'csvA': 'syntheaA.csv',
        'csvB': 'syntheaB_with_dups.csv',
        'gt': 'synthea_groundtruth.csv',
        'id_col': 'Id',
    },
    'medmentions': {
        'type': 'clean',
        'csvA': 'medmentionsA.csv',
        'csvB': 'medmentionsB.csv',
        'gt': 'medmentions_groundtruth.csv',
        'id_col': None,  # first column
    },
    'cms': {
        'type': 'clean',
        'csvA': 'cmsA.csv',
        'csvB': 'cmsB.csv',
        'gt': 'cms_groundtruth.csv',
        'id_col': None,
    },
    'umls': {
        'type': 'clean',
        'csvA': 'umlsA.csv',
        'csvB': 'umlsB.csv',
        'gt': 'umls_groundtruth.csv',
        'id_col': None,
    },
    'rxnorm': {
        'type': 'clean',
        'csvA': 'rxnormA.csv',
        'csvB': 'rxnormB.csv',
        'gt': 'rxnorm_groundtruth.csv',
        'id_col': None,
    },
}


def aggregate_row(row, id_col):
    """Concatenate all non-ID, non-NaN columns into a single text string."""
    vals = []
    for col in row.index:
        if col == id_col:
            continue
        val = row[col]
        if pd.notna(val) and str(val).strip():
            vals.append(str(val).strip())
    return ' '.join(vals) if vals else ''


def load_dataset(name, config, raw_dir):
    """Load raw CSVs and return (dfA, dfB, gt_df, id_col)."""
    id_col = config.get('id_col')

    if config['type'] == 'dirty':
        csv_path = os.path.join(raw_dir, config['csv'])
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing: {csv_path}")
        df = pd.read_csv(csv_path)
        if id_col is None:
            id_col = df.columns[0]
        # For Dirty ER: same collection is both A and B
        dfA = df.copy()
        dfB = df.copy()

    elif config['type'] == 'dirty_synthea':
        pathA = os.path.join(raw_dir, config['csvA'])
        pathB = os.path.join(raw_dir, config['csvB'])
        if not os.path.exists(pathA):
            raise FileNotFoundError(f"Missing: {pathA}")
        if not os.path.exists(pathB):
            raise FileNotFoundError(f"Missing: {pathB}")
        dfA = pd.read_csv(pathA)
        dfB = pd.read_csv(pathB)
        if id_col is None:
            id_col = dfA.columns[0]

    else:  # clean
        pathA = os.path.join(raw_dir, config['csvA'])
        pathB = os.path.join(raw_dir, config['csvB'])
        if not os.path.exists(pathA):
            raise FileNotFoundError(f"Missing: {pathA}")
        if not os.path.exists(pathB):
            raise FileNotFoundError(f"Missing: {pathB}")
        dfA = pd.read_csv(pathA)
        dfB = pd.read_csv(pathB)
        if id_col is None:
            id_col = dfA.columns[0]

    gt_path = os.path.join(raw_dir, config['gt'])
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Missing: {gt_path}")
    gt_df = pd.read_csv(gt_path)

    return dfA, dfB, gt_df, id_col


def prepare_dataset(name, config, raw_dir, output_dir, model):
    """Prepare a single dataset: generate embeddings + save all formats."""
    print(f"\n{'='*60}")
    print(f"  DATASET: {name}")
    print(f"{'='*60}")

    # Load
    dfA, dfB, gt_df, id_col = load_dataset(name, config, raw_dir)
    print(f"  Collection A: {len(dfA)} records")
    print(f"  Collection B: {len(dfB)} records")
    print(f"  Ground truth: {len(gt_df)} pairs")
    print(f"  ID column: {id_col}")

    # Create aggregate text
    print("  Creating aggregate text values...")
    aggA = dfA.apply(lambda row: aggregate_row(row, id_col), axis=1).tolist()
    aggB = dfB.apply(lambda row: aggregate_row(row, id_col), axis=1).tolist()

    # Filter out empty strings (replace with placeholder)
    aggA = [t if t else "empty" for t in aggA]
    aggB = [t if t else "empty" for t in aggB]

    # Generate embeddings
    print(f"  Encoding A ({len(aggA)} records)...")
    embA = model.encode(aggA, show_progress_bar=True, batch_size=256,
                        normalize_embeddings=True)
    print(f"  Encoding B ({len(aggB)} records)...")
    embB = model.encode(aggB, show_progress_bar=True, batch_size=256,
                        normalize_embeddings=True)

    # --- Save for FAISS/ScaNN ---
    emb_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    # Save embeddings as numpy arrays (much faster than CSV parsing)
    np.save(os.path.join(emb_dir, f"{name}_A.npy"), embA.astype(np.float32))
    np.save(os.path.join(emb_dir, f"{name}_B.npy"), embB.astype(np.float32))

    # Save ground truth in standard format
    gt_out = gt_df.copy()
    gt_out.columns = ['id1', 'id2']
    gt_out.to_csv(os.path.join(emb_dir, f"{name}_groundtruth.csv"), index=False)

    # Save ID mappings (original ID -> positional index)
    idsA = dfA[id_col].tolist()
    idsB = dfB[id_col].tolist()
    pd.DataFrame({'pos': range(len(idsA)), 'original_id': idsA}).to_csv(
        os.path.join(emb_dir, f"{name}_idsA.csv"), index=False)
    pd.DataFrame({'pos': range(len(idsB)), 'original_id': idsB}).to_csv(
        os.path.join(emb_dir, f"{name}_idsB.csv"), index=False)

    # --- Save for DeepBlocker ---
    db_dir = os.path.join(output_dir, "deepblocker")
    os.makedirs(db_dir, exist_ok=True)

    dfA_db = pd.DataFrame({
        'id': range(len(dfA)),
        'original_id': idsA,
        'Aggregate Value': aggA
    })
    dfB_db = pd.DataFrame({
        'id': range(len(dfB)),
        'original_id': idsB,
        'Aggregate Value': aggB
    })

    dfA_db.to_csv(os.path.join(db_dir, f"{name}A.csv"), sep='|', index=False)
    dfB_db.to_csv(os.path.join(db_dir, f"{name}B.csv"), sep='|', index=False)

    # Ground truth for DeepBlocker: needs positional IDs (ltable_id, rtable_id)
    # Map original IDs to positional indices
    id2posA = {orig_id: pos for pos, orig_id in enumerate(idsA)}
    id2posB = {orig_id: pos for pos, orig_id in enumerate(idsB)}

    gt_pairs = []
    skipped = 0
    for _, row in gt_df.iterrows():
        a_id, b_id = row.iloc[0], row.iloc[1]
        posA = id2posA.get(a_id)
        posB = id2posB.get(b_id)
        if posA is not None and posB is not None:
            gt_pairs.append({'ltable_id': posA, 'rtable_id': posB})
        else:
            skipped += 1

    if skipped > 0:
        print(f"  WARNING: {skipped} GT pairs skipped (ID not found in collections)")

    gt_db = pd.DataFrame(gt_pairs)
    gt_db.to_csv(os.path.join(db_dir, f"{name}_groundtruth.csv"), sep='|', index=False)

    print(f"  Saved {len(gt_pairs)} GT pairs (positional)")
    print(f"  Embeddings: {emb_dir}/{name}_A.npy ({embA.shape})")
    print(f"  DeepBlocker: {db_dir}/{name}A.csv")
    print(f"  DONE: {name}")

    return len(dfA), len(dfB), len(gt_pairs)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare medical datasets for NN methods (FAISS, ScaNN, DeepBlocker)")
    parser.add_argument('--raw_dir', default=RAW_DIR,
                        help=f"Directory with raw CSV files (default: {RAW_DIR})")
    parser.add_argument('--output_dir', default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument('--datasets', nargs='+', default=None,
                        help="Specific datasets to process (default: all)")
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                        help="Sentence-transformer model name (default: all-MiniLM-L6-v2)")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Encoding batch size (default: 256)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load embedding model
    print(f"Loading sentence-transformers model: {args.model}")
    model = SentenceTransformer(args.model)
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Process datasets
    datasets_to_run = args.datasets or list(DATASETS.keys())
    summary = []

    for name in datasets_to_run:
        if name not in DATASETS:
            print(f"  WARNING: Unknown dataset '{name}', skipping.")
            continue
        try:
            nA, nB, nGT = prepare_dataset(
                name, DATASETS[name], args.raw_dir, args.output_dir, model)
            summary.append((name, nA, nB, nGT, "OK"))
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            summary.append((name, 0, 0, 0, f"MISSING: {e}"))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            summary.append((name, 0, 0, 0, f"ERROR: {e}"))

    # Print summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} {'|A|':<10} {'|B|':<10} {'|GT|':<10} {'Status'}")
    print("-" * 60)
    for name, nA, nB, nGT, status in summary:
        print(f"{name:<15} {nA:<10} {nB:<10} {nGT:<10} {status}")


if __name__ == '__main__':
    main()
