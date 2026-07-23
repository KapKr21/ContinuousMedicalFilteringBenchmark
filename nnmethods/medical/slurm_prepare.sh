#!/bin/bash
#SBATCH --job-name=nn_prepare
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=nnmethods/medical/results/prepare_%j.log
#SBATCH --error=nnmethods/medical/results/prepare_%j.err

# =============================================================================
# Generate embeddings for all medical datasets using all-MiniLM-L6-v2
# Submit: sbatch nnmethods/medical/slurm_prepare.sh
# =============================================================================

echo "=== Prepare started at $(date) ==="
echo "Node: $(hostname)"

module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || \
module load python/3.11 2>/dev/null || \
module load Python 2>/dev/null

source /bigwork/nhkbkpkm/nnenv/bin/activate
cd /bigwork/nhkbkpkm/ContinuousMedicalFilteringBenchmark

mkdir -p nnmethods/medical/results

echo "=== Generating embeddings for all datasets ==="
python nnmethods/medical/prepare_medical_nn_data.py \
    --datasets febrl1 febrl2 febrl3 febrl4 rxnorm synthea medmentions cms umls

echo ""
echo "=== Prepare complete at $(date) ==="
echo "Files generated:"
ls -lh nnmethods/medical/data/embeddings/
ls -lh nnmethods/medical/data/deepblocker/
