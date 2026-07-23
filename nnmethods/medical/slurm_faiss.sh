#!/bin/bash
#SBATCH --job-name=nn_faiss
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=nnmethods/medical/results/faiss_%j.log
#SBATCH --error=nnmethods/medical/results/faiss_%j.err

# =============================================================================
# Run FAISS (Flat + HNSW) on all medical datasets
# Submit: sbatch nnmethods/medical/slurm_faiss.sh
# Depends on: slurm_prepare.sh must have completed first
# =============================================================================

echo "=== FAISS started at $(date) ==="
echo "Node: $(hostname)"

module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || \
module load python/3.11 2>/dev/null || \
module load Python 2>/dev/null

source /bigwork/nhkbkpkm/nnenv/bin/activate
cd /bigwork/nhkbkpkm/ContinuousMedicalFilteringBenchmark

python nnmethods/medical/run_faiss_medical.py \
    --k_values 1 5 10 50 100 200 1000

echo ""
echo "=== FAISS complete at $(date) ==="
