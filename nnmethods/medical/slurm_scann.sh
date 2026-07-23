#!/bin/bash
#SBATCH --job-name=nn_scann
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=nnmethods/medical/results/scann_%j.log
#SBATCH --error=nnmethods/medical/results/scann_%j.err

# =============================================================================
# Run ScaNN (AH + BF) on all medical datasets
# Submit: sbatch nnmethods/medical/slurm_scann.sh
# Depends on: slurm_prepare.sh must have completed first
# =============================================================================

echo "=== ScaNN started at $(date) ==="
echo "Node: $(hostname)"

module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || \
module load python/3.11 2>/dev/null || \
module load Python 2>/dev/null

source /bigwork/nhkbkpkm/nnenv/bin/activate
cd /bigwork/nhkbkpkm/ContinuousMedicalFilteringBenchmark

python nnmethods/medical/run_scann_medical.py \
    --k_values 1 5 10 50 100 200 1000

echo ""
echo "=== ScaNN complete at $(date) ==="
