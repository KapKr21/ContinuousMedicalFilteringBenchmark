#!/bin/bash
#SBATCH --job-name=nn_deepblock
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=nnmethods/medical/results/deepblocker_%j.log
#SBATCH --error=nnmethods/medical/results/deepblocker_%j.err

# =============================================================================
# Run DeepBlocker (AutoEncoder, CTT, Hybrid) on medical datasets
# Submit: sbatch nnmethods/medical/slurm_deepblocker.sh
# Depends on: slurm_prepare.sh must have completed first
# NOTE: Excludes UMLS (too large), CMS may be slow
# =============================================================================

echo "=== DeepBlocker started at $(date) ==="
echo "Node: $(hostname)"

module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || \
module load python/3.11 2>/dev/null || \
module load Python 2>/dev/null

source /bigwork/nhkbkpkm/nnenv/bin/activate
cd /bigwork/nhkbkpkm/ContinuousMedicalFilteringBenchmark

python nnmethods/medical/run_deepblocker_medical.py \
    --k_values 1 5 10 50 \
    --datasets febrl1 febrl2 febrl3 febrl4 rxnorm synthea medmentions cms

echo ""
echo "=== DeepBlocker complete at $(date) ==="
