#!/bin/bash
#SBATCH --job-name=nn_setup
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=nnmethods/medical/results/setup_%j.log
#SBATCH --error=nnmethods/medical/results/setup_%j.err

# =============================================================================
# One-time setup: Create venv and install all dependencies
# Submit: sbatch nnmethods/medical/slurm_setup.sh
# =============================================================================

echo "=== Setup started at $(date) ==="
echo "Node: $(hostname)"

# Load Python module (adjust version if needed)
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || \
module load python/3.11 2>/dev/null || \
module load Python 2>/dev/null || \
echo "WARNING: No python module loaded, using system python"

which python3
python3 --version

# Create venv in bigwork (more space)
VENV_DIR="/bigwork/nhkbkpkm/nnenv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip

echo "=== Installing core packages ==="
pip install numpy pandas psutil scikit-learn recordlinkage

echo "=== Installing PyTorch (CPU) ==="
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "=== Installing sentence-transformers ==="
pip install sentence-transformers

echo "=== Installing FAISS ==="
pip install faiss-cpu

echo "=== Installing ScaNN ==="
pip install scann

echo "=== Setting up DeepBlocker ==="
if [ ! -d "/bigwork/nhkbkpkm/DeepBlocker" ]; then
    git clone https://github.com/qcri/DeepBlocker.git /bigwork/nhkbkpkm/DeepBlocker
fi

# Add DeepBlocker to venv activation
grep -q "DeepBlocker" "$VENV_DIR/bin/activate" || \
    echo 'export PYTHONPATH=$PYTHONPATH:/bigwork/nhkbkpkm/DeepBlocker' >> "$VENV_DIR/bin/activate"

echo ""
echo "=== Setup complete at $(date) ==="
echo "Activate with: source /bigwork/nhkbkpkm/nnenv/bin/activate"
pip list
