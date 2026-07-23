#!/bin/bash
# =============================================================================
# run_all.sh — Master script to run NN methods on Pascal server
# =============================================================================
#
# Usage:
#   cd ~/ContinuousMedicalFilteringBenchmark
#   chmod +x nnmethods/medical/run_all.sh
#   ./nnmethods/medical/run_all.sh setup      # First time: create env + install deps
#   ./nnmethods/medical/run_all.sh prepare    # Generate embeddings (run once)
#   ./nnmethods/medical/run_all.sh faiss      # Run FAISS experiments
#   ./nnmethods/medical/run_all.sh scann      # Run ScaNN experiments
#   ./nnmethods/medical/run_all.sh deepblocker # Run DeepBlocker experiments
#   ./nnmethods/medical/run_all.sh all        # Run all three (sequentially)
#
# All long-running commands use:
#   nice -n 19   → low priority (avoids Perun killing the process)
#   nohup ... &  → survives SSH disconnection
#
# Monitor progress:
#   tail -f nnmethods/medical/results/results_faiss.log
#   tail -f nnmethods/medical/results/results_scann.log
#   tail -f nnmethods/medical/results/results_deepblocker.log
#
# =============================================================================

set -e

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
VENV_DIR="$HOME/nnenv"

# Create results directory
mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
activate_env() {
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
        echo "[OK] Virtual environment activated: $VENV_DIR"
    else
        echo "[ERROR] Virtual environment not found at $VENV_DIR"
        echo "        Run: $0 setup"
        exit 1
    fi
}

print_header() {
    echo ""
    echo "======================================================================"
    echo "  $1"
    echo "  $(date)"
    echo "======================================================================"
    echo ""
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
cmd_setup() {
    print_header "SETUP: Creating virtual environment and installing dependencies"

    # Create venv
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment at $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    else
        echo "Virtual environment already exists at $VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    pip install --upgrade pip

    # Install requirements
    echo "Installing requirements..."
    pip install -r "$SCRIPT_DIR/requirements.txt"

    # Clone DeepBlocker if not present
    if [ ! -d "$HOME/DeepBlocker" ]; then
        echo "Cloning DeepBlocker..."
        git clone https://github.com/qcri/DeepBlocker.git "$HOME/DeepBlocker"
        cd "$HOME/DeepBlocker"
        pip install -e .
        cd "$REPO_ROOT"
    else
        echo "DeepBlocker already cloned at ~/DeepBlocker"
    fi

    echo ""
    echo "[DONE] Setup complete. Activate with: source ~/nnenv/bin/activate"
    echo ""
    echo "Next steps:"
    echo "  1. $0 prepare              # Generate embeddings (15-30 min)"
    echo "  2. $0 faiss                # Run FAISS (background)"
    echo "  3. $0 scann                # Run ScaNN (background)"
    echo "  4. $0 deepblocker          # Run DeepBlocker (background)"
}

cmd_prepare() {
    print_header "PREPARE: Generating embeddings for all medical datasets"
    activate_env
    cd "$REPO_ROOT"

    echo "This will encode all datasets with all-MiniLM-L6-v2."
    echo "Estimated time: 15-30 minutes (depending on dataset sizes)."
    echo ""

    # Run preparation — start with small datasets, then large
    nice -n 19 python "$SCRIPT_DIR/prepare_medical_nn_data.py" \
        --datasets febrl1 febrl2 febrl3 febrl4 rxnorm synthea medmentions cms umls

    echo ""
    echo "[DONE] Embeddings saved to: $SCRIPT_DIR/data/"
    echo "  embeddings/   — .npy files for FAISS/ScaNN"
    echo "  deepblocker/  — text CSVs for DeepBlocker"
}

cmd_prepare_small() {
    print_header "PREPARE (small): Generating embeddings for small datasets only"
    activate_env
    cd "$REPO_ROOT"

    nice -n 19 python "$SCRIPT_DIR/prepare_medical_nn_data.py" \
        --datasets febrl1 febrl2 febrl3 febrl4 rxnorm synthea

    echo ""
    echo "[DONE] Small dataset embeddings ready."
}

cmd_faiss() {
    print_header "FAISS: Running Flat + HNSW on medical datasets"
    activate_env
    cd "$REPO_ROOT"

    echo "Running in background with nice -n 19..."
    echo "Log: $RESULTS_DIR/results_faiss.log"
    echo ""

    nice -n 19 nohup python "$SCRIPT_DIR/run_faiss_medical.py" \
        --k_values 1 5 10 50 100 \
        > "$RESULTS_DIR/results_faiss.log" 2>&1 &

    echo "[STARTED] PID: $!"
    echo "Monitor with: tail -f $RESULTS_DIR/results_faiss.log"
}

cmd_faiss_foreground() {
    print_header "FAISS (foreground): Running Flat + HNSW on medical datasets"
    activate_env
    cd "$REPO_ROOT"

    nice -n 19 python "$SCRIPT_DIR/run_faiss_medical.py" \
        --k_values 1 5 10 50 100
}

cmd_scann() {
    print_header "ScaNN: Running AH + BF on medical datasets"
    activate_env
    cd "$REPO_ROOT"

    echo "Running in background with nice -n 19..."
    echo "Log: $RESULTS_DIR/results_scann.log"
    echo ""

    nice -n 19 nohup python "$SCRIPT_DIR/run_scann_medical.py" \
        --k_values 1 5 10 50 100 \
        > "$RESULTS_DIR/results_scann.log" 2>&1 &

    echo "[STARTED] PID: $!"
    echo "Monitor with: tail -f $RESULTS_DIR/results_scann.log"
}

cmd_scann_foreground() {
    print_header "ScaNN (foreground): Running AH + BF on medical datasets"
    activate_env
    cd "$REPO_ROOT"

    nice -n 19 python "$SCRIPT_DIR/run_scann_medical.py" \
        --k_values 1 5 10 50 100
}

cmd_deepblocker() {
    print_header "DeepBlocker: Running AutoEncoder + CTT + Hybrid"
    activate_env
    cd "$REPO_ROOT"

    echo "Running in background with nice -n 19..."
    echo "Log: $RESULTS_DIR/results_deepblocker.log"
    echo "NOTE: Skipping UMLS (too large). CMS may be slow."
    echo ""

    nice -n 19 nohup python "$SCRIPT_DIR/run_deepblocker_medical.py" \
        --k_values 1 5 10 50 \
        --datasets febrl1 febrl2 febrl3 febrl4 rxnorm synthea medmentions \
        > "$RESULTS_DIR/results_deepblocker.log" 2>&1 &

    echo "[STARTED] PID: $!"
    echo "Monitor with: tail -f $RESULTS_DIR/results_deepblocker.log"
}

cmd_deepblocker_foreground() {
    print_header "DeepBlocker (foreground): Running AutoEncoder + CTT + Hybrid"
    activate_env
    cd "$REPO_ROOT"

    nice -n 19 python "$SCRIPT_DIR/run_deepblocker_medical.py" \
        --k_values 1 5 10 50 \
        --datasets febrl1 febrl2 febrl3 febrl4 rxnorm synthea medmentions
}

cmd_all() {
    print_header "ALL: Running FAISS, ScaNN, and DeepBlocker sequentially"
    activate_env
    cd "$REPO_ROOT"

    echo "Running all three methods sequentially in background..."
    echo "Log: $RESULTS_DIR/results_all.log"
    echo ""

    nice -n 19 nohup bash -c "
        echo '=== FAISS ===' && \
        python '$SCRIPT_DIR/run_faiss_medical.py' --k_values 1 5 10 50 100 && \
        echo '=== ScaNN ===' && \
        python '$SCRIPT_DIR/run_scann_medical.py' --k_values 1 5 10 50 100 && \
        echo '=== DeepBlocker ===' && \
        python '$SCRIPT_DIR/run_deepblocker_medical.py' --k_values 1 5 10 50 \
            --datasets febrl1 febrl2 febrl3 febrl4 rxnorm synthea medmentions
    " > "$RESULTS_DIR/results_all.log" 2>&1 &

    echo "[STARTED] PID: $!"
    echo "Monitor with: tail -f $RESULTS_DIR/results_all.log"
}

cmd_status() {
    print_header "STATUS: Checking running NN experiments"

    echo "Running Python processes:"
    ps aux | grep -E "run_(faiss|scann|deepblocker|prepare)_medical" | grep -v grep || echo "  None running."
    echo ""

    echo "Result files:"
    ls -lh "$RESULTS_DIR"/*.csv 2>/dev/null || echo "  No CSV results yet."
    echo ""

    echo "Log files:"
    ls -lh "$RESULTS_DIR"/*.log 2>/dev/null || echo "  No logs yet."
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
case "${1:-help}" in
    setup)
        cmd_setup
        ;;
    prepare)
        cmd_prepare
        ;;
    prepare-small)
        cmd_prepare_small
        ;;
    faiss)
        cmd_faiss
        ;;
    faiss-fg)
        cmd_faiss_foreground
        ;;
    scann)
        cmd_scann
        ;;
    scann-fg)
        cmd_scann_foreground
        ;;
    deepblocker)
        cmd_deepblocker
        ;;
    deepblocker-fg)
        cmd_deepblocker_foreground
        ;;
    all)
        cmd_all
        ;;
    status)
        cmd_status
        ;;
    help|*)
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  setup           Create venv, install deps, clone DeepBlocker"
        echo "  prepare         Generate embeddings for ALL datasets (run once)"
        echo "  prepare-small   Generate embeddings for small datasets only"
        echo "  faiss           Run FAISS in background (nohup)"
        echo "  faiss-fg        Run FAISS in foreground"
        echo "  scann           Run ScaNN in background (nohup)"
        echo "  scann-fg        Run ScaNN in foreground"
        echo "  deepblocker     Run DeepBlocker in background (nohup)"
        echo "  deepblocker-fg  Run DeepBlocker in foreground"
        echo "  all             Run all three sequentially in background"
        echo "  status          Check running experiments and results"
        echo ""
        echo "Typical workflow:"
        echo "  $0 setup              # First time only"
        echo "  $0 prepare            # Generate embeddings (~15-30 min)"
        echo "  $0 faiss              # Start FAISS (background)"
        echo "  $0 scann              # Start ScaNN (background)"
        echo "  $0 deepblocker        # Start DeepBlocker (background)"
        echo "  $0 status             # Check progress"
        echo ""
        echo "Monitor logs:"
        echo "  tail -f nnmethods/medical/results/results_faiss.log"
        echo "  tail -f nnmethods/medical/results/results_scann.log"
        echo "  tail -f nnmethods/medical/results/results_deepblocker.log"
        ;;
esac
