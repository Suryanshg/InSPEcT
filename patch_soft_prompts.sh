#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16g
#SBATCH -J "TREC"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -o logs_trec.out
#SBATCH -e logs_trec.out

# -----------------------------
# Load Required Modules
# -----------------------------
module load python/3.12.3
module load cuda/12.9.0

# -----------------------------
# Create / Activate venv
# -----------------------------
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Activating existing virtual environment..."
    source "$VENV_DIR/bin/activate"
fi

# -----------------------------
# Run Patch Scopes Experiment
# -----------------------------
python -m scripts.create_patching_outputs -m meta-llama/Meta-Llama-3-8B-Instruct -d SetFit/TREC-QC -n 56 -c to_patch_trec_56tokens -t description_and_classes -i 1 -max 1000