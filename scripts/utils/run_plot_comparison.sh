#!/bin/bash
#SBATCH --job-name=plot_comp
#SBATCH --output=logs/plot_comp_%j.out
#SBATCH --error=logs/plot_comp_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate evalhub_env

# ==============================================================================
# 1. AUTO-DISCOVER ALL AVAILABLE PATHS
# ==============================================================================
LIST_FILE="available_paths.txt"
echo "[INFO] Scanning for available summary.json files..."

if [ -d "results" ]; then
    find results -type f -name "summary.json" | sort > "$LIST_FILE"
    echo "[INFO] All available paths have been saved to: $LIST_FILE"
else
    echo "[WARNING] 'results' directory not found. Skipping path discovery."
fi

# ==============================================================================
# 2. TARGET PATHS FOR COMPARISON
# ==============================================================================
PATHS=(
    # Paste your desired paths from available_paths.txt here, enclosed in quotes.
    # "results/judgments/Qwen3.5-4B-Base_evaluated_by_Qwen3.5-35B-A3B/aime2026_t0.9/summary.json"
    # "results/evals/Qwen3.5-4B-Base/gsm8k/summary.json"
)

# ==============================================================================
# 3. EXECUTION
# ==============================================================================
if [ ${#PATHS[@]} -eq 0 ]; then
    echo "[WARNING] No paths defined in the PATHS array."
    echo "[INFO] Please check '$LIST_FILE', copy the paths you want to compare, add them to the PATHS array, and run again."
    exit 0
fi

# Dynamically generate filename to prevent overwriting previous plots
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_FILE="comparison_plots/comparison_${TIMESTAMP}.png"

echo "[INFO] Generating comparison plot for ${#PATHS[@]} configurations..."

python scripts/utils/plot_comparison.py "${PATHS[@]}" --out "$OUT_FILE"

echo "[INFO] Plotting execution completed."