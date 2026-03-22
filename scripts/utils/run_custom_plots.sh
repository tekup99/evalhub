#!/bin/bash
#SBATCH --job-name=custom_plots
#SBATCH --output=logs/custom_plots_%j.out
#SBATCH -e logs/custom_plots_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate evalhub_env

# Ortak Stiller ve Renkler
C1="tab:blue"
C2="tab:orange"
C3="tab:green"
C4="tab:red"
C5="tab:purple"

# L1=Düz (Solid), L2=Kesikli (Dashed)
L1="solid"
L2="dashed"

# Markerlar
M1="o"
M2="s"

OUT_DIR="comparison_plots"
mkdir -p "$OUT_DIR"

echo "[INFO] Plot 1 uretiliyor..."
python scripts/utils/plot_custom_group.py \
    --paths \
        "results/Qwen3.5-4B/aime2025/aime2025_summary.json" \
        "results/Qwen3.5-4B-Base/aime2025/aime2025_summary.json" \
        "results/judgments/Qwen3.5-4B-Base_evaluated_by_Qwen3.5-35B-A3B/aime2025/summary.json" \
        "results/judgments/Qwen3.5-4B_evaluated_by_Qwen3.5-35B-A3B/aime2025/summary.json" \
    --labels \
        "(a) 4B (Raw)" \
        "(b) 4B-Base (Raw)" \
        "(c) 4B-Base Judged" \
        "(d) 4B Judged" \
    --colors     "$C1" "$C2" "$C1" "$C2" \
    --linestyles "$L1" "$L1" "$L2" "$L2" \
    --markers    "$M1" "$M1" "$M2" "$M2" \
    --title "AIME 2025: Qwen3.5-4B vs Base (Raw vs Judged)" \
    --out "${OUT_DIR}/plot1_aime2025.png"


echo "[INFO] Plot 2 uretiliyor..."
python scripts/utils/plot_custom_group.py \
    --paths \
        "results/Qwen3.5-4B/aime2026/aime2026_summary.json" \
        "results/Qwen3.5-4B-Base/aime2026/aime2026_summary.json" \
        "results/judgments/Qwen3.5-4B-Base_evaluated_by_Qwen3.5-35B-A3B/aime2026/summary.json" \
        "results/judgments/Qwen3.5-4B_evaluated_by_Qwen3.5-35B-A3B/aime2026/summary.json" \
    --labels \
        "(a) 4B (Raw)" \
        "(b) 4B-Base (Raw)" \
        "(c) 4B-Base Judged" \
        "(d) 4B Judged" \
    --colors     "$C1" "$C2" "$C1" "$C2" \
    --linestyles "$L1" "$L1" "$L2" "$L2" \
    --markers    "$M1" "$M1" "$M2" "$M2" \
    --title "AIME 2026: Qwen3.5-4B vs Base (Raw vs Judged)" \
    --out "${OUT_DIR}/plot2_aime2026.png"


echo "[INFO] Plot 3 uretiliyor..."
python scripts/utils/plot_custom_group.py \
    --paths \
        "results/judgments/Qwen3.5-4B-Base_evaluated_by_Qwen3.5-35B-A3B/aime2026/summary.json" \
        "results/judgments/Qwen3.5-4B-Base_evaluated_by_Qwen3.5-35B-A3B/aime2025/summary.json" \
        "results/judgments/Qwen3.5-4B-Base_evaluated_by_Qwen3.5-35B-A3B_16384/aime2025/summary.json" \
        "results/judgments/Qwen3.5-4B-Base_evaluated_by_Qwen3.5-35B-A3B_16384/aime2026/summary.json" \
    --labels \
        "(a) 8192 Tokens / AIME 2026" \
        "(b) 8192 Tokens / AIME 2025" \
        "(c) 16384 Tokens / AIME 2025" \
        "(d) 16384 Tokens / AIME 2026" \
    --colors     "$C1" "$C2" "$C1" "$C2" \
    --linestyles "$L1" "$L1" "$L2" "$L2" \
    --markers    "$M1" "$M1" "$M2" "$M2" \
    --title "Max Tokens Effect: 8192 vs 16384" \
    --out "${OUT_DIR}/plot3_max_tokens_comparison.png"


echo "[INFO] Plot 4 uretiliyor..."
python scripts/utils/plot_custom_group.py \
    --paths \
        "results/judgments/Qwen3.5-4B-Base_evaluated_by_Qwen3.5-35B-A3B/aime2026_t0.0_k64/summary.json" \
        "results/judgments/Qwen3.5-4B-Base_evaluated_by_Qwen3.5-35B-A3B/aime2026_t0.3_k64/summary.json" \
        "results/judgments/Qwen3.5-4B-Base_evaluated_by_Qwen3.5-35B-A3B/aime2026/summary.json" \
        "results/judgments/Qwen3.5-4B-Base_evaluated_by_Qwen3.5-35B-A3B/aime2026_t1.0_k64/summary.json" \
        "results/judgments/Qwen3.5-4B-Base_evaluated_by_Qwen3.5-35B-A3B/aime2026_t1.2_k64/summary.json" \
    --labels \
        "(a) Temp = 0.0" \
        "(b) Temp = 0.3" \
        "(c) Temp = 0.6" \
        "(d) Temp = 1.0" \
        "(e) Temp = 1.2" \
    --colors     "$C1" "$C2" "$C3" "$C4" "$C5" \
    --linestyles "$L1" "$L1" "$L1" "$L1" "$L1" \
    --markers    "o"   "X"   "D"   "^"   "v" \
    --title "Temperature Sweep (Base, AIME 2026)" \
    --out "${OUT_DIR}/plot4_temperature_sweep.png"

echo "[INFO] Tum grafikler basariyla uretildi ve '$OUT_DIR' klasorune eklendi."