#!/bin/bash
#SBATCH --job-name=plot_all
#SBATCH --output=logs/plot_all_%j.out
#SBATCH -e logs/plot_all_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate evalhub_env

echo "Tüm base result'lar için Pass@K grafikleri oluşturuluyor..."
python scripts/utils/plot_all_metrics.py
echo "İşlem tamamlandı."