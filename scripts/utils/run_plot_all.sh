#!/bin/bash
#SBATCH --job-name=plot_all
#SBATCH --output=logs/plot_all_%j.out
#SBATCH --error=logs/plot_all_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate evalhub_env

python scripts/utils/plot_all_metrics.py