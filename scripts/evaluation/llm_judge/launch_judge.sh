#!/bin/bash
set -euo pipefail

# Config yükle
CONFIG_FILE="scripts/configs/judge_config.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: $CONFIG_FILE not found."
    exit 1
fi
source "$CONFIG_FILE"

# Listeleri diziye çevir
IFS=' ' read -r -a SUFFIX_LIST <<< "${SUFFIXES}"
IFS=' ' read -r -a TEMP_LIST <<< "${TEMPERATURES}"

mkdir -p logs

for base_model in ${BASE_MODELS}; do
    for judge_model in ${JUDGE_MODELS}; do
        for benchmark in ${BENCHMARKS}; do
            for suffix in "${SUFFIX_LIST[@]}"; do
                for temp in "${TEMP_LIST[@]}"; do
                    # Veri yolu için model ismini temizle (Qwen/ kısmını at)
                    BASE_SAFE=$(basename "$base_model")
                    INPUT_FILE="data/passatk_filtered/${BASE_SAFE}/${benchmark}${suffix}_corrects.jsonl"
                    
                    echo "--- Submitting: ${benchmark}${suffix} | Temp: ${temp} ---"
                    
                    # sbatch komutu
                    sbatch --wait \
                        --job-name="judge_$(basename "$judge_model")" \
                        --output="logs/judge_%j.out" \
                        --error="logs/judge_%j.err" \
                        scripts/evaluation/llm_judge/run_judge.sh \
                        "$judge_model" "$base_model" "$benchmark" "$INPUT_FILE" "$temp"
                done
            done
        done
    done
done