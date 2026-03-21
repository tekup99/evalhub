#!/bin/bash
set -euo pipefail

CONFIG_FILE="scripts/configs/cot_judge.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Configuration file not found: $CONFIG_FILE"
    exit 1
fi
source "$CONFIG_FILE"

IFS=' ' read -r -a SUFFIX_LIST <<< "${SUFFIXES}"
IFS=' ' read -r -a TEMP_LIST <<< "${TEMPERATURES}"

mkdir -p logs

echo "[INFO] Initializing LLM-as-a-Judge batch submissions..."

for base_model in ${BASE_MODELS}; do
    for judge_model in ${JUDGE_MODELS}; do
        for benchmark in ${BENCHMARKS}; do
            for suffix in "${SUFFIX_LIST[@]}"; do
                for temp in "${TEMP_LIST[@]}"; do
                    BASE_SAFE=$(basename "$base_model")
                    INPUT_FILE="data/passatk_filtered/${BASE_SAFE}/${benchmark}${suffix}_corrects.jsonl"
                    
                    echo "[INFO] Submitting -> Benchmark: ${benchmark}${suffix} | Temp: ${temp} | Base: ${BASE_SAFE}"
                    
                    sbatch --wait \
                        --job-name="judge_$(basename "$judge_model")" \
                        --output="logs/judge_%j.out" \
                        --error="logs/judge_%j.err" \
                        scripts/cot_judge_pipeline/03_run_judge_worker.sh \
                        "$judge_model" "$base_model" "$benchmark" "$INPUT_FILE" "$temp"
                done
            done
        done
    done
done

echo "[INFO] All judge jobs submitted successfully."