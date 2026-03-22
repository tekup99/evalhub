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

echo "[INFO] Initializing LLM-as-a-Judge batch submissions (Sequential mode with dependencies)..."

# Önceki işin ID'sini tutacağımız değişken
PREV_JOB_ID=""

for base_model in ${BASE_MODELS}; do
    for judge_model in ${JUDGE_MODELS}; do
        for benchmark in ${BENCHMARKS}; do
            for suffix in "${SUFFIX_LIST[@]}"; do
                # Suffix'in başındaki alt çizgiyi kaldırarak k değerini alıyoruz (örn: "_64" -> "64")
                K_VAL="${suffix#_}"

                for temp in "${TEMP_LIST[@]}"; do
                    BASE_SAFE=$(basename "$base_model")
                    INPUT_FILE="data/passatk_filtered/${BASE_SAFE}/${benchmark}${suffix}_corrects.jsonl"
                    
                    echo "[INFO] Submitting -> Benchmark: ${benchmark}${suffix} | Temp: ${temp} | Base: ${BASE_SAFE} | K: ${K_VAL}"
                    
                    # Eğer daha önce gönderilmiş bir iş varsa, ona bağımlılık (dependency) ekle
                    DEPENDENCY_ARG=""
                    if [[ -n "$PREV_JOB_ID" ]]; then
                        DEPENDENCY_ARG="--dependency=afterany:${PREV_JOB_ID}"
                    fi
                    
                    # sbatch çıktısını sadece Job ID verecek şekilde (--parsable) alıyoruz
                    # ve 6. argüman olarak K_VAL değerini worker scripte gönderiyoruz
                    PREV_JOB_ID=$(sbatch $DEPENDENCY_ARG \
                        --parsable \
                        --job-name="judge_$(basename "$judge_model")" \
                        --output="logs/judge_%j.out" \
                        --error="logs/judge_%j.err" \
                        scripts/cot_judge_pipeline/03_run_judge_worker.sh \
                        "$judge_model" "$base_model" "$benchmark" "$INPUT_FILE" "$temp" "$K_VAL")
                    
                    echo "[INFO] Submitted Job ID: $PREV_JOB_ID (Waiting for: ${DEPENDENCY_ARG:-None})"
                done
            done
        done
    done
done

echo "[INFO] All sequential judge jobs submitted successfully."