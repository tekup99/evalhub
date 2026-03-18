#!/bin/bash

# Log klasörünün var olduğundan emin ol (SLURM patlamaması için önemli)
mkdir -p logs

# Config'i yeni konumundan oku
CONFIG_FILE="scripts/configs/judge_config.env"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi
source "$CONFIG_FILE"

for JUDGE_MODEL_ID in $JUDGE_MODELS; do
    JUDGE_SAFE_NAME=$(basename "$JUDGE_MODEL_ID")
    
    for BASE_MODEL_ID in $BASE_MODELS; do
        BASE_SAFE_NAME=$(basename "$BASE_MODEL_ID")
        
        for TASK in $BENCHMARKS; do
            INPUT_FILE="$PWD/data/passatk_filtered/$BASE_SAFE_NAME/${TASK}_corrects.jsonl"
            
            # Eğer o base modelin o task için jsonl dosyası yoksa atla
            if [ ! -f "$INPUT_FILE" ]; then
                echo "Skipping (No file): Judge: $JUDGE_SAFE_NAME | Base: $BASE_SAFE_NAME | Task: $TASK"
                continue
            fi
            
            # SLURM İş İsmini Belirle: Ornek "J_Qwen3.5-35B-A3B_B_Qwen3.5-4B_aime2026"
            JOB_NAME="J_${JUDGE_SAFE_NAME}_B_${BASE_SAFE_NAME}_${TASK}"
            
            echo "Submitting SLURM Job: $JOB_NAME"
            
            # Sbatch ile worker scriptini kuyruğa yolla ve argümanları aktar (yol güncellendi)
            sbatch --job-name="$JOB_NAME" scripts/evaluation/llm_judge/run_judge.sh "$JUDGE_MODEL_ID" "$BASE_MODEL_ID" "$TASK" "$INPUT_FILE"
            
            # SLURM isteklerini çok hızlı atıp sistemi yormamak için ufak bir bekleme
            sleep 1
        done
    done
done

echo "All valid jobs have been submitted to SLURM queue!"