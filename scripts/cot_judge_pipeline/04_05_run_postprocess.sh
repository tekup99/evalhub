#!/bin/bash
#SBATCH --job-name=judge_postprocess
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate evalhub_env

BASE_DIR="results"
JUDGMENT_DIR="results/judgments"

# Bunları .env dosyasından okumak daha sağlıklıdır, ancak standalone bırakıldı.
BASE_MODELS=("Qwen3.5-4B-Base" "Qwen3.5-4B")
JUDGE_MODELS=("Qwen3.5-35B-A3B")
BENCHMARKS=("aime2025" "aime2026")
INPUT_FILENAME="math_judge.jsonl"
OUTPUT_FILENAME="math_judge_majority.jsonl"

echo "[INFO] Starting judge post-processing pipeline..."

for BASE_MODEL in "${BASE_MODELS[@]}"; do
    for JUDGE_MODEL in "${JUDGE_MODELS[@]}"; do
        for BENCH in "${BENCHMARKS[@]}"; do
            
            TARGET_DIR="${JUDGMENT_DIR}/${BASE_MODEL}_evaluated_by_${JUDGE_MODEL}/${BENCH}"
            INPUT_FILE="${TARGET_DIR}/${INPUT_FILENAME}"
            OUTPUT_FILE="${TARGET_DIR}/${OUTPUT_FILENAME}"
            
            BASE_RESULTS_FILE="${BASE_DIR}/${BASE_MODEL}/${BENCH}/${BENCH}_results.jsonl"
            JUDGED_RESULTS_FILE="${TARGET_DIR}/${BENCH}_results_judged.jsonl"
            SUMMARY_FILE="${TARGET_DIR}/summary.json"

            if [[ -f "$INPUT_FILE" ]]; then
                echo "[INFO] Aggregating majority votes for: Base=${BASE_MODEL} | Judge=${JUDGE_MODEL} | Bench=${BENCH}"
                python scripts/cot_judge_pipeline/04_aggregate_votes.py \
                    --input_file "$INPUT_FILE" \
                    --output_file "$OUTPUT_FILE"
                
                if [[ -f "$BASE_RESULTS_FILE" ]] && [[ -f "$OUTPUT_FILE" ]]; then
                    echo "[INFO] Applying judge metrics to base results..."
                    PYTHONPATH=. python scripts/cot_judge_pipeline/05_apply_metrics.py \
                        --base_results_file "$BASE_RESULTS_FILE" \
                        --judge_majority_file "$OUTPUT_FILE" \
                        --output_file "$JUDGED_RESULTS_FILE" \
                        --summary_file "$SUMMARY_FILE"
                else
                    echo "[WARNING] Base results or generated majority file missing for $BENCH. Skipping metric application."
                fi
            else
                echo "[WARNING] Input file $INPUT_FILE not found. Skipping aggregation."
            fi
            
        done
    done
done

echo "[INFO] Post-processing pipeline completed successfully."