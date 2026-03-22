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

JUDGMENT_ROOT="results/judgments"
BASE_DIR="results"
OUTPUT_FILENAME="math_judge_majority.jsonl"

echo "[INFO] Scanning for judgment directories in $JUDGMENT_ROOT ..."

find "$JUDGMENT_ROOT" -type d | while read -r TARGET_DIR; do
    # Öncelik math_judge.jsonl dosyasında, eğer yoksa raw olana bakar
    if [[ -f "${TARGET_DIR}/math_judge.jsonl" ]]; then
        INPUT_FILE="${TARGET_DIR}/math_judge.jsonl"
    elif [[ -f "${TARGET_DIR}/math_judge_raw.jsonl" ]]; then
        INPUT_FILE="${TARGET_DIR}/math_judge_raw.jsonl"
    else
        continue
    fi
    
    echo "[INFO] Processing folder: $TARGET_DIR"
    
    BENCH_FULL=$(basename "$TARGET_DIR")
    EVAL_DIR_NAME=$(basename $(dirname "$TARGET_DIR"))
    BASE_MODEL="${EVAL_DIR_NAME%%_evaluated_by_*}"
    
    OUTPUT_FILE="${TARGET_DIR}/${OUTPUT_FILENAME}"
    
    BASE_RESULTS_FILE="${BASE_DIR}/${BASE_MODEL}/${BENCH_FULL}/${BENCH_FULL}_results.jsonl"
    
    # Eğer _t veya _k uzantıları varsa onları silip orijinal benchmark adıyla aramayı dene
    if [[ ! -f "$BASE_RESULTS_FILE" ]]; then
        BENCH_CORE="${BENCH_FULL%%_t*}"
        BENCH_CORE="${BENCH_CORE%%_k*}"
        BASE_RESULTS_FILE="${BASE_DIR}/${BASE_MODEL}/${BENCH_CORE}/${BENCH_CORE}_results.jsonl"
    fi
    
    JUDGED_RESULTS_FILE="${TARGET_DIR}/${BENCH_FULL}_results_judged.jsonl"
    SUMMARY_FILE="${TARGET_DIR}/summary.json"

    python scripts/cot_judge_pipeline/04_aggregate_votes.py \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE"
    
    if [[ -f "$BASE_RESULTS_FILE" ]] && [[ -f "$OUTPUT_FILE" ]]; then
        PYTHONPATH=. python scripts/cot_judge_pipeline/05_apply_metrics.py \
            --base_results_file "$BASE_RESULTS_FILE" \
            --judge_majority_file "$OUTPUT_FILE" \
            --output_file "$JUDGED_RESULTS_FILE" \
            --summary_file "$SUMMARY_FILE"
    else
        echo "[WARNING] Base results not found for $BENCH_FULL at $BASE_RESULTS_FILE"
    fi
done

echo "[INFO] Post-processing completed."