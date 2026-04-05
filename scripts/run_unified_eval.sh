#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------------------------
# EvalHub Master Orchestrator: Professional Sequential DAG Implementation
# ------------------------------------------------------------------------------

CONFIG="scripts/unified_pipeline.env"
if [[ -f "$CONFIG" ]]; then
    source "$CONFIG"
fi


mkdir -p logs data results plots

echo "-----------------------------------------------------------------------"
echo "Initializing Pipeline Orchestration"
echo "-----------------------------------------------------------------------"

GLOBAL_LAST_JOB_ID=""

get_node_config() {
    [[ -n "$1" ]] && echo "--nodelist=$1" || echo ""
}

for MODEL in $BASE_MODELS; do
    CLEAN_MODEL=$(basename "$MODEL")
    for BENCHMARK in $BENCHMARKS; do
        for B_TEMP in $BASE_TEMPERATURES; do
            
            DYNAMIC_SUFFIX="_t${B_TEMP}_max${BASE_MAX_COMPLETION_TOKENS}_k${BASE_N_SAMPLES}"
            RUN_ID="${CLEAN_MODEL}_${BENCHMARK}${DYNAMIC_SUFFIX}"
            
            echo "[PIPELINE] Processing Combination: $RUN_ID"

            # --- PHASE 1: Base Evaluation ---
            GLOBAL_DEP=""
            [[ -n "$GLOBAL_LAST_JOB_ID" ]] && GLOBAL_DEP="--dependency=afterany:${GLOBAL_LAST_JOB_ID}"

            BASE_JOB=$(sbatch --parsable ${GLOBAL_DEP} \
                --job-name="eval_${RUN_ID}" \
                --output="logs/eval_${RUN_ID}_%j.out" \
                --error="logs/eval_${RUN_ID}_%j.err" \
                --ntasks=1 --cpus-per-task=${BASE_SLURM_CPUS} --mem=${BASE_SLURM_MEM} \
                --gres=${BASE_SLURM_GRES} --time=${BASE_SLURM_TIME} \
                $(get_node_config "$BASE_SLURM_NODELIST") \
                --export=ALL,\
TARGET_MODEL="$MODEL",TASKS="$BENCHMARK",TEMPERATURE="$B_TEMP",N_SAMPLES="$BASE_N_SAMPLES",\
TOP_P="$BASE_TOP_P",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS",\
FREQUENCY_PENALTY="$BASE_FREQUENCY_PENALTY",PRESENCE_PENALTY="$BASE_PRESENCE_PENALTY",\
NUM_WORKERS="$BASE_NUM_WORKERS",TIMEOUT="$BASE_TIMEOUT",\
STOP="$BASE_STOP",SYSTEM_PROMPT="$BASE_SYSTEM_PROMPT",OVERRIDE_ARGS="$BASE_OVERRIDE_ARGS",\
ENABLE_MULTITURN="$BASE_ENABLE_MULTITURN",MAX_TURNS="$BASE_MAX_TURNS",\
TOOL_CONFIG="$BASE_TOOL_CONFIG",CALLBACK="$BASE_CALLBACK",RESUME="$BASE_RESUME",\
HF_TOKEN="$HF_TOKEN",PORT="$PORT",DYNAMIC_SUFFIX="$DYNAMIC_SUFFIX" \
                scripts/pass_k_pipeline/02_run_eval_worker.sh)

            # --- PHASE 2: Extraction ---
            EXTRACT_JOB=$(sbatch --parsable --dependency=afterok:${BASE_JOB} \
                --job-name="extr_${RUN_ID}" \
                --output="logs/extr_${RUN_ID}_%j.out" \
                --ntasks=1 --cpus-per-task=${EXTRACT_SLURM_CPUS} --mem=${EXTRACT_SLURM_MEM} \
                --time=${EXTRACT_SLURM_TIME} \
                --export=ALL,MODEL="$MODEL",BENCHMARK="$BENCHMARK",SUFFIX="$DYNAMIC_SUFFIX" \
                scripts/cot_judge_pipeline/01_run_extraction.sh)

            INTERNAL_DEP=$EXTRACT_JOB

            # --- PHASE 3: CoT-Judge Evaluation ---
            FILTERED_FILE="data/passatk_filtered/${CLEAN_MODEL}/${BENCHMARK}${DYNAMIC_SUFFIX}_corrects.jsonl"
            
            for JUDGE in $JUDGE_MODELS; do
                CLEAN_JUDGE=$(basename "$JUDGE")
                JUDGE_RUN_ID="${RUN_ID}_J-${CLEAN_JUDGE}"

                JUDGE_JOB=$(sbatch --parsable --dependency=afterok:${INTERNAL_DEP} \
                    --job-name="jdg_${JUDGE_RUN_ID}" \
                    --output="logs/jdg_${JUDGE_RUN_ID}_%j.out" \
                    --error="logs/jdg_${JUDGE_RUN_ID}_%j.err" \
                    --ntasks=1 --cpus-per-task=${JUDGE_SLURM_CPUS} --mem=${JUDGE_SLURM_MEM} \
                    --gres=${JUDGE_SLURM_GRES} --time=${JUDGE_SLURM_TIME} \
                    $(get_node_config "$JUDGE_SLURM_NODELIST") \
                    --export=ALL,\
TOP_P="$JUDGE_TOP_P",MAX_COMPLETION_TOKENS="$JUDGE_MAX_COMPLETION_TOKENS",\
FREQUENCY_PENALTY="$JUDGE_FREQUENCY_PENALTY",PRESENCE_PENALTY="$JUDGE_PRESENCE_PENALTY",\
N_SAMPLES="$JUDGE_N_SAMPLES",NUM_WORKERS="$JUDGE_NUM_WORKERS",TIMEOUT="$JUDGE_TIMEOUT",\
STOP="$JUDGE_STOP",SYSTEM_PROMPT="$JUDGE_SYSTEM_PROMPT",OVERRIDE_ARGS="$JUDGE_OVERRIDE_ARGS",\
ENABLE_MULTITURN="$JUDGE_ENABLE_MULTITURN",MAX_TURNS="$JUDGE_MAX_TURNS",\
TOOL_CONFIG="$JUDGE_TOOL_CONFIG",CALLBACK="$JUDGE_CALLBACK",RESUME="$JUDGE_RESUME",\
HF_TOKEN="$HF_TOKEN",PORT="$PORT" \
                    scripts/cot_judge_pipeline/03_run_judge_worker.sh \
                    "$JUDGE" "$MODEL" "$BENCHMARK" "$FILTERED_FILE" "$JUDGE_TEMPERATURES" "$BASE_N_SAMPLES")

                # --- PHASE 4: Post-Processing ---
                POST_JOB=$(sbatch --parsable --dependency=afterok:${JUDGE_JOB} \
                    --job-name="post_${JUDGE_RUN_ID}" \
                    --output="logs/post_${JUDGE_RUN_ID}_%j.out" \
                    --ntasks=1 --cpus-per-task=${POST_SLURM_CPUS} --mem=${POST_SLURM_MEM} \
                    --time=${POST_SLURM_TIME} \
                    --export=ALL,JUDGE_MODEL="$JUDGE",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",SUFFIX="$DYNAMIC_SUFFIX" \
                    scripts/cot_judge_pipeline/04_05_run_postprocess.sh)
                
                INTERNAL_DEP=$POST_JOB
            done
            
            GLOBAL_LAST_JOB_ID=$INTERNAL_DEP
            echo "[SUBMITTED] Chain for $RUN_ID. Tail Job ID: $GLOBAL_LAST_JOB_ID"
        done
    done
donez