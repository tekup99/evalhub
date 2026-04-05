#!/bin/bash
# ==============================================================================
# EvalHub Master Orchestrator: Monolithic Pipeline Implementation
# ==============================================================================
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

eval "$(conda shell.bash hook)"
conda activate evalhub_env
export SGLANG_DISABLE_CUDNN_CHECK=1

# ------------------------------------------------------------------------------
# 1. Single Source of Truth Configuration
# ------------------------------------------------------------------------------
CONFIG_FILE="scripts/unified_pipeline.env"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Configuration file $CONFIG_FILE not found."
    exit 1
fi

source "$CONFIG_FILE"

# Determine execution mode (defaults to orchestrator if not provided via SLURM)
RUN_MODE="${RUN_MODE:-orchestrator}"

# ------------------------------------------------------------------------------
# 2. Shared Utilities
# ------------------------------------------------------------------------------
start_server_and_wait() {
    local model_path="$1"
    local port="$2"
    
    echo "[INFRA] Initializing sglang server for model: $model_path on port $port"
    python -m sglang.launch_server --model-path "$model_path" --port "$port" \
        --mem-fraction-static 0.85 \
        --schedule-conservativeness 1.0 \
        --chunked-prefill-size 4096 \
        --max-running-requests 32 &
    SERVER_PID=$!
    echo "[INFRA] Waiting for health check on http://127.0.0.1:$port/health..."
    local retries=0
    local max_retries=60 # 5 minutes total wait
    
    while ! curl -s "http://127.0.0.1:$port/health" > /dev/null; do
        sleep 5
        retries=$((retries + 1))
        if [[ "$retries" -ge "$max_retries" ]]; then
            echo "[ERROR] Server failed to pass health check within timeout."
            kill "$SERVER_PID" 2>/dev/null || true
            exit 1
        fi
    done
    echo "[INFRA] Server is healthy and accepting requests."
}

# ------------------------------------------------------------------------------
# 3. Worker Modes
# ------------------------------------------------------------------------------
run_eval_worker() {
    echo "======================================================================="
    echo "[EVAL WORKER] Starting Base Evaluation"
    echo "Target: $TARGET_MODEL | Benchmark: $BENCHMARK"
    echo "======================================================================="
    
    start_server_and_wait "$TARGET_MODEL" "$PORT"
    

    # Export API details so evalhub hosted_vllm backend can find the local server
    export HOSTED_VLLM_API_BASE="http://127.0.0.1:$PORT/v1"
    export HOSTED_VLLM_API_KEY="EMPTY"

    local clean_model=$(basename "$TARGET_MODEL")
    # Base format: results/<model_name>_t<temp>_max<tokens>_k<n>/<benchmark>
    local out_dir="results/${clean_model}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}/${BENCHMARK}"
    mkdir -p "$out_dir"
    
    echo "[EVAL WORKER] Executing generation phase (evalhub gen)..."
    python -m evalhub.cli gen \
        --model "hosted_vllm/$TARGET_MODEL" \
        --tasks "$BENCHMARK" \
        --temperature "$TEMPERATURE" \
        --n-samples "$N_SAMPLES" \
        --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
        --output-dir "$out_dir"

    # CRITICAL FIX: Ensure the raw outputs are formatted before evaluation
    local solutions_file="$out_dir/${BENCHMARK}.jsonl"
    if [[ ! -f "$solutions_file" ]]; then
        echo "[WARNING] $solutions_file not found. Falling back to raw JSONL."
        solutions_file="$out_dir/${BENCHMARK}_raw.jsonl"
    fi

    echo "[EVAL WORKER] Executing evaluation phase (evalhub eval)..."
    python -m evalhub.cli eval \
        --tasks "$BENCHMARK" \
        --solutions "$solutions_file" \
        --output-dir "$out_dir"

    echo "[INFRA] Tearing down Base Model server (PID: $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
}

run_extract_worker() {
    echo "======================================================================="
    echo "[EXTRACT WORKER] Running Pass@K Extraction"
    echo "Target: $TARGET_MODEL | Benchmark: $BENCHMARK"
    echo "======================================================================="
    
    local clean_model=$(basename "$TARGET_MODEL")
    local filtered_dir="data/passatk_filtered/${clean_model}"
    mkdir -p "$filtered_dir"
    
    local out_dir="results/${clean_model}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}/${BENCHMARK}"    
    # Evalhub outputs
    local results_file="${out_dir}/${BENCHMARK}_results.jsonl"
    local raw_file="${out_dir}/${BENCHMARK}_raw.jsonl"
    local output_filtered="${filtered_dir}/${BENCHMARK}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}_corrects.jsonl"    
    
    python scripts/cot_judge_pipeline/01_extract_corrects.py \
        --results_file "$results_file" \
        --raw_file "$raw_file" \
        --output_file "$output_filtered"
}

run_judge_worker() {
    echo "======================================================================="
    echo "[JUDGE WORKER] Starting CoT Judging"
    echo "Judge: $JUDGE_MODEL | Target: $TARGET_MODEL | Benchmark: $BENCHMARK"
    echo "======================================================================="
    
    start_server_and_wait "$JUDGE_MODEL" "$PORT"

    export HOSTED_VLLM_API_BASE="http://127.0.0.1:$PORT/v1"
    export HOSTED_VLLM_API_KEY="EMPTY"

    local clean_target=$(basename "$TARGET_MODEL")
    local clean_judge=$(basename "$JUDGE_MODEL")
    # CoT format: results/judgments/<target_model>_evaluated_by_<judge_model>_<max_tokens>/<benchmark>_t<temp>_k<n>
    local out_dir="results/judgments/${clean_target}_evaluated_by_${clean_judge}_${JUDGE_MAX_COMPLETION_TOKENS}/${BENCHMARK}_t${JUDGE_TEMPERATURES}"   
    mkdir -p "$out_dir"

    local final_override="{\"file_path\": \"$FILTERED_FILE\"}"

    echo "[JUDGE WORKER] Executing judgement generation phase..."
    python -m evalhub.cli gen \
        --model "hosted_vllm/$JUDGE_MODEL" \
        --tasks "math_judge" \
        --temperature "$JUDGE_TEMPERATURES" \
        --n-samples "$JUDGE_N_SAMPLES" \
        --max-completion-tokens "$JUDGE_MAX_COMPLETION_TOKENS" \
        --output-dir "$out_dir" \
            --override-args "$final_override"

    echo "[JUDGE WORKER] Executing judgement evaluation phase..."
    
    local judge_solutions_file="$out_dir/math_judge.jsonl"
    if [[ ! -f "$judge_solutions_file" ]]; then
        judge_solutions_file="$out_dir/math_judge_raw.jsonl"
    fi

    python -m evalhub.cli eval \
        --tasks "math_judge" \
        --solutions "$judge_solutions_file" \
        --output-dir "$out_dir" \
        --override-args "$final_override"


    echo "[INFRA] Tearing down Judge Model server (PID: $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
}

run_post_worker() {
    echo "======================================================================="
    echo "[POST WORKER] Post-Processing Judge Outputs"
    echo "Target: $TARGET_MODEL | Judge: $JUDGE_MODEL"
    echo "======================================================================="
    
    local clean_target=$(basename "$TARGET_MODEL")
    local clean_judge=$(basename "$JUDGE_MODEL")
    
    local out_dir="results/judgments/${clean_target}_evaluated_by_${clean_judge}_${JUDGE_MAX_COMPLETION_TOKENS}/${BENCHMARK}_t${JUDGE_TEMPERATURES}"
    
    local eval_judge_output="${out_dir}/math_judge.jsonl"
    [[ ! -f "$eval_judge_output" ]] && eval_judge_output="${out_dir}/math_judge_raw.jsonl"

    local majority_file="${out_dir}/math_judge_majority.jsonl"

    local base_results_path="results/${clean_target}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}/${BENCHMARK}"
    local base_results_file="${base_results_path}/${BENCHMARK}_results.jsonl"

    local judged_results_file="${out_dir}/${BENCHMARK}_results_judged.jsonl"
    local summary_file="${out_dir}/summary.json"

    echo "[POST WORKER] Aggregating votes..."
    python scripts/cot_judge_pipeline/04_aggregate_votes.py \
        --input_file "$eval_judge_output" \
        --output_file "$majority_file"

    if [[ -f "$base_results_file" ]]; then
        echo "[POST WORKER] Applying CoT filters to base results..."
        echo "Source: $base_results_file"
        
        python scripts/cot_judge_pipeline/05_apply_metrics.py \
            --base_results_file "$base_results_file" \
            --judge_majority_file "$majority_file" \
            --output_file "$judged_results_file" \
            --summary_file "$summary_file"
            
        echo "[POST WORKER] Success! Judged results saved to: $judged_results_file"
    else
        echo "[ERROR] Base results file not found at: $base_results_file"
        exit 1
    fi
}

# ------------------------------------------------------------------------------
# 4. Orchestrator Logic (DAG Submission)
# ------------------------------------------------------------------------------
run_orchestrator() {
    echo "======================================================================="
    echo "Initializing Unified Pipeline Orchestrator (Monolithic Mode)"
    echo "======================================================================="

    mkdir -p logs results/judgments data/passatk_filtered plots

    local script_path
    script_path=$(realpath "$0")
    local global_last_job_id=""

    for MODEL in $BASE_MODELS; do
        local clean_model
        clean_model=$(basename "$MODEL")
        
        for BENCHMARK in $BENCHMARKS; do
            for B_TEMP in $BASE_TEMPERATURES; do
                
                local dynamic_suffix="_t${B_TEMP}_max${BASE_MAX_COMPLETION_TOKENS}"
                local run_id="${clean_model}_${BENCHMARK}${dynamic_suffix}"
                
                echo "[ORCHESTRATOR] Submitting DAG for Combination: $run_id"

                # --- Phase 1: Base Evaluation ---
                local global_dep=""
                if [[ -n "$global_last_job_id" ]]; then
                    global_dep="--dependency=afterany:${global_last_job_id}"
                fi

                local base_job
                base_job=$(sbatch --parsable $global_dep \
                    --job-name="eval_${run_id}" \
                    --output="logs/eval_${run_id}_%j.out" \
                    --error="logs/eval_${run_id}_%j.err" \
                    --ntasks=1 --cpus-per-task="${BASE_SLURM_CPUS}" --mem="${BASE_SLURM_MEM}" \
                    --gres="${BASE_SLURM_GRES}" --time="${BASE_SLURM_TIME}" \
                    $([[ -n "$BASE_SLURM_NODELIST" ]] && echo "--nodelist=$BASE_SLURM_NODELIST") \
                    --export=ALL,RUN_MODE="eval_worker",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",N_SAMPLES="$BASE_N_SAMPLES",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS" \
                    "$script_path")

                # --- Phase 2: Extraction ---
                local extract_job
                extract_job=$(sbatch --parsable --dependency=afterok:"${base_job}" \
                    --job-name="extr_${run_id}" \
                    --output="logs/extr_${run_id}_%j.out" \
                    --ntasks=1 --cpus-per-task="${EXTRACT_SLURM_CPUS}" --mem="${EXTRACT_SLURM_MEM}" \
                    --time="${EXTRACT_SLURM_TIME}" \
                    --export=ALL,RUN_MODE="extract_worker",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",N_SAMPLES="$BASE_N_SAMPLES",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS" \
                    "$script_path")

                local internal_dep=$extract_job
                local filtered_file="data/passatk_filtered/${clean_model}/${BENCHMARK}_t${B_TEMP}_max${BASE_MAX_COMPLETION_TOKENS}_corrects.jsonl"

                # --- Phase 3 & 4: CoT Judge & Post-Processing ---
                for JUDGE in $JUDGE_MODELS; do
                    local clean_judge
                    clean_judge=$(basename "$JUDGE")
                    local judge_run_id="${run_id}_J-${clean_judge}"

                    local judge_job
                    judge_job=$(sbatch --parsable --dependency=afterok:"${internal_dep}" \
                        --job-name="jdg_${judge_run_id}" \
                        --output="logs/jdg_${judge_run_id}_%j.out" \
                        --error="logs/jdg_${judge_run_id}_%j.err" \
                        --ntasks=1 --cpus-per-task="${JUDGE_SLURM_CPUS}" --mem="${JUDGE_SLURM_MEM}" \
                        --gres="${JUDGE_SLURM_GRES}" --time="${JUDGE_SLURM_TIME}" \
                        $([[ -n "$JUDGE_SLURM_NODELIST" ]] && echo "--nodelist=$JUDGE_SLURM_NODELIST") \
                        --export=ALL,RUN_MODE="judge_worker",JUDGE_MODEL="$JUDGE",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",FILTERED_FILE="$filtered_file" \
                        "$script_path")

                    local post_job
                    post_job=$(sbatch --parsable --dependency=afterok:"${judge_job}" \
                        --job-name="post_${judge_run_id}" \
                        --output="logs/post_${judge_run_id}_%j.out" \
                        --ntasks=1 --cpus-per-task="${POST_SLURM_CPUS}" --mem="${POST_SLURM_MEM}" \
                        --time="${POST_SLURM_TIME}" \
                        --export=ALL,RUN_MODE="post_worker",JUDGE_MODEL="$JUDGE",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS" \
                        "$script_path")
                    
                    internal_dep=$post_job
                done
                
                global_last_job_id=$internal_dep
                echo "[ORCHESTRATOR] DAG Submitted. Tail Job ID: $global_last_job_id"
            done
        done
    done
}

# ------------------------------------------------------------------------------
# 5. Script Entry Point Routing
# ------------------------------------------------------------------------------
case "$RUN_MODE" in
    orchestrator)
        run_orchestrator
        ;;
    eval_worker)
        run_eval_worker
        ;;
    extract_worker)
        run_extract_worker
        ;;
    judge_worker)
        run_judge_worker
        ;;
    post_worker)
        run_post_worker
        ;;
    *)
        echo "[ERROR] Unknown RUN_MODE mapping: $RUN_MODE"
        exit 1
        ;;
esac

exit 0