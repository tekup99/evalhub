#!/bin/bash
# ==============================================================================
# EvalHub Master Orchestrator: Monolithic Pipeline Implementation (Gemma 4 / vLLM)
# ==============================================================================
set -euo pipefail

# 1. Path Independence: Establish project root dynamically
if [[ -z "${PROJECT_ROOT:-}" ]]; then
    export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Triton Fallback: Disable custom triton kernels if SparseMatrix issues persist on H200
export VLLM_USE_TRITON_FLASH_ATTN=0 

# Conda activation
eval "$(conda shell.bash hook 2>/dev/null || echo '')"
conda activate evalhub_env || echo "[WARNING] Conda environment failed to activate, falling back to system Python."

CONFIG_FILE="${PROJECT_ROOT}/scripts/gemma4_pipeline.env"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Configuration file $CONFIG_FILE not found at $PROJECT_ROOT/scripts." >&2
    exit 1
fi

# 2. Environment Variable Integration: Force export of all variables in the .env
set -a
source "$CONFIG_FILE"
set +a

# Ensure HF_TOKEN is exposed to vLLM
export HF_TOKEN="${HF_TOKEN:-}"

RUN_MODE="${RUN_MODE:-orchestrator}"
BASE_PORT="${PORT:-30000}"

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
start_server_and_wait() {
    local model_path="$1"
    local port="$2"
    local p_count="$3"
    
    mkdir -p logs
    local log_file="logs/vllm_server_error_${port}.log"
    touch "$log_file" # Log dosyasının kesin oluştuğundan emin olalım
    
    echo "[INFO] Initializing vLLM server for model: $model_path on port $port"
    echo "[INFO] Hardware Strategy: --tensor-parallel-size $p_count"

    # 1. vLLM argümanlarını güvenli bir şekilde dizi (array) içinde topluyoruz
    local vllm_args=(
        --model "$model_path"
        --port "$port"
        --tensor-parallel-size "$p_count"
        --trust-remote-code
    )

    # --- GEMMA MODİFİKASYONU: Jinja içine enable_thinking=true enjekte etme ---
    local is_gemma=false
    if [[ "${model_path,,}" == *"gemma"* ]]; then
        is_gemma=true
    fi

    if [[ "$is_gemma" == true ]]; then
        echo "[INFO] Gemma model detected ($model_path). Generating dynamic template with enable_thinking=true..."
        local template_file="${PROJECT_ROOT}/logs/gemma_thinking_${port}.jinja"
        local py_script="${PROJECT_ROOT}/logs/gen_template_${port}.py"
        
        # Python scripti ile modelin orjinal template'ini çekip modifiye ediyoruz
        cat << 'EOF' > "$py_script"
import sys
from transformers import AutoTokenizer

model_path = sys.argv[1]
out_file = sys.argv[2]

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    template = tokenizer.chat_template
    
    if not template:
        template = getattr(tokenizer, "default_chat_template", None)
        
    if not template:
        # Eğer tokenizer'da hiçbir template yoksa düz passthrough
        template = "{% for message in messages %}{{ message['content'] + '\\n\\n' }}{% endfor %}"

    
    template = template.replace("kwargs.get('enable_thinking', False)", "True")
    template = template.replace('kwargs.get("enable_thinking", False)', "True")
    template = template.replace("kwargs.get('enable_thinking',False)", "True")    
    # vLLM'de kwargs veremediğimiz için doğrudan Jinja değişkeni olarak ayarlıyoruz
    final_template = "{% set enable_thinking = true %}\n" + template
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(final_template)
except Exception as e:
    print(f"WARN: Template generation failed: {e}")
EOF
        
        # Scripti çalıştır ve şablon dosyasını vLLM'e ver
        python3 "$py_script" "$model_path" "$template_file"
        
        if [[ -f "$template_file" ]]; then
            vllm_args+=( --chat-template "$template_file" )
            echo "[INFO] Dynamic thinking template applied for Gemma."
        fi

    # 2. Eğer model Gemma DEĞİLSE ve BASE modelse
    elif [[ "$model_path" != *"-it"* ]] && [[ "$model_path" != *"-Instruct"* ]] && [[ "$model_path" != *"-Chat"* ]]; then
        
        if [[ "${model_path,,}" == *"qwen"* ]]; then
            vllm_args+=( --chat-template "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{{ '<|im_start|>assistant\n' }}" )
            echo "[INFO] Dynamic Injection: Inline ChatML applied for Qwen Base: $model_path"
        else
            vllm_args+=( --chat-template "{% for message in messages %}{{ message['content'] + '\n\n' }}{% endfor %}" )
            echo "[INFO] Dynamic Injection: Inline Passthrough applied for Base Model: $model_path"
        fi
        
    # Eğer model Gemma DEĞİLSE ve Instruct/Chat/Judge modelse
    else
        echo "[INFO] Non-Gemma Instruction-Tuned/Judge Model Detected ($model_path). Using native template."
    fi
    # -----------------------------------------------------------------

    # 3. Array'i vLLM'e iletiyoruz ("${vllm_args[@]}" tırnakları şablon boşluklarını korur)
    python -m vllm.entrypoints.openai.api_server "${vllm_args[@]}" >> "$log_file" 2>&1 &
    
    SERVER_PID=$!
    local retries=0
    local max_retries=360
    
    echo "[INFO] Waiting for server healthcheck... (Logs are in $log_file)"
    while ! curl -s "http://127.0.0.1:$port/health" > /dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[ERROR] vLLM server process ($SERVER_PID) died unexpectedly." >&2
            cat "$log_file" >&2
            exit 1
        fi
        
        sleep 5
        retries=$((retries + 1))
        
        if [ "$retries" -ge "$max_retries" ]; then
            echo "[ERROR] Server failed health check timeout on port $port." >&2
            tail -n 100 "$log_file" >&2
            kill "$SERVER_PID" 2>/dev/null || true
            exit 1
        fi
    done
    echo "[INFO] Server is healthy and accepting requests."
}

cleanup_server_and_cache() {
    echo "[INFO] Cleaning up server (PID: ${SERVER_PID:-}) and cache directory..."
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
    if [[ -n "${EVALHUB_CACHE_DIR:-}" ]]; then
        rm -rf "$EVALHUB_CACHE_DIR"
    fi
}

# ------------------------------------------------------------------------------
# Workers
# ------------------------------------------------------------------------------
run_eval_worker() {
    echo "[INFO] --- Starting Base Evaluation ---"
    
    # 3. Reconciled Port Logic: Based on configuration PORT
    local worker_port=$((BASE_PORT + RANDOM % 10000))
    export EVALHUB_CACHE_DIR="${PROJECT_ROOT}/data/cache/eval_${RANDOM}"
    mkdir -p "$EVALHUB_CACHE_DIR" "${PROJECT_ROOT}/logs"
    
    trap cleanup_server_and_cache EXIT

    start_server_and_wait "$TARGET_MODEL" "$worker_port" "$BASE_PARALLEL_COUNT"
    
    export HOSTED_VLLM_API_BASE="http://127.0.0.1:$worker_port/v1"
    export HOSTED_VLLM_API_KEY="EMPTY"

    local clean_model=$(basename "$TARGET_MODEL")
    local out_dir="${PROJECT_ROOT}/results/${clean_model}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}/${BENCHMARK}"
    mkdir -p "$out_dir"
    
    local cmd_args=(
        --model "hosted_vllm/$TARGET_MODEL"
        --tasks "$BENCHMARK"
        --temperature "$TEMPERATURE"
        --n-samples "$N_SAMPLES"
        --num-workers "$BASE_NUM_WORKERS"
        --max-completion-tokens "$MAX_COMPLETION_TOKENS"
        --output-dir "$out_dir"
    )

    [[ "${BASE_RESUME:-false}" == "true" ]] && cmd_args+=(--resume)
    
    echo "[INFO] Executing generation phase with full parameters..."
    python -m evalhub.cli gen "${cmd_args[@]}"

    local solutions_file="$out_dir/${BENCHMARK}.jsonl"
    [[ ! -f "$solutions_file" ]] && solutions_file="$out_dir/${BENCHMARK}_raw.jsonl"

    echo "[INFO] Executing evaluation phase..."
    python -m evalhub.cli eval \
        --tasks "$BENCHMARK" \
        --solutions "$solutions_file" \
        --output-dir "$out_dir"
}

run_extract_worker() {
    echo "[INFO] --- Running Pass@K Extraction ---"
    local clean_model=$(basename "$TARGET_MODEL")
    local filtered_dir="${PROJECT_ROOT}/data/passatk_filtered/${clean_model}"
    mkdir -p "$filtered_dir" "${PROJECT_ROOT}/logs"
    
    local out_dir="${PROJECT_ROOT}/results/${clean_model}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}/${BENCHMARK}"    
    local results_file="${out_dir}/${BENCHMARK}_results.jsonl"
    local raw_file="${out_dir}/${BENCHMARK}_raw.jsonl"
    local output_filtered="${filtered_dir}/${BENCHMARK}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}_corrects.jsonl"    
    
    python "${PROJECT_ROOT}/scripts/cot_judge_pipeline/01_extract_corrects.py" \
        --results_file "$results_file" \
        --raw_file "$raw_file" \
        --output_file "$output_filtered"
}

run_judge_worker() {
    echo "[INFO] --- Starting CoT Judging ---"
    
    local worker_port=$((BASE_PORT + 10000 + RANDOM % 10000))
    export EVALHUB_CACHE_DIR="${PROJECT_ROOT}/data/cache/judge_${RANDOM}"
    mkdir -p "$EVALHUB_CACHE_DIR" "${PROJECT_ROOT}/logs"

    trap cleanup_server_and_cache EXIT

    start_server_and_wait "$JUDGE_MODEL" "$worker_port" "$JUDGE_PARALLEL_COUNT"

    export HOSTED_VLLM_API_BASE="http://127.0.0.1:$worker_port/v1"
    export HOSTED_VLLM_API_KEY="EMPTY"

    local clean_target=$(basename "$TARGET_MODEL")
    local clean_judge=$(basename "$JUDGE_MODEL")
    local out_dir="${PROJECT_ROOT}/results/judgments/${clean_target}evaluated_by${clean_judge}_${JUDGE_MAX_COMPLETION_TOKENS}/${BENCHMARK}_t${JUDGE_TEMP}"   
    mkdir -p "$out_dir"

    if [[ -z "${FILTERED_FILE:-}" ]]; then
        FILTERED_FILE="${PROJECT_ROOT}/data/passatk_filtered/${clean_target}/${BENCHMARK}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}_corrects.jsonl"
    fi

    local final_override="{\"file_path\": \"$FILTERED_FILE\"}"
    
    local cmd_args=(
        --model "hosted_vllm/$JUDGE_MODEL"
        --tasks "math_judge"
        --temperature "$JUDGE_TEMP"
        --n-samples "$JUDGE_N_SAMPLES"
        --num-workers "$JUDGE_NUM_WORKERS"
        --max-completion-tokens "$JUDGE_MAX_COMPLETION_TOKENS"
        --output-dir "$out_dir"
        --override-args "$final_override"
    )

    echo "[INFO] Executing judgement generation phase..."
    python -m evalhub.cli gen "${cmd_args[@]}"

    local judge_solutions_file="$out_dir/math_judge.jsonl"
    [[ ! -f "$judge_solutions_file" ]] && judge_solutions_file="$out_dir/math_judge_raw.jsonl"

    python -m evalhub.cli eval \
        --tasks "math_judge" \
        --solutions "$judge_solutions_file" \
        --output-dir "$out_dir" \
        --override-args "$final_override"
}

run_post_worker() {
    echo "[INFO] --- Post-Processing Judge Outputs ---"
    local clean_target=$(basename "$TARGET_MODEL")
    local clean_judge=$(basename "$JUDGE_MODEL")
    
    local out_dir="${PROJECT_ROOT}/results/judgments/${clean_target}evaluated_by${clean_judge}_${JUDGE_MAX_COMPLETION_TOKENS}/${BENCHMARK}_t${JUDGE_TEMP}"
    rm -f "${out_dir}/math_judge_summary.json"

    local eval_judge_output="${out_dir}/math_judge.jsonl"
    [[ ! -f "$eval_judge_output" ]] && eval_judge_output="${out_dir}/math_judge_raw.jsonl"

    local majority_file="${out_dir}/math_judge_majority.jsonl"
    local base_results_path="${PROJECT_ROOT}/results/${clean_target}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}/${BENCHMARK}"
    local base_results_file="${base_results_path}/${BENCHMARK}_results.jsonl"

    local judged_results_file="${out_dir}/${BENCHMARK}_results.jsonl"
    local summary_file="${out_dir}/${BENCHMARK}_summary.json"
    local stats_file="${out_dir}/${BENCHMARK}_generation_stats.json"

    python "${PROJECT_ROOT}/scripts/cot_judge_pipeline/04_aggregate_votes.py" --input_file "$eval_judge_output" --output_file "$majority_file"
    python "${PROJECT_ROOT}/scripts/cot_judge_pipeline/05_apply_metrics.py" --base_results_file "$base_results_file" --judge_majority_file "$majority_file" --output_file "$judged_results_file" --summary_file "$summary_file" --stats_file "$stats_file"
}

# ------------------------------------------------------------------------------
# Orchestrator Logic
# ------------------------------------------------------------------------------
run_orchestrator() {
    echo "[INFO] Initializing Unified Pipeline Orchestrator (Monolithic Mode)"

    mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results/judgments" "${PROJECT_ROOT}/data/passatk_filtered" "${PROJECT_ROOT}/plots" "${PROJECT_ROOT}/data/cache"

    local script_path=$(realpath "$0")
    local global_last_job_id=""

    for MODEL in $BASE_MODELS; do
        local clean_model=$(basename "$MODEL")
        for BENCHMARK in $BENCHMARKS; do
            for B_TEMP in $BASE_TEMPERATURES; do
                
                local run_id="${clean_model}_${BENCHMARK}_t${B_TEMP}_max${BASE_MAX_COMPLETION_TOKENS}"
                echo "[INFO] Submitting Base Job for Combination: $run_id"

                local global_dep=""
                [[ -n "$global_last_job_id" ]] && global_dep="--dependency=afterany:${global_last_job_id}"

                local nodelist_param=""
                [[ -n "${BASE_SLURM_NODELIST:-}" ]] && nodelist_param="--nodelist=$BASE_SLURM_NODELIST"

                # 4. Strict Slurm Exports: Ensure HF_TOKEN and paths are explicitly passed to worker nodes
                local base_job=$(sbatch --parsable $global_dep \
                    --job-name="eval_${run_id}" \
                    --output="${PROJECT_ROOT}/logs/eval_${run_id}_%j.out" \
                    --error="${PROJECT_ROOT}/logs/eval_${run_id}_%j.err" \
                    --ntasks=1 --cpus-per-task="${BASE_SLURM_CPUS}" --mem="${BASE_SLURM_MEM}" \
                    --gres="${BASE_SLURM_GRES}" --time="${BASE_SLURM_TIME}" \
                    $nodelist_param \
                    --export=ALL,RUN_MODE="eval_worker",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",N_SAMPLES="$BASE_N_SAMPLES",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS",HF_TOKEN="${HF_TOKEN}",PROJECT_ROOT="${PROJECT_ROOT}" \
                    "$script_path" || echo "FAILED")

                if [[ "$base_job" == "FAILED" ]]; then
                    echo "[ERROR] Slurm sbatch failed to submit base_job. Check your slurm configuration."
                    exit 1
                fi

                echo "[SUCCESS] Base Eval Job Submitted: $base_job"
                
                local extract_job=$(sbatch --parsable --dependency=afterok:"${base_job}" \
                    --job-name="extr_${run_id}" --output="${PROJECT_ROOT}/logs/extr_${run_id}_%j.out" \
                    --ntasks=1 --time="${EXTRACT_SLURM_TIME}" \
                    --export=ALL,RUN_MODE="extract_worker",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS",PROJECT_ROOT="${PROJECT_ROOT}" \
                    "$script_path" || echo "FAILED")

                global_last_job_id="${extract_job}"
            done
        done
    done
}

case "$RUN_MODE" in
    orchestrator)   run_orchestrator ;;
    eval_worker)    run_eval_worker ;;
    extract_worker) run_extract_worker ;;
    judge_worker)   run_judge_worker ;;
    post_worker)    run_post_worker ;;
    *)              echo "[ERROR] Invalid RUN_MODE: $RUN_MODE" >&2; exit 1 ;;
esac

exit 0