#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --nodelist=nsdl2

set -euo pipefail

JUDGE_MODEL_ID=$1
BASE_MODEL_ID=$2
TASK=$3
INPUT_FILE=$4
TEMPERATURE=$5

eval "$(conda shell.bash hook)"
conda activate evalhub_env

CONFIG_FILE="${SLURM_SUBMIT_DIR}/scripts/configs/cot_judge.env"
source "$CONFIG_FILE"

JUDGE_SAFE_NAME=$(basename "$JUDGE_MODEL_ID")
BASE_SAFE_NAME=$(basename "$BASE_MODEL_ID")

PORT=$((30000 + RANDOM % 10000))
PROM_PORT=$((40000 + RANDOM % 10000))

export HF_TOKEN
export SGLANG_DISABLE_CUDNN_CHECK=1
export HOSTED_VLLM_API_BASE="http://127.0.0.1:${PORT}/v1"
export HOSTED_VLLM_API_KEY="EMPTY"

echo "[INFO] Starting Judge server (SGLang) for model $JUDGE_SAFE_NAME on port $PORT"

python -m sglang_router.launch_server \
    --model-path "$JUDGE_MODEL_ID" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dp 1 \
    --router-balance-abs-threshold 1 \
    --mem-fraction-static 0.85 \
    --router-prometheus-port "$PROM_PORT" \
    --trust-remote-code &
SERVER_PID=$!

echo "[INFO] Waiting for the judge server to be ready..."
while ! curl -s -f "http://127.0.0.1:${PORT}/v1/models" | grep -q "$JUDGE_SAFE_NAME"; do
    sleep 15
done
sleep 15 # Ekstra tampon süresi
echo "[INFO] Judge server is online and responding."

OUT_DIR="results/judgments/${BASE_SAFE_NAME}_evaluated_by_${JUDGE_SAFE_NAME}/${TASK}_t${TEMPERATURE}"
mkdir -p "$OUT_DIR"

if [[ -n "${OVERRIDE_ARGS:-}" ]] && [[ "$OVERRIDE_ARGS" == *\} ]]; then
    FINAL_OVERRIDE="${OVERRIDE_ARGS%\}}, \"file_path\": \"$INPUT_FILE\"}"
else
    FINAL_OVERRIDE="{\"file_path\": \"$INPUT_FILE\"}"
fi

GEN_ARGS=(
    --model "hosted_vllm/$JUDGE_MODEL_ID"
    --tasks "math_judge"
    --output-dir "$OUT_DIR"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --max-completion-tokens "$MAX_COMPLETION_TOKENS"
    --frequency-penalty "$FREQUENCY_PENALTY"
    --presence-penalty "$PRESENCE_PENALTY"
    --n-samples "$N_SAMPLES"
    --num-workers "$NUM_WORKERS"
    --timeout "$TIMEOUT"
    --override-args "$FINAL_OVERRIDE"
)

[[ -n "${STOP:-}" ]] && GEN_ARGS+=(--stop "$STOP")
[[ -n "${SYSTEM_PROMPT:-}" ]] && GEN_ARGS+=(--system-prompt "$SYSTEM_PROMPT")
[[ -n "${TOOL_CONFIG:-}" ]] && GEN_ARGS+=(--tool-config "$TOOL_CONFIG")

echo "[INFO] Initiating Judge Generation Phase..."
evalhub gen "${GEN_ARGS[@]}"

echo "[INFO] Initiating Judge Evaluation Phase..."
evalhub eval \
    --tasks "math_judge" \
    --solutions "$OUT_DIR/math_judge.jsonl" \
    --output-dir "$OUT_DIR"

echo "[INFO] Shutting down Judge server..."
kill "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true
echo "[INFO] Judge worker execution completed."