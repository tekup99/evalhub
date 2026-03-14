#!/bin/bash
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH --nodelist=nsdl2

# Parametreler launcher script üzerinden otomatik gelecek
JUDGE_MODEL_ID=$1
BASE_MODEL_ID=$2
TASK=$3
INPUT_FILE=$4

# Ortam hazırlığı
eval "$(conda shell.bash hook)"
conda activate evalhub_env

# Config dosyasını yükle (Hyperparametreler için)
CONFIG_FILE="$SLURM_SUBMIT_DIR/scripts/judge_config.env"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

JUDGE_SAFE_NAME=$(basename "$JUDGE_MODEL_ID")
BASE_SAFE_NAME=$(basename "$BASE_MODEL_ID")

# Port çakışmasını engellemek için İKİ FARKLI rastgele port seçiyoruz
PORT=$((30000 + RANDOM % 10000))
PROM_PORT=$((40000 + RANDOM % 10000))

export HF_TOKEN
export SGLANG_DISABLE_CUDNN_CHECK=1
export HOSTED_VLLM_API_BASE="http://127.0.0.1:${PORT}/v1"
export HOSTED_VLLM_API_KEY="EMPTY"

echo "Starting server for Judge: $JUDGE_SAFE_NAME | Base: $BASE_SAFE_NAME | Task: $TASK"
echo "Assigned SGLang API Port: $PORT | Prometheus Port: $PROM_PORT"

# DP 2 ve Dinamik Prometheus Portu eklendi!
python -m sglang_router.launch_server \
    --model-path "$JUDGE_MODEL_ID" \
    --host 127.0.0.1 \
    --port $PORT \
    --dp 2 \
    --router-balance-abs-threshold 1 \
    --mem-fraction-static 0.85 \
    --router-prometheus-port $PROM_PORT \
    --trust-remote-code &

SERVER_PID=$!

# Sunucunun hazır olmasını bekle
while ! curl -s -f http://127.0.0.1:${PORT}/v1/models | grep -q "$JUDGE_SAFE_NAME"; do
    sleep 15
done
sleep 30 # SGLang'in bellek atamalarını tamamen bitirmesi için ufak bir pay

OUT_DIR="$HOME/evalhub/results/judgments/${BASE_SAFE_NAME}_evaluated_by_${JUDGE_SAFE_NAME}/$TASK"
mkdir -p "$OUT_DIR"

# OVERRIDE_ARGS Merge Mantığı
if [ -n "$OVERRIDE_ARGS" ] && [[ "$OVERRIDE_ARGS" == *\} ]]; then
    FINAL_OVERRIDE="${OVERRIDE_ARGS%\}}, \"file_path\": \"$INPUT_FILE\"}"
else
    FINAL_OVERRIDE="{\"file_path\": \"$INPUT_FILE\"}"
fi

echo "Evaluating $BASE_SAFE_NAME on $TASK using Judge $JUDGE_SAFE_NAME..."

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
)

# Ekstra argümanlar
[ -n "$STOP" ] && GEN_ARGS+=(--stop "$STOP")
[ -n "$SYSTEM_PROMPT" ] && GEN_ARGS+=(--system-prompt "$SYSTEM_PROMPT")
[ -n "$FINAL_OVERRIDE" ] && GEN_ARGS+=(--override-args "$FINAL_OVERRIDE")
[ -n "$TOOL_CONFIG" ] && GEN_ARGS+=(--tool-config "$TOOL_CONFIG")
[ -n "$CALLBACK" ] && GEN_ARGS+=(--callback "$CALLBACK")
[ "$ENABLE_MULTITURN" = true ] && GEN_ARGS+=(--enable-multiturn --max-turns "$MAX_TURNS")
[ "$RESUME" = true ] && GEN_ARGS+=(--resume)

# Generation
evalhub gen "${GEN_ARGS[@]}"

# Evaluation
evalhub eval \
    --tasks "math_judge" \
    --solutions "$OUT_DIR/math_judge.jsonl" \
    --output-dir "$OUT_DIR"

# View
evalhub view \
    --results "$OUT_DIR/math_judge_results.jsonl" \
    --max-display -1

# Sunucuyu Kapat
echo "Killing SGLang server (PID: $SERVER_PID)"
kill $SERVER_PID
sleep 5