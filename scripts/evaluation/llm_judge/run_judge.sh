#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --nodelist=nsdl2

# Parametreler
JUDGE_MODEL_ID=$1
BASE_MODEL_ID=$2
TASK=$3
INPUT_FILE=$4
TEMPERATURE=$5

# Ortam hazırlığı
eval "$(conda shell.bash hook)"
conda activate evalhub_env

# Config yükle
CONFIG_FILE="$SLURM_SUBMIT_DIR/scripts/configs/judge_config.env"
source "$CONFIG_FILE"

JUDGE_SAFE_NAME=$(basename "$JUDGE_MODEL_ID")
BASE_SAFE_NAME=$(basename "$BASE_MODEL_ID")

# Port çakışmasını engellemek için rastgele portlar
PORT=$((30000 + RANDOM % 10000))
PROM_PORT=$((40000 + RANDOM % 10000))

export HF_TOKEN
export SGLANG_DISABLE_CUDNN_CHECK=1
export HOSTED_VLLM_API_BASE="http://127.0.0.1:${PORT}/v1"
export HOSTED_VLLM_API_KEY="EMPTY"

# Sunucu Başlat (SGLang)
python -m sglang_router.launch_server \
    --model-path "$JUDGE_MODEL_ID" \
    --host 127.0.0.1 \
    --port $PORT \
    --dp 1 \
    --router-balance-abs-threshold 1 \
    --mem-fraction-static 0.85 \
    --router-prometheus-port $PROM_PORT \
    --trust-remote-code &
SERVER_PID=$!

# Hazır olana kadar bekle
while ! curl -s -f http://127.0.0.1:${PORT}/v1/models | grep -q "$JUDGE_SAFE_NAME"; do
    sleep 15
done
sleep 30

# Çıktı klasörü
OUT_DIR="results/judgments/${BASE_SAFE_NAME}_evaluated_by_${JUDGE_SAFE_NAME}/${TASK}_t${TEMPERATURE}"
mkdir -p "$OUT_DIR"

# OVERRIDE_ARGS hazırlığı
if [ -n "$OVERRIDE_ARGS" ] && [[ "$OVERRIDE_ARGS" == *\} ]]; then
    FINAL_OVERRIDE="${OVERRIDE_ARGS%\}}, \"file_path\": \"$INPUT_FILE\"}"
else
    FINAL_OVERRIDE="{\"file_path\": \"$INPUT_FILE\"}"
fi

# GEN_ARGS dizisini oluştur
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

# Diğer opsiyonel argümanları ekle
[ -n "$STOP" ] && GEN_ARGS+=(--stop "$STOP")
[ -n "$SYSTEM_PROMPT" ] && GEN_ARGS+=(--system-prompt "$SYSTEM_PROMPT")
[ -n "$TOOL_CONFIG" ] && GEN_ARGS+=(--tool-config "$TOOL_CONFIG")
[ -n "$CALLBACK" ] && GEN_ARGS+=(--callback "$CALLBACK")
[ "$ENABLE_MULTITURN" = true ] && GEN_ARGS+=(--enable-multiturn --max-turns "$MAX_TURNS")
[ "$RESUME" = true ] && GEN_ARGS+=(--resume)

# Adım 1: Generation
evalhub gen "${GEN_ARGS[@]}"

# Adım 2: Evaluation
evalhub eval \
    --tasks "math_judge" \
    --solutions "$OUT_DIR/math_judge.jsonl" \
    --output-dir "$OUT_DIR"

# Adım 3: View
evalhub view \
    --results "$OUT_DIR/math_judge_results.jsonl" \
    --max-display -1

# Sunucuyu Kapat
kill $SERVER_PID
sleep 5