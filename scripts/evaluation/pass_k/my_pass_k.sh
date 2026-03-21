#!/bin/bash
#SBATCH --job-name=evalhub_run      # Varsayılan iş adı (Dışarıdan sbatch ile ezilebilir)
#SBATCH --output=logs/%x-%j.out     # Çıktı: logs/evalhub_run-123456.out (veya judge_Llama-123456.out)
#SBATCH --error=logs/%x-%j.err      # Hata: logs/evalhub_run-123456.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32  
#SBATCH --mem=256G            
#SBATCH --nodelist=nsdl2
#SBATCH --gres=gpu:h200:1           # 1 adet H200 GPU tahsisi
#SBATCH --time=08:00:00

eval "$(conda shell.bash hook)"
conda activate evalhub_env

# load config
CONFIG_FILE="$SLURM_SUBMIT_DIR/scripts/configs/config_v1.env"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config dosyasi bulunamadi: $CONFIG_FILE"
    exit 1
fi
source "$CONFIG_FILE"

# submit_all.sh tarafından iletilen hedef modeli al
HF_MODEL_ID=$TARGET_MODEL
MODEL_SAFE_NAME=$(basename "$HF_MODEL_ID")

# Port Çakışmasını Önleme (Aynı node'da çalışan birden fazla job için)
DYNAMIC_PORT=$((30000 + SLURM_JOB_ID % 10000))

# env setup
export HF_TOKEN
export SGLANG_DISABLE_CUDNN_CHECK=1
export HOSTED_VLLM_API_BASE="http://127.0.0.1:${DYNAMIC_PORT}/v1"
export HOSTED_VLLM_API_KEY="EMPTY"

# GPU sayısını manuel olarak belirtiyoruz (SLURM parametresiyle aynı olmalı)
NUM_GPUS=2

echo "Model $HF_MODEL_ID icin $NUM_GPUS GPU ile $DYNAMIC_PORT portunda sglang_router baslatiliyor..."

# start server (Çoklu GPU için sglang_router ve --dp parametresi eklendi)
python -m sglang_router.launch_server \
    --model-path "$HF_MODEL_ID" \
    --host 127.0.0.1 \
    --port $DYNAMIC_PORT \
    --dp $NUM_GPUS \
    --router-balance-abs-threshold 1 &
SERVER_PID=$!

# wait for server
while ! curl -s http://127.0.0.1:${DYNAMIC_PORT}/v1/models > /dev/null; do
    sleep 10
done

sleep 60

for TASK in $TASKS; do
    BASE_DIR="$HOME/evalhub/results/$MODEL_SAFE_NAME/$TASK"
    mkdir -p "$BASE_DIR"

    GEN_ARGS=(
        --model "hosted_vllm/$HF_MODEL_ID"
        --tasks "$TASK"
        --output-dir "$BASE_DIR"
        --temperature "$TEMPERATURE"
        --top-p "$TOP_P"
        --max-completion-tokens "$MAX_COMPLETION_TOKENS"
        --frequency-penalty "$FREQUENCY_PENALTY"
        --presence-penalty "$PRESENCE_PENALTY"
        --n-samples "$N_SAMPLES"
        --num-workers "$NUM_WORKERS"
        --timeout "$TIMEOUT"
    )

    # append optional args
    [ -n "$STOP" ] && GEN_ARGS+=(--stop "$STOP")
    [ -n "$SYSTEM_PROMPT" ] && GEN_ARGS+=(--system-prompt "$SYSTEM_PROMPT")
    [ -n "$OVERRIDE_ARGS" ] && GEN_ARGS+=(--override-args "$OVERRIDE_ARGS")
    [ -n "$TOOL_CONFIG" ] && GEN_ARGS+=(--tool-config "$TOOL_CONFIG")
    [ -n "$CALLBACK" ] && GEN_ARGS+=(--callback "$CALLBACK")
    [ "$ENABLE_MULTITURN" = true ] && GEN_ARGS+=(--enable-multiturn --max-turns "$MAX_TURNS")
    [ "$RESUME" = true ] && GEN_ARGS+=(--resume)

    evalhub gen "${GEN_ARGS[@]}"

    evalhub eval \
        --tasks "$TASK" \
        --solutions "$BASE_DIR/$TASK.jsonl" \
        --output-dir "$BASE_DIR"

    evalhub view \
        --results "$BASE_DIR/${TASK}_results.jsonl" \
        --max-display -1
done

# cleanup before exiting
kill $SERVER_PID
sleep 15