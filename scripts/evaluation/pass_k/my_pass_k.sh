#!/bin/bash
#SBATCH --job-name=evalhub_run
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task 32
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:4
#SBATCH --nodelist=nscluster

eval "$(conda shell.bash hook)"
conda activate evalhub_env

# load config (Yol güncellendi)
CONFIG_FILE="$SLURM_SUBMIT_DIR/scripts/configs/config_v1.env"
if [ ! -f "$CONFIG_FILE" ]; then
    exit 1
fi
source "$CONFIG_FILE"

# env setup
export HF_TOKEN
export SGLANG_DISABLE_CUDNN_CHECK=1
export HOSTED_VLLM_API_BASE="http://127.0.0.1:${PORT}/v1"
export HOSTED_VLLM_API_KEY="EMPTY"

for HF_MODEL_ID in $MODELS; do
    MODEL_SAFE_NAME=$(basename "$HF_MODEL_ID")
    
    # start server
python -m sglang_router.launch_server \
      --model-path "$HF_MODEL_ID" \
      --host 127.0.0.1 \
      --port $PORT \
      --dp 4 \
      --router-balance-abs-threshold 1 &
    SERVER_PID=$!

    # wait for server
    while ! curl -s http://127.0.0.1:${PORT}/v1/models > /dev/null; do
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

    # cleanup before next model
    kill $SERVER_PID
    sleep 15
done