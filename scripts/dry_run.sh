#!/bin/bash
#SBATCH --job-name=dry_run
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32  
#SBATCH --mem=30G            
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --nodelist=nscluster
#SBATCH --time=00:30:00
# ==============================================================================
# EvalHub Template & Generation Dry Run Script
# ==============================================================================
set -euo pipefail

# 1. Proje kök dizinini belirle
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Conda aktivasyonu
eval "$(conda shell.bash hook 2>/dev/null || echo '')"
conda activate evalhub_env || echo "[WARNING] Conda environment failed to activate."

CONFIG_FILE="${PROJECT_ROOT}/scripts/vllm.env"
if [[ -f "$CONFIG_FILE" ]]; then
    set -a
    source "$CONFIG_FILE"
    set +a
else
    echo "[WARNING] .env dosyası bulunamadı, script içindeki değişkenler kullanılacak."
fi

export HF_TOKEN="${HF_TOKEN:-}"

# Test edilecek modeller
BASE_MODELS="google/gemma-4-E2B Qwen/Qwen3.5-0.8B-Base mistralai/Ministral-3-3B-Base-2512"
JUDGE_MODELS="Qwen/Qwen3.5-0.8B google/gemma-4-E2B-it  mistralai/Ministral-3-3B-Reasoning-2512"

ALL_MODELS="$BASE_MODELS $JUDGE_MODELS"
TEMPLATE_DIR="${PROJECT_ROOT}/scripts/templates"

# Tekil soruyu çözecek ve formatı basacak geçici Python betiğini oluşturuyoruz
cat << 'EOF' > /tmp/dry_run_evalhub.py
import os
import sys
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_id = sys.argv[1]
template_path = sys.argv[2]
# Test için örnek bir AIME sorusu (Kısa)
question = "Let $x$ and $y$ be real numbers such that $x+y=10$ and $xy=21$. What is the value of $x^2+y^2$?"

print(f"\n{'='*60}")
print(f"🚀 BAŞLATILIYOR: {model_id}")
print(f"📄 TEMPLATE: {template_path}")
print(f"{'='*60}")

try:
    # 1. Tokenizer ve Chat Template Testi
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
    
    with open(template_path, 'r', encoding='utf-8') as f:
        chat_template_str = f.read()

    messages = [{"role": "user", "content": question}]
    
    # Template'i uygula
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        chat_template=chat_template_str, 
        tokenize=False,
        add_generation_prompt=True
    )
    
    print("\n--- 📥 MODELİN GÖRECEĞİ TAM İNPUT (RAW PROMPT) ---")
    print(formatted_prompt)
    print("--------------------------------------------------\n")

    # 2. vLLM ile Dry Run (Hızlı Generation Testi)
    print("⏳ VRAM'e Yükleniyor ve Generation Başlıyor (Max 50 Token)...")
    # H200 için tek GPU'da sorunsuz çalışması adına tensor_parallel_size=1 kullanıyoruz. 35B modeller H200'ün 141GB belleğine sığacaktır.
    llm = LLM(model=model_id, trust_remote_code=True, tensor_parallel_size=1, max_model_len=4096)
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
    outputs = llm.generate([formatted_prompt], sampling_params, use_tqdm=False)
    
    print("\n--- 📤 MODELİN ÇIKTISI (GENERATION) ---")
    print(outputs[0].outputs[0].text)
    print("---------------------------------------\n")
    
except Exception as e:
    print(f"\n[HATA] {model_id} işlenirken bir sorun oluştu:")
    print(e)
EOF

# Modellerin üzerinde döngüye girip Python scriptini çağırıyoruz
# (Her model bitişinde Python kapandığı için VRAM otomatik temizlenecektir)
for MODEL in $ALL_MODELS; do
    model_lower=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
    
    if [[ "$model_lower" == *"gemma-4"* ]]; then
        template_file="${TEMPLATE_DIR}/gemma4.jinja"
    elif [[ "$model_lower" == *"ministral"* ]]; then
        template_file="${TEMPLATE_DIR}/ministral3.jinja"
    elif [[ "$model_lower" == *"qwen"* ]]; then
        template_file="${TEMPLATE_DIR}/qwen3.5.jinja"
    else
        echo "[HATA] Desteklenmeyen model tespit edildi, dry run atlanıyor: $MODEL"
        continue
    fi

    if [[ ! -f "$template_file" ]]; then
        echo "[HATA] Template bulunamadı: $template_file"
        continue
    fi

    # Her modeli ayrı bir proses olarak çağır (OOM riskini sıfırlamak için)
    python /tmp/dry_run_evalhub.py "$MODEL" "$template_file"
    
    echo "🧹 VRAM Temizleniyor... Sıradaki modele geçiliyor."
    sleep 2
done

# Geçici dosyayı temizle
rm -f /tmp/dry_run_evalhub.py
echo "✅ Dry run testi başarıyla tamamlandı!"