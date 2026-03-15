#!/bin/bash
#SBATCH --job-name=apply_judge_results    # İşin adı

#SBATCH --output=logs/%x-%j.out     # Standart çıktı logları
#SBATCH -e logs/%x-%j.err           # Hata logları
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=1           
#SBATCH --mem=4G                    
#SBATCH --time=00:30:00   

eval "$(conda shell.bash hook)"
conda activate evalhub_env


BASE_DIR="results"
JUDGMENT_DIR="results/judgments"

BASE_MODELS=(
    "Qwen3.5-4B-Base"
    "Qwen3.5-4B"

)

JUDGE_MODELS=(
    "Qwen3.5-35B-A3B"
)

BENCHMARKS=(
    "aime2025"
    "aime2026"
)

echo "Jüri sonuçları orijinal metrik dosyalarına entegre ediliyor ve summary çıkarılıyor..."
echo "--------------------------------------------------------"

for BASE_MODEL in "${BASE_MODELS[@]}"; do
    for JUDGE_MODEL in "${JUDGE_MODELS[@]}"; do
        for BENCH in "${BENCHMARKS[@]}"; do
            
            # 1. Base modelin orijinal result jsonl'si
            BASE_RESULTS_FILE="${BASE_DIR}/${BASE_MODEL}/${BENCH}/${BENCH}_results.jsonl"
            
            # 2. Önceden oluşturduğumuz majority correct dosyası
            TARGET_DIR="${JUDGMENT_DIR}/${BASE_MODEL}_evaluated_by_${JUDGE_MODEL}/${BENCH}"
            JUDGE_MAJORITY_FILE="${TARGET_DIR}/math_judge_majority.jsonl"
            
            # 3. Yeniden hesaplanmış metrikler ve summary dosyası
            OUTPUT_FILE="${TARGET_DIR}/${BENCH}_results_judged.jsonl"
            SUMMARY_FILE="${TARGET_DIR}/summary.json"
            
            if [[ -f "$BASE_RESULTS_FILE" ]] && [[ -f "$JUDGE_MAJORITY_FILE" ]]; then
                echo "[İŞLENİYOR] Model: $BASE_MODEL | Bench: $BENCH"
                
                PYTHONPATH=. python scripts/data_prep/apply_judge_results.py \
                    --base_results_file "$BASE_RESULTS_FILE" \
                    --judge_majority_file "$JUDGE_MAJORITY_FILE" \
                    --output_file "$OUTPUT_FILE" \
                    --summary_file "$SUMMARY_FILE"
            else
                echo "[ATLANDI] Eksik dosya var: ${BASE_MODEL} - ${BENCH}"
            fi
            
        done
    done
done

echo "--------------------------------------------------------"
echo "Tüm entegrasyon ve yeniden hesaplama işlemleri tamamlandı!"