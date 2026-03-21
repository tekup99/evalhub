#!/bin/bash
#SBATCH --job-name=aggregate_majority    # İşin adı

#SBATCH --output=logs/%x-%j.out     # Standart çıktı logları
#SBATCH -e logs/%x-%j.err           # Hata logları
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=1           
#SBATCH --mem=4G                    
#SBATCH --time=00:30:00   
# Projenin kök dizininden (evalhub/) çalıştırılmalıdır.
JUDGMENT_DIR="results/judgments"

# Değerlendirilen (Base) modellerin listesi
BASE_MODELS=(
    "Qwen3.5-4B-Base"
    "Qwen3.5-4B"


)

# Değerlendiren (Judge) modellerin listesi
JUDGE_MODELS=(
    "Qwen3.5-35B-A3B"
)

# Benchmark listesi
BENCHMARKS=(
    "aime2026"
    "aime2025"

)

INPUT_FILENAME="math_judge.jsonl"
OUTPUT_FILENAME="math_judge_majority.jsonl"

echo "Çoğunluk oylaması (majority vote) hesaplamaları başlatılıyor..."
echo "--------------------------------------------------------"

for BASE_MODEL in "${BASE_MODELS[@]}"; do
    for JUDGE_MODEL in "${JUDGE_MODELS[@]}"; do
        for BENCH in "${BENCHMARKS[@]}"; do
            
            # evalhub/results/judgments/Qwen3.5-4B-Base_evaluated_by_Qwen3.5-35B-A3B/aime2026/ vb.
            TARGET_DIR="${JUDGMENT_DIR}/${BASE_MODEL}_evaluated_by_${JUDGE_MODEL}/${BENCH}"
            INPUT_FILE="${TARGET_DIR}/${INPUT_FILENAME}"
            OUTPUT_FILE="${TARGET_DIR}/${OUTPUT_FILENAME}"
            INPUT_FILENAME="math_judge.jsonl"
            OUTPUT_FILENAME="math_judge_majority.jsonl"

            if [ -f "$INPUT_FILE" ]; then
                echo "[İŞLENİYOR] Base: ${BASE_MODEL} | Judge: ${JUDGE_MODEL} | Bench: ${BENCH}"
                
                # Python scriptini çağır
                python scripts/cot_judge_pipeline/04_aggregate_votes.py \
                    --input_file "$INPUT_FILE" \
                    --output_file "$OUTPUT_FILE"
            else
                echo "[BULUNAMADI] Yol: $INPUT_FILE - Atlanıyor."
            fi
            
        done
    done
done

echo "--------------------------------------------------------"
echo "Tüm işlemler başarıyla tamamlandı!"

# --- JÜRİ SONUÇLARINI UYGULAMA ADIMI ---


