#!/bin/bash
#SBATCH --job-name=extract_pass_at_k    # İşin adı
#SBATCH --output=logs/%x-%j.out     # Standart çıktı logları
#SBATCH -e logs/%x-%j.err           # Hata logları
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=1           
#SBATCH --mem=4G                    
#SBATCH --time=00:30:00             

# 1. Adım: Çalışma dizinine (Evalhub kök dizini) geçiş yapın
cd $SLURM_SUBMIT_DIR

# 2. Adım: Sanal ortamı aktif edin
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "[Slurm] Sanal ortam aktif edildi."
fi

# 3. Adım: Çevre değişkenlerini (.env) yükle
ENV_FILE="scripts/data_prep/.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo "[Slurm] .env dosyası başarıyla yüklendi."
else
    echo "[HATA] $ENV_FILE bulunamadı! Lütfen .env dosyasının varlığından emin olun."
    exit 1
fi

echo "============================================================"
echo "[Extractor] PRM için doğru cevaplar filtreleniyor..."
echo "Tanımlı Modeller: $MODELS"
echo "Tanımlı Benchmarklar: $BENCHMARKS"
echo "============================================================"

# 4. Adım: Modeller ve Benchmarklar üzerinde iç içe döngü (loop)
for MODEL in $MODELS; do
    for BENCHMARK in $BENCHMARKS; do
        echo "------------------------------------------------------------"
        echo "-> İşleniyor: Model=$MODEL | Benchmark=$BENCHMARK"
        echo "------------------------------------------------------------"
        
        # Python betiğini çalıştır
        python scripts/data_prep/extract_corrects.py --model "$MODEL" --benchmark "$BENCHMARK"
    done
done

echo "============================================================"
echo "[Extractor] Tüm kombinasyonlar için işlemler tamamlandı!"
echo "============================================================"