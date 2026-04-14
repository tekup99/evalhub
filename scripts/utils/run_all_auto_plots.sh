#!/bin/bash
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
eval "$(conda shell.bash hook)"
conda activate evalhub_env

# Ana klasör yolları
RESULTS_DIR="results"
PLOTS_DIR="results/plots/auto_comparisons"

# Plot klasörünü oluştur
mkdir -p "$PLOTS_DIR"

echo "[INFO] 'results' klasoru taranıyor, otomatik plotlar olusturulacak..."
echo "================================================================"

# results/ içindeki tüm summary dosyalarını bul
# -type d \( -name "judgments" -o -name "plots*" \) -prune : judgments ve plots klasörlerinin İÇİNE GİRME
# -o -type f -name "*summary.json" -print : summary.json ile biten dosyaları yazdır
find "$RESULTS_DIR" -type d \( -name "judgments" -o -name "plots*" \) -prune -o -type f -name "*summary.json" -print | while read -r base_summary; do
    
    # Örnek path: results/Qwen2.5-32B_t0.6_max16384/aime2026/aime2026_summary.json
    
    # Path'i bölerek model klasörü adını ve benchmark klasörü adını çıkarıyoruz
    model_dir=$(echo "$base_summary" | awk -F'/' '{print $(NF-2)}')
    benchmark_dir=$(echo "$base_summary" | awk -F'/' '{print $(NF-1)}')
    
    echo "[PROCESS] Bulunan Base: $model_dir"
    echo "[PROCESS] Benchmark   : $benchmark_dir"
    
    # Otomatik başlık ve çıktı yolu oluşturma
    title="${model_dir}"
    subtitle="Benchmark: ${benchmark_dir} | Base vs Judgments"
    out_file="${PLOTS_DIR}/${model_dir}_${benchmark_dir}_comparison.png"
    
    # Python scriptini çağır
    python scripts/utils/plot_custom_group.py \
        --base_summary "$base_summary" \
        --title "$title" \
        --subtitle "$subtitle" \
        --out "$out_file"
        
    echo "----------------------------------------------------------------"
done

echo "[SUCCESS] Islem tamamlandi! Tum plotlar suraya kaydedildi: $PLOTS_DIR"