import json
import os
import argparse
from pathlib import Path

def main():
    # Argümanları tanımla
    parser = argparse.ArgumentParser(description="EvalHub sonuçlarındaki üretim (generation) istatistiklerini analiz eder.")
    parser.add_argument("--model", type=str, required=True, help="Modelin adı (Örn: Qwen3.5-4B-Base)")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark adı (Örn: aime2026)")
    parser.add_argument("--base_dir", type=str, default="/user/home/t.tuna/evalhub/results", help="Sonuçların bulunduğu ana dizin")

    args = parser.parse_args()

    # Dosya yollarını oluştur
    results_dir = Path(args.base_dir) / args.model / args.benchmark
    input_file = results_dir / f"{args.benchmark}_results.jsonl"
    output_file = results_dir / f"{args.benchmark}_generation_stats.json"

    # Dosyanın var olup olmadığını kontrol et
    if not input_file.exists():
        print(f"❌ HATA: Girdi dosyası bulunamadı: {input_file}")
        print("Lütfen model ve benchmark isimlerinin doğru olduğundan emin olun.")
        return

    stats_list = []

    print("=" * 70)
    print(f"📊 Analiz Edilen Dosya: {input_file}")
    print("=" * 70)
    print(f"{'Task ID':<25} | {'Generated (k)':<15} | {'Correct':<10} | {'Wrong':<10}")
    print("-" * 70)

    # JSONL dosyasını satır satır oku
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                task_id = data.get("task_id", "Unknown")
                
                # "correct" listesi her bir üretimin (sample) doğru(True)/yanlış(False) durumunu tutar
                correct_list = data.get("correct", [])
                
                # İstatistikleri hesapla
                generated_k = len(correct_list)
                correct_answers = sum(1 for x in correct_list if x is True)
                wrong_answers = generated_k - correct_answers
                
                # Konsola yazdır
                print(f"{task_id:<25} | {generated_k:<15} | {correct_answers:<10} | {wrong_answers:<10}")
                
                # Listeye ekle
                stats_list.append({
                    "task_id": task_id,
                    "generated_answers_k": generated_k,
                    "correct_answers": correct_answers,
                    "wrong_answers": wrong_answers
                })
            except json.JSONDecodeError:
                print(f"Hatalı JSON satırı atlandı.")

    # Sonuçları yeni JSON dosyasına kaydet
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(stats_list, f_out, indent=4, ensure_ascii=False)
        
    print("=" * 70)
    print(f"✅ İşlem tamamlandı! Toplam {len(stats_list)} görev (task) analiz edildi.")
    print(f"📁 Sonuçlar kaydedildi: {output_file}")

if __name__ == "__main__":
    main()