import json
import argparse
from pathlib import Path
from collections import defaultdict

# Orijinal evalhub yapısındaki hazır metrik fonksiyonları
from evalhub.utils.metrics import compute_pass_at_k, get_majority_vote

def main():
    parser = argparse.ArgumentParser(description="Jüri sonuçlarına göre metrikleri yeniden hesaplar ve summary oluşturur.")
    parser.add_argument("--base_results_file", type=str, required=True, help="Orijinal sonuç dosyası (örn: aime2025_results.jsonl)")
    parser.add_argument("--judge_majority_file", type=str, required=True, help="Jüri çoğunluk oyu dosyası (örn: math_judge_majority.jsonl)")
    parser.add_argument("--output_file", type=str, required=True, help="Yeni üretilecek güncellenmiş JSONL dosyası")
    parser.add_argument("--summary_file", type=str, required=True, help="Genel ortalamaların yazılacağı JSON dosyası (örn: summary.json)")
    
    args = parser.parse_args()
    
    # 1. Jüri sonuçlarını (majority_correct) sözlüğe al
    judge_dict = {}
    with open(args.judge_majority_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            task_id = data.get("task_id")
            majority_correct = data.get("majority_correct", False)
            judge_dict[task_id] = majority_correct

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Summary metrikleri için toplayıcılar
    sum_pass_at_k = defaultdict(float)
    correct_majority_count = 0
    total_tasks = 0
    
    # 2. Base result dosyasını oku, güncelle ve yeni dosyaya yaz
    with open(args.base_results_file, 'r', encoding='utf-8') as f_in, \
         open(args.output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            data = json.loads(line)
            
            task_id = data["task_id"]
            solutions = data["solutions"]  # Orijinal metinlere HİÇ dokunulmayacak
            correct = data["correct"]
            ground_truth = data.get("ground_truth", "")
            
            n = len(correct)
            
            # Her bir generation (üretim) için jüri kararına bak
            for i in range(n):
                gen_id = f"{task_id}_gen_{i}"
                
                if gen_id in judge_dict:
                    is_valid = judge_dict[gen_id]
                    # Sadece orijinalinde True olup, jüri tarafından False'a çevrilmesi gerekenleri değiştir
                    if correct[i] is True and not is_valid:
                        correct[i] = "cot_false"
            
            # 3. Yeni c (doğru sayısı) hesapla
            # Sadece değeri strict olarak 'True' (boolean) olanları sayıyoruz.
            c = sum(1 for x in correct if x is True)
            
            # 4. pass@k değerlerini yeniden hesapla ve summary için topla
            new_pass_at_k = {}
            for k_str in data.get("pass_at_k", {}).keys():
                k_val = int(k_str)
                if k_val <= n:
                    val = compute_pass_at_k(n, c, k_val)
                else:
                    val = 1.0
                
                new_pass_at_k[k_str] = val
                sum_pass_at_k[k_str] += val  # Ortalama hesabı için biriktir
            
            # 5. Yeni çoğunluk oyu (majority vote) hesapla (Orijinal solutions dizisi üzerinden)
            new_majority_vote = get_majority_vote(solutions)
            new_is_correct_majority = (new_majority_vote == ground_truth)
            
            if new_is_correct_majority:
                correct_majority_count += 1
                
            total_tasks += 1
            
            # 6. JSONL verisini güncelle
            data["correct"] = correct
            data["pass_at_k"] = new_pass_at_k
            data["majority_vote"] = new_majority_vote
            data["is_correct_majority"] = new_is_correct_majority
            
            f_out.write(json.dumps(data) + '\n')
            
    # 7. Summary (Ortalama) Değerlerini Hesapla ve Yazdır
    if total_tasks > 0:
        avg_pass_at_k = {k: (v / total_tasks) for k, v in sum_pass_at_k.items()}
        cons_at_k = correct_majority_count / total_tasks
        
        summary_data = {
            "pass_at_k": avg_pass_at_k,
            "cons_at_k": cons_at_k
        }
        
        summary_path = Path(args.summary_file)
        with open(summary_path, 'w', encoding='utf-8') as f_sum:
            json.dump(summary_data, f_sum)
            
        print(f"✅ Güncellenmiş sonuçlar: {args.output_file}")
        print(f"📊 Özet (Summary) raporu kaydedildi: {args.summary_file}")
    else:
        print("Uyarı: İşlenecek hiçbir görev (task) bulunamadı!")

if __name__ == "__main__":
    main()