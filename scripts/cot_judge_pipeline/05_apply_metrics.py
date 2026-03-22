import json
import argparse
from pathlib import Path
from collections import defaultdict
import math

# Pass@K formülü
def compute_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_results_file", type=str, required=True)
    parser.add_argument("--judge_majority_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--summary_file", type=str, required=True)
    args = parser.parse_args()
    
    # 1. Majority oylarını oku
    judge_dict = {}
    with open(args.judge_majority_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            judge_dict[data["task_id"]] = data["majority_correct"]

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    sum_pass_at_k = defaultdict(float)
    total_tasks = 0
    
    # 2. Base sonuçları güncelle ve Pass@K hesapla
    with open(args.base_results_file, "r", encoding="utf-8") as f_in, \
         open(args.output_file, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            data = json.loads(line)
            task_id = data["task_id"]
            correct_arr = data.get("correct", [])
            n_generations = len(correct_arr)
            
            # True ama hakem tarafından reddedilenleri "cot_false" yap
            for i in range(n_generations):
                gen_id = f"{task_id}_gen_{i}"
                if correct_arr[i] is True and gen_id in judge_dict:
                    if not judge_dict[gen_id]:
                        correct_arr[i] = "cot_false"
            
            # Geriye kalan GERÇEK True'ları say
            true_count = sum(1 for x in correct_arr if x is True)
            
            # Yeni Pass@K değerlerini hesapla
            new_pass_at_k = {}
            for k_str in data.get("pass_at_k", {}).keys():
                k_val = int(k_str)
                if k_val <= n_generations:
                    val = compute_pass_at_k(n_generations, true_count, k_val)
                else:
                    val = 1.0
                new_pass_at_k[k_str] = val
                sum_pass_at_k[k_str] += val
                
            total_tasks += 1
            data["correct"] = correct_arr
            data["pass_at_k"] = new_pass_at_k
            
            f_out.write(json.dumps(data) + "\n")
            
    # 3. Summary dosyasını oluştur
    if total_tasks > 0:
        summary_data = {
            "pass_at_k": {k: (v / total_tasks) for k, v in sum_pass_at_k.items()}
        }
        with open(args.summary_file, "w", encoding="utf-8") as f_sum:
            json.dump(summary_data, f_sum, indent=4)
        print(f"[BAŞARILI] {args.summary_file} oluşturuldu. Toplam Task: {total_tasks}")

if __name__ == "__main__":
    main()