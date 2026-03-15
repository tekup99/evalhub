import json
import argparse
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge sonuçlarını her bir spesifik generation için çoğunluk oylaması ile birleştirir.")
    parser.add_argument("--input_file", type=str, required=True, help="Girdi JSONL dosyasının yolu (örn: math_judge.jsonl)")
    parser.add_argument("--output_file", type=str, required=True, help="Çıktı JSONL dosyasının yolu (örn: math_judge_majority.jsonl)")
    
    args = parser.parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    
    if not input_path.exists():
        print(f"Uyarı: {input_path} bulunamadı, atlanıyor.")
        return

    task_groups = defaultdict(list)

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                task_id = data.get("task_id", "")
                solution = data.get("solution", "")
                
                # task_id'yi hiçbir şekilde kırpmadan doğrudan kullanıyoruz
                if task_id:
                    task_groups[task_id].append(solution)
            except json.JSONDecodeError:
                print(f"Hatalı JSON satırı atlandı: {line.strip()}")
                
    # Sonuçları JSONL olarak yazdır
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for task_id, solutions in task_groups.items():
            yes_count = solutions.count("yes")
            
            # Çoğunluk "yes" ise True, değilse (eşitlik durumu dahil) False
            majority_correct = yes_count > (len(solutions) / 2)
            
            output_data = {
                "task_id": task_id,
                "solutions": solutions,
                "majority_correct": majority_correct
            }
            f_out.write(json.dumps(output_data) + '\n')
            
    print(f"Tamamlandı: Toplam {len(task_groups)} benzersiz generation '{output_path.name}' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()