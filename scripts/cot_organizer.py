import json
import os
from collections import defaultdict

# ==============================================================
# AYARLAR
# ==============================================================
# evalhub kök dizininden çalıştırıldığını varsayıyoruz:
klasor_yolu = "results/Qwen3.5-4B/aime2025"
veri_seti_adi = "aime2025"

results_file = os.path.join(klasor_yolu, f"{veri_seti_adi}_results.jsonl")
raw_file = os.path.join(klasor_yolu, f"{veri_seti_adi}_raw.jsonl")
output_file = os.path.join(klasor_yolu, f"{veri_seti_adi}_correct_filtered_1024.jsonl")

# İlk 1024 soruyu (görevi) işleyecek sınır
MAX_TASKS_TO_PROCESS = 1024

correct_indices = defaultdict(list)
ground_truths = {}
generated_solutions = defaultdict(dict)
task_counter = 0

print("="*60)
print(f"--- ADIM 1: İlk {MAX_TASKS_TO_PROCESS} Görev İçin Doğruları Tespit Etme ---")
print(f"Okunan dosya: {results_file}")
print("="*60)

try:
    with open(results_file, "r", encoding="utf-8") as f_res:
        for line in f_res:
            if task_counter >= MAX_TASKS_TO_PROCESS:
                break
            
            data = json.loads(line)
            task_id = data.get("task_id")
            
            # Gerçek cevabı (ground truth) hafızaya alıyoruz
            ground_truths[task_id] = data.get("ground_truth", "")
            
            for idx, is_correct in enumerate(data.get("correct", [])):
                if is_correct:
                    # Doğru cevabın indeksini kaydet
                    correct_indices[task_id].append(idx)
                    
                    # Modelin ürettiği (extracted) cevabı hafızaya al
                    if "solutions" in data and len(data["solutions"]) > idx:
                        generated_solutions[task_id][idx] = data["solutions"][idx]
                    else:
                        generated_solutions[task_id][idx] = ""
            
            task_counter += 1
            
    print(f"[BİLGİ] Adım 1 tamamlandı. Toplam {task_counter} görev incelendi.")
    
except FileNotFoundError:
    print(f"[HATA] {results_file} dosyası bulunamadı! Lütfen yolları kontrol edin.")
    exit(1)


print("\n" + "="*60)
print("--- ADIM 2: Ham (Raw) Verileri Okuma ve Filtreleme ---")
print(f"Okunan dosya: {raw_file}")
print("="*60)

current_task_indices = defaultdict(int)
saved_count = 0

try:
    with open(raw_file, "r", encoding="utf-8") as f_raw, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_raw:
            data = json.loads(line)
            task_id = data.get("task_id")
            
            # Eğer task_id, Adım 1'de listeye aldığımız ilk 1024 görev içinde değilse veya hiç doğrusu yoksa atla
            if task_id not in correct_indices:
                continue
                
            current_idx = current_task_indices[task_id]
            
            # Eğer okuduğumuz bu indeks, doğru (True) olarak işaretlenmişse dosyaya yaz
            if current_idx in correct_indices[task_id]:
                output_data = {
                    "task_id": task_id,
                    "ground_truth": ground_truths.get(task_id, ""),
                    "generated_answer": generated_solutions[task_id].get(current_idx, ""),
                    "raw_response": data.get("response", {})
                }
                
                f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                saved_count += 1
                
            current_task_indices[task_id] += 1
            
except FileNotFoundError:
    print(f"[HATA] {raw_file} dosyası bulunamadı! Lütfen yolları kontrol edin.")
    exit(1)

print("\n" + "="*60)
print("--- İŞLEM BAŞARIYLA TAMAMLANDI ---")
print("="*60)
print(f"İncelenen maksimum görev sayısı : {MAX_TASKS_TO_PROCESS}")
print(f"Toplam kaydedilen DOĞRU cevap   : {saved_count}")
print(f"Filtrelenmiş yeni dosya         : {output_file}")