import os
from evalhub.benchmarks.alignment.math_judge import MathJudgeDataset

def test_dynamic_pipeline():
    print("1. Dinamik Veri Seti Başlatılıyor...")
    
    # Test için kullanılacak örnek dosya (sisteminde var olan bir jsonl dosyasını belirt)
    test_file = "data/passatk_filtered/Qwen3.5-4B/aime2025_corrects.jsonl"
    
    if not os.path.exists(test_file):
        print(f"❌ HATA: Test dosyası bulunamadı: {test_file}")
        print("Lütfen dosya yolunun doğru olduğundan emin ol.")
        return

    test_meta = {
        "file_path": test_file
    }
    
    try:
        dataset = MathJudgeDataset(meta_data=test_meta)
        print("Veriler ve orijinal benchmarklar (sorular) yükleniyor, bu işlem 3-5 saniye sürebilir...")
        dataset.load_tasks()
        
        task_count = len(dataset.tasks)
        print(f"\n✅ BAŞARILI! Toplam {task_count} görev (soru) dinamik olarak yüklendi.\n")
        
        print("2. İlk Sorunun Prompt Formatı Test Ediliyor:")
        print("-" * 50)
        
        # Sözlük (dict) formatından ilk elemanı güvenli şekilde çekme
        if isinstance(dataset.tasks, dict):
            first_task = list(dataset.tasks.values())[0]
        else:
            first_task = dataset.tasks[0]
            
        prompt_text = first_task.prompt
        # Orijinal sorunun ve çözümün başını görmek için ilk 800 karakteri yazdıralım
        print(prompt_text[:800] + "\n\n... [PROMPT DEVAM EDİYOR] ...")
        print("-" * 50)
        
        print("\n3. Dinamik Soru Eşleşme Kontrolü:")
        if "MISSING_QUESTION_IN_DATA" in prompt_text:
            print("❌ BAŞARISIZ: Soru Evalhub'dan dinamik olarak çekilemedi! Lütfen ID formatlarını kontrol et.")
        else:
            print("✅ BAŞARILI: Orijinal soru Evalhub'dan dinamik olarak çekilip Judge promptuna kusursuzca yerleştirilmiş!")

        print("\n4. Metadata Test Ediliyor:")
        print(first_task.metadata)
        
        print("\n5. Regex (Cevap Çıkarma) Test Ediliyor:")
        dummy_model_response = "Here is my step-by-step reasoning... Therefore, the final answer is \\boxed{no}."
        extracted = dataset.extract_solution("dummy_id", dummy_model_response)
        print(f"Modelin cevabından çıkarılan sonuç: '{extracted}' (Beklenen: 'no')")
        
        if extracted in ["yes", "no"]:
            print("\n🎉 TÜM DİNAMİK PYTHON MANTIĞI KUSURSUZ ÇALIŞIYOR!")
        else:
            print("\n❌ Regex kısmında bir sorun var!")
            
    except Exception as e:
        import traceback
        print(f"\n❌ HATA OLUŞTU:\n")
        traceback.print_exc()

if __name__ == "__main__":
    test_dynamic_pipeline()