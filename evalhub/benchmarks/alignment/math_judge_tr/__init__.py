import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset, DATASET_MAP
from evalhub.benchmarks.math.verifier.rllm import extract_boxed_answer
from evalhub.utils.logger import logger
from evalhub.utils.metrics import compute_pass_at_k, get_majority_vote
from evalhub.utils.pbar import get_progress_bar

MATH_JUDGE_TR = "math_judge_tr"

# Türkçe değerlendirme istemi
JUDGE_PROMPT_TEMPLATE_TR = """Sen matematik ve mantıksal akıl yürütme konusunda bir uzmansın. Görevin, verilen bir matematik problemine ait çözümün doğruluğunu değerlendirmektir. Bunu yaparken sadece nihai sonuca değil, **akıl yürütme sürecine de güçlü bir vurgu yapmalısın**.

Aşağıda **Problem** ve **(Başka bir yapay zeka modeli tarafından sunulan) Çözüm** bulunuyor:

—

**Problem**:

{question}

**(Başka bir yapay zeka modeli tarafından sunulan) Çözüm**:

{solution}

—

Lütfen aşağıdaki görevleri yerine getir:

1. Şunlara çok dikkat ederek **çözümü adım adım analiz et**:
   - Hesaplama doğruluğu
   - Mantıksal tutarlılık
   - Kavramsal anlayış
   - Akıl yürütmenin geçerli ve eksiksiz olup olmadığı
2. Nihai sonuç doğru olsa bile, **akıl yürütmedeki her türlü sorunu veya hatayı tespit et**. Bunları aşağıdaki kategorilere ayır (eğer geçerliyse):
   - **Hesaplama Hatası**: Aritmetik, cebirsel işlemler veya sayısal hesaplamalardaki yanlışlar.
   - **Mantıksal Hata**: Geçersiz akıl yürütme, kusurlu mantık veya yanlış çıkarım.
   - **Kavramsal Hata**: Matematiksel kavramların veya tanımların yanlış anlaşılması veya yanlış kullanılması.
   - **Eksiklik / Noksanlık**: Eksik adımlar, yetersiz gerekçelendirme veya sorunun tüm kısımlarını ele almama.
   - **Diğer**: Yukarıdaki kategorilere uymayan diğer her türlü hata.
3. Çözümün mantıksal olarak sağlam ve akıl yürütme açısından hatasız olup olmadığına dair **nihai bir karara var**.

Lütfen yanıtını aşağıdaki gibi biçimlendir:

—

**Tespit Edilen Sorunlar:**

- [Sorun 1]: [Sınıflandırma] - [Kısa açıklama]
- [Sorun 2]: [Sınıflandırma] - [Kısa açıklama]
- ...

Adım adım düşün ve nihai kararını \boxed{{yes}} veya \boxed{{no}} formatında belirt."""

MATH_JUDGE_TR_META_DATA = {
    "file_path": "" 
}

DEFAULT_KS = [2**i for i in range(11)]

@register_dataset((MATH_JUDGE_TR, "local/math_judge_tr", True))
class MathJudgeTRDataset(MathDataset):
    """LLM-as-a-Judge modeli için Türkçe değerlendirme veri seti sınıfı."""

    def __init__(self, name: str = MATH_JUDGE_TR, meta_data: dict[str, Any] = None, **kwargs):
        if meta_data is None:
            meta_data = MATH_JUDGE_TR_META_DATA.copy()
            
        file_path = meta_data.get("file_path")
        if file_path and os.path.exists(file_path):
            meta_data["file_mtime"] = os.path.getmtime(file_path)
            
        super().__init__(name, meta_data=meta_data, **kwargs)

    def load_tasks(self) -> None:
        file_path = self.meta_data.get("file_path")
        if not file_path or not os.path.exists(file_path):
            logger.error(f"File not found or not provided: {file_path}. Please provide a valid file_path via --override-args.")
            return

        loaded_benchmarks = {}

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                
                orig_id = item.get("original_task_id") 
                
                if orig_id:
                    benchmark_name = orig_id.split('/')[0].lower()
                    
                    if benchmark_name not in loaded_benchmarks:
                        try:
                            dataset_cls = DATASET_MAP.get(benchmark_name)
                            if dataset_cls is None:
                                raise ValueError(f"'{benchmark_name}' was not found in DATASET_MAP.")
                                
                            benchmark_ds = dataset_cls()
                            benchmark_ds.load_tasks()
                            loaded_benchmarks[benchmark_name] = benchmark_ds
                        except Exception as e:
                            logger.error(f"Failed to load benchmark {benchmark_name}: {e}")
                            loaded_benchmarks[benchmark_name] = None
                    
                    benchmark_ds = loaded_benchmarks.get(benchmark_name)
                    if benchmark_ds:
                        original_task = benchmark_ds.tasks.get(orig_id)
                        
                        if not original_task:
                            for k, v in benchmark_ds.tasks.items():
                                if str(k).lower() == str(orig_id).lower():
                                    original_task = v
                                    break
                        
                        if original_task:
                            item["question"] = original_task.prompt
                        else:
                            item["question"] = "MISSING_QUESTION_IN_DATA"
                            logger.warning(f"Could not find original task for {orig_id} in {benchmark_name}")
                    else:
                        item["question"] = "MISSING_QUESTION_IN_DATA"
                else:
                    item["question"] = "MISSING_QUESTION_IN_DATA"

                task = Task(
                    task_id=item["task_id"],
                    prompt=self.format_prompt(item),
                    metadata={
                        "original_task_id": item.get("original_task_id"),
                        "generation_idx": item.get("generation_idx")
                    }
                )
                
                groundtruth = GroundTruth(
                    task_id=item["task_id"],
                    answer="evet" 
                )
                
                self.add_task(task)
                self.add_groundtruth(groundtruth)

    def format_prompt(self, item: dict[str, Any]) -> str:
        question = item.get("question", "MISSING_QUESTION_IN_DATA")
        
        try:
            solution = item["raw_response"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            solution = item.get("generated_answer", "")

        return JUDGE_PROMPT_TEMPLATE_TR.format(
            question=question,
            solution=solution
        )

    def extract_solution(self, task_id: str, response: str) -> str:
        answer = extract_boxed_answer(response)
        if answer:
            return answer.lower().strip()
        return "invalid_format"
        
    def check_correct(self, extracted_answer: str | None, ground_truth: str, task_id: str = None) -> bool:
        if extracted_answer is None:
            return False
        return extracted_answer in ["evet", "yes"]

    def evaluate(self, solution: str | os.PathLike, output_dir: str | os.PathLike) -> None:
        """Override edilmiş evaluate metodu: evet, yes ve total için ayrı pass@k hesaplar."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        id2solutions = self._load_solutions(solution)
        if not id2solutions:
            logger.warning("Değerlendirilecek çözüm bulunamadı.")
            return

        assert len(id2solutions) == len(self.groundtruth), (
            f"Predictions ({len(id2solutions)}) must match groundtruths ({len(self.groundtruth)})"
        )

        results, correct_total, total = [], 0, len(id2solutions)
        progress = get_progress_bar()
        
        with progress:
            eval_task = progress.add_task("[bold blue]Evaluating Judge (TR)", total=total)

            for task_id, solutions in id2solutions.items():
                
                # Ayrı ayrı doğruluk matrisleri oluşturuluyor
                is_correct_evet = [sol == "evet" for sol in solutions]
                is_correct_yes = [sol == "yes" for sol in solutions]
                is_correct_total = [sol in ["evet", "yes"] for sol in solutions]

                n_solutions = len(solutions)

                # Ayrı ayrı pass@k sözlükleri
                pass_at_k_evet = defaultdict(float)
                pass_at_k_yes = defaultdict(float)
                pass_at_k_total = defaultdict(float)

                for k in DEFAULT_KS:
                    if k > n_solutions:
                        continue
                    pass_at_k_evet[str(k)] = compute_pass_at_k(n_solutions, sum(is_correct_evet), k)
                    pass_at_k_yes[str(k)] = compute_pass_at_k(n_solutions, sum(is_correct_yes), k)
                    pass_at_k_total[str(k)] = compute_pass_at_k(n_solutions, sum(is_correct_total), k)

                # Çoğunluk oyu (Majority Vote) hesaplaması
                majority_vote = get_majority_vote(solutions)
                is_correct_majority = majority_vote in ["evet", "yes"]

                result = {
                    "task_id": task_id,
                    "solutions": solutions,
                    "ground_truth": "evet/yes",
                    "correct_evet": is_correct_evet,
                    "correct_yes": is_correct_yes,
                    "correct_total": is_correct_total,
                    "pass_at_k_evet": pass_at_k_evet,
                    "pass_at_k_yes": pass_at_k_yes,
                    "pass_at_k_total": pass_at_k_total,
                    "pass_at_k": pass_at_k_total, # Pipeline'ın kırılmaması için geriye dönük uyumluluk
                    "majority_vote": majority_vote,
                    "is_correct_majority": is_correct_majority,
                }

                progress.update(eval_task, advance=1)
                results.append(result)
                correct_total += int(is_correct_majority)

        # Tüm görevler bittikten sonra metriklerin agregasyonu
        agg_pass_at_k_evet = {k: sum(r["pass_at_k_evet"].get(k, 0) for r in results) / total for k in results[0]["pass_at_k_evet"]}
        agg_pass_at_k_yes = {k: sum(r["pass_at_k_yes"].get(k, 0) for r in results) / total for k in results[0]["pass_at_k_yes"]}
        agg_pass_at_k_total = {k: sum(r["pass_at_k_total"].get(k, 0) for r in results) / total for k in results[0]["pass_at_k_total"]}
        cons_at_k = correct_total / total

        # Terminal Çıktıları
        logger.info("\n--- Pass@K Total (Evet + Yes) ---")
        for k, value in agg_pass_at_k_total.items():
            logger.info(f"Pass@{k} (Total): {value:.2%}")
            
        logger.info("\n--- Pass@K (Sadece 'Evet') ---")
        for k, value in agg_pass_at_k_evet.items():
            logger.info(f"Pass@{k} (Evet): {value:.2%}")

        logger.info("\n--- Pass@K (Sadece 'Yes') ---")
        for k, value in agg_pass_at_k_yes.items():
            logger.info(f"Pass@{k} (Yes): {value:.2%}")

        logger.info(f"\nCons@{len(results[0]['solutions'])} (Total): {cons_at_k:.2%}")

        # Detaylı JSONL kaydı
        result_path = output_dir / f"{self.name}_results.jsonl"
        with open(result_path, "wb") as f:
            for result in results:
                try:
                    f.write(orjson.dumps(result) + b"\n")
                except Exception as e:
                    logger.error(f"Error dumping result: {result.keys()}")
                    logger.error(f"Error: {e}")
                    exit(1)
        logger.info(f"Evaluation results saved to {result_path}")

        # Özet JSON kaydı
        summary_path = output_dir / f"{self.name}_summary.json"
        with open(summary_path, "wb") as f:
            f.write(orjson.dumps({
                "pass_at_k_evet": agg_pass_at_k_evet,
                "pass_at_k_yes": agg_pass_at_k_yes,
                "pass_at_k_total": agg_pass_at_k_total,
                "pass_at_k": agg_pass_at_k_total,  # Post-process adımları için uyumluluk
                "cons_at_k": cons_at_k
            }))
        logger.info(f"Evaluation summary saved to {summary_path}")