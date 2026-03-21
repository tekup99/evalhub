import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract correct solutions from benchmark results.")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    parser.add_argument("--suffix", type=str, default="", help="Directory suffix")
    parser.add_argument("--max_tasks", type=int, default=None, help="Maximum number of tasks to process")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    base_dir = Path(f"results/{args.model}/{args.benchmark}{args.suffix}")
    results_file = base_dir / f"{args.benchmark}{args.suffix}_results.jsonl"
    raw_file = base_dir / f"{args.benchmark}{args.suffix}_raw.jsonl"
    
    out_dir = Path(f"data/passatk_filtered/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f"{args.benchmark}{args.suffix}_corrects.jsonl"

    if not results_file.exists() or not raw_file.exists():
        logger.error("Missing required input files in %s", base_dir)
        return

    correct_indices: Dict[str, List[int]] = defaultdict(list)
    ground_truths: Dict[str, str] = {}
    generated_solutions: Dict[str, Dict[int, str]] = defaultdict(dict)

    with results_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_tasks and i >= args.max_tasks:
                break
            
            data = json.loads(line)
            task_id = data.get("task_id")
            if not task_id:
                continue

            ground_truths[task_id] = data.get("ground_truth", "")
            for idx, is_correct in enumerate(data.get("correct", [])):
                if is_correct:
                    correct_indices[task_id].append(idx)
                    solutions = data.get("solutions", [])
                    generated_solutions[task_id][idx] = solutions[idx] if idx < len(solutions) else ""

    current_task_indices: Dict[str, int] = defaultdict(int)
    saved_count = 0
    
    with raw_file.open("r", encoding="utf-8") as f_raw, output_file.open("w", encoding="utf-8") as f_out:
        for line in f_raw:
            data = json.loads(line)
            task_id = data.get("task_id")
            
            if not task_id or task_id not in correct_indices:
                continue
                
            current_idx = current_task_indices[task_id]
            if current_idx in correct_indices[task_id]:
                output_data = {
                    "task_id": f"{task_id}_gen_{current_idx}",
                    "original_task_id": task_id,
                    "generation_idx": current_idx,
                    "ground_truth": ground_truths.get(task_id, ""),
                    "generated_answer": generated_solutions[task_id].get(current_idx, ""),
                    "raw_response": data.get("response", {})
                }
                f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                saved_count += 1
                
            current_task_indices[task_id] += 1

    logger.info("Extraction complete. Processed %d valid entries.", saved_count)

if __name__ == "__main__":
    main()
