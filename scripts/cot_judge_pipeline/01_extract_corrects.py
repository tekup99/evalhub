import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

# Configure professional logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract correct solutions from benchmark results.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., Qwen3.5-4B-Base)")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name (e.g., aime2026)")
    parser.add_argument("--suffix", type=str, default="", help="Directory suffix (e.g., _64 or empty)")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # Dynamic input paths
    base_dir = Path(f"results/{args.model}/{args.benchmark}{args.suffix}")
    results_file = base_dir / f"{args.benchmark}{args.suffix}_results.jsonl"
    raw_file = base_dir / f"{args.benchmark}{args.suffix}_raw.jsonl"

    # Dynamic output path
    out_dir = Path(f"data/passatk_filtered/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = out_dir / f"{args.benchmark}{args.suffix}_corrects.jsonl"

    if not results_file.exists():
        logging.error(f"Results file not found: {results_file}")
        return

    correct_indices = defaultdict(list)
    ground_truths = {}
    generated_solutions = defaultdict(dict)

    # Step 1: Identify which generations are correct
    with results_file.open("r", encoding="utf-8") as f:
        for line in f:
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

    if not raw_file.exists():
        logging.error(f"Raw file not found: {raw_file}")
        return

    current_task_indices = defaultdict(int)
    
    # Step 2: Extract raw responses for correct generations only
    with raw_file.open("r", encoding="utf-8") as f_raw, output_file.open("w", encoding="utf-8") as f_out:
        for line in f_raw:
            data = json.loads(line)
            task_id = data.get("task_id")
            
            if task_id not in correct_indices:
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
                
            current_task_indices[task_id] += 1

    logging.info(f"Extraction complete. Filtered output saved to {output_file}")

if __name__ == "__main__":
    main()