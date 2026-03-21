import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyzes generation statistics from EvalHub results.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., Qwen3.5-4B-Base)")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name (e.g., aime2026)")
    parser.add_argument("--base_dir", type=str, default="results", help="Base directory for results")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    results_dir = Path(args.base_dir) / args.model / args.benchmark
    input_file = results_dir / f"{args.benchmark}_results.jsonl"
    output_file = results_dir / f"{args.benchmark}_generation_stats.json"

    if not input_file.exists():
        logging.error(f"Input file not found: {input_file}")
        return

    stats_list: List[Dict[str, Any]] = []
    logging.info(f"Analyzing: {input_file}")
    logging.info(f"{'Task ID':<25} | {'Generated (k)':<15} | {'Correct':<10} | {'Wrong':<10}")
    logging.info("-" * 70)

    with input_file.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                task_id = data.get("task_id", "Unknown")
                correct_list = data.get("correct", [])
                
                generated_k = len(correct_list)
                correct_answers = sum(1 for x in correct_list if x is True)
                wrong_answers = generated_k - correct_answers
                
                logging.info(f"{task_id:<25} | {generated_k:<15} | {correct_answers:<10} | {wrong_answers:<10}")
                
                stats_list.append({
                    "task_id": task_id,
                    "generated_answers_k": generated_k,
                    "correct_answers": correct_answers,
                    "wrong_answers": wrong_answers
                })
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON at line {line_num}")

    with output_file.open('w', encoding='utf-8') as f_out:
        json.dump(stats_list, f_out, indent=4, ensure_ascii=False)
        
    logging.info(f"Analysis complete. Processed {len(stats_list)} tasks. Saved to {output_file}")

if __name__ == "__main__":
    main()