import json
import logging
import argparse
from evalhub.benchmarks.registry import DATASET_MAP
from typing import Dict

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsamples a dataset and generates solution files.")
    parser.add_argument("--task", required=True, help="Task name (e.g., aime2026)")
    parser.add_argument("--input", required=True, help="Input raw JSONL file")
    parser.add_argument("--out_raw", required=True, help="Output downsampled raw JSONL")
    parser.add_argument("--out_sol", required=True, help="Output extracted solutions JSONL")
    parser.add_argument("--n", type=int, default=256, help="Maximum samples per task ID")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    if args.task not in DATASET_MAP:
        logging.error(f"Task '{args.task}' not found in DATASET_MAP.")
        return

    dataset_class = DATASET_MAP[args.task]()
    task_counts: Dict[str, int] = {}
    processed_count = 0

    logging.info(f"Starting downsampling for task: {args.task} (Max samples: {args.n})")

    with open(args.input, 'r', encoding='utf-8') as file_in, \
         open(args.out_raw, 'w', encoding='utf-8') as file_raw, \
         open(args.out_sol, 'w', encoding='utf-8') as file_sol:
        
        for line_num, line in enumerate(file_in, 1):
            if not line.strip(): continue
            
            try:
                data = json.loads(line)
                task_id = data["task_id"]
                task_counts[task_id] = task_counts.get(task_id, 0)
                
                if task_counts[task_id] < args.n:
                    file_raw.write(json.dumps(data, ensure_ascii=False) + "\n")
                    
                    response_data = data.get("response", {})
                    content = response_data.get("content") or \
                              response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                              
                    solution_data = {
                        "task_id": task_id,
                        "solution": dataset_class.extract_solution(task_id, content)
                    }
                    file_sol.write(json.dumps(solution_data, ensure_ascii=False) + "\n")
                    
                    task_counts[task_id] += 1
                    processed_count += 1
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON at line {line_num}")
            except KeyError as e:
                logging.warning(f"Missing expected key {e} at line {line_num}")

    logging.info(f"Downsampling complete. Wrote {processed_count} lines to outputs.")

if __name__ == "__main__":
    main()