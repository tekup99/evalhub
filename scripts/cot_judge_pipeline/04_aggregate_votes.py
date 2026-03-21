import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

# Profesyonel loglama ayarları
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregates LLM-as-a-Judge results using majority voting.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file (e.g., math_judge.jsonl)")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file (e.g., math_judge_majority.jsonl)")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        logging.warning(f"Input file not found, skipping: {input_path}")
        return

    task_groups: Dict[str, List[str]] = defaultdict(list)

    logging.info(f"Reading judge responses from: {input_path}")
    with input_path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                task_id = data.get("task_id", "")
                solution = data.get("solution", "")
                
                if task_id:
                    task_groups[task_id].append(solution)
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON at line {line_num}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Calculating majority votes for {len(task_groups)} unique generations...")
    with output_path.open('w', encoding='utf-8') as f_out:
        for task_id, solutions in task_groups.items():
            yes_count = solutions.count("yes")
            majority_correct = yes_count > (len(solutions) / 2)

            output_data = {
                "task_id": task_id,
                "solutions": solutions,
                "majority_correct": majority_correct
            }
            f_out.write(json.dumps(output_data) + '\n')

    logging.info(f"Aggregation complete. Saved to: {output_path.name}")

if __name__ == "__main__":
    main()