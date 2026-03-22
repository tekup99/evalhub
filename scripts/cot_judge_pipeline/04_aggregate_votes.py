import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return

    task_groups: Dict[str, List[str]] = defaultdict(list)

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if task_id := data.get("task_id"):
                    sol = data.get("solution", "")
                    if isinstance(sol, list):
                        task_groups[task_id].extend([str(s) for s in sol])
                    else:
                        task_groups[task_id].append(str(sol))
            except json.JSONDecodeError:
                continue

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f_out:
        for task_id, solutions in task_groups.items():
            yes_count = sum(1 for s in solutions if "yes" in s.lower())
            majority_correct = yes_count > (len(solutions) / 2) if len(solutions) > 0 else False
            output_data = {
                "task_id": task_id,
                "solutions": solutions,
                "majority_correct": majority_correct
            }
            f_out.write(json.dumps(output_data) + "\n")

    logger.info("Aggregation complete. Saved output to %s", output_path.name)

if __name__ == "__main__":
    main()