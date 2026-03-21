import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict

from evalhub.utils.metrics import compute_pass_at_k, get_majority_vote

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply judge results and recalculate metrics.")
    parser.add_argument("--base_results_file", type=str, required=True, help="Original results JSONL")
    parser.add_argument("--judge_majority_file", type=str, required=True, help="Judge majority vote JSONL")
    parser.add_argument("--output_file", type=str, required=True, help="Updated output JSONL")
    parser.add_argument("--summary_file", type=str, required=True, help="Summary JSON output")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    judge_dict: Dict[str, bool] = {}
    try:
        with open(args.judge_majority_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                judge_dict[data.get("task_id")] = data.get("majority_correct", False)
    except FileNotFoundError:
        logger.error("Judge majority file not found: %s", args.judge_majority_file)
        return

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    sum_pass_at_k: Dict[str, float] = defaultdict(float)
    correct_majority_count = 0
    total_tasks = 0
    
    with open(args.base_results_file, "r", encoding="utf-8") as f_in, \
         open(args.output_file, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
                
            data = json.loads(line)
            task_id = data["task_id"]
            solutions = data["solutions"]
            correct = data["correct"]
            ground_truth = data.get("ground_truth", "")
            n_generations = len(correct)
            
            for i in range(n_generations):
                gen_id = f"{task_id}_gen_{i}"
                if gen_id in judge_dict and correct[i] is True and not judge_dict[gen_id]:
                    correct[i] = "cot_false"
            
            correct_count = sum(1 for x in correct if x is True)
            new_pass_at_k = {}
            
            for k_str in data.get("pass_at_k", {}).keys():
                k_val = int(k_str)
                val = compute_pass_at_k(n_generations, correct_count, k_val) if k_val <= n_generations else 1.0
                new_pass_at_k[k_str] = val
                sum_pass_at_k[k_str] += val
            
            new_majority_vote = get_majority_vote(solutions)
            is_correct_majority = (new_majority_vote == ground_truth)
            
            if is_correct_majority:
                correct_majority_count += 1
                
            total_tasks += 1
            data.update({
                "correct": correct,
                "pass_at_k": new_pass_at_k,
                "majority_vote": new_majority_vote,
                "is_correct_majority": is_correct_majority
            })
            f_out.write(json.dumps(data) + "\n")
            
    if total_tasks > 0:
        summary_data = {
            "pass_at_k": {k: (v / total_tasks) for k, v in sum_pass_at_k.items()},
            "cons_at_k": correct_majority_count / total_tasks
        }
        with open(args.summary_file, "w", encoding="utf-8") as f_sum:
            json.dump(summary_data, f_sum, indent=4)
        logger.info("Metrics successfully applied and summarized.")
    else:
        logger.warning("No tasks were processed.")

if __name__ == "__main__":
    main()
