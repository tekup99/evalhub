import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from evalhub.utils.metrics import compute_pass_at_k, get_majority_vote

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Applies judge results and recalculates metrics.")
    parser.add_argument("--base_results_file", type=str, required=True, help="Original results JSONL")
    parser.add_argument("--judge_majority_file", type=str, required=True, help="Judge majority vote JSONL")
    parser.add_argument("--output_file", type=str, required=True, help="Updated output JSONL")
    parser.add_argument("--summary_file", type=str, required=True, help="Summary JSON output")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    judge_dict = {}
    with open(args.judge_majority_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            judge_dict[data.get("task_id")] = data.get("majority_correct", False)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    sum_pass_at_k: Dict[str, float] = defaultdict(float)
    correct_majority_count = 0
    total_tasks = 0
    
    logging.info(f"Updating metrics based on judge file: {args.judge_majority_file}")
    
    with open(args.base_results_file, 'r', encoding='utf-8') as f_in, \
         open(args.output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            data = json.loads(line)
            
            task_id = data["task_id"]
            solutions = data["solutions"]
            correct = data["correct"]
            ground_truth = data.get("ground_truth", "")
            n = len(correct)
            
            for i in range(n):
                gen_id = f"{task_id}_gen_{i}"
                if gen_id in judge_dict:
                    is_valid = judge_dict[gen_id]
                    if correct[i] is True and not is_valid:
                        correct[i] = "cot_false"
            
            c = sum(1 for x in correct if x is True)
            
            new_pass_at_k = {}
            for k_str in data.get("pass_at_k", {}).keys():
                k_val = int(k_str)
                val = compute_pass_at_k(n, c, k_val) if k_val <= n else 1.0
                new_pass_at_k[k_str] = val
                sum_pass_at_k[k_str] += val
            
            new_majority_vote = get_majority_vote(solutions)
            new_is_correct_majority = (new_majority_vote == ground_truth)
            
            if new_is_correct_majority:
                correct_majority_count += 1
                
            total_tasks += 1
            data.update({
                "correct": correct,
                "pass_at_k": new_pass_at_k,
                "majority_vote": new_majority_vote,
                "is_correct_majority": new_is_correct_majority
            })
            f_out.write(json.dumps(data) + '\n')
            
    if total_tasks > 0:
        summary_data = {
            "pass_at_k": {k: (v / total_tasks) for k, v in sum_pass_at_k.items()},
            "cons_at_k": correct_majority_count / total_tasks
        }
        with open(args.summary_file, 'w', encoding='utf-8') as f_sum:
            json.dump(summary_data, f_sum, indent=4)
        logging.info(f"Updated results saved to {args.output_file}")
        logging.info(f"Summary report generated at {args.summary_file}")
    else:
        logging.warning("No tasks processed. Output may be empty.")

if __name__ == "__main__":
    main()