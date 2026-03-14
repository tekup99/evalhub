import json
import os
import argparse
from collections import defaultdict

def main():
    # ==============================================================
    # DYNAMIC SETTINGS (Parameters from bash script)
    # ==============================================================
    parser = argparse.ArgumentParser(description="Extract dynamically correct solutions for PRM Judge.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model (e.g., Qwen3.5-4B)")
    parser.add_argument("--benchmark", type=str, required=True, help="Name of the benchmark (e.g., aime2025)")
    args = parser.parse_args()

    # Dynamic paths for input files
    base_dir = f"results/{args.model}/{args.benchmark}"
    results_file = os.path.join(base_dir, f"{args.benchmark}_results.jsonl")
    raw_file = os.path.join(base_dir, f"{args.benchmark}_raw.jsonl")
    
    # Dynamic path for the output destination
    out_dir = f"data/passatk_filtered/{args.model}"
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"{args.benchmark}_corrects.jsonl")

    # ==============================================================
    # START PROCESSING
    # ==============================================================
    correct_indices = defaultdict(list)
    generation_counts = {}
    ground_truths = {}
    generated_solutions = defaultdict(dict)
    task_counter = 0

    print("="*70)
    print("--- STEP 1: Identifying Correct Solutions and Statistics ---")
    print(f"Model: {args.model} | Benchmark: {args.benchmark}")
    print(f"Reading file: {results_file}")
    print("="*70)

    try:
        with open(results_file, "r", encoding="utf-8") as f_res:
            for line in f_res:
                data = json.loads(line)
                task_id = data.get("task_id")
                
                # Store the ground truth answer
                ground_truths[task_id] = data.get("ground_truth", "")
                
                # Count total generations for this task
                correct_list = data.get("correct", [])
                generation_counts[task_id] = len(correct_list)
                
                for idx, is_correct in enumerate(correct_list):
                    if is_correct:
                        # Save the index of the correct answer
                        correct_indices[task_id].append(idx)
                        
                        # Store the extracted generated solution
                        if "solutions" in data and len(data["solutions"]) > idx:
                            generated_solutions[task_id][idx] = data["solutions"][idx]
                        else:
                            generated_solutions[task_id][idx] = ""
                
                # Print individual task stats
                correct_count = len(correct_indices[task_id])
                total_gen = generation_counts[task_id]
                print(f"Task: {task_id:<20} | Generated: {total_gen:<5} | Correct: {correct_count}")
                
                task_counter += 1
                
        print("-" * 70)
        print(f"[INFO] Step 1 complete. Processed {task_counter} total tasks.")
        
    except FileNotFoundError:
        print(f"[ERROR] File {results_file} not found! Please check the paths.")
        exit(1)


    print("\n" + "="*70)
    print("--- STEP 2: Reading and Filtering Raw Data ---")
    print(f"Reading file: {raw_file}")
    print("="*70)

    current_task_indices = defaultdict(int)
    saved_count = 0

    try:
        with open(raw_file, "r", encoding="utf-8") as f_raw, open(output_file, "w", encoding="utf-8") as f_out:
            for line in f_raw:
                data = json.loads(line)
                task_id = data.get("task_id")
                
                # Skip if the task has no correct answers recorded
                if task_id not in correct_indices:
                    continue
                    
                current_idx = current_task_indices[task_id]
                
                # If this specific index was marked as correct, write it to the new file
                if current_idx in correct_indices[task_id]:
                    output_data = {
                        "task_id": f"{task_id}_gen_{current_idx}",  # EvalHub için benzersiz ID
                        "original_task_id": task_id,                # Orijinal AIME soru ID'si
                        "generation_idx": current_idx,              # Base Model'in kaçıncı üretimi olduğu
                        "ground_truth": ground_truths.get(task_id, ""),
                        "generated_answer": generated_solutions[task_id].get(current_idx, ""),
                        "raw_response": data.get("response", {})
                    }
                    
                    f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                    saved_count += 1
                    
                current_task_indices[task_id] += 1
                
    except FileNotFoundError:
        print(f"[ERROR] File {raw_file} not found! Please check the paths.")
        exit(1)

    print("\n" + "="*70)
    print("--- PROCESS COMPLETED SUCCESSFULLY ---")
    print("="*70)
    print(f"Total unique tasks processed : {task_counter}")
    print(f"Total correct samples saved  : {saved_count}")
    print(f"Filtered output file         : {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()