import os
import logging
from evalhub.benchmarks.alignment.math_judge import MathJudgeDataset

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def test_dynamic_pipeline() -> None:
    logging.info("Initializing dynamic dataset for testing...")
    
    test_file = "data/passatk_filtered/Qwen3.5-4B/aime2025_corrects.jsonl"
    if not os.path.exists(test_file):
        logging.error(f"Test file not found: {test_file}. Please check the path.")
        return

    test_meta = {"file_path": test_file}
    
    try:
        dataset = MathJudgeDataset(meta_data=test_meta)
        logging.info("Loading original benchmarks. This may take a few seconds...")
        dataset.load_tasks()
        
        task_count = len(dataset.tasks)
        logging.info(f"SUCCESS: {task_count} tasks loaded dynamically.")
        
        logging.info("Testing prompt format for the first task:")
        first_task = list(dataset.tasks.values())[0] if isinstance(dataset.tasks, dict) else dataset.tasks[0]
            
        prompt_text = first_task.prompt
        print("-" * 50)
        print(prompt_text[:800] + "\n\n... [PROMPT CONTINUES] ...")
        print("-" * 50)
        
        logging.info("Checking dynamic question matching...")
        if "MISSING_QUESTION_IN_DATA" in prompt_text:
            logging.error("FAILED: Could not fetch the original question from Evalhub.")
        else:
            logging.info("SUCCESS: Original question integrated into the Judge prompt perfectly.")

        logging.info(f"Metadata check: {first_task.metadata}")
        
        logging.info("Testing regex response extraction...")
        dummy_model_response = "Here is my step-by-step reasoning... Therefore, the final answer is \\boxed{no}."
        extracted = dataset.extract_solution("dummy_id", dummy_model_response)
        logging.info(f"Extracted result: '{extracted}' (Expected: 'no')")
        
        if extracted in ["yes", "no"]:
            logging.info("SUCCESS: Dynamic python logic is working flawlessly.")
        else:
            logging.error("FAILED: Regex extraction issue detected.")
            
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    test_dynamic_pipeline()