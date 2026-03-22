#!/usr/bin/env python3
import json
import re
import math
from pathlib import Path
import matplotlib.pyplot as plt

def extract_metadata(path: Path) -> tuple[str, bool]:
    is_cot = "judge" in path.parts or "judgment" in str(path)
    task, model = path.parent.name, path.parent.parent.name
    if is_cot:
        temp = re.search(r"_t([\d\.]+)", task)
        t_str = f" (t={temp.group(1)})" if temp else ""
        return f"{model} | Task: {task.split('_t')[0]}{t_str}", True
    return f"Model: {model} | Task: {task}", False

def process_summary(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        pass_k = {int(k): float(v) for k, v in json.load(f).get("pass_at_k", {}).items()}
    
    if not pass_k:
        return

    label, is_cot = extract_metadata(path)
    k_vals = sorted(pass_k.keys())
    
    # X eksenindeki sayilari k'nin log2 degerine ceviriyoruz (2^x = k)
    log2_k_vals = [int(math.log2(k)) for k in k_vals]
    
    plt.figure(figsize=(10, 6))
    plt.plot(log2_k_vals, [pass_k[k] for k in k_vals], label=label, linewidth=2,
             marker='s' if is_cot else 'o', linestyle='--' if is_cot else '-')
    
    # xticks olarak donusturulmus kuvvetleri veriyoruz
    plt.xticks(log2_k_vals, log2_k_vals)
    plt.title(f"Performance: {label}", fontweight="bold")
    plt.xlabel(r"$\log_2(k)$")
    plt.ylabel("Pass@k Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    out_path = path.parent / "pass_at_k_plot.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    base_dir = Path("results")
    if not base_dir.exists():
        return
        
    for summary_path in base_dir.rglob("summary.json"):
        try:
            process_summary(summary_path)
        except Exception:
            pass

if __name__ == "__main__":
    main()