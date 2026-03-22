#!/usr/bin/env python3
import json
import re
import math
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def extract_label(path: Path) -> tuple[str, bool]:
    is_cot = "judge" in path.parts or "judgment" in str(path)
    task = path.parent.name
    model = path.parent.parent.name
    if is_cot:
        temp = re.search(r"_t([\d\.]+)", task)
        t_str = f" (t={temp.group(1)})" if temp else ""
        return f"{model} | {task.split('_t')[0]}{t_str}", True
    return f"{model} | {task}", False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=str, help="Paths to summary.json files")
    parser.add_argument("--out", type=str, required=True, help="Output plot filename")
    args = parser.parse_args()

    plt.figure(figsize=(12, 8))
    
    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"[WARNING] File not found: {path}")
            continue
            
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            pass_k = {int(k): float(v) for k, v in data.get("pass_at_k", {}).items()}
            
        if not pass_k:
            continue

        label, is_cot = extract_label(path)
        k_vals = sorted(pass_k.keys())
        
        # X eksenindeki sayilari k'nin log2 degerine ceviriyoruz
        log2_k_vals = [int(math.log2(k)) for k in k_vals]
        
        plt.plot(log2_k_vals, [pass_k[k] for k in k_vals], label=label, linewidth=2,
                 marker='s' if is_cot else 'o', linestyle='--' if is_cot else '-')

    plt.title("Pass@k Comparison", fontweight="bold")
    plt.xlabel(r"$\log_2(k)$")
    plt.ylabel("Pass@k Score")
    # x ekseni çizgilerinin sadece tam sayı olmasını sağlıyoruz
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Plot saved to {out_path}")

if __name__ == "__main__":
    main()