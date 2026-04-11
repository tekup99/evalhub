import json
import math
from pathlib import Path
import matplotlib.pyplot as plt

def extract_metadata(path: Path) -> str:
    task = path.parent.name
    model = path.parent.parent.name
    return f"Model: {model} | Task: {task}"

def process_summary(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            pass_k = {int(k): float(v) for k, v in data.get("pass_at_k", {}).items()}
    except Exception as e:
        print(f"[ERROR] Dosya okunamadi {path}: {e}")
        return
    
    if not pass_k:
        return

    label = extract_metadata(path)
    k_vals = sorted(pass_k.keys())
    
    log2_k_vals = [int(math.log2(k)) for k in k_vals]
    
    plt.figure(figsize=(10, 6))
    plt.plot(log2_k_vals, [pass_k[k] for k in k_vals], label=label, linewidth=2,
             marker='o', linestyle='-')
    
    plt.xticks(log2_k_vals, log2_k_vals)
    plt.title(f"Performance: {label}", fontweight="bold")
    plt.xlabel(r"$\log_2(k)$")
    plt.ylabel("Pass@k Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    out_path = path.parent / f"{path.parent.name}_pass_at_k_plot.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Gorsel olusturuldu: {out_path}")

def main():
    base_dir = Path("results")
    if not base_dir.exists():
        print("[WARNING] 'results' klasoru bulunamadi.")
        return
        
    for summary_path in base_dir.rglob("*summary.json"):
        process_summary(summary_path)

if __name__ == "__main__":
    main()