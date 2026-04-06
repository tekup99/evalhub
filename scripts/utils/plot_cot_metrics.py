import json
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt

def plot_metrics(summary_file: Path, benchmark: str, base_model: str, judge_model: str, out_dir: Path) -> None:
    if not summary_file.exists():
        print(f"[ERROR] Summary file not found: {summary_file}", file=sys.stderr)
        sys.exit(1)

    try:
        with summary_file.open("r", encoding="utf-8") as f:
            line = f.readline().strip()
            if not line:
                raise ValueError("Summary file is empty.")
            data = json.loads(line)
    except Exception as e:
        print(f"[ERROR] Failed to read summary file: {e}", file=sys.stderr)
        sys.exit(1)

    pass_at_k = data.get("pass_at_k", {})
    cons_at_k = data.get("cons_at_k")

    if not pass_at_k:
        print("[WARNING] No pass_at_k data found. Skipping plot.", file=sys.stderr)
        return

    k_values = sorted([int(k) for k in pass_at_k.keys()])
    scores = [pass_at_k[str(k)] for k in k_values]

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, scores, marker="o", linestyle="-", color="#1f77b4", linewidth=2, markersize=8, label="CoT Pass@K")

    if cons_at_k is not None:
        plt.axhline(y=cons_at_k, color="#d62728", linestyle="--", linewidth=2, label=f"CoT Cons@K ({cons_at_k:.3f})")

    plt.title(f"CoT Evaluation Results\nBase: {base_model} | Judge: {judge_model} | Bench: {benchmark}", fontsize=12, pad=15)
    plt.xlabel("K (Number of Samples)", fontsize=11)
    plt.ylabel("Score (Probability)", fontsize=11)
    plt.xticks(k_values)
    
    max_score = max(max(scores), cons_at_k if cons_at_k else 0)
    plt.ylim(0, max_score * 1.2)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="upper left", fontsize=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{base_model}_judged_by_{judge_model}_{benchmark}_cot_metrics.png"
    
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot successfully saved to: {out_file}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CoT metrics plot.")
    parser.add_argument("--summary_file", type=Path, required=True, help="Path to JSONL summary file")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    parser.add_argument("--model", type=str, required=True, help="Base model name")
    parser.add_argument("--judge", type=str, required=True, help="Judge model name")
    parser.add_argument("--out_dir", type=Path, default=Path("plots"), help="Output directory")
    
    args = parser.parse_args()
    plot_metrics(args.summary_file, args.benchmark, args.model, args.judge, args.out_dir)

if __name__ == "__main__":
    main()