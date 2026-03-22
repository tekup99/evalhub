#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt

def load_data(path_str):
    path = Path(path_str)
    if not path.exists():
        print(f"[WARNING] Dosya bulunamadi: {path_str}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        pass_k = {int(k): float(v) for k, v in data.get("pass_at_k", {}).items()}
    return pass_k

def main():
    parser = argparse.ArgumentParser(description="Custom Plot Generator")
    parser.add_argument("--paths", nargs='+', required=True, help="Liste halinde JSON yollari")
    parser.add_argument("--labels", nargs='+', required=True, help="Liste halinde etiketler")
    parser.add_argument("--colors", nargs='+', required=True, help="Liste halinde renkler")
    parser.add_argument("--linestyles", nargs='+', required=True, help="Liste halinde cizgi stilleri (solid/dashed)")
    parser.add_argument("--markers", nargs='+', required=True, help="Liste halinde marker stilleri")
    parser.add_argument("--title", type=str, required=True, help="Grafik basligi")
    parser.add_argument("--out", type=str, required=True, help="Kaydedilecek gorselin yolu")
    args = parser.parse_args()

    assert all(len(lst) == len(args.paths) for lst in [args.labels, args.colors, args.linestyles, args.markers]), "Verilen ozellik listelerinin uzunluklari esit olmali!"

    style_map = {"solid": "-", "dashed": "--"}

    plt.figure(figsize=(10, 6))
    has_data = False

    for i in range(len(args.paths)):
        pass_k = load_data(args.paths[i])
        if not pass_k:
            continue
        
        has_data = True
        k_vals = sorted(pass_k.keys())
        log2_k_vals = [int(math.log2(k)) for k in k_vals]
        
        # solid/dashed metnini matplotlib sembolune ceviriyoruz
        ls = style_map.get(args.linestyles[i], "-")
        
        plt.plot(log2_k_vals, [pass_k[k] for k in k_vals], 
                 label=args.labels[i], 
                 color=args.colors[i], 
                 linestyle=ls, 
                 marker=args.markers[i], 
                 linewidth=2.5, markersize=8)

    if not has_data:
        print(f"[SKIP] Cizilecek veri bulunamadi: {args.out}")
        plt.close()
        return

    plt.title(args.title, fontweight="bold", fontsize=14)
    plt.xlabel(r"$\log_2(k)$", fontsize=12)
    plt.ylabel("Pass@k Score", fontsize=12)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gorsel kaydedildi: {out_path}")

if __name__ == "__main__":
    main()