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

def get_judge_label(j_path):
    """Path üzerinden judge modelinin ismini ve sıcaklık/benchmark değerini çıkarır"""
    parts = j_path.parts
    eval_part = next((p for p in parts if 'evaluated_by' in p), None)
    if eval_part:
        judge_name = eval_part.split('evaluated_by')[-1]
    else:
        judge_name = "Judge"
    
    parent_name = j_path.parent.name
    # Ornek parent_name: aime2026_t0.6 -> t0.6 kismini al
    temp_str = parent_name.split('_')[-1] if '_' in parent_name else parent_name
    return f"Judge: {judge_name} ({temp_str})"

def finish_plot(args, has_data):
    """Grafigi formatlayip kaydetme islemini gerceklestirir"""
    if not has_data:
        print(f"[SKIP] Cizilecek veri bulunamadi: {args.out}")
        plt.close()
        return

    # EĞER SUBTITLE VERİLDİYSE ANA BAŞLIK ÜSTE, ALT BAŞLIK ALTA YAZILIR
    if args.subtitle:
        plt.suptitle(args.title, fontweight="bold", fontsize=15, y=0.98)
        plt.title(args.subtitle, fontsize=11, color="dimgray", pad=10)
    else:
        plt.title(args.title, fontweight="bold", fontsize=14)

    plt.xlabel(r"$\log_2(k)$", fontsize=12)
    plt.ylabel("Pass@k Score", fontsize=12)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # --- LEGEND ALTA ALMA AYARLARI ---
    # loc ve bbox_to_anchor ile plotun hemen altina ortaliyoruz.
    # ncol=2 ile legend ögelerini 2 sütun halinde yan yana diziyoruz ki çok fazla satır olup aşağı uzamasın.
    plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15), 
        ncol=2, 
        fontsize=10 # Orijinal iyi dediğin yazı boyutu
    )
    # ---------------------------------
    
    plt.tight_layout()
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # bbox_inches="tight" sayesinde alttaki legend'in kesilmesini engelliyoruz
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gorsel kaydedildi: {out_path}")
    
def auto_mode(args):
    """1., 2. ve 3. maddelerdeki otomatik Base & Judge eslestirme mantigi"""
    base_path = Path(args.base_summary)
    if not base_path.exists():
        print(f"[ERROR] Base dosya bulunamadi: {base_path}")
        return

    # Base modeli tespit et 
    # (Ornek: results/Qwen2.5-32B_t0.6_max16384/aime2026/aime2026_summary.json)
    if len(base_path.parts) >= 3:
        base_dir_name = base_path.parts[-3]
        benchmark_name = base_path.parts[-2]
    else:
        base_dir_name = base_path.parent.parent.name
        benchmark_name = base_path.parent.name

    judgments_base = Path(args.judgments_dir)
    judge_paths = []

    # 2. Madde: Base ile baslayan ve evaluated_by iceren judgement klasorlerini bul
    if judgments_base.exists():
        for j_dir in judgments_base.iterdir():
            if not j_dir.is_dir(): continue
            if 'evaluated_by' not in j_dir.name: continue
            
            base_prefix = j_dir.name.split('evaluated_by')[0]
            
            # Eger bu judgment klasoru bizim base modelimize aitse
            if base_dir_name.startswith(base_prefix):
                # 3. Madde: Klasorun altindaki tum t0.6, t0.0 vb. dosyalari bul
                for summary_file in j_dir.rglob("*summary.json"):
                    # Ayni benchmark (or. aime2026) dosyasi oldugunu garantiye al
                    if benchmark_name in summary_file.name or benchmark_name in summary_file.parent.name:
                        judge_paths.append(summary_file)
    
    plt.figure(figsize=(10, 6))
    has_data = False

    # 1. Madde (Base -> Mavi ve Duz cizgi)
    base_data = load_data(base_path)
    if base_data:
        k_vals = sorted(base_data.keys())
        log2_k_vals = [int(math.log2(k)) for k in k_vals]
        plt.plot(log2_k_vals, [base_data[k] for k in k_vals], 
                 label=f"Base: {base_dir_name}", 
                 color="blue", 
                 linestyle="-", 
                 marker="o", 
                 linewidth=2.5, markersize=8)
        has_data = True

    # 1. & 3. Madde (Judge -> Turuncu, birden fazlaysa farkli marker/renk tonu)
    colors = ["darkorange", "forestgreen", "crimson", "purple", "saddlebrown", "magenta", "teal"]
    markers = ["s", "^", "D", "v", "p", "*", "X"]
    
    judge_paths = sorted(judge_paths) # Sabit siralama icin

    for i, j_path in enumerate(judge_paths):
        j_data = load_data(j_path)
        if not j_data:
            continue
        
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        
        k_vals = sorted(j_data.keys())
        log2_k_vals = [int(math.log2(k)) for k in k_vals]
        
        plt.plot(log2_k_vals, [j_data[k] for k in k_vals], 
                 label=get_judge_label(j_path), 
                 color=c, 
                 linestyle="--", 
                 marker=m, 
                 linewidth=2.5, markersize=8)
        has_data = True

    finish_plot(args, has_data)

def manual_mode(args):
    """Eski script kullanim aliskanligini bozmamak icin orijinal calisma sekli"""
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
        
        ls = style_map.get(args.linestyles[i], "-")
        
        plt.plot(log2_k_vals, [pass_k[k] for k in k_vals], 
                 label=args.labels[i], 
                 color=args.colors[i], 
                 linestyle=ls, 
                 marker=args.markers[i], 
                 linewidth=2.5, markersize=8)

    finish_plot(args, has_data)

def main():
    parser = argparse.ArgumentParser(description="Custom Plot Generator")
    
    # Yeni Otomatik Mod Argumanlari
    parser.add_argument("--base_summary", type=str, help="Otomatik mod icin Base model summary json yolu")
    parser.add_argument("--judgments_dir", type=str, default="results/judgments", help="Otomatik mod icin Judgments klasoru")
    
    # Eski Manuel Mod Argumanlari (required=False yapildi)
    parser.add_argument("--paths", nargs='+', help="Liste halinde JSON yollari (Manuel Mod)")
    parser.add_argument("--labels", nargs='+', help="Liste halinde etiketler (Manuel Mod)")
    parser.add_argument("--colors", nargs='+', help="Liste halinde renkler (Manuel Mod)")
    parser.add_argument("--linestyles", nargs='+', help="Liste halinde cizgi stilleri (Manuel Mod)")
    parser.add_argument("--markers", nargs='+', help="Liste halinde marker stilleri (Manuel Mod)")
    
    # Ortak Argumanlar
    parser.add_argument("--title", type=str, required=True, help="Grafik basligi")
    parser.add_argument("--subtitle", type=str, default="", help="Grafik alt basligi (Opsiyonel)")
    parser.add_argument("--out", type=str, required=True, help="Kaydedilecek gorselin yolu")
    args = parser.parse_args()

    if args.base_summary:
        auto_mode(args)
    elif args.paths:
        manual_mode(args)
    else:
        print("[ERROR] Lutfen ya --base_summary (otomatik mod) ya da --paths (manuel mod) saglayin.")

if __name__ == "__main__":
    main()