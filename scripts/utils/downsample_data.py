import json, argparse
from evalhub.benchmarks.registry import DATASET_MAP

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_raw", required=True)
    parser.add_argument("--out_sol", required=True)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()

    # Benchmark sınıfını repodan dinamik al
    ds = DATASET_MAP[args.task]()
    counts = {}

    with open(args.input) as f_in, open(args.out_raw, "w") as f_r, open(args.out_sol, "w") as f_s:
        for line in f_in:
            d = json.loads(line)
            tid = d["task_id"]
            counts[tid] = counts.get(tid, 0)
            
            if counts[tid] < args.n:
                f_r.write(json.dumps(d) + "\n")
                # Mevcut extract_solution metodunu kullan
                res = d.get("response", {})
                content = res.get("content") or res.get("choices", [{}])[0].get("message", {}).get("content", "")
                f_s.write(json.dumps({"task_id": tid, "solution": ds.extract_solution(tid, content)}) + "\n")
                counts[tid] += 1

if __name__ == "__main__":
    main()