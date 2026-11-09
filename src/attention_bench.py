"""Timing and peak-memory benchmark: reference attention vs tiled online-softmax vs PyTorch SDPA.
Run from repo root:  python src/attention_bench.py            (GPU recommended; falls back to CPU, memory then reported as 0)
Writes outputs/attention_bench.json.  Shapes mimic Qwen2.5-1.5B: 12 query heads, head_dim 128, batch 1.
"""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention import ref_attention, tiled_attention

ap = argparse.ArgumentParser()
ap.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
ap.add_argument("--heads", type=int, default=12)
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--dtype", default="float16")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--out", default="outputs/attention_bench.json")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = getattr(torch, a.dtype) if dev == "cuda" else torch.float32


def measure(fn, q, k, v):
    fn(q, k, v)                                   # warm-up
    if dev == "cuda":
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(a.reps):
        t = time.perf_counter(); fn(q, k, v)
        if dev == "cuda": torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    mem = (torch.cuda.max_memory_allocated() - base) / 2**20 if dev == "cuda" else 0.0
    return 1000 * sorted(ts)[len(ts) // 2], mem


sdpa = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
res = []
for T in a.lengths:
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, a.heads, T, a.dim, device=dev, dtype=dt) for _ in range(3))
    row = {"T": T}
    for name, fn in [("ref", ref_attention), ("tiled", tiled_attention), ("sdpa", sdpa)]:
        try:
            row[f"{name}_ms"], row[f"{name}_mem_MiB"] = measure(fn, q, k, v)
        except torch.cuda.OutOfMemoryError:
            row[f"{name}_ms"], row[f"{name}_mem_MiB"] = float("nan"), float("nan")
            torch.cuda.empty_cache()
    err = (tiled_attention(q, k, v).float() - ref_attention(q, k, v).float()).abs().max().item()
    row["tiled_vs_ref_max_err"] = err
    print(row)
    res.append(row)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump({"settings": {"device": dev, "dtype": str(dt), "heads": a.heads, "head_dim": a.dim, "reps": a.reps,
                        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}, "results": res}, open(a.out, "w"), indent=2)
