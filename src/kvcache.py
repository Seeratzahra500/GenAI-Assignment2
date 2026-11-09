import os; os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import argparse, json, time, statistics, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from common import load_jsonl

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

@torch.no_grad()
def generate_nocache(model, ids, n_new):
    """Reference: re-runs the FULL sequence at every step (use_cache=False), greedy argmax."""
    seq, logs, t_first, t_all = ids.clone(), [], 0.0, time.perf_counter()
    sync(); t0 = time.perf_counter()
    for s in range(n_new):
        lg = model(input_ids=seq, use_cache=False).logits[:, -1, :].float().clone()
        logs.append(lg); seq = torch.cat([seq, lg.argmax(-1, keepdim=True)], 1)
        if s == 0: sync(); t_first = time.perf_counter() - t0
    sync(); total = time.perf_counter() - t0
    return seq[0, ids.shape[1]:].tolist(), logs, {"prefill_s": t_first, "decode_s": total - t_first}

@torch.no_grad()
def generate_cached(model, ids, n_new):
    """Explicit cache flow: prefill once, then feed ONE token per step with explicit position_ids,
    a growing attention_mask, and the cache carried between calls."""
    T, dev = ids.shape[1], ids.device
    cache = DynamicCache()
    sync(); t0 = time.perf_counter()
    out = model(input_ids=ids, position_ids=torch.arange(T, device=dev)[None], past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    lg = out.logits[:, -1, :].float().clone(); logs = [lg]; nxt = lg.argmax(-1, keepdim=True); toks = [nxt]
    sync(); t_pre = time.perf_counter() - t0
    for s in range(1, n_new):
        cur = T + s - 1                                   # absolute position of the token being fed
        assert cache.get_seq_length() == cur, (cache.get_seq_length(), cur)   # cache grows by exactly 1/step
        out = model(input_ids=nxt, position_ids=torch.tensor([[cur]], device=dev),
                    attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=dev),
                    past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        lg = out.logits[:, -1, :].float().clone(); logs.append(lg)
        nxt = lg.argmax(-1, keepdim=True); toks.append(nxt)
    sync(); total = time.perf_counter() - t0
    return torch.cat(toks, 1)[0].tolist(), logs, {"prefill_s": t_pre, "decode_s": total - t_pre}

def measure(fn, model, ids, n_new, reps):
    fn(model, ids, n_new)                                  # warm-up (not counted)
    runs, peak = [], 0
    for _ in range(reps):
        torch.cuda.reset_peak_memory_stats()
        toks, logs, t = fn(model, ids, n_new)
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**20); runs.append(t)
    pre = statistics.median(r["prefill_s"] for r in runs); dec = statistics.median(r["decode_s"] for r in runs)
    return toks, logs, {"prefill_s": pre, "decode_s": dec, "total_s": pre + dec, "tok_per_s_total": n_new / (pre + dec),
                        "decode_tok_per_s": (n_new - 1) / dec if n_new > 1 else None, "peak_mem_MiB": peak}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct"); ap.add_argument("--adapter", default=None)
    ap.add_argument("--dtype", default="float16"); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--in_lens", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--out_lens", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--out", default="outputs/kv_bench.json"); a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map={"": 0})
    if a.adapter:
        from peft import PeftModel; model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.eval()
    cfg = model.config; head_dim = cfg.hidden_size // cfg.num_attention_heads
    bytes_el = torch.finfo(getattr(torch, a.dtype)).bits // 8
    base_mem = torch.cuda.memory_allocated() / 2**20
    text = "\n".join(r["prompt"] for r in load_jsonl("data/val.jsonl")[:60])       # validation text only
    all_ids = tok(text, return_tensors="pt").input_ids[0].cuda()

    results = []
    for T in a.in_lens:
        for n in a.out_lens:
            ids = all_ids[:T][None]
            t_ref, l_ref, m_ref = measure(generate_nocache, model, ids, n, a.reps)
            t_kv, l_kv, m_kv = measure(generate_cached, model, ids, n, a.reps)
            match = t_ref == t_kv
            first_mis = next((i for i, (x, y) in enumerate(zip(t_ref, t_kv)) if x != y), None)
            upto = n if first_mis is None else first_mis       # logits are comparable up to the first mismatch
            maxdiff = max((l_ref[i] - l_kv[i]).abs().max().item() for i in range(upto)) if upto else None
            kv_mib = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * (T + n) * bytes_el / 2**20
            r = {"input_len": T, "new_tokens": n, "tokens_identical": match, "first_mismatch_step": first_mis,
                 "max_abs_logit_diff_before_mismatch": maxdiff, "kv_cache_MiB_at_end": round(kv_mib, 2),
                 "no_cache": m_ref, "kv_cache": m_kv, "speedup_total": m_ref["total_s"] / m_kv["total_s"]}
            results.append(r)
            print(f"T={T:4d} n={n:3d} identical={match} first_mismatch={first_mis} maxlogitdiff={maxdiff} "
                  f"| nocache {m_ref['total_s']:.2f}s {m_ref['tok_per_s_total']:.1f} tok/s {m_ref['peak_mem_MiB']:.0f}MiB "
                  f"| cached {m_kv['total_s']:.2f}s {m_kv['tok_per_s_total']:.1f} tok/s {m_kv['peak_mem_MiB']:.0f}MiB "
                  f"| speedup x{r['speedup_total']:.1f}")
    json.dump({"settings": {"model": a.model, "adapter": a.adapter, "dtype": a.dtype, "reps": a.reps,
                            "decoding": "greedy, fixed n_new (EOS ignored), batch 1", "weights_MiB": round(base_mem)},
               "results": results}, open(a.out, "w"), indent=2)
