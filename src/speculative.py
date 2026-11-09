import os; os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import argparse, json, time, statistics, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from common import load_jsonl, chat_prompt
from kvcache import generate_cached, sync

def crop_to(cache, keep):
    """Truncate a DynamicCache to `keep` tokens (negative-count form works across transformers versions)."""
    rm = cache.get_seq_length() - keep
    if rm > 0:
        cache.crop(-rm)

def fwd(model, cache, tokens, start, dev):
    """Feed `tokens` at absolute positions start.. with explicit position ids, mask and cache position."""
    n = len(tokens)
    o = model(input_ids=torch.tensor([tokens], device=dev),
              position_ids=torch.arange(start, start + n, device=dev)[None],
              attention_mask=torch.ones(1, start + n, dtype=torch.long, device=dev),
              cache_position=torch.arange(start, start + n, device=dev),
              past_key_values=cache, use_cache=True)
    return o.logits[0], o.past_key_values

@torch.no_grad()
def generate_spec(target, draft, ids, n_new, k=4):
    """Greedy speculative decoding. Invariant at the top of each round: seq = prompt + toks, and BOTH caches
    hold KV for seq[:-1]; the last token `last` is still to be fed."""
    dev, T = ids.device, ids.shape[1]
    cT, cD = DynamicCache(), DynamicCache()
    st = dict(rounds=0, drafted=0, accepted=0, target_calls=0, draft_calls=0)
    sync(); t0 = time.perf_counter()
    pos = torch.arange(T, device=dev)[None]
    o = target(input_ids=ids, position_ids=pos, past_key_values=cT, use_cache=True); cT = o.past_key_values
    cD = draft(input_ids=ids, position_ids=pos, past_key_values=cD, use_cache=True).past_key_values
    last = o.logits[0, -1].argmax().item(); toks = [last]
    st["target_calls"] += 1; st["draft_calls"] += 1
    sync(); t_pre = time.perf_counter() - t0
    while len(toks) < n_new:
        L = T + len(toks)
        assert cT.get_seq_length() == L - 1 and cD.get_seq_length() == L - 1, "cache out of sync"
        k_eff = min(k, n_new - len(toks) - 1)
        drafts, cur = [], last
        for j in range(k_eff):                                   # 1) draft k tokens autoregressively
            lg, cD = fwd(draft, cD, [cur], L - 1 + j, dev)
            cur = lg[-1].argmax().item(); drafts.append(cur); st["draft_calls"] += 1
        lg, cT = fwd(target, cT, [last] + drafts, L - 1, dev)    # 2) ONE target pass verifies all drafts
        st["target_calls"] += 1
        tp = lg.argmax(-1).tolist()                              # target's greedy token after each fed position
        n_acc = 0
        while n_acc < k_eff and tp[n_acc] == drafts[n_acc]: n_acc += 1   # 3) accept matching prefix
        new = drafts[:n_acc] + [tp[n_acc]]                       # 4) + correction (or bonus token if all matched)
        toks += new
        st["rounds"] += 1; st["drafted"] += k_eff; st["accepted"] += n_acc
        crop_to(cT, L + n_acc)                                       # 5) drop KV of rejected drafts
        if n_acc == k_eff:                                       # draft never consumed its last token: feed it
            _, cD = fwd(draft, cD, [cur], L - 1 + k_eff, dev); st["draft_calls"] += 1
        else:
            crop_to(cD, L + n_acc)
        last = new[-1]
    sync(); total = time.perf_counter() - t0
    return toks[:n_new], st, {"prefill_s": t_pre, "decode_s": total - t_pre, "total_s": total}

def timed(fn, reps):
    fn()                                                         # warm-up
    ts, peak = [], 0
    for _ in range(reps):
        torch.cuda.reset_peak_memory_stats(); sync(); t = time.perf_counter()
        r = fn(); sync(); ts.append(time.perf_counter() - t)
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**20)
    return r, statistics.median(ts), peak

def load(name, adapter, dtype):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype), device_map={"": 0})
    if adapter:
        from peft import PeftModel; m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    return m.eval()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct"); ap.add_argument("--target_adapter", default=None)
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--draft_adapter", default=None)
    ap.add_argument("--dtype", default="float16"); ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--n_new", type=int, default=48); ap.add_argument("--n_prompts", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2); ap.add_argument("--out", default="outputs/spec_bench.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.target); tok_d = AutoTokenizer.from_pretrained(a.draft)
    assert tok.get_vocab() == tok_d.get_vocab(), "tokenizers differ"
    target = load(a.target, a.target_adapter, a.dtype); draft = load(a.draft, a.draft_adapter, a.dtype)
    assert target.config.vocab_size == draft.config.vocab_size
    mem = {"both_models_MiB": round(torch.cuda.memory_allocated() / 2**20)}
    rows = load_jsonl("data/val.jsonl")                          # validation prompts only
    step = max(1, len(rows) // a.n_prompts)
    prompts = [tok(chat_prompt(tok, rows[i]), add_special_tokens=False, return_tensors="pt").input_ids.cuda()
               for i in range(0, len(rows), step)][:a.n_prompts]

    ref = []
    for ids in prompts:
        (t, _, _), sec, peak = timed(lambda: generate_cached(target, ids, a.n_new), a.reps)
        ref.append({"tokens": t, "sec": sec, "peak": peak})
    base_sec = sum(r["sec"] for r in ref)
    summary = [{"mode": "greedy_kvcache", "k": 0, "sec_total": base_sec, "tok_per_s": a.n_new * len(prompts) / base_sec,
                "peak_MiB": max(r["peak"] for r in ref), "all_tokens_equal": True}]
    print(f"baseline greedy+cache: {base_sec:.2f}s  {summary[0]['tok_per_s']:.1f} tok/s")
    for k in a.ks:
        eq, agg, sec_sum, peak = 0, dict(rounds=0, drafted=0, accepted=0, target_calls=0, draft_calls=0), 0.0, 0
        mism = []
        for i, ids in enumerate(prompts):
            (t, st, _), sec, pk = timed(lambda: generate_spec(target, draft, ids, a.n_new, k), a.reps)
            ok = t == ref[i]["tokens"]; eq += ok
            if not ok: mism.append({"prompt": i, "first_diff": next(j for j, (x, y) in enumerate(zip(t, ref[i]["tokens"])) if x != y)})
            for key in agg: agg[key] += st[key]
            sec_sum += sec; peak = max(peak, pk)
        r = {"mode": "speculative", "k": k, "sec_total": sec_sum, "tok_per_s": a.n_new * len(prompts) / sec_sum,
             "speedup_vs_greedy_cache": base_sec / sec_sum, "peak_MiB": peak, "all_tokens_equal": eq == len(prompts),
             "prompts_equal": f"{eq}/{len(prompts)}", "mismatches": mism,
             "acceptance_rate": agg["accepted"] / max(1, agg["drafted"]),
             "mean_accepted_per_round": agg["accepted"] / max(1, agg["rounds"]),
             "target_forward_calls": agg["target_calls"], "draft_forward_calls": agg["draft_calls"],
             "tokens_generated": a.n_new * len(prompts)}
        summary.append(r)
        print(f"k={k}: equal {r['prompts_equal']} | accept {r['acceptance_rate']:.2f} | {r['tok_per_s']:.1f} tok/s | "
              f"speedup x{r['speedup_vs_greedy_cache']:.2f} | target calls {agg['target_calls']} | peak {peak:.0f}MiB")
    json.dump({"settings": {**vars(a), **mem, "decoding": "greedy, fixed n_new, batch 1, EOS ignored"}, "results": summary},
              open(a.out, "w"), indent=2)
