import math, json, torch
import torch.nn.functional as F

def causal_mask(Tq, Tk, device="cpu"):
    # query i sits at global position i + (Tk - Tq): it may attend to keys j <= i + (Tk - Tq)
    return torch.ones(Tq, Tk, dtype=torch.bool, device=device).tril(diagonal=Tk - Tq)

def ref_attention(q, k, v, causal=True):
    """Reference scaled dot-product attention with a hand-written stable softmax. q:(B,H,Tq,D) k,v:(B,H,Tk,D)"""
    s = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        s = s.masked_fill(~causal_mask(q.shape[-2], k.shape[-2], q.device), float("-inf"))
    e = torch.exp(s - s.amax(-1, keepdim=True))          # subtract row max for stability
    return ((e / e.sum(-1, keepdim=True)) @ v.float()).to(q.dtype)

def tiled_attention(q, k, v, causal=True, block_q=64, block_k=64):
    """Blocked attention with online softmax: never materialises the full Tq x Tk score matrix."""
    B, H, Tq, D = q.shape; Tk = k.shape[-2]; off = Tk - Tq
    scale = 1.0 / math.sqrt(D)
    out = torch.empty(B, H, Tq, D, dtype=torch.float32, device=q.device)
    for i0 in range(0, Tq, block_q):
        i1 = min(i0 + block_q, Tq)
        qb = q[..., i0:i1, :].float()
        m = torch.full((B, H, i1 - i0), float("-inf"), device=q.device)   # running row max
        l = torch.zeros((B, H, i1 - i0), device=q.device)                 # running normaliser
        acc = torch.zeros((B, H, i1 - i0, D), device=q.device)            # running weighted sum of V
        for j0 in range(0, Tk, block_k):
            j1 = min(j0 + block_k, Tk)
            if causal and j0 > (i1 - 1) + off:      # whole key block is in the future: skip
                break
            s = (qb @ k[..., j0:j1, :].float().transpose(-2, -1)) * scale
            if causal:
                qi = torch.arange(i0, i1, device=q.device)[:, None] + off
                kj = torch.arange(j0, j1, device=q.device)[None, :]
                s = s.masked_fill(kj > qi, float("-inf"))
            m_new = torch.maximum(m, s.amax(-1))
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)  # guard -inf - -inf
            p = torch.exp(s - m_safe[..., None])
            alpha = torch.exp(m - m_safe)            # rescale what was accumulated under the old max
            l = l * alpha + p.sum(-1)
            acc = acc * alpha[..., None] + p @ v[..., j0:j1, :].float()
            m = m_new
        out[..., i0:i1, :] = acc / l[..., None]
    return out.to(q.dtype)

def run_tests(out_path="outputs/attention_tests.json"):
    torch.manual_seed(0); rows = []
    cases = [(1,2,8,8,16), (2,4,37,37,32), (1,4,64,64,64), (2,2,100,100,64), (1,2,1,50,32), (1,2,5,50,32), (1,8,257,257,64)]
    for (B, H, Tq, Tk, D) in cases:
        for causal in (True, False):
            for mag in (1.0, 30.0):          # mag=30 -> huge logits, tests numerical stability
                q = torch.randn(B,H,Tq,D) * mag; k = torch.randn(B,H,Tk,D); v = torch.randn(B,H,Tk,D)
                r = ref_attention(q, k, v, causal)
                t = tiled_attention(q, k, v, causal, block_q=16, block_k=16)
                mask = causal_mask(Tq, Tk) if causal else None
                sd = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                rows.append({"shape(B,H,Tq,Tk,D)": [B,H,Tq,Tk,D], "causal": causal, "logit_scale": mag,
                             "tiled_vs_ref_max": float((t-r).abs().max()), "tiled_vs_ref_mean": float((t-r).abs().mean()),
                             "ref_vs_sdpa_max": float((r-sd).abs().max())})
    for x in rows: print(x)
    worst = max(x["tiled_vs_ref_max"] for x in rows)
    print("WORST tiled-vs-ref max abs error:", worst)
    assert worst < 1e-4, "tiled attention does not match reference"
    json.dump(rows, open(out_path, "w"), indent=2); print("PASS")

if __name__ == "__main__":
    run_tests()
