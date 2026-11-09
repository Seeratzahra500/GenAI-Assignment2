"""Builds every figure and LaTeX table used by report/main.tex from the files in outputs/ and data/.

Run from the repository root AFTER all experiments finished:   python src/report_assets.py
Missing input files are skipped with a message (main.tex then shows a visible TODO for that item).

Writes:  report/figures/*.pdf      report/generated/*.tex      report/generated/errors.json
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--outputs", default="outputs")
ap.add_argument("--data", default="data")
ap.add_argument("--report", default="report")
ap.add_argument("--n_errors", type=int, default=12)
A = ap.parse_args()
FIG, GEN = os.path.join(A.report, "figures"), os.path.join(A.report, "generated")
os.makedirs(FIG, exist_ok=True); os.makedirs(GEN, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "figure.dpi": 150})

ABBR = {"included": "Incl", "not included": "NotIncl", "excluded": "Excl", "not excluded": "NotExcl",
        "not enough information": "NEI", "not applicable": "NA", "invalid": "Invalid"}
REPL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "≥": r"$\ge$", "≤": r"$\le$",
        "℃": r"\textdegree{}C", "°": r"\textdegree{}", "µ": r"$\mu$", "μ": r"$\mu$",
        "–": "--", "—": "---", "‘": "'", "’": "'", "“": '"', "”": '"', "×": r"$\times$"}


def esc(s, n=None):
    s = str(s)
    if n and len(s) > n:
        s = s[:n - 3].rstrip() + "..."
    s = "".join(REPL.get(c, c) for c in s)
    return s.encode("ascii", "ignore").decode()


def load(name, folder=None):
    p = os.path.join(folder or A.outputs, name)
    if not os.path.exists(p):
        print(f"SKIP  missing {p}")
        return None
    return json.load(open(p))


def write(name, text):
    open(os.path.join(GEN, name), "w").write(text)
    print("wrote", os.path.join(GEN, name))


def savefig(name):
    plt.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close()
    print("wrote", os.path.join(FIG, name))


def f3(x):
    return f"{x:.3f}"


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


MODELS = [("Base (no fine-tuning)", "base_{s}.json"), ("v1 LoRA (final)", "ft_{s}.json"),
          ("v2 LoRA (oversampled)", "ft_v2_{s}.json")]

# ---------------------------------------------------------------- label distribution (data)
rows_by_split = {}
for s in ["train", "val", "test"]:
    p = os.path.join(A.data, f"{s}.jsonl")
    if os.path.exists(p):
        rows_by_split[s] = [json.loads(l) for l in open(p)]
if len(rows_by_split) == 3:
    labs = [("inclusion", "included"), ("inclusion", "not included"), ("inclusion", "not enough information"),
            ("inclusion", "not applicable"), ("exclusion", "excluded"), ("exclusion", "not excluded"),
            ("exclusion", "not enough information"), ("exclusion", "not applicable")]
    t = ["\\begin{tabular}{llrrr}\\toprule", "Type & Label & Train & Val & Test \\\\ \\midrule"]
    for ct, lb in labs:
        c = [sum(1 for r in rows_by_split[s] if r["criterion_type"] == ct and r["gold_label"] == lb) for s in ["train", "val", "test"]]
        t.append(f"{ct} & {lb} & {c[0]} & {c[1]} & {c[2]} \\\\")
    tot = [len(rows_by_split[s]) for s in ["train", "val", "test"]]
    t += ["\\midrule", f"\\multicolumn{{2}}{{l}}{{All annotations}} & {tot[0]} & {tot[1]} & {tot[2]} \\\\", "\\bottomrule\\end{tabular}"]
    write("labels.tex", "\n".join(t))

# ---------------------------------------------------------------- summary tables (val + test)
for split in ["val", "test"]:
    lines = ["\\begin{tabular}{lrrrrrrr}\\toprule",
             "Model & Acc. & Macro-F1 & Incl. acc. & Incl. F1 & Excl. acc. & Excl. F1 & Evid. F1 \\\\ \\midrule"]
    n = 0
    for label, pat in MODELS:
        d = load(pat.format(s=split))
        if not d:
            continue
        m = d["metrics"]; n += 1
        lines.append(f"{label} & {f3(m['overall']['accuracy'])} & {f3(m['overall']['macro_f1_present_classes'])} & "
                     f"{f3(m['inclusion']['accuracy'])} & {f3(m['inclusion']['macro_f1_present_classes'])} & "
                     f"{f3(m['exclusion']['accuracy'])} & {f3(m['exclusion']['macro_f1_present_classes'])} & "
                     f"{f3(m['evidence_all_records']['micro']['f1'])} \\\\")
    if n:
        write(f"summary_{split}.tex", "\n".join(lines + ["\\bottomrule\\end{tabular}"]))

# ---------------------------------------------------------------- evidence table
for split in ["val", "test"]:
    lines = ["\\begin{tabular}{lrrrrrrr}\\toprule",
             " & \\multicolumn{3}{c}{All records (micro)} & Macro & \\multicolumn{3}{c}{Non-empty gold only (micro)} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){6-8}",
             "Model & P & R & F1 & F1 & P & R & F1 \\\\ \\midrule"]
    n = 0
    for label, pat in MODELS:
        d = load(pat.format(s=split))
        if not d:
            continue
        m = d["metrics"]; a, b = m["evidence_all_records"], m["evidence_nonempty_gold_only"]; n += 1
        lines.append(f"{label} & {f3(a['micro']['precision'])} & {f3(a['micro']['recall'])} & {f3(a['micro']['f1'])} & "
                     f"{f3(a['macro_per_record']['f1'])} & {f3(b['micro']['precision'])} & {f3(b['micro']['recall'])} & "
                     f"{f3(b['micro']['f1'])} \\\\")
        last_empty = m["evidence_n_empty_gold"]; nrows = len(d["rows"])
    if n:
        write(f"evidence_{split}.tex", "\n".join(lines + ["\\bottomrule\\end{tabular}"]))
        write(f"evidence_{split}_note.tex", f"{last_empty} of {nrows}")

# ---------------------------------------------------------------- per-class tables and confusion matrices
for split in ["val", "test"]:
    d = load(f"ft_{split}.json")
    if not d:
        continue
    m = d["metrics"]
    lines = ["\\begin{tabular}{llrrrr}\\toprule", "Type & Label & P & R & F1 & Support \\\\ \\midrule"]
    for ct in ["inclusion", "exclusion"]:
        first = True
        for lb, v in m[ct]["per_class"].items():
            cells = ("n/a", "n/a", "n/a") if v["support"] == 0 else (f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1']:.2f}")
            lines.append(f"{ct if first else ''} & {lb} & {cells[0]} & {cells[1]} & {cells[2]} & {v['support']} \\\\")
            first = False
        if ct == "inclusion":
            lines.append("\\midrule")
    write(f"perclass_{split}.tex", "\n".join(lines + ["\\bottomrule\\end{tabular}"]))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, ct in zip(axes, ["inclusion", "exclusion"]):
        cm = np.array(m[ct]["confusion_matrix"]["rows_gold_cols_pred"]); labels = [ABBR[l] for l in m[ct]["confusion_matrix"]["labels"]]
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if norm[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(f"{ct} criteria (n={int(cm.sum())})")
    plt.tight_layout(); savefig(f"cm_{split}.pdf")

# ---------------------------------------------------------------- training curves
logs = [("1.5B v1 (final)", "train_log.json"), ("1.5B v2 (oversampled)", "train_log_v2.json"), ("0.5B draft", "train_log_draft.json")]
logs = [(t, load(f)) for t, f in logs]
logs = [(t, h) for t, h in logs if h]
if logs:
    fig, axes = plt.subplots(1, len(logs), figsize=(7.0, 2.2), squeeze=False)
    for ax, (title, h) in zip(axes[0], logs):
        tr = [(x["step"], x["loss"]) for x in h if "loss" in x and "eval_loss" not in x]
        ev = [(x["step"], x["eval_loss"]) for x in h if "eval_loss" in x]
        ax.plot(*zip(*tr), label="train loss", lw=1.2)
        ax.plot(*zip(*ev), "o-", label="validation loss", lw=1.2, ms=2.5)
        best = min(ev, key=lambda z: z[1]); ax.axvline(best[0], color="gray", ls=":", lw=0.8)
        ax.set_yscale("log"); ax.set_xlabel("optimizer step"); ax.set_title(f"{title}\nbest val loss {best[1]:.4f} @ step {best[0]}")
    axes[0][0].set_ylabel("loss (answer tokens)"); axes[0][0].legend()
    plt.tight_layout(); savefig("train_curves.pdf")

# ---------------------------------------------------------------- KV-cache benchmark
kv = load("kv_bench.json")
if kv:
    R = kv["results"]
    lines = ["\\begin{tabular}{rrcrrrrrr}\\toprule",
             "$T_{in}$ & $n$ & Tokens equal & Max $|\\Delta$logit$|$ & tok/s no cache & tok/s cached & Speedup & Peak MiB (no cache/cached) & KV MiB \\\\ \\midrule"]
    for r in R:
        eq = "yes" if r["tokens_identical"] else f"no (step {r['first_mismatch_step']})"
        lines.append(f"{r['input_len']} & {r['new_tokens']} & {eq} & {r['max_abs_logit_diff_before_mismatch']:.3f} & "
                     f"{r['no_cache']['tok_per_s_total']:.1f} & {r['kv_cache']['tok_per_s_total']:.1f} & {r['speedup_total']:.1f}$\\times$ & "
                     f"{r['no_cache']['peak_mem_MiB']:.0f}/{r['kv_cache']['peak_mem_MiB']:.0f} & {r['kv_cache_MiB_at_end']:.1f} \\\\")
    write("kv.tex", "\n".join(lines + ["\\bottomrule\\end{tabular}"]))
    cfg = [f"{r['input_len']}/{r['new_tokens']}" for r in R]; x = np.arange(len(R))
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.bar(x - 0.2, [r["no_cache"]["tok_per_s_total"] for r in R], 0.4, label="no cache (reference)")
    ax.bar(x + 0.2, [r["kv_cache"]["tok_per_s_total"] for r in R], 0.4, label="KV cache")
    ax.set_xticks(x); ax.set_xticklabels(cfg); ax.set_xlabel("input tokens / new tokens"); ax.set_ylabel("tokens per second")
    ax.legend(); plt.tight_layout(); savefig("kv_bench.pdf")

# ---------------------------------------------------------------- speculative decoding benchmark
sp = load("spec_bench.json")
if sp:
    R = sp["results"]; base = R[0]
    lines = ["\\begin{tabular}{lrrrrrrr}\\toprule",
             "Mode & Tokens equal & tok/s & Speedup & Accept. rate & Acc./round & Target passes & Peak MiB \\\\ \\midrule",
             f"Greedy + KV cache & -- & {base['tok_per_s']:.1f} & 1.00$\\times$ & -- & -- & -- & {base['peak_MiB']:.0f} \\\\"]
    for r in R[1:]:
        lines.append(f"Speculative $k={r['k']}$ & {r['prompts_equal']} & {r['tok_per_s']:.1f} & {r['speedup_vs_greedy_cache']:.2f}$\\times$ & "
                     f"{r['acceptance_rate']:.2f} & {r['mean_accepted_per_round']:.2f} & {r['target_forward_calls']} & {r['peak_MiB']:.0f} \\\\")
    write("spec.tex", "\n".join(lines + ["\\bottomrule\\end{tabular}"]))
    ks = [r["k"] for r in R[1:]]
    fig, ax = plt.subplots(figsize=(3.4, 2.3)); ax2 = ax.twinx()
    ax.bar(["greedy"] + [f"k={k}" for k in ks], [base["tok_per_s"]] + [r["tok_per_s"] for r in R[1:]], color="#9ecae1")
    ax.set_ylabel("tokens per second"); ax2.plot(range(1, len(ks) + 1), [r["acceptance_rate"] for r in R[1:]], "o-", color="#d94801")
    ax2.set_ylabel("draft acceptance rate", color="#d94801"); ax2.set_ylim(0, 1)
    plt.tight_layout(); savefig("spec_bench.pdf")

# ---------------------------------------------------------------- attention: correctness + timing
at = load("attention_tests.json")
if at:
    by = defaultdict(list)
    for r in at:
        by[tuple(r["shape(B,H,Tq,Tk,D)"])].append(r)
    lines = ["\\begin{tabular}{lrrr}\\toprule", "Shape $(B,H,T_q,T_k,D)$ & Max abs. err. & Mean abs. err. & Max err. vs SDPA \\\\ \\midrule"]
    for shp, rs in by.items():
        lines.append(f"{tuple(shp)} & {sci(max(x['tiled_vs_ref_max'] for x in rs))} & {sci(float(np.mean([x['tiled_vs_ref_mean'] for x in rs])))} & "
                     f"{sci(max(x['ref_vs_sdpa_max'] for x in rs))} \\\\")
    write("attn_tests.tex", "\n".join(lines + ["\\bottomrule\\end{tabular}"]))
    write("attn_tests_note.tex", f"{len(at)} cases, worst tiled-vs-reference error {sci(max(x['tiled_vs_ref_max'] for x in at))}")

ab = load("attention_bench.json")
if ab:
    R = ab["results"]
    lines = ["\\begin{tabular}{rrrrrrr}\\toprule", "$T$ & \\multicolumn{3}{c}{Time (ms)} & \\multicolumn{3}{c}{Peak extra memory (MiB)} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", " & ref. & tiled & SDPA & ref. & tiled & SDPA \\\\ \\midrule"]
    for r in R:
        lines.append(f"{r['T']} & {r['ref_ms']:.2f} & {r['tiled_ms']:.2f} & {r['sdpa_ms']:.2f} & "
                     f"{r['ref_mem_MiB']:.1f} & {r['tiled_mem_MiB']:.1f} & {r['sdpa_mem_MiB']:.1f} \\\\")
    write("attn_bench.tex", "\n".join(lines + ["\\bottomrule\\end{tabular}"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3)); T = [r["T"] for r in R]
    for key, lab in [("ref", "reference"), ("tiled", "tiled online-softmax"), ("sdpa", "PyTorch SDPA (fused)")]:
        axes[0].plot(T, [r[f"{key}_ms"] for r in R], "o-", label=lab, ms=3)
        axes[1].plot(T, [max(r[f"{key}_mem_MiB"], 0.01) for r in R], "o-", label=lab, ms=3)
    for ax, yl in zip(axes, ["time per call (ms)", "peak extra GPU memory (MiB)"]):
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("sequence length $T$"); ax.set_ylabel(yl)
    axes[0].legend(); plt.tight_layout(); savefig("attn_bench.pdf")

# ---------------------------------------------------------------- error analysis (test split only)
PRIORITY = {"excluded": 0, "not included": 0, "not applicable": 1, "not enough information": 2}


def split_numbered(note):
    out, cur, i = [], 0, 0
    marks = []
    while True:
        m = re.compile(rf"(?:^|\s){i}\.\s").search(note, cur)
        if not m:
            break
        marks.append((m.start(), m.end())); cur = m.end(); i += 1
    for j, (_, e) in enumerate(marks):
        out.append(note[e:(marks[j + 1][0] if j + 1 < len(marks) else len(note))].strip())
    return out


def explain(kind, g, p):
    if kind == "evidence":
        return "correct label, cited sentences miss the gold evidence"
    if p == "invalid":
        return "invalid / unparsable output"
    if g == "not enough information" and p in ("not excluded", "included"):
        return "missing information treated as absence or eligibility"
    if g in ("excluded", "not included"):
        return "contradicting evidence missed (contradictory case)"
    if g == "not applicable":
        return "non-applicability not recognised"
    if p == "not enough information":
        return "supporting evidence not found (over-cautious)"
    return "label confusion"


ft = load("ft_test.json")
test_rows = rows_by_split.get("test")
if ft and test_rows:
    byid = {r["annotation_id"]: r for r in test_rows}
    groups = defaultdict(list)
    for r in ft["rows"]:
        g, p = r["gold_label"], (r["pred"]["label"] or "invalid")
        ge, pe = set(r["gold_evidence"]), set(r["pred"]["evidence"])
        if g != p:
            groups[("label", g, p)].append(r)
        elif ge and not (ge & pe):
            groups[("evidence", g, p)].append(r)
    order = sorted(groups, key=lambda k: (k[0] != "label", PRIORITY.get(k[1], 3), -len(groups[k])))
    chosen = []
    for kind_pass, cap in (("label", A.n_errors - 2), ("evidence", A.n_errors)):
        for k in [k for k in order if k[0] == kind_pass]:
            seen = set()
            for r in groups[k]:
                if r["patient_id"] in seen or len(chosen) >= cap:
                    continue
                chosen.append((k, r)); seen.add(r["patient_id"])
                if len(seen) >= 2:
                    break
    detail, lines = [], ["\\begin{tabular}{rllp{2.6cm}llll}\\toprule", "\\# & Type & Gold $\\rightarrow$ predicted & Criterion & Gold ev. & Pred. ev. & Category \\\\ \\midrule"]
    for n, (k, r) in enumerate(chosen, 1):
        src = byid[r["annotation_id"]]; pr = src["prompt"]
        note = pr.split("Patient note (sentences are numbered):\n", 1)[1].split("\n\nCriterion type:")[0]
        crit = pr.split("Criterion: ", 1)[1].split("\n\nDecide")[0]
        sents = split_numbered(note); p = r["pred"]["label"] or "invalid"
        detail.append({"n": n, "annotation_id": r["annotation_id"], "patient_id": r["patient_id"], "type": r["criterion_type"],
                       "criterion": crit, "gold": r["gold_label"], "pred": p, "gold_evidence": r["gold_evidence"],
                       "pred_evidence": r["pred"]["evidence"], "category": explain(k[0], r["gold_label"], p), "raw_output": r["raw"],
                       "gold_evidence_text": [sents[i] for i in r["gold_evidence"] if i < len(sents)],
                       "pred_evidence_text": [sents[i] for i in r["pred"]["evidence"] if i < len(sents)]})
        ge = ",".join(map(str, r["gold_evidence"])) or "--"; pe = ",".join(map(str, r["pred"]["evidence"])) or "--"
        lines.append(f"{n} & {r['criterion_type'][:4]}. & {esc(r['gold_label'])} $\\rightarrow$ {esc(p)} & {esc(crit, 60)} & {ge} & {pe} & {esc(explain(k[0], r['gold_label'], p))} \\\\")
    write("errors.tex", "\n".join(lines + ["\\bottomrule\\end{tabular}"]))
    json.dump(detail, open(os.path.join(GEN, "errors.json"), "w"), indent=2)
    cnt = Counter(f"{d['gold']} -> {d['pred']}" for d in detail)
    print(f"error analysis: {len(detail)} examples from {len(groups)} distinct (kind, gold, pred) groups;", dict(cnt))
    if len(detail) < 8:
        print("WARNING: fewer than 8 distinct errors found - the rubric asks for at least 8")
print("done.")
