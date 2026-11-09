import os, sys, re, json, time, datetime, argparse, threading, statistics
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import torch, pandas as pd, gradio as gr
from transformers import AutoTokenizer
from common import parse_output, chat_prompt
from kvcache import generate_nocache, generate_cached, sync
from speculative import generate_spec, load

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter", default="Seerat500/trialgpt-qwen2.5-1.5b-lora")
ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--draft_adapter", default="Seerat500/trialgpt-qwen2.5-0.5b-draft-lora")
ap.add_argument("--port", type=int, default=7860)
ap.add_argument("--share", action="store_true")
args = ap.parse_args()

print("Loading models (first start downloads weights) ...", flush=True)
TOK = AutoTokenizer.from_pretrained(args.target)
TARGET = load(args.target, args.adapter, "float16")
DRAFT = load(args.draft, args.draft_adapter, "float16")
EOS = TOK.convert_tokens_to_ids("<|im_end|>")
LOCK = threading.Lock()

SYSTEM = "You are a clinical trial eligibility assistant. Use only the patient note."
LABELS = {"inclusion": ["included", "not included", "not enough information", "not applicable"],
          "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"]}
MODES = ["KV cache (greedy)", "No cache (reference greedy)", "Speculative decoding (greedy)"]
N_NEW, MAX_PROMPT_TOKENS = 64, 1400
COLS = ["#", "Type", "Criterion", "Prediction", "Evidence IDs", "Evidence text", "Seconds"]
EMPTY = pd.DataFrame(columns=COLS)
DISCLAIMER = ("**Educational screening prototype, not a clinical enrollment decision.** Outputs come from a small "
              "fine-tuned language model, can be wrong, and must never be used to enroll or exclude a patient. "
              "Missing evidence is not proof of eligibility or ineligibility.")

def build_prompt(note_numbered, ctype, ctext):      # identical to the training-data prompt
    labels = " | ".join(LABELS[ctype])
    return (f"Patient note (sentences are numbered):\n{note_numbered}\n\n"
            f"Criterion type: {ctype}\nCriterion: {ctext}\n\n"
            f"Decide the patient's status for this criterion. Allowed labels: {labels}.\n"
            'Answer with JSON only: {"label": "<label>", "evidence": [<sentence ids>]}')

def split_sentences(note):
    note = note.strip()
    if re.match(r"^0\.\s", note):                     # already numbered like the dataset: keep numbering
        marks, cur, i = [], 0, 0
        while True:
            m = re.compile(rf"(?:^|\s){i}\.\s").search(note, cur)
            if not m: break
            marks.append((m.start(), m.end())); cur = m.end(); i += 1
        if marks:
            out = [note[e:(marks[j + 1][0] if j + 1 < len(marks) else len(note))].strip() for j, (s, e) in enumerate(marks)]
            return out, "numbered input kept as given"
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", note) if s.strip()]
    return sents, "automatic regex sentence splitting"

def parse_criteria(text, ctype):
    out = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", line).strip()
        if line: out.append((ctype, line))
    return out

def prepare(note, inc, exc):
    note = (note or "").strip()
    if len(note) < 30: return None, "The patient description is empty or too short (at least ~30 characters needed)."
    if len(note) > 8000: return None, "The patient description is too long (limit 8000 characters)."
    sents, how = split_sentences(note)
    if not sents or len(sents) > 80: return None, "Could not find 1-80 sentences in the patient description."
    crit = parse_criteria(inc, "inclusion") + parse_criteria(exc, "exclusion")
    if not crit: return None, "Add at least one inclusion or exclusion criterion (one per line)."
    if len(crit) > 25: return None, "Too many criteria (limit 25 per run)."
    if any(len(c[1]) > 1000 for c in crit): return None, "A criterion is longer than 1000 characters."
    numbered = " ".join(f"{i}. {s}" for i, s in enumerate(sents))
    return dict(sents=sents, how=how, numbered=numbered, crit=crit), None

def encode(numbered, ctype, ctext):
    x = {"system": SYSTEM, "prompt": build_prompt(numbered, ctype, ctext)}
    return TOK(chat_prompt(TOK, x), add_special_tokens=False, return_tensors="pt").input_ids.cuda()

def decode(mode, ids, k):
    if mode == MODES[0]: return generate_cached(TARGET, ids, N_NEW)[0]
    if mode == MODES[1]: return generate_nocache(TARGET, ids, N_NEW)[0]
    return generate_spec(TARGET, DRAFT, ids, N_NEW, int(k))[0]

def aggregate(res):
    bad = [r for r in res if r["label"] is None]
    exc = [r for r in res if r["type"] == "exclusion" and r["label"] == "excluded"]
    inc_no = [r for r in res if r["type"] == "inclusion" and r["label"] == "not included"]
    unk = [r for r in res if r["label"] == "not enough information"]
    if exc: head = "LIKELY INELIGIBLE: at least one exclusion criterion is predicted to be met"
    elif inc_no: head = "LIKELY INELIGIBLE: at least one inclusion criterion is predicted not to be met"
    elif unk or bad: head = "UNDETERMINED: some criteria lack enough information or could not be parsed"
    else: head = "NO DISQUALIFYING FINDING: every assessed criterion is satisfied or not applicable"
    detail = f"{len(exc)} exclusion(s) met, {len(inc_no)} inclusion(s) not met, {len(unk)} with insufficient information, {len(bad)} unparsed"
    rule = ("Aggregation rule (explicit, not a ground-truth label): any 'excluded' or 'not included' -> likely ineligible; "
            "else any 'not enough information' or unparsed output -> undetermined; else no disqualifying finding. "
            "'not applicable' criteria are ignored.")
    return head, detail, rule

def make_report(prep, res, head, detail, rule, mode, k):
    ts = datetime.datetime.now()
    L = ["# Eligibility screening report", "", "> " + DISCLAIMER, "", "## Provenance",
         f"- Generated: {ts:%Y-%m-%d %H:%M:%S}", f"- Base model: {args.target}", f"- Fine-tuned adapter: {args.adapter}",
         f"- Decoding: {mode}" + (f", k={int(k)} (draft: {args.draft} + {args.draft_adapter})" if mode == MODES[2] else "") + ", greedy",
         f"- Sentence handling: {prep['how']}", "", "## Preliminary outcome", f"**{head}**", "", detail, "", f"_{rule}_", "",
         "## Criterion-level results", "", "| # | Type | Criterion | Prediction | Evidence |", "|---|---|---|---|---|"]
    for n, r in enumerate(res, 1):
        ev = "; ".join(f"[{i}] {prep['sents'][i]}" for i in r["evidence"]) or "none cited"
        L.append(f"| {n} | {r['type']} | {r['criterion'].replace('|', '/')} | {r['label'] or 'invalid output'} | {ev.replace('|', '/')} |")
    L += ["", "## Patient description (numbered sentences)", ""] + [f"{i}. {s}" for i, s in enumerate(prep["sents"])]
    L += ["", "## Raw model outputs", ""] + [f"{n}. `{r['raw']}`" for n, r in enumerate(res, 1)]
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    path = os.path.join(ROOT, "reports", f"screening_report_{ts:%Y%m%d_%H%M%S}.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    return path

def fail(msg):
    gr.Warning(msg)
    return f"**Input problem:** {msg}", EMPTY, "", None

def screen(note, inc, exc, mode, k, progress=gr.Progress()):
    prep, err = prepare(note, inc, exc)
    if err: return fail(err)
    res = []
    with LOCK:
        for n, (ctype, ctext) in enumerate(prep["crit"]):
            progress(n / len(prep["crit"]), desc=f"Criterion {n + 1}/{len(prep['crit'])}")
            ids = encode(prep["numbered"], ctype, ctext)
            if ids.shape[1] > MAX_PROMPT_TOKENS: return fail(f"Input too long ({ids.shape[1]} tokens, limit {MAX_PROMPT_TOKENS}). Shorten the note.")
            sync(); t0 = time.perf_counter()
            toks = decode(mode, ids, k)
            sync(); dt = time.perf_counter() - t0
            raw = TOK.decode(toks[:toks.index(EOS) if EOS in toks else len(toks)], skip_special_tokens=True)
            label, ev, valid = parse_output(raw)
            ok = bool(valid and label in LABELS[ctype])
            res.append({"type": ctype, "criterion": ctext, "label": label if ok else None, "raw": raw, "sec": dt,
                        "evidence": sorted({i for i in ev if 0 <= i < len(prep["sents"])}), "bad_ids": [i for i in ev if not 0 <= i < len(prep["sents"])]})
    head, detail, rule = aggregate(res)
    rows = [[n, r["type"], r["criterion"], r["label"] or "invalid output",
             ", ".join(map(str, r["evidence"])) or "none cited",
             " | ".join(f"[{i}] {prep['sents'][i]}" for i in r["evidence"]) or "(no sentence cited)", round(r["sec"], 2)]
            for n, r in enumerate(res, 1)]
    det = "\n\n".join(f"**{n}. [{r['type']}] {r['criterion']}**\n```\n{r['raw']}\n```" + (f"\nIgnored out-of-range sentence IDs: {r['bad_ids']}" if r["bad_ids"] else "")
                      for n, r in enumerate(res, 1))
    overall = f"## {head}\n{detail}\n\n_{rule}_\n\n_Sentence handling: {prep['how']}. {len(prep['sents'])} sentences._"
    return overall, pd.DataFrame(rows, columns=COLS), "### Raw model outputs\n" + det, make_report(prep, res, head, detail, rule, mode, k)

def bench(note, inc, exc, ks_sel, reps):
    prep, err = prepare(note, inc, exc)
    if err: gr.Warning(err); return f"**Input problem:** {err}", pd.DataFrame(), pd.DataFrame()
    ctype, ctext = prep["crit"][0]
    ids = encode(prep["numbered"], ctype, ctext)
    if ids.shape[1] > MAX_PROMPT_TOKENS: return "Input too long.", pd.DataFrame(), pd.DataFrame()
    n = 48
    def run(fn):
        fn(); ts, peak, out = [], 0, None                      # one warm-up, not counted
        for _ in range(int(reps)):
            torch.cuda.reset_peak_memory_stats(); sync(); t = time.perf_counter()
            out = fn(); sync(); ts.append(time.perf_counter() - t)
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**20)
        return out, statistics.median(ts), peak
    rows = []
    with LOCK:
        (ref, _, _), s_ref, p_ref = run(lambda: generate_nocache(TARGET, ids, n))
        (tkv, _, _), s_kv, p_kv = run(lambda: generate_cached(TARGET, ids, n))
        def first_diff(a, b): return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        rows.append(["No cache", s_ref, n / s_ref, s_kv / s_ref, p_ref, "reference", "-", n])
        rows.append(["KV cache", s_kv, n / s_kv, 1.0, p_kv, "equal" if tkv == ref else f"differs at step {first_diff(ref, tkv)}", "-", n])
        for k in [int(x) for x in (ks_sel or ["4"])]:
            (ts_, st, _), s, p = run(lambda: generate_spec(TARGET, DRAFT, ids, n, k))
            rows.append([f"Speculative k={k}", s, n / s, s_kv / s, p, "equal" if ts_ == ref else f"differs at step {first_diff(ref, ts_)}",
                         round(st["accepted"] / max(1, st["drafted"]), 2), st["target_calls"]])
    df = pd.DataFrame(rows, columns=["Mode", "Seconds", "Tokens/s", "Speedup vs KV cache", "Peak GPU MiB (incl. weights)",
                                     "Tokens vs no-cache reference", "Draft acceptance rate", "Target forward passes"]).round(3)
    msg = (f"Benchmarked criterion 1 ({ctype}): **{ctext}**. Prompt length {ids.shape[1]} tokens, {n} new tokens, greedy, batch 1, "
           f"median of {int(reps)} timed runs after 1 warm-up. In fp16, near-tied logits can make cached or speculative paths differ from the reference.")
    return msg, df, df[["Mode", "Tokens/s"]]

def stored_results():
    rows = []
    for name, f in [("Base model (no fine-tuning)", "base_test.json"), ("Fine-tuned LoRA", "ft_test.json")]:
        p = os.path.join(ROOT, "outputs", f)
        if os.path.exists(p):
            m = json.load(open(p))["metrics"]
            rows.append([name, round(m["overall"]["accuracy"], 3), round(m["overall"]["macro_f1_present_classes"], 3),
                         round(m["evidence_all_records"]["micro"]["f1"], 3), round(m["parse_failure_rate"], 3)])
    return pd.DataFrame(rows, columns=["Model", "Accuracy", "Macro-F1 (classes present)", "Evidence micro-F1", "Parse failure rate"])

EX_NOTE = ("A 67-year-old man with a 20-year history of type 2 diabetes and hypertension presents with exertional chest pain for 3 weeks. "
           "He quit smoking 10 years ago. He takes metformin and lisinopril. Blood pressure is 152/90 mmHg. ECG shows ST depression in leads V4 to V6. "
           "Serum creatinine is 1.1 mg/dL. He has no known drug allergies and agrees to follow the study protocol.")
EX_INC = "Age 40 to 75 years\nDiagnosis of type 2 diabetes mellitus\nHbA1c above 9%"
EX_EXC = "Currently pregnant\nSerum creatinine above 3.0 mg/dL\nKnown allergy to metformin\nPrevious myocardial infarction"

with gr.Blocks(title="Clinical Trial Eligibility Screener") as demo:
    gr.Markdown("# Clinical trial eligibility screener (educational prototype)")
    gr.Markdown(DISCLAIMER)
    with gr.Tab("Screening"):
        with gr.Row():
            note = gr.Textbox(label="Patient description (plain text, or dataset-style '0. ... 1. ...')", lines=12)
            with gr.Column():
                inc = gr.Textbox(label="Inclusion criteria (one per line)", lines=5)
                exc = gr.Textbox(label="Exclusion criteria (one per line)", lines=5)
        with gr.Row():
            mode = gr.Dropdown(MODES, value=MODES[0], label="Decoding mode")
            k = gr.Slider(1, 8, value=4, step=1, label="Draft tokens k (speculative mode only)")
        with gr.Row():
            run_btn = gr.Button("Run screening", variant="primary"); ex_btn = gr.Button("Load example"); clr_btn = gr.Button("Clear")
        overall = gr.Markdown()
        table = gr.Dataframe(value=EMPTY, headers=COLS, interactive=False, wrap=True)
        details = gr.Markdown()
        report = gr.File(label="Download screening report (Markdown)")
        run_btn.click(screen, [note, inc, exc, mode, k], [overall, table, details, report])
        ex_btn.click(lambda: (EX_NOTE, EX_INC, EX_EXC), None, [note, inc, exc])
        clr_btn.click(lambda: ("", "", "", "", EMPTY, "", None), None, [note, inc, exc, overall, table, details, report])
    with gr.Tab("Inference benchmark"):
        gr.Markdown("Runs the **first criterion** from the Screening tab live through no-cache, KV-cache and speculative decoding.")
        ks = gr.CheckboxGroup(["2", "4", "6", "8"], value=["4"], label="Speculative k values")
        reps = gr.Slider(1, 3, value=2, step=1, label="Timed repeats (median)")
        b_btn = gr.Button("Run live benchmark", variant="primary")
        b_msg = gr.Markdown(); b_table = gr.Dataframe(interactive=False, wrap=True)
        b_plot = gr.BarPlot(x="Mode", y="Tokens/s", title="Throughput (tokens/s, higher is better)", height=300)
        b_btn.click(bench, [note, inc, exc, ks, reps], [b_msg, b_table, b_plot])
    with gr.Tab("Held-out test results"):
        gr.Markdown("Criterion-level results on the fixed 8-patient test split (206 annotations), loaded from `outputs/`.")
        gr.Dataframe(value=stored_results(), interactive=False)

demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
