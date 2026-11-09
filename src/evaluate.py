import os; os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import argparse, json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import load_jsonl, chat_prompt, parse_output, compute_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter", default=None)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--max_new", type=int, default=64)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map={"": 0})
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.eval()

rows = load_jsonl(f"data/{a.split}.jsonl")
order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"]))
raw = [None] * len(rows); t0 = time.time()
for s in range(0, len(order), a.bs):
    idx = order[s:s + a.bs]
    enc = tok([chat_prompt(tok, rows[i]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    for i, d in zip(idx, tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
        raw[i] = d
preds = []
for r in raw:
    l, e, v = parse_output(r); preds.append({"label": l, "evidence": e, "valid": v})
m = compute_metrics(rows, preds)
res = {"settings": {"model": a.model, "adapter": a.adapter, "split": a.split, "decoding": "greedy (do_sample=False)",
                    "max_new_tokens": a.max_new, "dtype": "float16", "batch_size": a.bs, "seconds": round(time.time() - t0, 1)},
       "metrics": m,
       "rows": [{"annotation_id": r["annotation_id"], "patient_id": r["patient_id"], "criterion_type": r["criterion_type"],
                 "gold_label": r["gold_label"], "gold_evidence": r["gold_evidence"], "raw": raw[i], "pred": preds[i]}
                for i, r in enumerate(rows)]}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
print(json.dumps({k: m[k] for k in ["parse_failure_rate"]}),
      "| overall acc", round(m["overall"]["accuracy"], 3), "macroF1", round(m["overall"]["macro_f1_present_classes"], 3),
      "| evidence micro-F1", round(m["evidence_all_records"]["micro"]["f1"], 3))
