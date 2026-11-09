import json, re, numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

LABELS = {"inclusion": ["included","not included","not enough information","not applicable"],
          "exclusion": ["excluded","not excluded","not enough information","not applicable"]}
ALL_LABELS = ["included","not included","excluded","not excluded","not enough information","not applicable"]

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

def chat_prompt(tok, x):
    return tok.apply_chat_template(
        [{"role":"system","content":x["system"]},{"role":"user","content":x["prompt"]}],
        tokenize=False, add_generation_prompt=True)

def parse_output(text):
    """Returns (label or None, evidence ids, valid_json)."""
    m = re.search(r"\{.*?\}", text, re.S)
    if not m: return None, [], False
    try:
        d = json.loads(m.group(0))
        return str(d.get("label","")).strip().lower(), [int(i) for i in d.get("evidence",[])], True
    except Exception:
        return None, [], False

def _cls(gold, pred, labels):
    if not gold: return None
    present = [l for l in labels if l in set(gold)]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    return {"n": len(gold),
            "accuracy": float(np.mean([g == q for g, q in zip(gold, pred)])),
            "macro_f1_present_classes": float(f1_score(gold, pred, labels=present, average="macro", zero_division=0)),
            "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                          for i, l in enumerate(labels)},
            "confusion_matrix": {"labels": labels + ["invalid"], "rows_gold_cols_pred":
                                 confusion_matrix(gold, pred, labels=labels + ["invalid"]).tolist()}}

def _evid(golds, preds):
    tp = fp = fn = 0; P = []; R = []; F = []
    for g, p in zip(golds, preds):
        g, p = set(g), set(p)
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        if not g and not p: pr = rc = f = 1.0          # both empty = correct "no evidence"
        else:
            pr = len(g & p) / len(p) if p else 0.0
            rc = len(g & p) / len(g) if g else 0.0
            f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        P.append(pr); R.append(rc); F.append(f)
    mp = tp / (tp + fp) if tp + fp else 0.0
    mr = tp / (tp + fn) if tp + fn else 0.0
    return {"micro": {"precision": mp, "recall": mr, "f1": 2*mp*mr/(mp+mr) if mp+mr else 0.0},
            "macro_per_record": {"precision": float(np.mean(P)), "recall": float(np.mean(R)), "f1": float(np.mean(F))}}

def compute_metrics(rows, preds):
    mapped = [p["label"] if p["label"] in LABELS[r["criterion_type"]] else "invalid" for r, p in zip(rows, preds)]
    out = {"parse_failure_rate": float(np.mean([not p["valid"] for p in preds]))}
    out["overall"] = _cls([r["gold_label"] for r in rows], mapped, ALL_LABELS)
    for t in ["inclusion", "exclusion"]:
        ix = [i for i, r in enumerate(rows) if r["criterion_type"] == t]
        out[t] = _cls([rows[i]["gold_label"] for i in ix], [mapped[i] for i in ix], LABELS[t])
    g = [r["gold_evidence"] for r in rows]; p = [q["evidence"] for q in preds]
    out["evidence_all_records"] = _evid(g, p)
    ne = [i for i in range(len(rows)) if g[i]]
    out["evidence_nonempty_gold_only"] = _evid([g[i] for i in ne], [p[i] for i in ne])
    out["evidence_n_empty_gold"] = len(rows) - len(ne)
    return out
