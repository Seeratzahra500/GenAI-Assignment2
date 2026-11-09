"""Data preparation for the TrialGPT criterion-level task.

Creates (all under --out_dir, default data/):
  split_manifest.json   patient-grouped split manifest (seed 42, 37/8/8) with leakage checks
  train.jsonl / val.jsonl / test.jsonl   prompt + target + gold fields, one criterion annotation per line
  token_length_stats.json                token lengths (truncation analysis)

Only the expert columns are used. The GPT-4 columns (gpt4_explanation, gpt4_sentences, gpt4_eligibility,
explanation_correctness) and the original 'training' flag are never read.

Usage:  python src/prepare_data.py                      (downloads the dataset from Hugging Face)
        python src/prepare_data.py --parquet file.parquet   (offline)
"""
import argparse
import ast
import json
import os

import numpy as np
import pandas as pd

LABELS = {"inclusion": ["included", "not included", "not enough information", "not applicable"],
          "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"]}
SYSTEM = "You are a clinical trial eligibility assistant. Use only the patient note."
TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_ids(s):
    return [int(x) for x in (s if isinstance(s, list) else ast.literal_eval(s))]


def build_prompt(r):
    labels = " | ".join(LABELS[r.criterion_type])
    return (f"Patient note (sentences are numbered):\n{r.note}\n\n"
            f"Criterion type: {r.criterion_type}\nCriterion: {r.criterion_text}\n\n"
            f"Decide the patient's status for this criterion. Allowed labels: {labels}.\n"
            'Answer with JSON only: {"label": "<label>", "evidence": [<sentence ids>]}')


def build_target(r):
    return json.dumps({"label": r.expert_eligibility, "evidence": parse_ids(r.expert_sentences)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=None, help="local parquet file instead of downloading")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--skip_token_stats", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if a.parquet:
        df = pd.read_parquet(a.parquet)
    else:
        from datasets import load_dataset
        df = load_dataset("ncbi/TrialGPT-Criterion-Annotations", split="train").to_pandas()
    df["criterion_text"] = df["criterion_text"].fillna("")
    print("rows:", len(df), "| patients:", df.patient_id.nunique())

    # ---- patient-grouped split: sort unique IDs, shuffle with default_rng(42), 37 / 8 / 8 ----
    pids = np.array(sorted(df.patient_id.unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(pids)
    splits = {"train": list(pids[:37]), "val": list(pids[37:45]), "test": list(pids[45:])}
    assert [len(v) for v in splits.values()] == [37, 8, 8], "unexpected number of patients"
    names = list(splits)
    for i in range(3):                                   # leakage check: no patient in two splits
        for j in range(i + 1, 3):
            assert not set(splits[names[i]]) & set(splits[names[j]]), f"patient leakage {names[i]}/{names[j]}"
    manifest = {"seed": 42,
                "method": "sorted unique patient_id -> np.random.default_rng(42).shuffle -> 37/8/8",
                "patients": {k: [str(x) for x in v] for k, v in splits.items()},
                "counts": {k: {"patients": len(v), "annotations": int(df.patient_id.isin(v).sum())}
                           for k, v in splits.items()}}
    assert sum(c["annotations"] for c in manifest["counts"].values()) == len(df)
    json.dump(manifest, open(os.path.join(a.out_dir, "split_manifest.json"), "w"), indent=2)
    print("split counts:", manifest["counts"])

    # ---- prompts / targets ----
    tok = None
    if not a.skip_token_stats:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
    stats = {}
    for name, ids in splits.items():
        sub = df[df.patient_id.isin(ids)]
        rows, lens = [], []
        for r in sub.itertuples():
            assert r.expert_eligibility in LABELS[r.criterion_type], (r.annotation_id, r.expert_eligibility)
            prompt, target = build_prompt(r), build_target(r)
            rows.append({"annotation_id": int(r.annotation_id), "patient_id": str(r.patient_id),
                         "criterion_type": r.criterion_type, "system": SYSTEM, "prompt": prompt, "target": target,
                         "gold_label": r.expert_eligibility, "gold_evidence": parse_ids(r.expert_sentences)})
            if tok is not None:
                text = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                                {"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True)
                lens.append(len(tok(text).input_ids) + len(tok(target).input_ids))
        with open(os.path.join(a.out_dir, f"{name}.jsonl"), "w") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
        if lens:
            lens = np.array(lens)
            stats[name] = {"n": len(rows), "mean": int(lens.mean()), "p95": int(np.percentile(lens, 95)),
                           "max": int(lens.max()), "over_1024": int((lens > 1024).sum()),
                           "over_2048": int((lens > 2048).sum())}
    if stats:
        json.dump(stats, open(os.path.join(a.out_dir, "token_length_stats.json"), "w"), indent=2)
        print("token lengths:", stats)


if __name__ == "__main__":
    main()
