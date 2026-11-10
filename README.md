# Generative AI – Assignment 2: Clinical-trial criterion eligibility with a LoRA-tuned decoder

Author: Seerat E Zahra (23i-0640), FAST-NUCES Islamabad. Individual assignment.

Task: given a patient note and one trial criterion (inclusion or exclusion), predict the eligibility label and the supporting sentence IDs
(`{"label": "...", "evidence": [ids]}`). Dataset: `ncbi/TrialGPT-Criterion-Annotations` (only expert columns are used; GPT-4 columns and the original `training` flag are never read).

* Target model: `Qwen/Qwen2.5-1.5B-Instruct` + LoRA (r=16, alpha=32). Draft model for speculative decoding: `Qwen/Qwen2.5-0.5B-Instruct` + LoRA (same tokenizer).
* Adapters (Hugging Face): **ADD LINKS HERE**   · Demo video: **ADD LINK**   · Report: `report/main.pdf`

## Layout
| Path | Purpose |
|---|---|
| `src/prepare_data.py` | patient-grouped split (seed 42, 37/8/8), manifest, prompts/targets, token stats |
| `src/common.py` | output parser and metrics (accuracy, macro-F1, per-class, confusion, evidence P/R/F1) |
| `src/train.py`, `src/train_draft.py`, `src/train_v2.py` | LoRA training (1.5B v1, 0.5B draft, 1.5B v2 oversampling variant) |
| `src/evaluate.py` | greedy batch evaluation for any split / adapter |
| `src/attention.py` | reference attention + tiled online-softmax attention + tests |
| `src/attention_bench.py` | timing / peak-memory benchmark of reference vs tiled vs SDPA |
| `src/kvcache.py` | KV-cache decoder, no-cache reference decoder, benchmark |
| `src/speculative.py` | greedy speculative decoding + benchmark |
| `src/report_assets.py` | builds every figure / LaTeX table of the report from `outputs/` |
| `app/app.py` | Gradio app (screening, decoding comparison, results/report export) |
| `data/` | split manifest, train/val/test jsonl, token statistics |
| `outputs/` | all metrics / benchmark JSON files and training logs |
| `report/` | IEEE LaTeX report |

## Reproduce (Kaggle, 1x T4/P100 16 GB, ~2 h GPU in total)
```bash
pip install -r requirements.txt            # on Kaggle: pip uninstall -y torchao  (conflicts with peft)
python src/prepare_data.py                 # writes data/ (identical to the committed files)
python src/evaluate.py --split val  --out outputs/base_val.json            # base model, no adapter
python src/train.py                        # LoRA 1.5B  -> outputs/lora_1.5b/best  (best checkpoint by val loss)
python src/evaluate.py --split val  --adapter outputs/lora_1.5b/best --out outputs/ft_val.json
python src/train_draft.py                  # LoRA 0.5B draft
python src/attention.py                    # correctness tests -> outputs/attention_tests.json
python src/attention_bench.py              # -> outputs/attention_bench.json
python src/kvcache.py                      # -> outputs/kv_bench.json
python src/speculative.py                  # -> outputs/spec_bench.json
# test split: run ONCE, after the model was chosen on validation
python src/evaluate.py --split test --out outputs/base_test.json
python src/evaluate.py --split test --adapter outputs/lora_1.5b/best --out outputs/ft_test.json
python src/report_assets.py                # figures + tables -> report/
cd report && pdflatex main && pdflatex main
python app/app.py                          # Gradio app; prints a local (and optional public) URL
```
Check each script's `--help` for the exact flags. Seeds: split 42, training 42. fp16 greedy decoding is deterministic on the same GPU but can differ across GPU models/driver versions on near-ties.

## Honest notes
* Test split was used only for final reporting; model selection used validation loss/metrics.
* Validation/test have very small supports for rare classes (e.g. `excluded`, `not included`, `not applicable`), so per-class numbers are noisy; macro-F1 averages only classes present in the split.
* Speculative decoding was *slower* than plain KV-cache greedy decoding in our setting (see report); outputs are identical to the target's greedy output.
* The app is an educational prototype, not a medical or enrolment decision tool. A public Gradio link only lives while the Kaggle session runs.
* Parts of the code and report were written with AI assistance; see the AI-use appendix in the report.
