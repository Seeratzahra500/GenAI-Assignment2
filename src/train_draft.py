import os; os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from common import load_jsonl, chat_prompt

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"; MAXLEN = 1024; OUT = "outputs/lora_0.5b"
tok = AutoTokenizer.from_pretrained(MODEL)

def encode(x):
    p = tok(chat_prompt(tok, x), add_special_tokens=False).input_ids
    t = tok(x["target"] + "<|im_end|>", add_special_tokens=False).input_ids
    full, lab = p + t, [-100] * len(p) + t          # loss only on the JSON answer
    return {"input_ids": full[:MAXLEN], "labels": lab[:MAXLEN], "truncated": len(full) > MAXLEN}

train = [encode(x) for x in load_jsonl("data/train.jsonl")]
val = [encode(x) for x in load_jsonl("data/val.jsonl")]
print("truncated train/val:", sum(e["truncated"] for e in train), sum(e["truncated"] for e in val))

def collate(b):
    m = max(len(x["input_ids"]) for x in b); pad = tok.pad_token_id
    return {"input_ids": torch.tensor([x["input_ids"] + [pad] * (m - len(x["input_ids"])) for x in b]),
            "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (m - len(x["input_ids"])) for x in b]),
            "labels": torch.tensor([x["labels"] + [-100] * (m - len(x["labels"])) for x in b])}

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                 target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model = get_peft_model(model, cfg)
for p in model.parameters():
    if p.requires_grad: p.data = p.data.float()     # fp32 adapters for fp16 AMP
model.config.use_cache = False
model.print_trainable_parameters()

args = TrainingArguments(output_dir=OUT, num_train_epochs=3, per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=16, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=6, fp16=True,
    logging_steps=5, eval_strategy="steps", eval_steps=10, save_strategy="steps", save_steps=10, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    report_to="none", gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, seed=42, remove_unused_columns=False)
tr = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val, data_collator=collate)
tr.train()
tr.save_model(f"{OUT}/best")
json.dump(tr.state.log_history, open("outputs/train_log_draft.json", "w"), indent=2)
json.dump({"model": MODEL, "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": "all linear proj"},
           "lr": 2e-4, "epochs": 3, "eff_batch": 16, "per_device_batch": 1, "grad_accum": 16, "grad_checkpointing": True, "max_len": MAXLEN, "fp16": True, "seed": 42,
           "best_checkpoint": tr.state.best_model_checkpoint}, open("outputs/train_config_draft.json", "w"), indent=2)
print("best:", tr.state.best_model_checkpoint, "best val loss:", tr.state.best_metric)
