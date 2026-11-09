import os, sys; os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from common import load_jsonl
from kvcache import generate_nocache, generate_cached

dtype = getattr(torch, sys.argv[1] if len(sys.argv) > 1 else "float16")
name = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map={"": 0})
model = PeftModel.from_pretrained(model, "outputs/lora_1.5b/best").merge_and_unload().eval()
text = "\n".join(r["prompt"] for r in load_jsonl("data/val.jsonl")[:60])
ids = tok(text, return_tensors="pt").input_ids[0].cuda()[:256][None]
t_ref, l_ref, _ = generate_nocache(model, ids, 64)
t_kv, l_kv, _ = generate_cached(model, ids, 64)
print(dtype, "| tokens identical over 64 steps:", t_ref == t_kv)
i = next((j for j, (a, b) in enumerate(zip(t_ref, t_kv)) if a != b), None)
if i is not None:
    print("first mismatch step:", i, "| no-cache token", repr(tok.decode(t_ref[i])), "| cached token", repr(tok.decode(t_kv[i])))
    for label, lg in [("no-cache", l_ref[i]), ("cached", l_kv[i])]:
        top = lg[0].topk(2)
        print(label, "top-2:", [tok.decode(t) for t in top.indices],
              "logits:", [round(v, 4) for v in top.values.tolist()],
              "margin:", round((top.values[0] - top.values[1]).item(), 4))
