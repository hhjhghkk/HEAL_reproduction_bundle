"""Minimal local-generation template.

HEAL/EAI prompts are strings. Feed the generated prompt to your chosen local LLM,
then save the raw model response in the format expected by the EAI evaluator.
Adjust model name and output serialization to your experiment.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = "Replace this string with one generated EAI/HEAL goal-interpretation prompt."

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
messages = [{"role": "user", "content": PROMPT}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
answer = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(answer)
