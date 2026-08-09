from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heal_repro.data import VARIANT_FILES, prompt_from_row, read_rows


DEFAULT_DATA = ROOT / "workspace" / "HEAL_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local model over HEAL CSV prompts.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--environment", choices=("virtualhome", "behavior"), default="virtualhome")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_FILES),
        default=[
            "distractor_injection",
            "object_removal",
            "scene_object_synonymous",
            "scene_task_contradiction",
        ],
    )
    parser.add_argument("--backend", choices=("transformers", "empty"), default="transformers")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "predictions.jsonl")
    parser.add_argument("--max-samples", type=int, help="Maximum rows per variant; omit for the full set.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def existing_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                keys.add((item["environment"], item["variant"], int(item["row_index"])))
    return keys


def build_generator(args: argparse.Namespace):
    if args.backend == "empty":
        return lambda _prompt: '{"node goals": [], "edge goals": [], "action goals": []}'

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            output[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )

    return generate


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise SystemExit(f"Output exists: {args.output}. Use --resume or choose another path.")
    completed = existing_keys(args.output) if args.resume else set()
    generate = build_generator(args)
    mode = "a" if args.resume else "w"

    with args.output.open(mode, encoding="utf-8") as handle:
        for variant in args.variants:
            rows = read_rows(args.data_root, args.environment, variant)
            if args.max_samples is not None:
                rows = rows[: args.max_samples]
            for row_index, row in enumerate(rows):
                key = (args.environment, variant, row_index)
                if key in completed:
                    continue
                response = generate(prompt_from_row(row, variant))
                record = {
                    "environment": args.environment,
                    "variant": variant,
                    "row_index": row_index,
                    "task_id": row.get("task_id", ""),
                    "model": "empty-structured-baseline" if args.backend == "empty" else args.model,
                    "response": response,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{variant}] {row_index + 1}/{len(rows)} {row.get('task_id', '')}")
    print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
