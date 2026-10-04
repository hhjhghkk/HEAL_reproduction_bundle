from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help="Number of candidate responses per selected sample.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable random sampling. Required when --num-generations is greater than 1.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. Each sample/candidate receives a stable derived seed.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--row-indices",
        type=int,
        nargs="+",
        help="Only run these zero-based CSV row indices.",
    )
    selection.add_argument(
        "--task-ids",
        nargs="+",
        help="Only run rows whose task_id is listed here.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_generations < 1:
        raise SystemExit("--num-generations must be at least 1.")
    if args.num_generations > 1 and not args.do_sample:
        raise SystemExit(
            "Multiple greedy generations would normally be identical. "
            "Add --do-sample when --num-generations is greater than 1."
        )
    if args.do_sample and args.temperature <= 0:
        raise SystemExit("--temperature must be greater than 0 when sampling.")
    if not 0 < args.top_p <= 1:
        raise SystemExit("--top-p must be in the interval (0, 1].")
    if args.top_k < 0:
        raise SystemExit("--top-k must be zero or greater.")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be at least 1 when supplied.")
    if args.row_indices and any(index < 0 for index in args.row_indices):
        raise SystemExit("--row-indices only accepts zero-based non-negative integers.")


def existing_keys(path: Path) -> set[tuple[str, str, int, int]]:
    """Read completed candidate keys from JSONL.

    Older prediction files do not contain ``generation_id``.  They are treated
    as candidate 0, so the new resume behavior remains backward compatible.
    """

    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                keys.add(
                    (
                        item["environment"],
                        item["variant"],
                        int(item["row_index"]),
                        int(item.get("generation_id", 0)),
                    )
                )
    return keys


def candidate_seed(
    base_seed: int,
    environment: str,
    variant: str,
    row_index: int,
    generation_id: int,
) -> int:
    """Create a stable, distinct seed for one sample candidate."""

    identity = f"{base_seed}|{environment}|{variant}|{row_index}|{generation_id}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def select_rows(
    rows: list[dict[str, str]],
    row_indices: list[int] | None,
    task_ids: list[str] | None,
    max_samples: int | None,
    variant: str,
) -> list[tuple[int, dict[str, str]]]:
    """Select rows while preserving their original CSV indices."""

    indexed_rows = list(enumerate(rows))
    if row_indices is not None:
        requested = set(row_indices)
        missing = sorted(requested - set(range(len(rows))))
        if missing:
            raise SystemExit(
                f"Row indices outside {variant} (rows={len(rows)}): {missing}"
            )
        indexed_rows = [item for item in indexed_rows if item[0] in requested]
    elif task_ids is not None:
        requested = set(task_ids)
        present = {row.get("task_id", "") for _, row in indexed_rows}
        missing = sorted(requested - present)
        if missing:
            raise SystemExit(f"task_id values not found in {variant}: {missing}")
        indexed_rows = [
            item for item in indexed_rows if item[1].get("task_id", "") in requested
        ]

    if max_samples is not None:
        indexed_rows = indexed_rows[:max_samples]
    return indexed_rows


def build_generator(args: argparse.Namespace):
    if args.backend == "empty":
        return lambda _prompt, _seed: '{"node goals": [], "edge goals": [], "action goals": []}'

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

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

    def generate(prompt: str, seed: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if args.do_sample:
            generation_kwargs.update(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            set_seed(seed)
        else:
            # Qwen's bundled generation_config.json contains sampling values.
            # Reset them to neutral defaults during greedy decoding so
            # Transformers does not print the same harmless warning per row.
            generation_kwargs.update(temperature=1.0, top_p=1.0, top_k=50)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                **generation_kwargs,
            )
        return tokenizer.decode(
            output[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )

    #这里是返回build_generator()里定义的generate()函数，以后要改模型啥的只用改
    #build_generator()里面的generate()函数就行，外面主流程基本不用动，算是个接口
    return generate


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise SystemExit(f"Output exists: {args.output}. Use --resume or choose another path.")
    completed = existing_keys(args.output) if args.resume else set()

    #加载模型
    generate = build_generator(args)
    mode = "a" if args.resume else "w"

    with args.output.open(mode, encoding="utf-8") as handle:
        for variant in args.variants:

            #数据入口，读取CSV
            rows = read_rows(args.data_root, args.environment, variant)
            selected_rows = select_rows(
                rows,
                args.row_indices,
                args.task_ids,
                args.max_samples,
                variant,
            )
            pending: list[tuple[int, dict[str, str], int, int]] = []
            for row_index, row in selected_rows:
                for generation_id in range(args.num_generations):
                    key = (args.environment, variant, row_index, generation_id)
                    if key in completed:
                        continue
                    seed = candidate_seed(
                        args.seed,
                        args.environment,
                        variant,
                        row_index,
                        generation_id,
                    )
                    pending.append((row_index, row, generation_id, seed))

            if not pending:
                print(f"[{variant}] no pending candidates; everything requested is already saved")
                continue

            for progress, (row_index, row, generation_id, seed) in enumerate(pending, 1):

                #取prompt并调用模型，prompt_from_row为数据和模型间接口（prompt接口），拿到完整prompt字符串
                #generate为模型调用接口，加载调用模型输出回答
                response = generate(prompt_from_row(row, variant), seed)

                #结果格式化，保存为jsonl，每行一个json对象
                record = {
                    "environment": args.environment,
                    "variant": variant,
                    "row_index": row_index,
                    "task_id": row.get("task_id", ""),
                    "generation_id": generation_id,
                    "seed": seed,
                    "model": "empty-structured-baseline" if args.backend == "empty" else args.model,
                    "do_sample": args.do_sample,
                    "response": response,
                }
                if args.do_sample:
                    record.update(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )

                #保存结果，写入json
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                #强制写入磁盘，跑一条存一条，防止跑一半丢失
                handle.flush()
                print(
                    f"[{variant}] {progress}/{len(pending)} "
                    f"row={row_index} task={row.get('task_id', '')} "
                    f"candidate={generation_id} seed={seed}",
                    flush=True,
                )
    print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
