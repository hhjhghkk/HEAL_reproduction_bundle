from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heal_repro.data import (
    normalize_object,
    parse_scene,
    prompt_from_row,
    read_rows,
    split_probe_objects,
)
from heal_repro.metrics import load_predictions, score_predictions
from heal_repro.preferences import (
    ACTION_CONFLICT_REQUIREMENTS,
    ACTION_FEASIBLE_REQUIREMENTS,
    _gold_payload,
    load_virtualhome_gold_goals,
    project_virtualhome_gold_goals,
)
from heal_repro.synonyms import load_virtualhome_synonym_projections
from scripts.run_heal_inference import (
    build_generator,
    candidate_seed,
    existing_keys,
    select_rows,
    validate_args,
)


DEFAULT_DATA = ROOT / "workspace" / "HEAL_dataset"
DEFAULT_EAI = ROOT / "workspace" / "embodied-agent-interface"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine HEAL responses using deterministic verifier feedback."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--eai-root", type=Path, default=DEFAULT_EAI)
    parser.add_argument("--environment", default="virtualhome", choices=("virtualhome",))
    parser.add_argument(
        "--variant",
        default="scene_task_contradiction",
        choices=(
            "scene_task_contradiction",
            "distractor_injection",
            "scene_object_synonymous",
            "object_removal",
        ),
    )
    parser.add_argument("--source-generation-id", type=int, default=0)
    parser.add_argument(
        "--sample-kind",
        choices=("blocked", "executable"),
        default="blocked",
        help="Generate either verified refusals or verified executable goals.",
    )
    parser.add_argument("--backend", choices=("transformers", "empty"), default="transformers")
    parser.add_argument("--model", default="workspace/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--row-indices", type=int, nargs="+")
    selection.add_argument("--task-ids", nargs="+")
    exclusion = parser.add_mutually_exclusive_group()
    exclusion.add_argument(
        "--exclude-row-indices",
        type=int,
        nargs="+",
        help="Skip these dataset row indices after normal selection.",
    )
    exclusion.add_argument(
        "--exclude-task-ids",
        nargs="+",
        help="Skip these task IDs after normal selection.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help=(
            "Maximum verified blocked samples after automatic filtering; "
            "omit to process all blocked samples."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "qwen2.5_7b_refinement_pilot.jsonl",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def missing_required_objects(
    row: dict[str, str],
    scene_names: set[str],
    gold,
    variant: str,
) -> list[str]:
    """Return only requirements backed by EAI gold or an audited action rule."""

    missing = {
        goal.name for goal in gold.node_goals if goal.name not in scene_names
    }
    missing.update(
        endpoint
        for relation in gold.relations
        for endpoint in (relation.from_name, relation.to_name)
        if endpoint not in scene_names
    )
    task_name = normalize_object(row.get("task_name", ""))
    # The audited action-prerequisite rule is specific to scene/task conflict:
    # that transformation deliberately removes objects needed by the task.
    # Distractor injection keeps the original executable task and only adds
    # irrelevant object mentions to its natural-language description.
    if (
        variant == "scene_task_contradiction"
        and not missing
        and task_name in ACTION_CONFLICT_REQUIREMENTS
    ):
        required = ACTION_CONFLICT_REQUIREMENTS[task_name]
        absent = required - scene_names
        if absent == required:
            missing.update(absent)
    return sorted(missing)


def build_refinement_prompt(
    original_prompt: str,
    previous_response: str,
    missing_objects: list[str],
    hallucinated_objects: list[str],
    hallucinated_relations: list,
) -> str:
    """Append concise verifier evidence and demand a model-generated revision."""

    relation_text = [
        f"{item.from_name} --{item.relation}--> {item.to_name}"
        for item in hallucinated_relations
    ]
    feedback_lines = [
        "The task is infeasible in the current scene because these required objects are absent: "
        + ", ".join(missing_objects)
        + "."
    ]
    if hallucinated_objects:
        feedback_lines.append(
            "The previous response used objects absent from the current scene: "
            + ", ".join(hallucinated_objects)
            + "."
        )
    if relation_text:
        feedback_lines.append(
            "The previous response also used relations with absent endpoints: "
            + "; ".join(relation_text)
            + "."
        )
    feedback = "\n".join(f"- {line}" for line in feedback_lines)
    exact_missing = ", ".join(missing_objects)
    return f"""{original_prompt}

--- PREVIOUS RESPONSE ---
{previous_response}

--- VERIFIED SCENE FEEDBACK ---
{feedback}

Revise the previous response using the verified feedback above.
Do not assume, invent, fetch, or substitute objects that are absent from the current scene.
Because required objects are absent, do not create a plan. Return exactly one JSON object
with empty \"node goals\", \"edge goals\", and \"action goals\" lists, plus a concise
\"reason\" naming the absent required object or objects. The reason must copy at least one
exact object identifier from this verified missing list: {exact_missing}.
Output JSON only, with no markdown code fence and no explanation outside the JSON.
"""


def build_executable_refinement_prompt(
    original_prompt: str,
    previous_response: str,
    verified_goal: dict,
    forbidden_objects: list[str] | None = None,
) -> str:
    """Ask the model to revise against an executable EAI symbolic goal."""

    goal_json = json.dumps(verified_goal, ensure_ascii=False, indent=2)
    exact_copy = json.dumps(verified_goal, ensure_ascii=False, separators=(",", ":"))
    distractor_feedback = ""
    if forbidden_objects:
        distractor_feedback = (
            "\nThe following objects were deliberately injected as irrelevant distractors and "
            "must not appear in any goal: "
            + ", ".join(forbidden_objects)
            + ".\n"
        )
    return f"""{original_prompt}

--- PREVIOUS RESPONSE ---
{previous_response}

--- VERIFIED EXECUTABLE GOAL FEEDBACK ---
The current scene still supports the task. Do not refuse it.
{distractor_feedback}
The verified EAI symbolic goal for this task is:
{goal_json}

Revise the previous response so that its node, edge, and action goals match the verified
symbolic goal exactly. Do not add extra goals, objects, states, relations, or actions.
Return exactly one JSON object with the keys \"node goals\", \"edge goals\", and
\"action goals\". Copy object identifiers, states, relation names, and action names exactly
as shown in the verified goal. This is a strict serialization task, not a new planning task.
Do not improve, complete, or infer anything from the earlier task description.

Copy the following one-line JSON as the entire response, character for character:
{exact_copy}

Output that JSON only, with no markdown code fence and no explanation outside it.
"""


def main() -> None:
    args = parse_args()
    validate_args(args)
    source_predictions = load_predictions(args.predictions)
    source_index: dict[int, dict] = {}
    for record in source_predictions:
        if record.get("environment") != args.environment or record.get("variant") != args.variant:
            continue
        if int(record.get("generation_id", 0)) != args.source_generation_id:
            continue
        row_index = int(record["row_index"])
        if row_index in source_index:
            raise SystemExit(f"Duplicate source prediction for row_index={row_index}")
        source_index[row_index] = record

    rows = read_rows(args.data_root, args.environment, args.variant)
    selected_rows = select_rows(
        rows,
        args.row_indices,
        args.task_ids,
        None,
        args.variant,
    )
    excluded_row_indices = set(args.exclude_row_indices or [])
    excluded_task_ids = set(args.exclude_task_ids or [])
    selected_rows = [
        (index, row)
        for index, row in selected_rows
        if index not in excluded_row_indices
        and row.get("task_id", "") not in excluded_task_ids
    ]
    missing_sources = [index for index, _ in selected_rows if index not in source_index]
    if missing_sources:
        raise SystemExit(f"Source predictions are missing row indices: {missing_sources}")

    selected_sources = [source_index[index] for index, _ in selected_rows]
    scores = score_predictions(args.data_root, selected_sources, eai_root=args.eai_root)
    score_index = {score.row_index: score for score in scores}
    gold_index = load_virtualhome_gold_goals(args.eai_root)
    synonym_projections = (
        load_virtualhome_synonym_projections(args.data_root)
        if args.variant == "scene_object_synonymous"
        else {}
    )

    tasks = []
    skipped_other_kind = []
    for row_index, row in selected_rows:
        task_id = row.get("task_id", "")
        if task_id not in gold_index:
            raise SystemExit(f"No EAI gold found for task_id={task_id}")
        prompt = prompt_from_row(row, args.variant)
        scene_names = set(parse_scene(prompt))
        gold = gold_index[task_id]
        if args.variant == "scene_object_synonymous":
            gold = project_virtualhome_gold_goals(
                gold, synonym_projections[row_index]
            )
        missing = missing_required_objects(row, scene_names, gold, args.variant)
        task_name = normalize_object(row.get("task_name", ""))
        action_feasible = (
            task_name in ACTION_FEASIBLE_REQUIREMENTS
            and ACTION_FEASIBLE_REQUIREMENTS[task_name].issubset(scene_names)
        )
        executable = not missing and bool(
            gold.node_goals
            or gold.relations
            or action_feasible
            or (
                args.variant in {"distractor_injection", "scene_object_synonymous"}
                and gold.action_goals
            )
        )
        matches_kind = (
            args.sample_kind == "blocked" and bool(missing)
        ) or (
            args.sample_kind == "executable" and executable
        )
        if not matches_kind:
            skipped_other_kind.append((row_index, task_id))
            continue
        tasks.append((row_index, row, prompt, missing, score_index[row_index], gold))

    if args.max_samples is not None:
        tasks = tasks[: args.max_samples]
    if not tasks:
        raise SystemExit(
            f"No verified {args.sample_kind} samples were selected."
        )
    print(
        f"Verified {args.sample_kind} samples selected: {len(tasks)}; "
        f"other samples skipped: {len(skipped_other_kind)}",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise SystemExit(f"Output exists: {args.output}. Use --resume or choose another path.")
    completed = existing_keys(args.output) if args.resume else set()
    pending = []
    for row_index, row, prompt, missing, score, gold in tasks:
        for generation_id in range(args.num_generations):
            key = (args.environment, args.variant, row_index, generation_id)
            if key in completed:
                continue
            seed = candidate_seed(
                args.seed,
                args.environment,
                f"{args.variant}:verifier_refinement",
                row_index,
                generation_id,
            )
            pending.append(
                (row_index, row, prompt, missing, score, gold, generation_id, seed)
            )

    if not pending:
        print("No pending refinement candidates; everything requested is already saved.")
        return

    generate = build_generator(args)
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for progress, item in enumerate(pending, 1):
            row_index, row, prompt, missing, score, gold, generation_id, seed = item
            source = source_index[row_index]
            verified_goal = None
            forbidden_objects: list[str] = []
            if args.sample_kind == "blocked":
                refinement_prompt = build_refinement_prompt(
                    prompt,
                    str(source.get("response", "")),
                    missing,
                    score.hallucinated_objects,
                    score.hallucinated_relations,
                )
                target_behavior = "blocked_refusal"
                feedback_basis = "blocked_eai_goal_or_audited_action_requirement"
            else:
                verified_goal = _gold_payload(gold, prompt)
                forbidden_objects = (
                    sorted(split_probe_objects(row.get("distractors", "")))
                    if args.variant == "distractor_injection"
                    else []
                )
                refinement_prompt = build_executable_refinement_prompt(
                    prompt,
                    str(source.get("response", "")),
                    verified_goal,
                    forbidden_objects,
                )
                target_behavior = "executable_gold"
                feedback_basis = "executable_eai_node_edge_action_goal"
            response = generate(refinement_prompt, seed)
            record = {
                "environment": args.environment,
                "variant": args.variant,
                "row_index": row_index,
                "task_id": row.get("task_id", ""),
                "generation_id": generation_id,
                "seed": seed,
                "model": "empty-structured-baseline" if args.backend == "empty" else args.model,
                "do_sample": args.do_sample,
                "stage": "verifier_feedback_refinement",
                "target_behavior": target_behavior,
                "source_predictions": str(args.predictions),
                "source_generation_id": args.source_generation_id,
                "feedback_basis": feedback_basis,
                "missing_required_objects": missing,
                "previous_hallucinated_objects": score.hallucinated_objects,
                "response": response,
            }
            if verified_goal is not None:
                record["verified_goal"] = verified_goal
            if args.variant == "distractor_injection":
                record["forbidden_distractor_objects"] = forbidden_objects
            if args.do_sample:
                record.update(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[refinement] {progress}/{len(pending)} row={row_index} "
                f"task={row.get('task_id', '')} candidate={generation_id}",
                flush=True,
            )
    print(f"Refined predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
