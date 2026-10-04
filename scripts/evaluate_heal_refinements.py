from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heal_repro.data import normalize_object, parse_response, read_rows
from heal_repro.metrics import load_predictions, score_predictions
from heal_repro.preferences import (
    _compare_model_to_gold,
    load_virtualhome_gold_goals,
    project_virtualhome_gold_goals,
)
from heal_repro.refinement import validate_blocked_refinement
from heal_repro.synonyms import load_virtualhome_synonym_projections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly validate and select model-generated HEAL refinements."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "workspace" / "HEAL_dataset",
    )
    parser.add_argument(
        "--eai-root",
        type=Path,
        default=ROOT / "workspace" / "embodied-agent-interface",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_executable_refinement(record: dict, score, gold) -> dict:
    comparison = _compare_model_to_gold(str(record.get("response", "")), score, gold)
    payload = parse_response(str(record.get("response", "")))
    scene_clean = (
        not score.hallucinated_objects
        and score.hallucinated_states == 0
        and not score.hallucinated_relations
        and not score.invalid_relation_types
        and not score.invalid_relation_targets
        and score.relation_format_errors == 0
    )
    passed = bool(comparison["model_goal_exact_match"] and scene_clean)
    return {
        "passed": passed,
        "format_valid": payload is not None,
        "model_goal_exact_match": comparison["model_goal_exact_match"],
        "scene_and_relation_clean": scene_clean,
        "predicted_node_goals": comparison["predicted_node_goals"],
        "incorrect_node_goals": comparison["incorrect_node_goals"],
        "missing_node_goals": comparison["missing_node_goals"],
        "predicted_relations": comparison["matched_relations_to_gold"]
        + comparison["incorrect_relations_to_gold"],
        "incorrect_relations": comparison["incorrect_relations_to_gold"],
        "missing_relations": comparison["missing_relations_to_gold"],
        "predicted_action_goals": comparison["predicted_action_goals"],
        "incorrect_action_goals": comparison["incorrect_action_goals"],
        "missing_action_goals": comparison["missing_action_goals"],
    }


def main() -> None:
    args = parse_args()
    records = load_predictions(args.predictions)
    scores = score_predictions(args.data_root, records, eai_root=args.eai_root)
    gold_index = load_virtualhome_gold_goals(args.eai_root)
    synonym_projections = (
        load_virtualhome_synonym_projections(args.data_root)
        if any(
            record.get("environment") == "virtualhome"
            and record.get("variant") == "scene_object_synonymous"
            for record in records
        )
        else {}
    )
    object_removal_rows = (
        read_rows(args.data_root, "virtualhome", "object_removal")
        if any(
            record.get("environment") == "virtualhome"
            and record.get("variant") == "object_removal"
            for record in records
        )
        else []
    )
    candidate_results = []

    for record, score in zip(records, scores, strict=True):
        target_behavior = record.get("target_behavior")
        validation_record = record
        if not target_behavior:
            if score.variant == "object_removal":
                target_behavior = "blocked_refusal"
                row = object_removal_rows[score.row_index]
                validation_record = {
                    **record,
                    "missing_required_objects": [
                        normalize_object(row.get("removed_object", ""))
                    ],
                }
            else:
                target_behavior = (
                    "blocked_refusal"
                    if record.get("missing_required_objects")
                    else "executable_gold"
                )
        if target_behavior == "blocked_refusal":
            validation = validate_blocked_refinement(validation_record)
        elif target_behavior == "executable_gold":
            if score.task_id not in gold_index:
                raise SystemExit(f"No EAI gold found for task_id={score.task_id}")
            gold = gold_index[score.task_id]
            if score.variant == "scene_object_synonymous":
                gold = project_virtualhome_gold_goals(
                    gold, synonym_projections[score.row_index]
                )
            validation = validate_executable_refinement(
                record, score, gold
            )
        else:
            raise SystemExit(f"Unknown target_behavior={target_behavior!r}")

        candidate_results.append(
            {
                "environment": score.environment,
                "variant": score.variant,
                "row_index": score.row_index,
                "task_id": score.task_id,
                "generation_id": int(record.get("generation_id", 0)),
                "model": str(record.get("model", "")),
                "target_behavior": target_behavior,
                "validation": validation,
            }
        )

    by_task: dict[tuple[str, str, int], list[dict]] = {}
    for result in candidate_results:
        key = (result["environment"], result["variant"], result["row_index"])
        by_task.setdefault(key, []).append(result)

    selected = []
    unresolved = []
    for key, candidates in sorted(by_task.items()):
        passing = sorted(
            (item for item in candidates if item["validation"]["passed"]),
            key=lambda item: item["generation_id"],
        )
        if passing:
            selected.append(passing[0])
        else:
            unresolved.append(
                {
                    "environment": key[0],
                    "variant": key[1],
                    "row_index": key[2],
                    "task_id": candidates[0]["task_id"],
                    "candidate_generation_ids": sorted(
                        item["generation_id"] for item in candidates
                    ),
                }
            )

    reason_matches = Counter(
        item["validation"].get("reason_match", "not_applicable")
        for item in candidate_results
        if item["target_behavior"] == "blocked_refusal"
        and item["validation"]["passed"]
    )
    summary = {
        "source_records": len(records),
        "unique_tasks": len(by_task),
        "passing_candidates": sum(
            item["validation"]["passed"] for item in candidate_results
        ),
        "selected_tasks": len(selected),
        "unresolved_tasks": unresolved,
        "target_behaviors": dict(
            Counter(item["target_behavior"] for item in candidate_results)
        ),
        "selected_target_behaviors": dict(
            Counter(item["target_behavior"] for item in selected)
        ),
        "passing_refusal_reason_matches": dict(reason_matches),
        "selection_policy": "lowest generation_id among strictly passing candidates",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "candidate_scores.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for item in candidate_results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (args.output_dir / "selected_candidates.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Candidate scores: {args.output_dir / 'candidate_scores.jsonl'}")
    print(f"Selected tasks:   {args.output_dir / 'selected_candidates.jsonl'}")


if __name__ == "__main__":
    main()
