from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heal_repro.metrics import load_predictions
from heal_repro.preferences import build_full_conflict_preference_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact HEAL preference pairs with two real model responses."
    )
    parser.add_argument("--rejected-predictions", type=Path, required=True)
    parser.add_argument("--blocked-refinements", type=Path, required=True)
    parser.add_argument("--blocked-selection", type=Path, required=True)
    parser.add_argument("--executable-refinements", type=Path, required=True)
    parser.add_argument("--executable-selection", type=Path, required=True)
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


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def refinement_index(records: list[dict]) -> dict[tuple[int, int], dict]:
    result = {}
    for record in records:
        key = (int(record["row_index"]), int(record.get("generation_id", 0)))
        if key in result:
            raise ValueError(f"Duplicate refinement candidate key: {key}")
        result[key] = record
    return result


def compact_rejected_evidence(metrics: dict) -> dict:
    return {
        "format_valid": metrics["format_valid"],
        "refused_or_empty": metrics["refused_or_empty"],
        "goal_exact_match": metrics["model_goal_exact_match"],
        "hallucinated_objects": metrics["hallucinated_objects"],
        "hallucinated_states": metrics["hallucinated_states"],
        "hallucinated_relations": metrics["hallucinated_relations"],
        "invalid_relation_types": metrics["invalid_relation_types"],
        "invalid_relation_targets": metrics["invalid_relation_targets"],
        "incorrect_node_goals": metrics["incorrect_node_goals"],
        "missing_node_goals": metrics["missing_node_goals"],
        "incorrect_relations": metrics["incorrect_relations_to_gold"],
        "missing_relations": metrics["missing_relations_to_gold"],
        "incorrect_action_goals": metrics["incorrect_action_goals"],
        "missing_action_goals": metrics["missing_action_goals"],
    }


def main() -> None:
    args = parse_args()
    rejected_records = load_predictions(args.rejected_predictions)
    rejected_pairs, rejected_summary = build_full_conflict_preference_pairs(
        args.data_root,
        rejected_records,
        args.eai_root,
        environment="virtualhome",
        variant="scene_task_contradiction",
    )
    rejected_index = {int(item["row_index"]): item for item in rejected_pairs}
    if len(rejected_index) != 338:
        raise SystemExit(f"Expected 338 rejected rows, found {len(rejected_index)}")
    if any(item["rejected_source"] != "local_model_response" for item in rejected_pairs):
        raise SystemExit("At least one rejected side is not a real local-model response")

    blocked_records = load_predictions(args.blocked_refinements)
    executable_records = load_predictions(args.executable_refinements)
    chosen_index = {
        **refinement_index(blocked_records),
        **refinement_index(executable_records),
    }
    selected = load_jsonl(args.blocked_selection) + load_jsonl(args.executable_selection)
    if len(selected) != 338:
        raise SystemExit(f"Expected 338 selected chosen tasks, found {len(selected)}")

    selected_rows = set()
    final_pairs = []
    for selection in sorted(selected, key=lambda item: int(item["row_index"])):
        row_index = int(selection["row_index"])
        generation_id = int(selection["generation_id"])
        if row_index in selected_rows:
            raise SystemExit(f"Duplicate selected chosen row_index={row_index}")
        selected_rows.add(row_index)
        chosen_record = chosen_index[(row_index, generation_id)]
        rejected_pair = rejected_index[row_index]
        if chosen_record["task_id"] != rejected_pair["task_id"]:
            raise SystemExit(f"task_id mismatch at row_index={row_index}")
        validation = selection["validation"]
        if not validation.get("passed"):
            raise SystemExit(f"Selected chosen candidate did not pass at row_index={row_index}")
        rejected_evidence = compact_rejected_evidence(
            rejected_pair["model_metrics"]
        )
        if rejected_evidence["goal_exact_match"]:
            raise SystemExit(f"Rejected answer exactly matches gold at row_index={row_index}")
        target_behavior = chosen_record.get(
            "target_behavior", selection["target_behavior"]
        )

        final_pairs.append(
            {
                "environment": "virtualhome",
                "variant": "scene_task_contradiction",
                "row_index": row_index,
                "task_id": chosen_record["task_id"],
                "label_basis": rejected_pair["label_basis"],
                "chosen": {
                    "model": chosen_record["model"],
                    "stage": chosen_record["stage"],
                    "generation_id": generation_id,
                    "target_behavior": target_behavior,
                    "response": chosen_record["response"],
                    "validation": validation,
                },
                "rejected": {
                    "model": rejected_pair["model"],
                    "stage": "initial_inference",
                    "response": rejected_pair["rejected"],
                    "evidence": rejected_evidence,
                },
            }
        )

    if selected_rows != set(range(338)):
        raise SystemExit(
            f"Selected rows do not cover 0..337; missing={sorted(set(range(338)) - selected_rows)}"
        )

    summary = {
        "status": "complete_coverage_pending_final_human_review",
        "pairs": len(final_pairs),
        "environment": "virtualhome",
        "variant": "scene_task_contradiction",
        "chosen_models": dict(
            Counter(item["chosen"]["model"] for item in final_pairs)
        ),
        "rejected_models": dict(
            Counter(item["rejected"]["model"] for item in final_pairs)
        ),
        "chosen_source": "verifier-feedback refinement model response",
        "rejected_source": "initial model inference response",
        "both_sides_are_real_model_outputs": True,
        "both_sides_use_the_same_model": (
            {item["chosen"]["model"] for item in final_pairs}
            == {item["rejected"]["model"] for item in final_pairs}
        ),
        "chosen_target_behaviors": dict(
            Counter(item["chosen"]["target_behavior"] for item in final_pairs)
        ),
        "label_bases": dict(Counter(item["label_basis"] for item in final_pairs)),
        "chosen_generation_ids": dict(
            Counter(str(item["chosen"]["generation_id"]) for item in final_pairs)
        ),
        "rejected_model_exact_matches": sum(
            item["rejected"]["evidence"]["goal_exact_match"] for item in final_pairs
        ),
        "source_files": {
            "rejected_predictions": str(args.rejected_predictions),
            "blocked_refinements": str(args.blocked_refinements),
            "blocked_selection": str(args.blocked_selection),
            "executable_refinements": str(args.executable_refinements),
            "executable_selection": str(args.executable_selection),
        },
        "storage_note": (
            "Prompts are not duplicated. Recover each prompt from the HEAL CSV using "
            "environment, variant, and row_index."
        ),
        "rejected_audit": {
            "source_predictions": rejected_summary["source_predictions"],
            "all_rejected_are_local_model_responses": True,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "scene_task_contradiction_model_generated_pairs.jsonl"
    summary_path = args.output_dir / "summary.json"
    with pairs_path.open("w", encoding="utf-8") as handle:
        for item in final_pairs:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Preference pairs: {pairs_path}")
    print(f"Summary:          {summary_path}")


if __name__ == "__main__":
    main()
