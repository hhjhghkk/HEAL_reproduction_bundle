from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heal_repro.data import normalize_object, read_rows
from heal_repro.metrics import load_predictions, score_predictions
from heal_repro.refinement import validate_blocked_refinement


ENVIRONMENT = "virtualhome"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build compact blocked-task preference pairs from strictly validated "
            "real model responses."
        )
    )
    parser.add_argument("--variant", choices=("object_removal",), required=True)
    parser.add_argument("--initial-predictions", type=Path, required=True)
    parser.add_argument("--refinements", type=Path, required=True)
    parser.add_argument("--refinement-selection", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "workspace" / "HEAL_dataset"
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


def index_unique(records: list[dict], key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    result: dict[tuple, dict] = {}
    for record in records:
        key = tuple(
            int(record[field]) if field in {"row_index", "generation_id"} else record[field]
            for field in key_fields
        )
        if key in result:
            raise ValueError(f"Duplicate record key {key_fields}={key}")
        result[key] = record
    return result


def relation_lists(relations) -> list[list[str]]:
    return [
        [relation.from_name, relation.relation, relation.to_name]
        for relation in relations
    ]


def main() -> None:
    args = parse_args()
    rows = read_rows(args.data_root, ENVIRONMENT, args.variant)
    expected_rows = set(range(len(rows)))

    initial_records = [
        record
        for record in load_predictions(args.initial_predictions)
        if record.get("environment") == ENVIRONMENT
        and record.get("variant") == args.variant
        and int(record.get("generation_id", 0)) == 0
    ]
    initial_index = index_unique(initial_records, ("row_index",))
    initial_rows = {key[0] for key in initial_index}
    if initial_rows != expected_rows:
        raise SystemExit(
            "Initial predictions do not cover the dataset exactly; "
            f"missing={sorted(expected_rows - initial_rows)}, "
            f"extra={sorted(initial_rows - expected_rows)}"
        )
    initial_scores = score_predictions(
        args.data_root, initial_records, eai_root=args.eai_root
    )
    score_index = {score.row_index: score for score in initial_scores}

    initial_validations = {}
    for row_index in sorted(expected_rows):
        removed = normalize_object(rows[row_index].get("removed_object", ""))
        record = initial_index[(row_index,)]
        validation = validate_blocked_refinement(
            {**record, "missing_required_objects": [removed]}
        )
        if validation["passed"]:
            raise SystemExit(
                f"Initial response is already a valid refusal at row={row_index}; "
                "this builder currently expects every initial response to be rejected"
            )
        initial_validations[row_index] = validation

    refinement_records = load_predictions(args.refinements)
    refinement_index = index_unique(
        refinement_records, ("row_index", "generation_id")
    )
    selections = load_jsonl(args.refinement_selection)
    selection_index = index_unique(selections, ("row_index",))
    selected_rows = {key[0] for key in selection_index}
    if selected_rows != expected_rows:
        raise SystemExit(
            "Strict refinement selections do not cover the dataset exactly; "
            f"missing={sorted(expected_rows - selected_rows)}, "
            f"extra={sorted(selected_rows - expected_rows)}"
        )

    final_pairs = []
    for row_index in sorted(expected_rows):
        initial_record = initial_index[(row_index,)]
        initial_validation = initial_validations[row_index]
        score = score_index[row_index]
        selection = selection_index[(row_index,)]
        validation = selection.get("validation", {})
        if not validation.get("passed"):
            raise SystemExit(f"Selected chosen candidate failed at row={row_index}")
        generation_id = int(selection["generation_id"])
        chosen_record = refinement_index[(row_index, generation_id)]
        task_id = str(initial_record.get("task_id", ""))
        if str(chosen_record.get("task_id", "")) != task_id:
            raise SystemExit(f"Chosen task_id mismatch at row={row_index}")
        if chosen_record.get("target_behavior") != "blocked_refusal":
            raise SystemExit(f"Chosen target behavior is not blocked at row={row_index}")
        if str(initial_record.get("model", "")) != str(chosen_record.get("model", "")):
            raise SystemExit(f"Chosen/rejected model mismatch at row={row_index}")
        if str(initial_record.get("response", "")) == str(chosen_record.get("response", "")):
            raise SystemExit(f"Chosen/rejected responses are identical at row={row_index}")

        removed = normalize_object(rows[row_index].get("removed_object", ""))
        final_pairs.append(
            {
                "environment": ENVIRONMENT,
                "variant": args.variant,
                "row_index": row_index,
                "task_id": task_id,
                "removed_object": removed,
                "label_basis": "removed_object_is_direct_eai_node_or_relation_dependency",
                "pair_route": "initial_wrong_plan_plus_verified_blocked_refusal",
                "chosen": {
                    "model": str(chosen_record.get("model", "")),
                    "stage": "verifier_feedback_refinement",
                    "generation_id": generation_id,
                    "response": str(chosen_record.get("response", "")),
                    "validation": validation,
                },
                "rejected": {
                    "model": str(initial_record.get("model", "")),
                    "stage": "initial_inference",
                    "generation_id": int(initial_record.get("generation_id", 0)),
                    "response": str(initial_record.get("response", "")),
                    "evidence": {
                        "valid_blocked_refusal": initial_validation["passed"],
                        "format_valid": initial_validation["format_valid"],
                        "goals_empty": initial_validation["goals_empty"],
                        "reason_nonempty": initial_validation["reason_nonempty"],
                        "reason_match": initial_validation["reason_match"],
                        "missing_required_objects": [removed],
                        "hallucinated_objects": score.hallucinated_objects,
                        "hallucinated_states": score.hallucinated_states,
                        "hallucinated_relations": relation_lists(
                            score.hallucinated_relations
                        ),
                        "relation_format_errors": score.relation_format_errors,
                        "invalid_relation_types": relation_lists(
                            score.invalid_relation_types
                        ),
                        "invalid_relation_targets": relation_lists(
                            score.invalid_relation_targets
                        ),
                    },
                },
            }
        )

    summary = {
        "status": "complete_coverage_pending_final_human_review",
        "pairs": len(final_pairs),
        "environment": ENVIRONMENT,
        "variant": args.variant,
        "both_sides_are_real_model_outputs": True,
        "both_sides_use_the_same_model": all(
            item["chosen"]["model"] == item["rejected"]["model"]
            for item in final_pairs
        ),
        "chosen_models": dict(Counter(item["chosen"]["model"] for item in final_pairs)),
        "rejected_models": dict(
            Counter(item["rejected"]["model"] for item in final_pairs)
        ),
        "pair_routes": dict(Counter(item["pair_route"] for item in final_pairs)),
        "chosen_generation_ids": dict(
            Counter(str(item["chosen"]["generation_id"]) for item in final_pairs)
        ),
        "strictly_valid_chosen": sum(
            item["chosen"]["validation"]["passed"] for item in final_pairs
        ),
        "strictly_wrong_rejected": sum(
            not item["rejected"]["evidence"]["valid_blocked_refusal"]
            for item in final_pairs
        ),
        "chosen_reason_matches": dict(
            Counter(
                item["chosen"]["validation"]["reason_match"]
                for item in final_pairs
            )
        ),
        "source_files": {
            "initial_predictions": str(args.initial_predictions),
            "refinements": str(args.refinements),
            "refinement_selection": str(args.refinement_selection),
        },
        "storage_note": (
            "Prompts are omitted. Recover them from the HEAL CSV using environment, "
            "variant, and row_index. Responses and compact validation evidence are retained."
        ),
        "method_note": (
            "Every removed object was audited as a direct EAI node or relation dependency. "
            "Initial wrong plans are paired with 7B refusals generated after deterministic "
            "missing-object feedback."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / f"{args.variant}_model_generated_pairs.jsonl"
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
