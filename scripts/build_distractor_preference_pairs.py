from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heal_repro.data import read_rows
from heal_repro.metrics import load_predictions, score_predictions
from heal_repro.preferences import load_virtualhome_gold_goals
from heal_repro.preferences import project_virtualhome_gold_goals
from heal_repro.synonyms import load_virtualhome_synonym_projections
from scripts.evaluate_heal_refinements import validate_executable_refinement


ENVIRONMENT = "virtualhome"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build compact executable-variant preference pairs from strictly "
            "validated real model responses."
        )
    )
    parser.add_argument("--initial-predictions", type=Path, required=True)
    parser.add_argument("--refinements", type=Path, required=True)
    parser.add_argument("--refinement-selection", type=Path, required=True)
    parser.add_argument("--extra-rejected-candidates", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("distractor_injection", "scene_object_synonymous"),
        default="distractor_injection",
    )
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


def validate_records(
    records: list[dict], data_root: Path, eai_root: Path, variant: str
) -> dict[tuple[int, int], dict]:
    scores = score_predictions(data_root, records, eai_root=eai_root)
    gold_index = load_virtualhome_gold_goals(eai_root)
    synonym_projections = (
        load_virtualhome_synonym_projections(data_root)
        if variant == "scene_object_synonymous"
        else {}
    )
    result: dict[tuple[int, int], dict] = {}
    for record, score in zip(records, scores, strict=True):
        if score.task_id not in gold_index:
            raise ValueError(f"No EAI gold for task_id={score.task_id}")
        key = (score.row_index, int(record.get("generation_id", 0)))
        if key in result:
            raise ValueError(f"Duplicate validation key={key}")
        gold = gold_index[score.task_id]
        if variant == "scene_object_synonymous":
            gold = project_virtualhome_gold_goals(
                gold, synonym_projections[score.row_index]
            )
        result[key] = validate_executable_refinement(record, score, gold)
    return result


def compact_evidence(validation: dict) -> dict:
    return {
        "format_valid": validation["format_valid"],
        "goal_exact_match": validation["model_goal_exact_match"],
        "scene_and_relation_clean": validation["scene_and_relation_clean"],
        "incorrect_node_goals": validation["incorrect_node_goals"],
        "missing_node_goals": validation["missing_node_goals"],
        "incorrect_relations": validation["incorrect_relations"],
        "missing_relations": validation["missing_relations"],
        "incorrect_action_goals": validation["incorrect_action_goals"],
        "missing_action_goals": validation["missing_action_goals"],
    }


def response_side(
    record: dict,
    stage: str,
    validation: dict,
    include_validation: bool,
) -> dict:
    side = {
        "model": str(record.get("model", "")),
        "stage": stage,
        "generation_id": int(record.get("generation_id", 0)),
        "response": str(record.get("response", "")),
    }
    if record.get("seed") is not None:
        side["seed"] = int(record["seed"])
    side["validation" if include_validation else "evidence"] = (
        validation if include_validation else compact_evidence(validation)
    )
    return side


def main() -> None:
    args = parse_args()
    variant = args.variant
    rows = read_rows(args.data_root, ENVIRONMENT, variant)
    expected_rows = set(range(len(rows)))

    initial_records = [
        record
        for record in load_predictions(args.initial_predictions)
        if record.get("environment") == ENVIRONMENT
        and record.get("variant") == variant
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
    initial_validations = validate_records(
        initial_records, args.data_root, args.eai_root, variant
    )
    correct_initial_rows = {
        row_index
        for (row_index, generation_id), validation in initial_validations.items()
        if generation_id == 0 and validation["passed"]
    }

    refinement_records = load_predictions(args.refinements)
    refinement_index = index_unique(
        refinement_records, ("row_index", "generation_id")
    )
    selected = load_jsonl(args.refinement_selection)
    selected_index = index_unique(selected, ("row_index",))
    selected_rows = {key[0] for key in selected_index}
    expected_refinement_rows = expected_rows - correct_initial_rows
    if selected_rows != expected_refinement_rows:
        raise SystemExit(
            "Strict refinement selections do not cover exactly the initially wrong rows; "
            f"missing={sorted(expected_refinement_rows - selected_rows)}, "
            f"extra={sorted(selected_rows - expected_refinement_rows)}"
        )

    extra_records = load_predictions(args.extra_rejected_candidates)
    extra_validations = validate_records(
        extra_records, args.data_root, args.eai_root, variant
    )
    extra_by_row: dict[int, list[dict]] = {}
    for record in extra_records:
        extra_by_row.setdefault(int(record["row_index"]), []).append(record)

    sampled_rejected: dict[int, tuple[dict, dict]] = {}
    for row_index in sorted(correct_initial_rows):
        candidates = sorted(
            extra_by_row.get(row_index, []),
            key=lambda item: int(item.get("generation_id", 0)),
        )
        for candidate in candidates:
            generation_id = int(candidate.get("generation_id", 0))
            validation = extra_validations[(row_index, generation_id)]
            if (
                validation["format_valid"]
                and validation["scene_and_relation_clean"]
                and not validation["model_goal_exact_match"]
                and str(candidate.get("response", "")).strip()
            ):
                sampled_rejected[row_index] = (candidate, validation)
                break
        if row_index not in sampled_rejected:
            raise SystemExit(
                f"No clear model-generated wrong candidate for initially correct row={row_index}"
            )

    final_pairs = []
    for row_index in sorted(expected_rows):
        initial_record = initial_index[(row_index,)]
        initial_validation = initial_validations[(row_index, 0)]
        task_id = str(initial_record.get("task_id", ""))

        if row_index in correct_initial_rows:
            chosen = response_side(
                initial_record, "initial_inference", initial_validation, True
            )
            rejected_record, rejected_validation = sampled_rejected[row_index]
            rejected = response_side(
                rejected_record,
                "sampled_initial_inference",
                rejected_validation,
                False,
            )
            pair_route = "initial_correct_plus_sampled_wrong"
        else:
            if initial_validation["passed"]:
                raise SystemExit(f"Initially wrong row unexpectedly passed: {row_index}")
            selection = selected_index[(row_index,)]
            if not selection.get("validation", {}).get("passed"):
                raise SystemExit(f"Selected refinement did not pass for row={row_index}")
            generation_id = int(selection["generation_id"])
            chosen_record = refinement_index[(row_index, generation_id)]
            if str(chosen_record.get("task_id", "")) != task_id:
                raise SystemExit(f"Chosen task_id mismatch for row={row_index}")
            chosen = response_side(
                chosen_record,
                "verifier_feedback_refinement",
                selection["validation"],
                True,
            )
            rejected = response_side(
                initial_record, "initial_inference", initial_validation, False
            )
            pair_route = "initial_wrong_plus_verified_refinement"

        if chosen["validation"]["passed"] is not True:
            raise SystemExit(f"Chosen side failed strict validation for row={row_index}")
        if rejected["evidence"]["goal_exact_match"] is not False:
            raise SystemExit(f"Rejected side is not strictly wrong for row={row_index}")
        if chosen["model"] != rejected["model"]:
            raise SystemExit(f"Chosen/rejected model mismatch for row={row_index}")
        if chosen["response"] == rejected["response"]:
            raise SystemExit(f"Chosen/rejected responses are identical for row={row_index}")

        final_pairs.append(
            {
                "environment": ENVIRONMENT,
                "variant": variant,
                "row_index": row_index,
                "task_id": task_id,
                "label_basis": "strict_eai_full_goal_and_scene_validation",
                "pair_route": pair_route,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    summary = {
        "status": "complete_coverage_pending_final_human_review",
        "pairs": len(final_pairs),
        "environment": ENVIRONMENT,
        "variant": variant,
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
        "chosen_stages": dict(Counter(item["chosen"]["stage"] for item in final_pairs)),
        "rejected_stages": dict(
            Counter(item["rejected"]["stage"] for item in final_pairs)
        ),
        "chosen_generation_ids": dict(
            Counter(str(item["chosen"]["generation_id"]) for item in final_pairs)
        ),
        "rejected_generation_ids": dict(
            Counter(str(item["rejected"]["generation_id"]) for item in final_pairs)
        ),
        "strictly_valid_chosen": sum(
            item["chosen"]["validation"]["passed"] for item in final_pairs
        ),
        "strictly_wrong_rejected": sum(
            not item["rejected"]["evidence"]["goal_exact_match"]
            for item in final_pairs
        ),
        "source_files": {
            "initial_predictions": str(args.initial_predictions),
            "refinements": str(args.refinements),
            "refinement_selection": str(args.refinement_selection),
            "extra_rejected_candidates": str(args.extra_rejected_candidates),
        },
        "storage_note": (
            "Prompts are omitted. Recover them from the HEAL CSV using environment, "
            "variant, and row_index. Responses and compact strict evidence are retained."
        ),
        "method_note": (
            "Initially wrong responses are paired with EAI-gold-guided 7B refinements. "
            "The one initially correct response is paired with a strictly verified wrong "
            "7B sample from the original prompt."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / f"{variant}_model_generated_pairs.jsonl"
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
