from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heal_repro.metrics import load_predictions
from heal_repro.preferences import (
    build_full_conflict_preference_pairs,
    build_infeasible_preference_pairs,
    write_preference_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build compact, high-confidence HEAL refusal-vs-hallucinated-plan "
            "preference pairs."
        )
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
    parser.add_argument("--environment", default="virtualhome")
    parser.add_argument("--variant", default="scene_task_contradiction")
    parser.add_argument(
        "--selection-mode",
        choices=("full_gold_conflict", "strict_relation"),
        default="full_gold_conflict",
        help=(
            "full_gold_conflict uses blocked node or relation gold; strict_relation "
            "reproduces the earlier relation-only subset."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "preferences"
            / "scene_task_contradiction_preference_pairs"
        ),
    )
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    builder = (
        build_full_conflict_preference_pairs
        if args.selection_mode == "full_gold_conflict"
        else build_infeasible_preference_pairs
    )
    pairs, summary = builder(
        args.data_root,
        predictions,
        args.eai_root,
        environment=args.environment,
        variant=args.variant,
    )
    filename = (
        "scene_task_contradiction_preference_pairs.jsonl"
        if args.selection_mode == "full_gold_conflict"
        else "scene_task_contradiction_relation_only_pairs.jsonl"
    )
    pairs_path, summary_path = write_preference_pairs(
        pairs, summary, args.output_dir, filename=filename
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Preference pairs: {pairs_path}")
    print(f"Summary:          {summary_path}")


if __name__ == "__main__":
    main()
