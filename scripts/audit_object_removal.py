from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heal_repro.data import (
    normalize_object,
    parse_scene,
    prompt_from_row,
    read_rows,
)
from heal_repro.preferences import load_virtualhome_gold_goals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit VirtualHome object-removal rows against baseline scenes and EAI gold."
    )
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "workspace" / "HEAL_dataset"
    )
    parser.add_argument(
        "--eai-root",
        type=Path,
        default=ROOT / "workspace" / "embodied-agent-interface",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = read_rows(args.data_root, "virtualhome", "baseline")
    removals = read_rows(args.data_root, "virtualhome", "object_removal")
    baseline_by_task = {row["task_id"]: row for row in baseline}
    gold_index = load_virtualhome_gold_goals(args.eai_root)

    removed_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    task_row_counts: Counter[str] = Counter()
    integrity_issues = []
    indirect_rows = []

    for row_index, row in enumerate(removals):
        task_id = row["task_id"]
        task_row_counts[task_id] += 1
        removed = normalize_object(row.get("removed_object", ""))
        removed_counts[removed] += 1
        baseline_row = baseline_by_task[task_id]
        baseline_scene = set(parse_scene(prompt_from_row(baseline_row, "baseline")))
        modified_scene = set(parse_scene(prompt_from_row(row, "object_removal")))
        actual_removed = sorted(baseline_scene - modified_scene)
        actual_added = sorted(modified_scene - baseline_scene)
        if (
            not removed
            or removed not in baseline_scene
            or removed in modified_scene
            or actual_removed != [removed]
            or actual_added
        ):
            integrity_issues.append(
                {
                    "row_index": row_index,
                    "task_id": task_id,
                    "declared_removed": removed,
                    "actual_removed": actual_removed,
                    "actual_added": actual_added,
                }
            )

        gold = gold_index[task_id]
        blocked_nodes = [goal for goal in gold.node_goals if goal.name == removed]
        blocked_relations = [
            relation
            for relation in gold.relations
            if removed in {relation.from_name, relation.to_name}
        ]
        if blocked_nodes or blocked_relations:
            category = "direct_eai_node_or_relation_dependency"
        elif gold.action_goals:
            category = "action_goal_without_direct_object_endpoint"
        else:
            category = "removed_object_not_in_eai_goal"
        category_counts[category] += 1
        if category != "direct_eai_node_or_relation_dependency":
            indirect_rows.append(
                {
                    "row_index": row_index,
                    "task_id": task_id,
                    "task_name": row.get("task_name", ""),
                    "removed_object": removed,
                    "gold_node_goals": [
                        [goal.name, goal.state] for goal in gold.node_goals
                    ],
                    "gold_relations": [
                        [rel.from_name, rel.relation, rel.to_name]
                        for rel in gold.relations
                    ],
                    "gold_action_goals": list(gold.action_goals),
                }
            )

    report = {
        "rows": len(removals),
        "unique_tasks": len(task_row_counts),
        "rows_per_task_distribution": dict(
            sorted(Counter(task_row_counts.values()).items())
        ),
        "integrity_issues": integrity_issues,
        "dependency_categories": dict(sorted(category_counts.items())),
        "removed_object_counts": dict(sorted(removed_counts.items())),
        "indirect_rows": indirect_rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Audit report: {args.output}")


if __name__ == "__main__":
    main()
