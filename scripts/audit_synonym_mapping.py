from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heal_repro.data import parse_scene_object_sequence, prompt_from_row, read_rows
from heal_repro.preferences import load_virtualhome_gold_goals
from heal_repro.synonyms import load_virtualhome_synonym_projections, project_object_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover and audit VirtualHome synonym substitutions from paired HEAL CSVs."
    )
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "workspace" / "HEAL_dataset"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = read_rows(args.data_root, "virtualhome", "baseline")
    synonymous = read_rows(
        args.data_root, "virtualhome", "scene_object_synonymous"
    )
    baseline_by_task = {row["task_id"]: row for row in baseline}
    projections = load_virtualhome_synonym_projections(args.data_root)
    eai_root = ROOT / "workspace" / "embodied-agent-interface"
    gold_index = load_virtualhome_gold_goals(eai_root)
    pair_counts: Counter[tuple[str, str]] = Counter()
    length_shapes: Counter[tuple[int, int]] = Counter()
    row_alias_collisions = []
    unchanged_pairs: Counter[str] = Counter()
    gold_projection_issues = []

    for row_index, row in enumerate(synonymous):
        task_id = row["task_id"]
        if task_id not in baseline_by_task:
            raise SystemExit(f"No baseline row for task_id={task_id}")
        baseline_scene = parse_scene_object_sequence(
            prompt_from_row(baseline_by_task[task_id], "baseline")
        )
        modified_scene = parse_scene_object_sequence(
            prompt_from_row(row, "scene_object_synonymous")
        )
        length_shapes[(len(baseline_scene), len(modified_scene))] += 1
        if len(baseline_scene) != len(modified_scene):
            raise SystemExit(
                f"Scene line count mismatch at row={row_index}, task={task_id}: "
                f"baseline={len(baseline_scene)}, synonym={len(modified_scene)}"
            )
        for canonical, synonym in zip(baseline_scene, modified_scene, strict=True):
            if canonical == synonym:
                unchanged_pairs[canonical] += 1
            else:
                pair_counts[(canonical, synonym)] += 1
        aliases: dict[str, list[str]] = {}
        for canonical, synonym in zip(baseline_scene, modified_scene, strict=True):
            aliases.setdefault(synonym, []).append(canonical)
        collisions = {
            synonym: sorted(set(canonical_names))
            for synonym, canonical_names in aliases.items()
            if len(set(canonical_names)) > 1
        }
        if collisions:
            row_alias_collisions.append(
                {"row_index": row_index, "task_id": task_id, "collisions": collisions}
            )

        gold = gold_index[task_id]
        projection = projections[row_index]
        canonical_goal_names = {
            goal.name for goal in gold.node_goals
        } | {
            endpoint
            for relation in gold.relations
            for endpoint in (relation.from_name, relation.to_name)
        }
        missing_canonical = sorted(canonical_goal_names - set(projection))
        modified_name_set = set(modified_scene)
        missing_projected = sorted(
            {
                project_object_name(name, projection)
                for name in canonical_goal_names
            }
            - modified_name_set
        )
        if missing_canonical or missing_projected:
            gold_projection_issues.append(
                {
                    "row_index": row_index,
                    "task_id": task_id,
                    "gold_names_absent_from_baseline_mapping": missing_canonical,
                    "projected_gold_names_absent_from_synonym_scene": missing_projected,
                }
            )

    canonical_aliases: dict[str, set[str]] = {}
    synonym_canonicals: dict[str, set[str]] = {}
    for canonical, synonym in pair_counts:
        canonical_aliases.setdefault(canonical, set()).add(synonym)
        synonym_canonicals.setdefault(synonym, set()).add(canonical)

    report = {
        "baseline_rows": len(baseline),
        "synonym_rows": len(synonymous),
        "scene_line_count_pairs": {
            f"baseline_{baseline_count}_synonym_{synonym_count}": count
            for (baseline_count, synonym_count), count in sorted(length_shapes.items())
        },
        "mapping_counts": [
            {"canonical": canonical, "synonym": synonym, "rows": count}
            for (canonical, synonym), count in sorted(pair_counts.items())
        ],
        "unchanged_names": dict(sorted(unchanged_pairs.items())),
        "canonical_names_with_multiple_aliases": {
            canonical: sorted(aliases)
            for canonical, aliases in sorted(canonical_aliases.items())
            if len(aliases) > 1
        },
        "aliases_used_by_multiple_canonical_names": {
            synonym: sorted(canonical_names)
            for synonym, canonical_names in sorted(synonym_canonicals.items())
            if len(canonical_names) > 1
        },
        "row_alias_collisions": row_alias_collisions,
        "gold_projection_issues": gold_projection_issues,
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
