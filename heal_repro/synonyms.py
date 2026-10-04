from __future__ import annotations

from pathlib import Path

from .data import RelationTriple, parse_scene_object_sequence, prompt_from_row, read_rows


def load_virtualhome_synonym_projections(
    dataset_root: Path,
) -> dict[int, dict[str, str]]:
    """Recover canonical-to-synonym object mappings for every HEAL row.

    The public CSV has no explicit mapping column.  Baseline and synonym prompts
    retain the same task IDs and scene-line order, so each substitution can be
    recovered positionally.  Projection is intentionally canonical -> synonym:
    the reverse direction can be ambiguous when two objects share one alias.
    """

    baseline = read_rows(dataset_root, "virtualhome", "baseline")
    synonymous = read_rows(
        dataset_root, "virtualhome", "scene_object_synonymous"
    )
    baseline_by_task = {row["task_id"]: row for row in baseline}
    if len(baseline_by_task) != len(baseline):
        raise ValueError("VirtualHome baseline contains duplicate task IDs")

    projections: dict[int, dict[str, str]] = {}
    for row_index, row in enumerate(synonymous):
        task_id = row["task_id"]
        if task_id not in baseline_by_task:
            raise KeyError(f"No baseline row for synonym task_id={task_id}")
        canonical_names = parse_scene_object_sequence(
            prompt_from_row(baseline_by_task[task_id], "baseline")
        )
        synonym_names = parse_scene_object_sequence(
            prompt_from_row(row, "scene_object_synonymous")
        )
        if len(canonical_names) != len(synonym_names):
            raise ValueError(
                f"Scene line count mismatch at synonym row={row_index}, task={task_id}: "
                f"baseline={len(canonical_names)}, synonym={len(synonym_names)}"
            )
        projection: dict[str, str] = {}
        for canonical, synonym in zip(canonical_names, synonym_names, strict=True):
            previous = projection.setdefault(canonical, synonym)
            if previous != synonym:
                raise ValueError(
                    f"Conflicting synonym for {canonical!r} at row={row_index}: "
                    f"{previous!r} versus {synonym!r}"
                )
        projections[row_index] = projection
    return projections


def project_object_name(name: str, projection: dict[str, str]) -> str:
    """Project a canonical EAI name into the row's synonym vocabulary."""

    return projection.get(name, name)


def project_relation_triples(
    relations: list[RelationTriple], projection: dict[str, str]
) -> list[RelationTriple]:
    """Project relation endpoints and deduplicate alias-collapsed triples."""

    return sorted(
        {
            RelationTriple(
                project_object_name(relation.from_name, projection),
                relation.relation,
                project_object_name(relation.to_name, projection),
            )
            for relation in relations
        }
    )


def project_relation_constraints(
    constraints: dict[str, frozenset[str]], projection: dict[str, str]
) -> dict[str, frozenset[str]]:
    """Project the prompt's canonical relation-target ontology to synonym names."""

    return {
        relation: frozenset(
            project_object_name(target, projection) for target in targets
        )
        for relation, targets in constraints.items()
    }
