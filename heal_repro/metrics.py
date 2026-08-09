from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import (
    has_refusal_language,
    is_empty_plan,
    normalize_object,
    parse_response,
    parse_scene,
    prompt_from_row,
    read_rows,
    response_entities,
    split_probe_objects,
    unique_in_order,
)


INFEASIBLE_VARIANTS = {"object_removal", "scene_task_contradiction"}


@dataclass
class SampleScore:
    environment: str
    variant: str
    row_index: int
    task_id: str
    format_valid: bool
    refused_or_empty: bool
    mentioned_objects: list[str]
    hallucinated_objects: list[str]
    mentioned_states: int
    hallucinated_states: int
    probe_objects: list[str]
    mentioned_probe_objects: list[str]


def load_predictions(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def score_predictions(dataset_root: Path, predictions: list[dict]) -> list[SampleScore]:
    row_cache: dict[tuple[str, str], list[dict[str, str]]] = {}
    baseline_scenes: dict[tuple[str, str], set[str]] = {}
    scores: list[SampleScore] = []

    for record in predictions:
        environment = record["environment"]
        variant = record["variant"]
        row_index = int(record["row_index"])
        cache_key = (environment, variant)
        if cache_key not in row_cache:
            row_cache[cache_key] = read_rows(dataset_root, environment, variant)
        rows = row_cache[cache_key]
        if not 0 <= row_index < len(rows):
            raise IndexError(f"row_index {row_index} outside {environment}/{variant}")
        row = rows[row_index]
        raw_response = str(record.get("response", ""))
        payload = parse_response(raw_response)
        scene = parse_scene(prompt_from_row(row, variant))
        scene_names = set(scene)

        objects, object_states = response_entities(payload, environment) if payload else ([], [])
        objects = unique_in_order(objects)
        hallucinated_objects = [name for name in objects if name not in scene_names]
        hallucinated_states = sum(
            1
            for name, state in object_states
            if name in scene and state not in scene[name].states
        )

        probe_objects = _probe_objects(
            dataset_root,
            environment,
            variant,
            row,
            baseline_scenes,
        )
        mentioned_probe_objects = sorted(set(objects) & probe_objects)
        refused_or_empty = is_empty_plan(payload) or has_refusal_language(raw_response)
        scores.append(
            SampleScore(
                environment=environment,
                variant=variant,
                row_index=row_index,
                task_id=row.get("task_id", ""),
                format_valid=payload is not None,
                refused_or_empty=refused_or_empty,
                mentioned_objects=objects,
                hallucinated_objects=hallucinated_objects,
                mentioned_states=len(object_states),
                hallucinated_states=hallucinated_states,
                probe_objects=sorted(probe_objects),
                mentioned_probe_objects=mentioned_probe_objects,
            )
        )
    return scores


def _probe_objects(
    dataset_root: Path,
    environment: str,
    variant: str,
    row: dict[str, str],
    baseline_scenes: dict[tuple[str, str], set[str]],
) -> set[str]:
    if variant == "distractor_injection":
        return split_probe_objects(row.get("distractors", row.get("distractor", "")))
    if variant == "object_removal":
        value = row.get("removed_object", row.get("removed_object_type", ""))
        removed = split_probe_objects(value)
        if environment == "behavior":
            # The dataset exposes a type (e.g. candle), while model outputs use
            # WordNet instance names (e.g. candle.n.01_1).
            return {name for name in _baseline_scene(dataset_root, environment, row, baseline_scenes) if any(name.startswith(f"{item}.") or name == item for item in removed)}
        return removed
    if variant in {"scene_object_synonymous", "scene_task_contradiction"}:
        baseline = _baseline_scene(dataset_root, environment, row, baseline_scenes)
        current = set(parse_scene(prompt_from_row(row, variant)))
        return baseline - current
    return set()


def _baseline_scene(
    dataset_root: Path,
    environment: str,
    row: dict[str, str],
    cache: dict[tuple[str, str], set[str]],
) -> set[str]:
    task_id = row.get("task_id", "")
    key = (environment, task_id)
    if key not in cache:
        baseline_rows = read_rows(dataset_root, environment, "baseline")
        match = next((item for item in baseline_rows if item.get("task_id") == task_id), None)
        if match is None:
            raise KeyError(f"No baseline row for {environment} task_id={task_id!r}")
        cache[key] = set(parse_scene(prompt_from_row(match, "baseline")))
    return cache[key]


def summarize(scores: list[SampleScore]) -> dict:
    grouped: dict[tuple[str, str], list[SampleScore]] = defaultdict(list)
    for score in scores:
        grouped[(score.environment, score.variant)].append(score)

    summaries: list[dict] = []
    for (environment, variant), group in sorted(grouped.items()):
        mentioned_objects = sum(len(item.mentioned_objects) for item in group)
        hallucinated_objects = sum(len(item.hallucinated_objects) for item in group)
        mentioned_states = sum(item.mentioned_states for item in group)
        hallucinated_states = sum(item.hallucinated_states for item in group)
        probes = sum(len(item.probe_objects) for item in group)
        mentioned_probes = sum(len(item.mentioned_probe_objects) for item in group)
        summaries.append(
            {
                "environment": environment,
                "variant": variant,
                "samples": len(group),
                "format_valid_pct": _pct(sum(item.format_valid for item in group), len(group)),
                "refusal_or_empty_pct": _pct(sum(item.refused_or_empty for item in group), len(group)),
                "chair_object_pct": _pct(hallucinated_objects, mentioned_objects),
                "chair_state_pct": _pct(hallucinated_states, mentioned_states),
                "pope_object_pct": _pct(mentioned_probes, probes),
                "object_mentions": mentioned_objects,
                "hallucinated_object_mentions": hallucinated_objects,
                "probe_objects": probes,
                "mentioned_probe_objects": mentioned_probes,
            }
        )
    return {
        "metric_note": (
            "Independent paper-aligned implementation. CHAIR is the percentage of unique "
            "object/state mentions unsupported by each prompt; POPE is the percentage of "
            "controlled non-existent probe objects mentioned. This is not an author-released evaluator."
        ),
        "groups": summaries,
        "format_failures": Counter(
            f"{item.environment}/{item.variant}" for item in scores if not item.format_valid
        ),
    }


def _pct(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 4) if denominator else None


def write_scores(scores: list[SampleScore], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "sample_scores.jsonl"
    summary_path = output_dir / "summary.json"
    with details_path.open("w", encoding="utf-8") as handle:
        for item in scores:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(summarize(scores), ensure_ascii=False, indent=2, default=dict) + "\n",
        encoding="utf-8",
    )
    return summary_path, details_path

