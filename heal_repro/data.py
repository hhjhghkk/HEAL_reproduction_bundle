from __future__ import annotations

import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VARIANT_FILES = {
    "baseline": "baseline.csv",
    "distractor_injection": "distractor_injection.csv",
    "object_removal": "object_removal.csv",
    "scene_object_synonymous": "scene_object_synonymous.csv",
    "scene_task_contradiction": "scene_task_contradiction.csv",
}

EXPECTED_ROWS = {
    "virtualhome": {
        "baseline": 338,
        "distractor_injection": 338,
        "object_removal": 582,
        "scene_object_synonymous": 338,
        "scene_task_contradiction": 338,
    },
    # The public repository currently has 99 rather than the paper's reported
    # 100 examples in these three variants.  We validate the public artifact.
    "behavior": {
        "baseline": 99,
        "distractor_injection": 99,
        "object_removal": 678,
        "scene_object_synonymous": 99,
        "scene_task_contradiction": 99,
    },
}


@dataclass(frozen=True)
class SceneObject:
    name: str
    states: frozenset[str]


def read_rows(dataset_root: Path, environment: str, variant: str) -> list[dict[str, str]]:
    if environment not in EXPECTED_ROWS:
        raise ValueError(f"Unknown environment: {environment}")
    if variant not in VARIANT_FILES:
        raise ValueError(f"Unknown variant: {variant}")
    path = dataset_root / environment / VARIANT_FILES[variant]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prompt_from_row(row: dict[str, str], variant: str) -> str:
    key = "baseline_prompts" if variant == "baseline" else "modified_prompts"
    prompt = row.get(key, "")
    if not prompt.strip():
        raise ValueError(f"Row has no prompt in column {key!r}")
    return prompt


def parse_scene(prompt: str) -> dict[str, SceneObject]:
    """Parse the explicitly listed scene objects and their allowed states.

    Both VirtualHome (``name, initial states: ...``) and BEHAVIOR
    (``wordnet_name_id: [...]``) formats are supported.  Only the authoritative
    scene section is inspected; examples elsewhere in the prompt are ignored.
    """
    marker = "Relevant objects in the scene are:"
    if marker not in prompt:
        return {}
    section = prompt.split(marker, 1)[1]
    stop_markers = (
        "All possible relationships",
        "All initial states in the scene are:",
        "Symbolic goals format:",
    )
    stops = [section.find(item) for item in stop_markers if section.find(item) >= 0]
    if stops:
        section = section[: min(stops)]

    objects: dict[str, SceneObject] = {}
    vh_pattern = re.compile(
        r"^\s*([^,\r\n]+),\s*initial states:\s*(\[[^\r\n]*\]),\s*"
        r"possible states:\s*(\[[^\r\n]*\])\s*$",
        re.IGNORECASE,
    )
    behavior_pattern = re.compile(r"^\s*([^:\r\n]+):\s*(\[[^\r\n]*\])\s*$")

    for line in section.splitlines():
        line = line.strip()
        if not line:
            continue
        match = vh_pattern.match(line)
        if match:
            name = normalize_object(match.group(1))
            states = _literal_string_set(match.group(3))
            objects[name] = SceneObject(name, frozenset(normalize_state(x) for x in states))
            continue
        match = behavior_pattern.match(line)
        if match:
            name = normalize_object(match.group(1))
            states = _literal_string_set(match.group(2))
            objects[name] = SceneObject(name, frozenset(normalize_state(x) for x in states))
    return objects


def _literal_string_set(value: str) -> set[str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return set()
    return {str(item) for item in parsed} if isinstance(parsed, (list, tuple, set)) else set()


def normalize_object(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_state(value: object) -> str:
    return str(value).strip().upper()


def parse_response(text: str) -> dict | None:
    """Extract the first JSON object from a raw model response."""
    cleaned = text.strip().replace("<|eot_id|>", "")
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    starts = [match.start() for match in re.finditer(r"\{", cleaned)]
    for start in starts:
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
            if _looks_like_goal_payload(value):
                return value
        except json.JSONDecodeError:
            continue
    # Some EAI prompts show Python literals and models copy their single quotes.
    for start in starts:
        try:
            value = ast.literal_eval(cleaned[start:])
            if _looks_like_goal_payload(value):
                return value
        except (SyntaxError, ValueError):
            continue
    return None


def _looks_like_goal_payload(value: object) -> bool:
    """Reject nested example dictionaries extracted from truncated responses."""
    if not isinstance(value, dict):
        return False
    return bool({"node goals", "edge goals", "action goals"} & set(value))


def response_entities(payload: dict, environment: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Return mentioned objects and ``(object, state)`` pairs."""
    objects: list[str] = []
    object_states: list[tuple[str, str]] = []
    node_goals = payload.get("node goals", [])
    edge_goals = payload.get("edge goals", [])

    if environment == "virtualhome":
        for goal in node_goals if isinstance(node_goals, list) else []:
            if isinstance(goal, dict) and "name" in goal:
                name = normalize_object(goal["name"])
                objects.append(name)
                if "state" in goal:
                    object_states.append((name, normalize_state(goal["state"])))
        for goal in edge_goals if isinstance(edge_goals, list) else []:
            if isinstance(goal, dict):
                for key in ("from_name", "to_name"):
                    if key in goal:
                        objects.append(normalize_object(goal[key]))
    elif environment == "behavior":
        for goal in node_goals if isinstance(node_goals, list) else []:
            flat = _strip_not(goal)
            if isinstance(flat, list) and len(flat) >= 2:
                name = normalize_object(flat[1])
                objects.append(name)
                object_states.append((name, normalize_state(flat[0])))
        for goal in edge_goals if isinstance(edge_goals, list) else []:
            flat = _strip_not(goal)
            if isinstance(flat, list) and len(flat) >= 3:
                objects.extend((normalize_object(flat[1]), normalize_object(flat[2])))
    else:
        raise ValueError(f"Unknown environment: {environment}")
    return objects, object_states


def _strip_not(value: object) -> object:
    if isinstance(value, list) and len(value) == 2 and str(value[0]).lower() == "not":
        return value[1]
    return value


def is_empty_plan(payload: dict | None) -> bool:
    if payload is None:
        return False
    goal_keys = ("node goals", "edge goals", "action goals")
    present = [payload[key] for key in goal_keys if key in payload]
    return bool(present) and all(value in ([], {}, None, "") for value in present)


def has_refusal_language(text: str) -> bool:
    lowered = text.lower()
    phrases = (
        "cannot complete",
        "can't complete",
        "cannot perform",
        "can't perform",
        "cannot generate",
        "unable to",
        "not feasible",
        "infeasible",
        "required object",
        "missing object",
        "no valid plan",
    )
    return any(phrase in lowered for phrase in phrases)


def split_probe_objects(value: str) -> set[str]:
    if not value:
        return set()
    return {
        normalize_object(item)
        for item in re.split(r"\s*,\s*|\s*;\s*", value)
        if normalize_object(item)
    }


def unique_in_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
