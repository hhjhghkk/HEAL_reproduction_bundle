from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

from .data import (
    RelationTriple,
    normalize_object,
    normalize_state,
    parse_response,
    parse_scene,
    prompt_from_row,
    read_rows,
)
from .metrics import (
    VIRTUALHOME_RELATION_GROUND_TRUTH,
    SampleScore,
    load_virtualhome_relation_ground_truth,
    score_predictions,
)
from .synonyms import project_object_name


STRICT_INFEASIBLE_LABEL = "blocked_gold+hallucinated_relation"

# These two action-only task families have no object-bearing EAI node/edge goal.
# Their scene requirements were therefore audited against the natural-language
# task and the action ontology instead of being guessed from every baseline object.
ACTION_CONFLICT_REQUIREMENTS = {
    "write an email": frozenset({"keyboard"}),
    "wash dishes by hand": frozenset({"sink", "faucet", "water"}),
}
ACTION_FEASIBLE_REQUIREMENTS = {
    "wash hands": frozenset({"sink", "soap"}),
}


@dataclass(frozen=True, order=True)
class NodeGoal:
    name: str
    state: str


@dataclass(frozen=True)
class VirtualHomeGoldGoals:
    node_goals: tuple[NodeGoal, ...]
    relations: tuple[RelationTriple, ...]
    action_goals: tuple[str, ...]


def project_virtualhome_gold_goals(
    gold: VirtualHomeGoldGoals,
    projection: dict[str, str],
) -> VirtualHomeGoldGoals:
    """Project canonical EAI node/relation endpoints into synonym prompt names."""

    return VirtualHomeGoldGoals(
        node_goals=tuple(
            sorted(
                {
                    NodeGoal(project_object_name(goal.name, projection), goal.state)
                    for goal in gold.node_goals
                }
            )
        ),
        relations=tuple(
            sorted(
                {
                    RelationTriple(
                        project_object_name(relation.from_name, projection),
                        relation.relation,
                        project_object_name(relation.to_name, projection),
                    )
                    for relation in gold.relations
                }
            )
        ),
        action_goals=gold.action_goals,
    )


def load_virtualhome_gold_goals(eai_root: Path) -> dict[str, VirtualHomeGoldGoals]:
    """Load VirtualHome node, relation and action goals indexed by HEAL task_id."""

    path = eai_root / VIRTUALHOME_RELATION_GROUND_TRUTH
    if not path.exists():
        raise FileNotFoundError(f"VirtualHome goal ground truth not found: {path}")
    root = json.loads(path.read_text(encoding="utf-8"))
    relation_index = load_virtualhome_relation_ground_truth(eai_root)
    result: dict[str, VirtualHomeGoldGoals] = {}

    def visit(node: object) -> None:
        if not isinstance(node, dict):
            return
        for task_id, value in node.items():
            if isinstance(value, dict) and isinstance(value.get("vh_goal"), dict):
                vh_goal = value["vh_goal"]
                node_goals = sorted(
                    {
                        NodeGoal(
                            normalize_object(goal["class_name"]),
                            normalize_state(goal["state"]),
                        )
                        for goal in vh_goal.get("goal", [])
                        if isinstance(goal, dict)
                        and {"class_name", "state"}.issubset(goal)
                    }
                )
                action_goals = tuple(
                    sorted(
                        {
                            str(action).strip().upper()
                            for action in vh_goal.get("actions", [])
                            if str(action).strip()
                        }
                    )
                )
                goals = VirtualHomeGoldGoals(
                    tuple(node_goals),
                    tuple(relation_index.get(task_id, [])),
                    action_goals,
                )
                if task_id in result and result[task_id] != goals:
                    raise ValueError(f"Conflicting goal ground truth for task_id={task_id!r}")
                result[task_id] = goals
            else:
                visit(value)

    visit(root)
    return result


def build_full_conflict_preference_pairs(
    dataset_root: Path,
    predictions: list[dict],
    eai_root: Path,
    environment: str = "virtualhome",
    variant: str = "scene_task_contradiction",
) -> tuple[list[dict], dict]:
    """Build one evidence-backed pair for every resolvable conflict sample.

    A modified scene does not always make the original goal impossible: some
    public conflict prompts still contain every object used by the EAI gold.
    Those samples receive the executable EAI goal as ``chosen``; only samples
    with blocked goals or audited action prerequisites receive a refusal.
    """

    if environment != "virtualhome" or variant != "scene_task_contradiction":
        raise ValueError(
            "Full conflict preference construction currently supports only "
            "virtualhome/scene_task_contradiction"
        )
    selected_predictions = [
        record
        for record in predictions
        if record.get("environment") == environment and record.get("variant") == variant
    ]
    scores = score_predictions(dataset_root, selected_predictions, eai_root=eai_root)
    score_index = {
        (item.environment, item.variant, item.row_index): item for item in scores
    }
    if len(score_index) != len(scores):
        raise ValueError("Duplicate prediction keys found while building preference pairs")

    gold_index = load_virtualhome_gold_goals(eai_root)
    rows = read_rows(dataset_root, environment, variant)
    pairs: list[dict] = []
    unresolved: list[dict] = []
    selected_scores: list[SampleScore] = []
    blocked_node_total = 0
    pair_types: dict[str, int] = {}
    rejected_sources: dict[str, int] = {}

    for record in selected_predictions:
        row_index = int(record["row_index"])
        score = score_index[(environment, variant, row_index)]
        row = rows[row_index]
        if row.get("task_id", "") != score.task_id:
            raise ValueError(
                f"task_id mismatch at {environment}/{variant}/{row_index}: "
                f"dataset={row.get('task_id')!r}, score={score.task_id!r}"
            )
        if score.task_id not in gold_index:
            unresolved.append({"row_index": row_index, "task_id": score.task_id, "reason": "no_gold"})
            continue

        prompt = prompt_from_row(row, variant)
        scene_names = set(parse_scene(prompt))
        gold = gold_index[score.task_id]
        blocked_nodes = [goal for goal in gold.node_goals if goal.name not in scene_names]
        blocked_relations = [
            relation
            for relation in gold.relations
            if relation.from_name not in scene_names or relation.to_name not in scene_names
        ]
        task_name = normalize_object(row.get("task_name", ""))
        action_missing: list[str] = []
        pair_type = ""
        if blocked_nodes or blocked_relations:
            pair_type = "gold_goal_blocked_refusal"
        elif task_name in ACTION_CONFLICT_REQUIREMENTS:
            required = ACTION_CONFLICT_REQUIREMENTS[task_name]
            action_missing = sorted(required - scene_names)
            if action_missing != sorted(required):
                unresolved.append(
                    {
                        "row_index": row_index,
                        "task_id": score.task_id,
                        "reason": "action_conflict_requirement_partially_present",
                    }
                )
                continue
            pair_type = "action_prerequisite_blocked_refusal"
        elif gold.node_goals or gold.relations:
            pair_type = "eai_gold_executable"
        elif task_name in ACTION_FEASIBLE_REQUIREMENTS:
            required = ACTION_FEASIBLE_REQUIREMENTS[task_name]
            if not required.issubset(scene_names):
                unresolved.append(
                    {
                        "row_index": row_index,
                        "task_id": score.task_id,
                        "reason": "audited_action_feasibility_requirement_missing",
                    }
                )
                continue
            pair_type = "eai_action_gold_executable"
        else:
            unresolved.append(
                {
                    "row_index": row_index,
                    "task_id": score.task_id,
                    "reason": "no_audited_node_relation_or_action_decision",
                }
            )
            continue

        pair = build_full_conflict_pair(
            record,
            score,
            scene_names,
            gold,
            blocked_nodes,
            blocked_relations,
            prompt,
            pair_type,
            action_missing,
        )
        pairs.append(pair)
        selected_scores.append(score)
        blocked_node_total += len(blocked_nodes)
        pair_types[pair_type] = pair_types.get(pair_type, 0) + 1
        rejected_source = pair["rejected_source"]
        rejected_sources[rejected_source] = rejected_sources.get(rejected_source, 0) + 1

    summary = {
        "status": (
            "candidate_pairs_pending_manual_review"
            if unresolved
            else "complete_coverage_pending_manual_review"
        ),
        "environment": environment,
        "variant": variant,
        "source_predictions": len(selected_predictions),
        "selected_pairs": len(pairs),
        "selection_label": "variant_aware_gold_or_refusal_v2",
        "pair_types": pair_types,
        "rejected_sources": rejected_sources,
        "chosen_policy": (
            "Use structured refusal when gold goals or audited action prerequisites are blocked; "
            "otherwise serialize the executable EAI node/edge/action gold."
        ),
        "rejected_policy": (
            "Use the byte-for-byte local-model response when it differs from the valid chosen "
            "behavior; otherwise create a controlled wrong refusal or blocked-gold answer."
        ),
        "blocked_gold_node_goals": blocked_node_total,
        "selected_rejected_metrics": _selected_metrics(selected_scores),
        "unresolved_samples": unresolved,
        "storage_note": (
            "Full prompts are not duplicated. Recover each prompt from the HEAL CSV using "
            "environment, variant, and row_index."
        ),
    }
    return pairs, summary


def build_infeasible_preference_pairs(
    dataset_root: Path,
    predictions: list[dict],
    eai_root: Path,
    environment: str = "virtualhome",
    variant: str = "scene_task_contradiction",
) -> tuple[list[dict], dict]:
    """Build high-confidence refusal-vs-hallucinated-plan preference pairs.

    The full HEAL prompt is deliberately not copied into every pair.  A pair's
    ``environment``, ``variant`` and ``row_index`` form a lossless reference to
    the source CSV, while the original prediction JSONL remains the immutable
    source for the raw model response.
    """

    selected_predictions = [
        record
        for record in predictions
        if record.get("environment") == environment and record.get("variant") == variant
    ]
    scores = score_predictions(dataset_root, selected_predictions, eai_root=eai_root)
    score_index = {
        (item.environment, item.variant, item.row_index): item for item in scores
    }
    if len(score_index) != len(scores):
        raise ValueError("Duplicate prediction keys found while building preference pairs")

    rows = read_rows(dataset_root, environment, variant)
    pairs: list[dict] = []
    selected_scores: list[SampleScore] = []
    for record in selected_predictions:
        key = (environment, variant, int(record["row_index"]))
        score = score_index[key]
        if not is_strict_infeasible_candidate(score):
            continue

        row = rows[score.row_index]
        if row.get("task_id", "") != score.task_id:
            raise ValueError(
                f"task_id mismatch at {environment}/{variant}/{score.row_index}: "
                f"dataset={row.get('task_id')!r}, score={score.task_id!r}"
            )
        scene_names = set(parse_scene(prompt_from_row(row, variant)))
        pair = build_compact_pair(record, score, scene_names)
        pairs.append(pair)
        selected_scores.append(score)

    summary = {
        "status": "candidate_pairs_pending_manual_review",
        "environment": environment,
        "variant": variant,
        "source_predictions": len(selected_predictions),
        "selected_pairs": len(pairs),
        "selection_label": STRICT_INFEASIBLE_LABEL,
        "selection_rule": {
            "format_valid": True,
            "refused_or_empty": False,
            "blocked_gold_relations_nonempty": True,
            "hallucinated_relations_nonempty": True,
        },
        "chosen_policy": (
            "Deterministic structured refusal; missing objects are absent endpoints of "
            "blocked EAI gold relations."
        ),
        "rejected_policy": "Parsed goal payload from the original model response.",
        "selected_rejected_metrics": _selected_metrics(selected_scores),
        "storage_note": (
            "Full prompts are not duplicated. Recover each prompt from the HEAL CSV using "
            "environment, variant, and row_index; recover the byte-for-byte raw response from "
            "the source prediction JSONL using the same key."
        ),
    }
    return pairs, summary


def is_strict_infeasible_candidate(score: SampleScore) -> bool:
    return (
        score.format_valid
        and not score.refused_or_empty
        and bool(score.blocked_gold_relations)
        and bool(score.hallucinated_relations)
    )


def build_compact_pair(
    record: dict,
    score: SampleScore,
    scene_names: set[str],
) -> dict:
    """Create one compact, auditable preference record."""

    rejected = parse_response(str(record.get("response", "")))
    if rejected is None:
        raise ValueError("A strict preference candidate must have a parseable response")

    missing_objects = sorted(
        {
            endpoint
            for relation in score.blocked_gold_relations
            for endpoint in (relation.from_name, relation.to_name)
            if endpoint not in scene_names
        }
    )
    if not missing_objects:
        raise ValueError("Blocked gold relations did not yield any absent endpoint")

    chosen = {
        "node goals": [],
        "edge goals": [],
        "action goals": [],
        "reason": _refusal_reason(missing_objects),
    }
    predicted_count = len(score.predicted_relations)
    return {
        "environment": score.environment,
        "variant": score.variant,
        "row_index": score.row_index,
        "task_id": score.task_id,
        "model": str(record.get("model", "")),
        "chosen": chosen,
        "rejected": rejected,
        "metrics": {
            "format_valid": score.format_valid,
            "refused_or_empty": score.refused_or_empty,
            "chair_object_pct": _pct(
                len(score.hallucinated_objects), len(score.mentioned_objects)
            ),
            "chair_state_pct": _pct(
                score.hallucinated_states, score.mentioned_states
            ),
            "pope_object_pct": _pct(
                len(score.mentioned_probe_objects), len(score.probe_objects)
            ),
            "mentioned_objects": score.mentioned_objects,
            "hallucinated_objects": score.hallucinated_objects,
            "mentioned_states": score.mentioned_states,
            "hallucinated_states": score.hallucinated_states,
            "probe_objects": score.probe_objects,
            "mentioned_probe_objects": score.mentioned_probe_objects,
            "predicted_relations": _triple_lists(score.predicted_relations),
            "relation_endpoint_hallucination_pct": _pct(
                len(score.hallucinated_relations), predicted_count
            ),
            "relation_format_errors": score.relation_format_errors,
            "hallucinated_relations": _triple_lists(score.hallucinated_relations),
            "invalid_relation_types": _triple_lists(score.invalid_relation_types),
            "invalid_relation_targets": _triple_lists(score.invalid_relation_targets),
            "gold_relations": _triple_lists(score.gold_relations),
            "blocked_gold_relations": _triple_lists(score.blocked_gold_relations),
            "relation_goal_evaluable": score.relation_goal_evaluable,
        },
        "label_basis": STRICT_INFEASIBLE_LABEL,
    }


def build_full_conflict_pair(
    record: dict,
    score: SampleScore,
    scene_names: set[str],
    gold: VirtualHomeGoldGoals,
    blocked_nodes: list[NodeGoal],
    blocked_relations: list[RelationTriple],
    prompt: str,
    pair_type: str,
    action_missing: list[str],
) -> dict:
    """Create a full-coverage pair and retain the model response plus metrics."""

    missing_objects = sorted(
        {
            goal.name for goal in blocked_nodes if goal.name not in scene_names
        }
        | {
            endpoint
            for relation in blocked_relations
            for endpoint in (relation.from_name, relation.to_name)
            if endpoint not in scene_names
        }
        | set(action_missing)
    )
    is_refusal_pair = pair_type.endswith("blocked_refusal")
    if is_refusal_pair:
        if not missing_objects:
            raise ValueError("Blocked goals did not yield any absent object")
        chosen = {
            "node goals": [],
            "edge goals": [],
            "action goals": [],
            "reason": _refusal_reason(missing_objects),
        }
    else:
        chosen = _gold_payload(gold, prompt)

    raw_response = str(record.get("response", ""))
    comparison = _compare_model_to_gold(raw_response, score, gold)
    if is_refusal_pair:
        model_correct = score.refused_or_empty and not score.hallucinated_objects
    else:
        model_correct = comparison["model_goal_exact_match"]

    if model_correct and is_refusal_pair:
        rejected = json.dumps(_gold_payload(gold, prompt), ensure_ascii=False)
        rejected_source = "synthetic_blocked_gold_answer"
    elif model_correct:
        rejected = json.dumps(
            {
                "node goals": [],
                "edge goals": [],
                "action goals": [],
                "reason": "The task cannot be completed.",
            },
            ensure_ascii=False,
        )
        rejected_source = "synthetic_wrong_refusal"
    else:
        rejected = raw_response
        rejected_source = "local_model_response"

    metrics = _score_metrics(score)
    metrics.update(
        {
            "gold_node_goals": _node_goal_lists(gold.node_goals),
            "blocked_gold_node_goals": _node_goal_lists(blocked_nodes),
            "gold_action_goals": list(gold.action_goals),
            **comparison,
        }
    )
    pair = {
        "environment": score.environment,
        "variant": score.variant,
        "row_index": score.row_index,
        "task_id": score.task_id,
        "model": str(record.get("model", "")),
        "chosen": chosen,
        "rejected": rejected,
        "rejected_source": rejected_source,
        "model_metrics": metrics,
        "label_basis": pair_type,
    }
    if rejected_source != "local_model_response":
        pair["model_response"] = raw_response
    return pair


def _gold_payload(gold: VirtualHomeGoldGoals, prompt: str) -> dict:
    descriptions = _parse_action_descriptions(prompt)
    action_goals = []
    for alternatives in gold.action_goals:
        action = alternatives.split("|")[0].strip().upper()
        action_goals.append(
            {"action": action, "description": descriptions.get(action, "")}
        )
    return {
        "node goals": [
            {"name": goal.name, "state": goal.state} for goal in gold.node_goals
        ],
        "edge goals": [
            {
                "from_name": relation.from_name,
                "relation": relation.relation,
                "to_name": relation.to_name,
            }
            for relation in gold.relations
        ],
        "action goals": action_goals,
    }


def _compare_model_to_gold(
    raw_response: str,
    score: SampleScore,
    gold: VirtualHomeGoldGoals,
) -> dict:
    payload = parse_response(raw_response)
    predicted_nodes: set[NodeGoal] = set()
    predicted_actions: set[str] = set()
    node_format_errors = 0
    action_format_errors = 0
    if payload is not None:
        node_items = payload.get("node goals", [])
        if not isinstance(node_items, list):
            node_format_errors += 1
        else:
            for item in node_items:
                if not isinstance(item, dict) or not {"name", "state"}.issubset(item):
                    if item not in ({}, [], None, ""):
                        node_format_errors += 1
                    continue
                predicted_nodes.add(
                    NodeGoal(normalize_object(item["name"]), normalize_state(item["state"]))
                )
        action_items = payload.get("action goals", [])
        if not isinstance(action_items, list):
            action_format_errors += 1
        else:
            for item in action_items:
                if not isinstance(item, dict) or "action" not in item:
                    if item not in ({}, [], None, ""):
                        action_format_errors += 1
                    continue
                action = str(item["action"]).strip().upper()
                if action:
                    predicted_actions.add(action)

    gold_nodes = set(gold.node_goals)
    predicted_relations = set(score.predicted_relations)
    gold_relations = set(gold.relations)
    matched_actions, incorrect_actions, missing_actions = _compare_actions(
        predicted_actions, gold.action_goals
    )
    matched_nodes = sorted(predicted_nodes & gold_nodes)
    incorrect_nodes = sorted(predicted_nodes - gold_nodes)
    missing_nodes = sorted(gold_nodes - predicted_nodes)
    matched_relations = sorted(predicted_relations & gold_relations)
    incorrect_relations = sorted(predicted_relations - gold_relations)
    missing_relations = sorted(gold_relations - predicted_relations)
    exact = (
        payload is not None
        and node_format_errors == 0
        and action_format_errors == 0
        and score.relation_format_errors == 0
        and not incorrect_nodes
        and not missing_nodes
        and not incorrect_relations
        and not missing_relations
        and not incorrect_actions
        and not missing_actions
    )
    return {
        "model_goal_exact_match": exact,
        "node_goal_format_errors": node_format_errors,
        "predicted_node_goals": _node_goal_lists(sorted(predicted_nodes)),
        "matched_node_goals": _node_goal_lists(matched_nodes),
        "incorrect_node_goals": _node_goal_lists(incorrect_nodes),
        "missing_node_goals": _node_goal_lists(missing_nodes),
        "matched_relations_to_gold": _triple_lists(matched_relations),
        "incorrect_relations_to_gold": _triple_lists(incorrect_relations),
        "missing_relations_to_gold": _triple_lists(missing_relations),
        "action_goal_format_errors": action_format_errors,
        "predicted_action_goals": sorted(predicted_actions),
        "matched_action_goals": matched_actions,
        "incorrect_action_goals": incorrect_actions,
        "missing_action_goals": missing_actions,
    }


def _compare_actions(
    predicted: set[str], gold_groups: tuple[str, ...]
) -> tuple[list[str], list[str], list[str]]:
    remaining = set(predicted)
    matched: list[str] = []
    missing: list[str] = []
    for group in gold_groups:
        alternatives = {item.strip().upper() for item in group.split("|") if item.strip()}
        candidates = sorted(remaining & alternatives)
        if candidates:
            matched.append(candidates[0])
            remaining.remove(candidates[0])
        else:
            missing.append(group)
    return matched, sorted(remaining), missing


def _parse_action_descriptions(prompt: str) -> dict[str, str]:
    marker = "Below is a dictionary of possible actions"
    marker_index = prompt.find(marker)
    if marker_index < 0:
        return {}
    start = prompt.find("{", marker_index)
    if start < 0:
        return {}
    depth = 0
    end = -1
    for index in range(start, len(prompt)):
        if prompt[index] == "{":
            depth += 1
        elif prompt[index] == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end < 0:
        return {}
    try:
        parsed = ast.literal_eval(prompt[start:end])
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        str(action).strip().upper(): str(description).strip()
        for action, description in parsed.items()
        if str(action).strip()
    }


def write_preference_pairs(
    pairs: list[dict],
    summary: dict,
    output_dir: Path,
    filename: str = "scene_task_contradiction_preference_pairs.jsonl",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / filename
    summary_path = output_dir / "summary.json"
    with pairs_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return pairs_path, summary_path


def _refusal_reason(missing_objects: list[str]) -> str:
    if len(missing_objects) == 1:
        subject = f"the required object {missing_objects[0]} is"
    elif len(missing_objects) == 2:
        subject = f"the required objects {missing_objects[0]} and {missing_objects[1]} are"
    else:
        joined = ", ".join(missing_objects[:-1]) + f", and {missing_objects[-1]}"
        subject = f"the required objects {joined} are"
    return f"The task cannot be completed because {subject} absent from the current scene."


def _triple_lists(relations: list[RelationTriple]) -> list[list[str]]:
    return [[item.from_name, item.relation, item.to_name] for item in relations]


def _node_goal_lists(goals: tuple[NodeGoal, ...] | list[NodeGoal]) -> list[list[str]]:
    return [[item.name, item.state] for item in goals]


def _score_metrics(score: SampleScore) -> dict:
    predicted_count = len(score.predicted_relations)
    return {
        "format_valid": score.format_valid,
        "refused_or_empty": score.refused_or_empty,
        "chair_object_pct": _pct(
            len(score.hallucinated_objects), len(score.mentioned_objects)
        ),
        "chair_state_pct": _pct(score.hallucinated_states, score.mentioned_states),
        "pope_object_pct": _pct(
            len(score.mentioned_probe_objects), len(score.probe_objects)
        ),
        "mentioned_objects": score.mentioned_objects,
        "hallucinated_objects": score.hallucinated_objects,
        "mentioned_states": score.mentioned_states,
        "hallucinated_states": score.hallucinated_states,
        "probe_objects": score.probe_objects,
        "mentioned_probe_objects": score.mentioned_probe_objects,
        "predicted_relations": _triple_lists(score.predicted_relations),
        "relation_endpoint_hallucination_pct": _pct(
            len(score.hallucinated_relations), predicted_count
        ),
        "relation_format_errors": score.relation_format_errors,
        "hallucinated_relations": _triple_lists(score.hallucinated_relations),
        "invalid_relation_types": _triple_lists(score.invalid_relation_types),
        "invalid_relation_targets": _triple_lists(score.invalid_relation_targets),
        "gold_relations": _triple_lists(score.gold_relations),
        "blocked_gold_relations": _triple_lists(score.blocked_gold_relations),
        "relation_goal_evaluable": score.relation_goal_evaluable,
    }


def _pct(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 4) if denominator else None


def _selected_metrics(scores: list[SampleScore]) -> dict:
    object_mentions = sum(len(item.mentioned_objects) for item in scores)
    hallucinated_objects = sum(len(item.hallucinated_objects) for item in scores)
    state_mentions = sum(item.mentioned_states for item in scores)
    hallucinated_states = sum(item.hallucinated_states for item in scores)
    probes = sum(len(item.probe_objects) for item in scores)
    mentioned_probes = sum(len(item.mentioned_probe_objects) for item in scores)
    relation_mentions = sum(len(item.predicted_relations) for item in scores)
    hallucinated_relations = sum(len(item.hallucinated_relations) for item in scores)
    return {
        "samples": len(scores),
        "chair_object_pct": _pct(hallucinated_objects, object_mentions),
        "chair_state_pct": _pct(hallucinated_states, state_mentions),
        "pope_object_pct": _pct(mentioned_probes, probes),
        "object_mentions": object_mentions,
        "hallucinated_object_mentions": hallucinated_objects,
        "relation_mentions": relation_mentions,
        "hallucinated_relation_mentions": hallucinated_relations,
        "relation_endpoint_hallucination_pct": _pct(
            hallucinated_relations, relation_mentions
        ),
        "relation_format_errors": sum(item.relation_format_errors for item in scores),
        "invalid_relation_type_mentions": sum(
            len(item.invalid_relation_types) for item in scores
        ),
        "invalid_relation_target_mentions": sum(
            len(item.invalid_relation_targets) for item in scores
        ),
        "blocked_gold_relations": sum(
            len(item.blocked_gold_relations) for item in scores
        ),
    }
