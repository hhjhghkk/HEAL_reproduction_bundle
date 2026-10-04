from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import (
    has_refusal_language,
    is_empty_plan,
    normalize_object,
    normalize_relation,
    parse_relation_constraints,
    parse_response,
    parse_scene,
    prompt_from_row,
    read_rows,
    RelationTriple,
    response_entities,
    response_relations,
    split_probe_objects,
    unique_in_order,
)
from .synonyms import (
    load_virtualhome_synonym_projections,
    project_relation_constraints,
    project_relation_triples,
)


INFEASIBLE_VARIANTS = {"object_removal", "scene_task_contradiction"}
RELATION_GOAL_EVALUABLE_VARIANTS = {
    "baseline",
    "distractor_injection",
    "scene_object_synonymous",
}
VIRTUALHOME_RELATION_GROUND_TRUTH = (
    Path("src")
    / "virtualhome_eval"
    / "resources"
    / "virtualhome"
    / "task_state_LTL_formula_accurate.json"
)


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
    predicted_relations: list[RelationTriple]
    relation_format_errors: int
    invalid_relation_types: list[RelationTriple]
    invalid_relation_targets: list[RelationTriple]
    hallucinated_relations: list[RelationTriple]
    gold_relations: list[RelationTriple]
    blocked_gold_relations: list[RelationTriple]
    relation_goal_evaluable: bool
    matched_relations: list[RelationTriple]
    incorrect_relations: list[RelationTriple]
    missing_relations: list[RelationTriple]
    # Candidate identity is carried through evaluation so several responses for
    # the same HEAL row can be compared without losing their provenance.
    generation_id: int = 0
    seed: int | None = None
    model: str = ""

#读取前面run_heal_inference.py生成的predictions.jsonl文件，并返回一个包含所有预测记录的列表
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

#逐条预测评分，核心部分
def score_predictions(
    dataset_root: Path,
    predictions: list[dict],
    eai_root: Path | None = None,
) -> list[SampleScore]:
    row_cache: dict[tuple[str, str], list[dict[str, str]]] = {}
    baseline_scenes: dict[tuple[str, str], set[str]] = {}
    relation_ground_truth = (
        load_virtualhome_relation_ground_truth(eai_root) if eai_root is not None else {}
    )
    synonym_projections = (
        load_virtualhome_synonym_projections(dataset_root)
        if any(
            record.get("environment") == "virtualhome"
            and record.get("variant") == "scene_object_synonymous"
            for record in predictions
        )
        else {}
    )
    scores: list[SampleScore] = []

    for record in predictions:
        environment = record["environment"]
        variant = record["variant"]
        row_index = int(record["row_index"])

        #找回原始HEAL数据
        cache_key = (environment, variant)
        if cache_key not in row_cache:
            row_cache[cache_key] = read_rows(dataset_root, environment, variant)
        rows = row_cache[cache_key]
        if not 0 <= row_index < len(rows):
            raise IndexError(f"row_index {row_index} outside {environment}/{variant}")
        row = rows[row_index]
        raw_response = str(record.get("response", ""))
        #parse_response为data.py中函数，解析模型回答
        payload = parse_response(raw_response)
        #prompt_from_row()先从csv取当前variant的完整prompt，parse_scene()再从prompt中解析出场景中的物体
        prompt = prompt_from_row(row, variant)
        scene = parse_scene(prompt)
        scene_names = set(scene)
        relation_constraints = parse_relation_constraints(prompt)
        if variant == "scene_object_synonymous":
            relation_constraints = project_relation_constraints(
                relation_constraints, synonym_projections[row_index]
            )

        #从模型回答中提取object和state
        objects, object_states = response_entities(payload, environment) if payload else ([], [])
        predicted_relations, relation_format_errors = (
            response_relations(payload, environment) if payload else ([], 0)
        )

        #objects去重
        objects = unique_in_order(objects)

        #检测hallucinated object，模型提到了，但当前scene不存在
        hallucinated_objects = [name for name in objects if name not in scene_names]

        #检测hallucinated object state，对象本身必须存在，但模型说的状态不在这个对象允许状态中
        #这里hallucinated_objects先滤掉不存在的对象，避免重复计算
        hallucinated_states = sum(
            1
            for name, state in object_states
            if name in scene and state not in scene[name].states
        )

        # 关系任一端点不在修改后的场景中，即为确定性的关系幻觉。
        hallucinated_relations = [
            relation
            for relation in predicted_relations
            if relation.from_name not in scene_names or relation.to_name not in scene_names
        ]
        invalid_relation_types = [
            relation
            for relation in predicted_relations
            if relation_constraints and relation.relation not in relation_constraints
        ]
        invalid_relation_targets = [
            relation
            for relation in predicted_relations
            if relation.relation in relation_constraints
            and relation.to_name not in relation_constraints[relation.relation]
        ]

        task_id = row.get("task_id", "")
        gold_available = environment == "virtualhome" and task_id in relation_ground_truth
        gold_relations = relation_ground_truth.get(task_id, []) if gold_available else []
        if variant == "scene_object_synonymous" and gold_relations:
            gold_relations = project_relation_triples(
                gold_relations, synonym_projections[row_index]
            )

        # 对不可执行变体，原始 gold goal 只用于说明哪些必要关系已被场景破坏，
        # 不能把复述原始 gold goal 当作正确答案。
        blocked_gold_relations = (
            [
                relation
                for relation in gold_relations
                if relation.from_name not in scene_names or relation.to_name not in scene_names
            ]
            if variant in INFEASIBLE_VARIANTS
            else []
        )

        # 只有场景仍可执行时才计算 gold relation P/R/F1。synonym变体的
        # EAI关系端点已在上方正向投影到该行prompt的同义词空间。
        relation_goal_evaluable = (
            payload is not None
            and gold_available
            and variant in RELATION_GOAL_EVALUABLE_VARIANTS
        )
        if relation_goal_evaluable:
            predicted_set = set(predicted_relations)
            gold_set = set(gold_relations)
            matched_relations = sorted(predicted_set & gold_set)
            incorrect_relations = sorted(predicted_set - gold_set)
            missing_relations = sorted(gold_set - predicted_set)
        else:
            matched_relations = []
            incorrect_relations = []
            missing_relations = []

        probe_objects = _probe_objects(
            dataset_root,
            environment,
            variant,
            row,
            baseline_scenes,
        )
        #判断模型回答中是否提到了probe_objects中的物体
        mentioned_probe_objects = sorted(set(objects) & probe_objects)
        #判断拒绝或空计划，任满足其一就算，这里并不判断“拒绝是不是正确”
        refused_or_empty = is_empty_plan(payload) or has_refusal_language(raw_response)
        scores.append(
            SampleScore(
                environment=environment,
                variant=variant,
                row_index=row_index,
                task_id=task_id,
                format_valid=payload is not None,   #模型回答能否被解析
                refused_or_empty=refused_or_empty,
                mentioned_objects=objects,
                hallucinated_objects=hallucinated_objects,
                mentioned_states=len(object_states),   #这里保存的是数量
                hallucinated_states=hallucinated_states,
                probe_objects=sorted(probe_objects),
                mentioned_probe_objects=mentioned_probe_objects,
                predicted_relations=predicted_relations,
                relation_format_errors=relation_format_errors,
                invalid_relation_types=invalid_relation_types,
                invalid_relation_targets=invalid_relation_targets,
                hallucinated_relations=hallucinated_relations,
                gold_relations=gold_relations,
                blocked_gold_relations=blocked_gold_relations,
                relation_goal_evaluable=relation_goal_evaluable,
                matched_relations=matched_relations,
                incorrect_relations=incorrect_relations,
                missing_relations=missing_relations,
                generation_id=int(record.get("generation_id", 0)),
                seed=(int(record["seed"]) if record.get("seed") is not None else None),
                model=str(record.get("model", "")),
            )
        )
    return scores


def load_virtualhome_relation_ground_truth(eai_root: Path) -> dict[str, list[RelationTriple]]:
    """Index VirtualHome gold edge goals by HEAL ``task_id``.

    The upstream JSON stores relation endpoints as numeric IDs.  The matching
    ``tl_goal`` contains ``object_name.id`` tokens, which provide a lossless ID
    to name mapping for all relation goals in the public artifact.
    """

    path = eai_root / VIRTUALHOME_RELATION_GROUND_TRUTH
    if not path.exists():
        raise FileNotFoundError(f"VirtualHome relation ground truth not found: {path}")
    root = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[RelationTriple]] = {}

    def visit(node: object) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if isinstance(value, dict) and isinstance(value.get("vh_goal"), dict):
                relations = _relations_from_virtualhome_goal(key, value)
                if key in result and result[key] != relations:
                    raise ValueError(f"Conflicting relation ground truth for task_id={key!r}")
                result[key] = relations
            else:
                visit(value)

    visit(root)
    return result


def _relations_from_virtualhome_goal(
    task_id: str,
    record: dict,
) -> list[RelationTriple]:
    tl_goal = str(record.get("tl_goal") or "")
    id_to_name = {
        int(match.group(2)): normalize_object(match.group(1))
        for match in re.finditer(r"([A-Za-z][A-Za-z0-9_]*)\.([0-9]+)", tl_goal)
    }
    relations: list[RelationTriple] = []
    for goal in record.get("vh_goal", {}).get("goal", []):
        if not isinstance(goal, dict) or not {
            "from_id",
            "to_id",
            "relation_type",
        }.issubset(goal):
            continue
        from_id = int(goal["from_id"])
        to_id = int(goal["to_id"])
        if from_id not in id_to_name or to_id not in id_to_name:
            raise ValueError(
                f"Cannot resolve relation endpoint names for task_id={task_id!r}: {goal}"
            )
        relations.append(
            RelationTriple(
                id_to_name[from_id],
                normalize_relation(goal["relation_type"]),
                id_to_name[to_id],
            )
        )
    return sorted(set(relations))

#根据variant返回当前样本重点检查哪些对象
def _probe_objects(
    dataset_root: Path,
    environment: str,
    variant: str,
    row: dict[str, str],
    baseline_scenes: dict[tuple[str, str], set[str]],
) -> set[str]:
    if variant == "distractor_injection":
        #这条数据注入了哪些干扰对象
        return split_probe_objects(row.get("distractors", row.get("distractor", "")))
    if variant == "object_removal":
        #这条数据删除了哪些对象
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

#根据当前task_id找到同一个任务对应的baseline scene
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
        predicted_relations = sum(len(item.predicted_relations) for item in group)
        hallucinated_relations = sum(len(item.hallucinated_relations) for item in group)
        relation_format_errors = sum(item.relation_format_errors for item in group)
        invalid_relation_types = sum(len(item.invalid_relation_types) for item in group)
        invalid_relation_targets = sum(len(item.invalid_relation_targets) for item in group)
        blocked_gold_relations = sum(len(item.blocked_gold_relations) for item in group)
        relation_groups = [item for item in group if item.relation_goal_evaluable]
        relation_tp = sum(len(item.matched_relations) for item in relation_groups)
        relation_fp = sum(len(item.incorrect_relations) for item in relation_groups)
        relation_fn = sum(len(item.missing_relations) for item in relation_groups)
        relation_precision = _ratio(relation_tp, relation_tp + relation_fp)
        relation_recall = _ratio(relation_tp, relation_tp + relation_fn)
        summaries.append(
            {
                "environment": environment,
                "variant": variant,
                "samples": len(group),
                "format_valid_pct": _pct(sum(item.format_valid for item in group), len(group)),
                "refusal_or_empty_pct": _pct(sum(item.refused_or_empty for item in group), len(group)),
                "chair_object_pct": _pct(hallucinated_objects, mentioned_objects),    #object-level CHAIR指标
                "chair_state_pct": _pct(hallucinated_states, mentioned_states),
                "pope_object_pct": _pct(mentioned_probes, probes),
                "object_mentions": mentioned_objects,
                "hallucinated_object_mentions": hallucinated_objects,
                "probe_objects": probes,
                "mentioned_probe_objects": mentioned_probes,
                "relation_mentions": predicted_relations,
                "hallucinated_relation_mentions": hallucinated_relations,
                "relation_endpoint_hallucination_pct": _pct(
                    hallucinated_relations, predicted_relations
                ),
                "relation_format_errors": relation_format_errors,
                "invalid_relation_type_mentions": invalid_relation_types,
                "invalid_relation_target_mentions": invalid_relation_targets,
                "blocked_gold_relations": blocked_gold_relations,
                "relation_goal_evaluable_samples": len(relation_groups),
                "relation_goal_tp": relation_tp,
                "relation_goal_fp": relation_fp,
                "relation_goal_fn": relation_fn,
                "relation_goal_precision_pct": _as_pct(relation_precision),
                "relation_goal_recall_pct": _as_pct(relation_recall),
                "relation_goal_f1_pct": _as_pct(
                    _f1(relation_precision, relation_recall)
                ),
            }
        )
    return {
        "metric_note": (
            "Independent paper-aligned implementation. CHAIR is the percentage of unique "
            "object/state mentions unsupported by each prompt; POPE is the percentage of "
            "controlled non-existent probe objects mentioned. Relation type/target validation uses "
            "the HEAL prompt ontology; relation endpoint hallucination uses the modified HEAL scene; "
            "relation goal P/R/F1 uses EAI ground truth for executable variants; synonym-"
            "substitution gold endpoints are projected into each modified prompt's vocabulary "
            "before comparison. This is not an author-released evaluator."
        ),
        "groups": summaries,
        "format_failures": Counter(
            f"{item.environment}/{item.variant}" for item in scores if not item.format_valid
        ),
    }

#通用百分比函数
def _pct(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 4) if denominator else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _as_pct(value: float | None) -> float | None:
    return round(100.0 * value, 4) if value is not None else None

#把评估结果写入磁盘
def write_scores(scores: list[SampleScore], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "sample_scores.jsonl"  #逐样本结果
    summary_path = output_dir / "summary.json"      #汇总结果
    with details_path.open("w", encoding="utf-8") as handle:
        for item in scores:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(summarize(scores), ensure_ascii=False, indent=2, default=dict) + "\n",
        encoding="utf-8",
    )
    return summary_path, details_path
