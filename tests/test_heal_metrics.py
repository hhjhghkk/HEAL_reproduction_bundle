from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from heal_repro.data import (
    RelationTriple,
    has_refusal_language,
    is_empty_plan,
    parse_relation_constraints,
    parse_response,
    parse_scene,
    parse_scene_object_sequence,
    response_entities,
    response_relations,
)
from heal_repro.metrics import (
    SampleScore,
    load_virtualhome_relation_ground_truth,
    score_predictions,
    summarize,
)
from heal_repro.preferences import (
    NodeGoal,
    VirtualHomeGoldGoals,
    _compare_actions,
    _gold_payload,
    _refusal_reason,
    build_compact_pair,
    build_full_conflict_preference_pairs,
    is_strict_infeasible_candidate,
    load_virtualhome_gold_goals,
    project_virtualhome_gold_goals,
)
from heal_repro.refinement import reason_match_kind, validate_blocked_refinement
from heal_repro.synonyms import (
    project_relation_constraints,
    project_relation_triples,
)
from scripts.run_heal_inference import (
    candidate_seed,
    existing_keys,
    select_rows,
    validate_args,
)
from scripts.run_heal_refinement import (
    build_executable_refinement_prompt,
    build_refinement_prompt,
    missing_required_objects,
)


class HealMetricTests(unittest.TestCase):
    def test_synonym_projection_preserves_colliding_scene_lines(self) -> None:
        prompt = """Relevant objects in the scene are:
dish, initial states: ['CLEAN'], possible states: ['CLEAN']
dish, initial states: ['DIRTY'], possible states: ['DIRTY']

All possible relationships are the keys of a dictionary
"""
        self.assertEqual(parse_scene_object_sequence(prompt), ["dish", "dish"])
        self.assertEqual(list(parse_scene(prompt)), ["dish"])

        gold = VirtualHomeGoldGoals(
            node_goals=(NodeGoal("plate", "CLEAN"),),
            relations=(RelationTriple("bowl", "ON", "plate"),),
            action_goals=("GRAB",),
        )
        projected = project_virtualhome_gold_goals(
            gold, {"bowl": "dish", "plate": "dish"}
        )
        self.assertEqual(projected.node_goals, (NodeGoal("dish", "CLEAN"),))
        self.assertEqual(
            projected.relations, (RelationTriple("dish", "ON", "dish"),)
        )
        self.assertEqual(projected.action_goals, ("GRAB",))
        self.assertEqual(
            project_relation_triples(
                [
                    RelationTriple("bowl", "ON", "dishwasher"),
                    RelationTriple("plate", "ON", "dishwasher"),
                ],
                {"bowl": "dish", "plate": "dish", "dishwasher": "washer"},
            ),
            [RelationTriple("dish", "ON", "washer")],
        )
        self.assertEqual(
            project_relation_constraints(
                {"ON": frozenset({"dishwasher", "washing_machine"})},
                {"dishwasher": "washer", "washing_machine": "washer"},
            ),
            {"ON": frozenset({"washer"})},
        )

    def test_action_conflict_requirement_is_not_applied_to_distractor_injection(self) -> None:
        gold = VirtualHomeGoldGoals(
            node_goals=(),
            relations=(),
            action_goals=("GRAB", "WASH|RINSE|SCRUB|WIPE"),
        )
        row = {"task_name": "Wash dishes by hand"}
        scene_names = {"character", "dish_soap", "plate"}

        self.assertEqual(
            missing_required_objects(
                row, scene_names, gold, "scene_task_contradiction"
            ),
            ["faucet", "sink", "water"],
        )
        self.assertEqual(
            missing_required_objects(row, scene_names, gold, "distractor_injection"),
            [],
        )

    def test_refusal_reason_exact_equivalent_and_unrelated(self) -> None:
        self.assertEqual(reason_match_kind("couch", ["couch"]), "exact")
        self.assertEqual(reason_match_kind("The sofa is absent.", ["couch"]), "equivalent")
        self.assertEqual(reason_match_kind("book", ["novel"]), "equivalent")
        self.assertEqual(reason_match_kind("lights", ["light"]), "equivalent")
        self.assertEqual(reason_match_kind("coffee_maker", ["couch"]), "none")

        result = validate_blocked_refinement(
            {
                "missing_required_objects": ["stereo"],
                "response": json.dumps(
                    {
                        "node goals": [],
                        "edge goals": [],
                        "action goals": [],
                        "reason": "The radio is absent.",
                    }
                ),
            }
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["reason_match"], "equivalent")

    def test_executable_refinement_prompt_contains_exact_verified_goal(self) -> None:
        prompt = build_executable_refinement_prompt(
            "ORIGINAL PROMPT",
            "PREVIOUS RESPONSE",
            {
                "node goals": [{"name": "computer", "state": "ON"}],
                "edge goals": [],
                "action goals": [],
            },
            ["toy", "vase"],
        )
        self.assertIn("current scene still supports the task", prompt)
        self.assertIn('"name": "computer"', prompt)
        self.assertIn('"state": "ON"', prompt)
        self.assertIn("match the verified", prompt)
        self.assertIn("irrelevant distractors", prompt)
        self.assertIn("toy, vase", prompt)
        self.assertIn("strict serialization task", prompt)
        self.assertIn("character for character", prompt)
        self.assertIn(
            '{"node goals":[{"name":"computer","state":"ON"}],"edge goals":[],"action goals":[]}',
            prompt,
        )

    def test_refinement_prompt_contains_verifier_evidence_and_output_contract(self) -> None:
        prompt = build_refinement_prompt(
            "ORIGINAL PROMPT",
            '{"node goals": [{"name": "washer", "state": "ON"}]}',
            ["washer"],
            ["washer"],
            [RelationTriple("soap", "ON", "washer")],
        )
        self.assertIn("required objects are absent: washer", prompt)
        self.assertIn("soap --ON--> washer", prompt)
        self.assertIn('empty "node goals", "edge goals", and "action goals"', prompt)
        self.assertIn("exact object identifier", prompt)
        self.assertIn("verified missing list: washer", prompt)
        self.assertIn("Output JSON only", prompt)

    def test_candidate_resume_keys_and_stable_seeds(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "predictions.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "environment": "virtualhome",
                                "variant": "scene_task_contradiction",
                                "row_index": 3,
                            }
                        ),
                        json.dumps(
                            {
                                "environment": "virtualhome",
                                "variant": "scene_task_contradiction",
                                "row_index": 3,
                                "generation_id": 2,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                existing_keys(path),
                {
                    ("virtualhome", "scene_task_contradiction", 3, 0),
                    ("virtualhome", "scene_task_contradiction", 3, 2),
                },
            )

        seed_a = candidate_seed(42, "virtualhome", "scene_task_contradiction", 3, 0)
        seed_b = candidate_seed(42, "virtualhome", "scene_task_contradiction", 3, 1)
        self.assertEqual(
            seed_a,
            candidate_seed(42, "virtualhome", "scene_task_contradiction", 3, 0),
        )
        self.assertNotEqual(seed_a, seed_b)

    def test_candidate_row_selection_and_sampling_validation(self) -> None:
        rows = [
            {"task_id": "a"},
            {"task_id": "b"},
            {"task_id": "c"},
        ]
        self.assertEqual(
            select_rows(rows, [2, 0], None, None, "demo"),
            [(0, rows[0]), (2, rows[2])],
        )
        self.assertEqual(
            select_rows(rows, None, ["b"], None, "demo"),
            [(1, rows[1])],
        )
        invalid = Namespace(
            num_generations=2,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_samples=None,
            row_indices=None,
        )
        with self.assertRaises(SystemExit):
            validate_args(invalid)

    def test_executable_gold_payload_and_action_alternatives(self) -> None:
        prompt = """Below is a dictionary of possible actions:
{'RINSE': 'rinse sth', 'WASH': 'wash sth'}
Goal name and goal description:
"""
        gold = VirtualHomeGoldGoals(
            node_goals=(NodeGoal("computer", "ON"),),
            relations=(),
            action_goals=("RINSE|WASH",),
        )
        payload = _gold_payload(gold, prompt)
        self.assertEqual(payload["node goals"], [{"name": "computer", "state": "ON"}])
        self.assertEqual(
            payload["action goals"],
            [{"action": "RINSE", "description": "rinse sth"}],
        )
        self.assertEqual(
            _compare_actions({"WASH"}, gold.action_goals),
            (["WASH"], [], []),
        )

    def test_virtualhome_scene_and_hallucinated_entity_input(self) -> None:
        prompt = """Relevant objects in the scene are:
washer, initial states: ['OFF'], possible states: ['OFF', 'ON']
soap, initial states: ['CLEAN'], possible states: ['CLEAN', 'WET']

All possible relationships are the keys of a dictionary
"""
        scene = parse_scene(prompt)
        self.assertEqual(set(scene), {"washer", "soap"})
        payload = parse_response(
            '```json\n{"node goals": [{"name": "refrigerator", "state": "ON"}], '
            '"edge goals": [{"from_name": "soap", "relation": "INSIDE", "to_name": "washer"}]}\n```'
        )
        self.assertIsNotNone(payload)
        objects, states = response_entities(payload or {}, "virtualhome")
        self.assertEqual(objects, ["refrigerator", "soap", "washer"])
        self.assertEqual(states, [("refrigerator", "ON")])

    def test_behavior_nested_not(self) -> None:
        payload = {
            "node goals": [["not", ["Dusty", "table.n.02_1"]]],
            "edge goals": [["inside", "apple.n.01_1", "basket.n.01_1"]],
        }
        objects, states = response_entities(payload, "behavior")
        self.assertEqual(objects, ["table.n.02_1", "apple.n.01_1", "basket.n.01_1"])
        self.assertEqual(states, [("table.n.02_1", "DUSTY")])

    def test_refusal_and_empty(self) -> None:
        self.assertTrue(has_refusal_language("The task is infeasible because a required object is absent."))
        self.assertTrue(is_empty_plan({"node goals": [], "edge goals": []}))
        self.assertFalse(is_empty_plan(None))

    def test_truncated_outer_json_does_not_parse_nested_goal_item(self) -> None:
        truncated = '{"node goals": [{"name": "soap", "state": "CLEAN"}], "edge goals": ['
        self.assertIsNone(parse_response(truncated))

    def test_relation_extraction_normalizes_and_reports_malformed_entries(self) -> None:
        payload = {
            "edge goals": [
                {"from_name": "Soap", "relation": "ONTOP", "to_name": "Washer"},
                {"from_name": "Soap", "relation": "ON", "to_name": "Washer"},
                {"from_name": "soap", "to_name": "washer"},
                {},
            ]
        }
        relations, malformed = response_relations(payload, "virtualhome")
        self.assertEqual(relations, [RelationTriple("soap", "ON", "washer")])
        self.assertEqual(malformed, 1)

        prompt = """Each relation has a fixed set of objects to be its 'to_name' target.
Here is a dictionary where keys are 'relation' and values are target sets:
{'ON': {'table', 'washer'}, 'INSIDE': {'fridge'}}
"""
        self.assertEqual(
            parse_relation_constraints(prompt),
            {
                "ON": frozenset({"table", "washer"}),
                "INSIDE": frozenset({"fridge"}),
            },
        )

    def test_relation_ground_truth_and_variant_aware_scoring(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "HEAL_dataset"
            virtualhome = dataset_root / "virtualhome"
            virtualhome.mkdir(parents=True)
            baseline_prompt = """Relevant objects in the scene are:
washer, initial states: ['OFF'], possible states: ['OFF', 'ON']
soap, initial states: ['CLEAN'], possible states: ['CLEAN', 'WET']

All possible relationships are the keys of a dictionary
Each relation has a fixed set of objects to be its 'to_name' target.
Here is a dictionary where keys are 'relation' and values are target sets:
{'ON': {'washer'}, 'INSIDE': {'washer'}}
"""
            contradiction_prompt = """Relevant objects in the scene are:
soap, initial states: ['CLEAN'], possible states: ['CLEAN', 'WET']

All possible relationships are the keys of a dictionary
"""

            with (virtualhome / "baseline.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["task_id", "task_name", "baseline_prompts"],
                )
                writer.writeheader()
                writer.writerow(
                    {"task_id": "t1", "task_name": "demo", "baseline_prompts": baseline_prompt}
                )
            with (virtualhome / "scene_task_contradiction.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["task_id", "task_name", "modified_prompts"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "t1",
                        "task_name": "demo",
                        "modified_prompts": contradiction_prompt,
                    }
                )

            eai_root = root / "eai"
            ground_truth_path = (
                eai_root
                / "src"
                / "virtualhome_eval"
                / "resources"
                / "virtualhome"
                / "task_state_LTL_formula_accurate.json"
            )
            ground_truth_path.parent.mkdir(parents=True)
            ground_truth_path.write_text(
                json.dumps(
                    {
                        "scene_1": {
                            "demo": {
                                "t1": {
                                    "vh_goal": {
                                        "actions": ["TYPE"],
                                        "goal": [
                                            {
                                                "id": 1,
                                                "class_name": "washer",
                                                "state": "ON",
                                            },
                                            {
                                                "from_id": 2,
                                                "relation_type": "ON",
                                                "to_id": 1,
                                            }
                                        ],
                                    },
                                    "tl_goal": "ONTOP(soap.2, washer.1)",
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            relation_index = load_virtualhome_relation_ground_truth(eai_root)
            self.assertEqual(
                relation_index["t1"],
                [RelationTriple("soap", "ON", "washer")],
            )
            all_goals = load_virtualhome_gold_goals(eai_root)
            self.assertEqual(
                [(goal.name, goal.state) for goal in all_goals["t1"].node_goals],
                [("washer", "ON")],
            )
            self.assertEqual(all_goals["t1"].action_goals, ("TYPE",))

            baseline_response = json.dumps(
                {
                    "node goals": [],
                    "edge goals": [
                        {"from_name": "soap", "relation": "ON", "to_name": "washer"},
                        {"from_name": "washer", "relation": "INSIDE", "to_name": "soap"},
                        {"from_name": "ghost", "relation": "ON", "to_name": "washer"},
                    ],
                    "action goals": [],
                }
            )
            contradiction_response = json.dumps(
                {
                    "node goals": [],
                    "edge goals": [
                        {"from_name": "soap", "relation": "ON", "to_name": "washer"}
                    ],
                    "action goals": [],
                }
            )
            scores = score_predictions(
                dataset_root,
                [
                    {
                        "environment": "virtualhome",
                        "variant": "baseline",
                        "row_index": 0,
                        "response": baseline_response,
                    },
                    {
                        "environment": "virtualhome",
                        "variant": "scene_task_contradiction",
                        "row_index": 0,
                        "generation_id": 3,
                        "seed": 12345,
                        "model": "candidate-model",
                        "response": contradiction_response,
                    },
                ],
                eai_root=eai_root,
            )

            baseline_score, contradiction_score = scores
            self.assertTrue(baseline_score.relation_goal_evaluable)
            self.assertEqual(
                baseline_score.matched_relations,
                [RelationTriple("soap", "ON", "washer")],
            )
            self.assertEqual(len(baseline_score.incorrect_relations), 2)
            self.assertEqual(len(baseline_score.hallucinated_relations), 1)
            self.assertEqual(len(baseline_score.invalid_relation_targets), 1)
            self.assertEqual(baseline_score.invalid_relation_types, [])
            self.assertEqual(baseline_score.missing_relations, [])

            self.assertFalse(contradiction_score.relation_goal_evaluable)
            self.assertEqual(contradiction_score.generation_id, 3)
            self.assertEqual(contradiction_score.seed, 12345)
            self.assertEqual(contradiction_score.model, "candidate-model")
            self.assertEqual(
                contradiction_score.blocked_gold_relations,
                [RelationTriple("soap", "ON", "washer")],
            )
            self.assertEqual(len(contradiction_score.hallucinated_relations), 1)

            baseline_summary = next(
                group
                for group in summarize(scores)["groups"]
                if group["variant"] == "baseline"
            )
            self.assertEqual(baseline_summary["relation_goal_tp"], 1)
            self.assertEqual(baseline_summary["relation_goal_fp"], 2)
            self.assertEqual(baseline_summary["relation_goal_fn"], 0)
            self.assertEqual(baseline_summary["relation_goal_precision_pct"], 33.3333)
            self.assertEqual(baseline_summary["relation_goal_recall_pct"], 100.0)
            self.assertEqual(baseline_summary["relation_goal_f1_pct"], 50.0)

            pairs, pair_summary = build_full_conflict_preference_pairs(
                dataset_root,
                [
                    {
                        "environment": "virtualhome",
                        "variant": "scene_task_contradiction",
                        "row_index": 0,
                        "model": "demo-model",
                        "response": contradiction_response,
                    }
                ],
                eai_root,
            )
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pair_summary["selected_pairs"], 1)
            self.assertEqual(pair_summary["unresolved_samples"], [])
            self.assertIsInstance(pairs[0]["rejected"], str)
            self.assertEqual(
                pairs[0]["model_metrics"]["blocked_gold_node_goals"],
                [["washer", "ON"]],
            )
            self.assertEqual(pairs[0]["label_basis"], "gold_goal_blocked_refusal")

    def test_compact_infeasible_preference_pair_keeps_response_and_metrics(self) -> None:
        relation = RelationTriple("soap", "ON", "washer")
        score = SampleScore(
            environment="virtualhome",
            variant="scene_task_contradiction",
            row_index=0,
            task_id="t1",
            format_valid=True,
            refused_or_empty=False,
            mentioned_objects=["soap", "washer"],
            hallucinated_objects=["washer"],
            mentioned_states=0,
            hallucinated_states=0,
            probe_objects=["washer"],
            mentioned_probe_objects=["washer"],
            predicted_relations=[relation],
            relation_format_errors=0,
            invalid_relation_types=[],
            invalid_relation_targets=[],
            hallucinated_relations=[relation],
            gold_relations=[relation],
            blocked_gold_relations=[relation],
            relation_goal_evaluable=False,
            matched_relations=[],
            incorrect_relations=[],
            missing_relations=[],
        )
        response = json.dumps(
            {
                "node goals": [],
                "edge goals": [
                    {"from_name": "soap", "relation": "ON", "to_name": "washer"}
                ],
                "action goals": [],
            }
        )
        record = {"model": "demo-model", "response": response}

        self.assertTrue(is_strict_infeasible_candidate(score))
        pair = build_compact_pair(record, score, {"soap"})
        self.assertEqual(pair["rejected"]["edge goals"][0]["to_name"], "washer")
        self.assertEqual(pair["chosen"]["edge goals"], [])
        self.assertIn("washer", pair["chosen"]["reason"])
        self.assertEqual(pair["metrics"]["chair_object_pct"], 50.0)
        self.assertEqual(pair["metrics"]["relation_endpoint_hallucination_pct"], 100.0)
        self.assertEqual(
            pair["metrics"]["blocked_gold_relations"], [["soap", "ON", "washer"]]
        )
        self.assertIn(
            "soap and washer are absent",
            _refusal_reason(["soap", "washer"]),
        )
        self.assertNotIn("soap, and washer", _refusal_reason(["soap", "washer"]))


if __name__ == "__main__":
    unittest.main()
