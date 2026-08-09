from __future__ import annotations

import unittest

from heal_repro.data import (
    has_refusal_language,
    is_empty_plan,
    parse_response,
    parse_scene,
    response_entities,
)


class HealMetricTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
