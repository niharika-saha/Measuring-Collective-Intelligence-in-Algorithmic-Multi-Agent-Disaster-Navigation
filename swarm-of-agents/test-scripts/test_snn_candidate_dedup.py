import random
import unittest

from snn_core_v2 import SNNDecision


class CandidateDeduplicationTests(unittest.TestCase):
    def setUp(self):
        self.policy = SNNDecision(
            rng=random.Random(7),
            exploration_rate=0.0,
        )
        self.current = {"lat": 35.0, "lon": 139.0}
        self.goal = {"lat": 35.02, "lon": 139.0}

    def feature(self, node, progress, pheromone=0.0):
        return {
            "node": node,
            "dist": 1200.0 - progress,
            "hazard": 0.0,
            "visited": 0.0,
            "progress": progress,
            "is_prev_node": 0.0,
            "is_new": 1.0,
            "taboo": 0.0,
            "blacklist": 0.0,
            "two_hop_goal": 0.0,
            "pheromone": pheromone,
        }

    def test_duplicate_nodes_have_one_probability_entry(self):
        features = [
            self.feature(101, 100.0, 0.5),
            self.feature(101, 100.0, 0.5),
            self.feature(202, 80.0, 0.0),
            self.feature(202, 80.0, 0.0),
        ]
        self.policy.decide(features, self.current, self.goal)
        diagnostic = self.policy.last_decision

        self.assertEqual(diagnostic["raw_candidate_count"], 4)
        self.assertEqual(diagnostic["candidate_count"], 2)
        self.assertEqual(diagnostic["duplicate_candidate_count"], 2)
        self.assertEqual(
            [item["node"] for item in diagnostic["candidates"]],
            [101, 202],
        )
        self.assertAlmostEqual(
            sum(item["probability"] for item in diagnostic["candidates"]),
            1.0,
            places=12,
        )

    def test_score_gap_and_counterfactual_are_node_level(self):
        features = [
            self.feature(101, 100.0, 0.5),
            self.feature(101, 100.0, 0.5),
            self.feature(202, 80.0, 0.0),
        ]
        self.policy.decide(features, self.current, self.goal)
        diagnostic = self.policy.last_decision

        self.assertGreater(diagnostic["score_gap_best_second"], 0.0)
        self.assertEqual(len(diagnostic["candidates"]), 2)
        self.assertTrue(
            all("probability_without_pheromone" in item
                for item in diagnostic["candidates"])
        )
        self.assertTrue(
            diagnostic["candidates"][0]["pheromone_probability_effect"] is not None
        )


if __name__ == "__main__":
    unittest.main()
