"""Regression tests for the Priority-2 SNN diagnostics pipeline."""
from types import SimpleNamespace
import unittest

from agent_snn_v3 import SNNAgent
from collective_metrics import pheromone_metrics


class DecisionDiagnosticsTests(unittest.TestCase):
    def make_agent(self, last_decision, observer):
        agent = SNNAgent.__new__(SNNAgent)
        agent.current_node = 10
        agent.clock = 7
        agent.snn = SimpleNamespace(last_decision=last_decision)
        agent.decision_observer = observer
        return agent

    def test_notify_forwards_fresh_candidate_diagnostics(self):
        diagnostics = {
            "selected_node": 12,
            "pheromone_changed_score_argmax": True,
            "candidates": [{"node": 12, "pheromone_feature": 0.5,
                            "pheromone_score": 4.0,
                            "pheromone_probability_effect": 0.2}],
        }
        received = []
        agent = self.make_agent(diagnostics,
                                lambda *args: received.append(args))
        agent._notify_decision(12)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][:3], (10, 12, 7))
        self.assertEqual(received[0][3]["selected_node"], 12)
        self.assertIsNot(received[0][3], diagnostics)
        self.assertIsNot(received[0][3]["candidates"], diagnostics["candidates"])

    def test_fallback_without_fresh_decision_passes_none(self):
        received = []
        agent = self.make_agent(None, lambda *args: received.append(args))
        agent._notify_decision(13)
        self.assertEqual(received, [(10, 13, 7, None)])

    def test_metrics_use_scorer_choice_not_overridden_movement(self):
        sim = SimpleNamespace(
            _shared_memory_reads=[{"channel": "pheromones"}],
            _navigation_decisions=[{
                "policy_type": "default_snn",
                "next_node": 99,  # movement was overridden after scoring
                "decision_diagnostics": {
                    "selected_node": 12,
                    "pheromone_changed_score_argmax": True,
                    "candidates": [{
                        "node": 12, "pheromone_feature": 0.5,
                        "pheromone_score": 4.0,
                        "pheromone_probability_effect": 0.2,
                    }, {
                        "node": 13, "pheromone_feature": 0.0,
                        "pheromone_score": 0.0,
                        "pheromone_probability_effect": -0.2,
                    }],
                },
            }],
        )
        result = pheromone_metrics(sim)
        self.assertEqual(result["decisions_with_candidate_pheromone_diagnostics"], 1)
        self.assertEqual(result["decisions_with_comparable_no_pheromone_counterfactual"], 1)
        self.assertEqual(result["pheromone_changed_score_argmax"], 1)
        self.assertEqual(result["scored_choice_overridden_before_movement"], 1)
        self.assertEqual(result["scored_choice_matches_executed_next_hop"], 0)
        self.assertAlmostEqual(result["mean_selected_candidate_pheromone_score"], 4.0)


if __name__ == "__main__":
    unittest.main()
