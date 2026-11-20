
import unittest
from types import SimpleNamespace

from collective_metrics import (
    coverage_metrics,
    information_metrics,
    communication_efficiency_metrics,
    collective_metrics_from_sim,
    pheromone_metrics,
    aggregate_numeric_metrics,
    performance_metrics,
    hazard_exposure_metrics,
)
from swarm_simulation_dynamic import DynamicSwarmSimulation


class TestCollectiveMetrics(unittest.TestCase):

    def test_identical_coverage(self):
        agents = [
            SimpleNamespace(
                visited={1, 2, 3},
                cell_counts={(0, 0), (0, 1)}
            ),
            SimpleNamespace(
                visited={1, 2, 3},
                cell_counts={(0, 0), (0, 1)}
            ),
        ]
        result = coverage_metrics(agents)

        self.assertEqual(result["unique_nodes_covered"], 3)
        self.assertEqual(
            result["redundant_agent_node_memberships"], 3
        )
        self.assertEqual(
            result["mean_pairwise_node_jaccard"], 1.0
        )

    def test_disjoint_coverage(self):
        agents = [
            SimpleNamespace(visited={1, 2}, cell_counts={(0, 0)}),
            SimpleNamespace(visited={3, 4}, cell_counts={(1, 1)}),
        ]
        result = coverage_metrics(agents)

        self.assertEqual(result["unique_nodes_covered"], 4)
        self.assertEqual(
            result["redundant_agent_node_memberships"], 0
        )
        self.assertEqual(
            result["mean_pairwise_node_jaccard"], 0.0
        )

    def test_single_agent(self):
        agents = [
            SimpleNamespace(visited={1, 2}, cell_counts={(0, 0)})
        ]
        result = coverage_metrics(agents)

        self.assertEqual(result["agent_count"], 1)
        self.assertEqual(result["unique_nodes_covered"], 2)
        self.assertEqual(
            result["mean_pairwise_node_jaccard"], None
        )

    def test_empty_agents(self):
        result = coverage_metrics([])
        self.assertEqual(result["agent_count"], 0)
        self.assertEqual(result["unique_nodes_covered"], 0)

    def test_early_warning_censored_without_self_discovery(self):
        sim = SimpleNamespace(
            _hazard_event_records={
                "e1": {
                    "awareness": {
                        "0": {
                            "agent": 0,
                            "awareness_source": "teammate_message",
                            "awareness_step": 5,
                        }
                    }
                }
            },
            _hazard_discoveries=[],
            _shared_memory_reads=[],
            _navigation_decisions=[],
            _response={},
        )

        result = information_metrics(sim)
        early = result["early_warning"]

        self.assertEqual(
            early["censored_no_self_discovery"], 1
        )
        self.assertIsNone(
            early["mean_lead_steps_positive_means_teammate_was_earlier"]
        )

    def test_early_warning_lead(self):
        sim = SimpleNamespace(
            _hazard_event_records={
                "e1": {
                    "awareness": {
                        "0": {
                            "agent": 0,
                            "awareness_source": "teammate_message",
                            "awareness_step": 5,
                        }
                    }
                }
            },
            _hazard_discoveries=[
                {
                    "agent": 0,
                    "source": "self_observation",
                    "step": 9,
                    "event_ids": ["e1"],
                }
            ],
            _shared_memory_reads=[],
            _navigation_decisions=[],
            _response={},
        )

        result = information_metrics(sim)

        self.assertEqual(
            result["early_warning"][
                "mean_lead_steps_positive_means_teammate_was_earlier"
            ],
            4,
        )

    def test_zero_denominators(self):
        sim = SimpleNamespace(
            bus=SimpleNamespace(
                stats=lambda: {
                    "transmissions": 0,
                    "deliveries": 0,
                    "dropped": 0,
                    "bytes_tx": 0,
                }
            ),
            _response={},
        )

        result = communication_efficiency_metrics(sim)

        self.assertIsNone(result["deliveries_per_transmission"])
        self.assertIsNone(result["bytes_per_delivery"])
        self.assertIsNone(
            result["deliveries_per_attributed_planner_route_change"]
        )

    def test_empty_pairwise_coverage_is_undefined(self):
        agents = [
            SimpleNamespace(visited=set(), cell_counts={}),
            SimpleNamespace(visited=set(), cell_counts={}),
        ]
        result = coverage_metrics(agents)
        self.assertIsNone(result["mean_pairwise_node_jaccard"])
        self.assertIsNone(result["mean_pairwise_cell_jaccard"])

    def test_pheromone_counterfactual_diagnostics(self):
        sim = SimpleNamespace(
            _shared_memory_reads=[
                {"channel": "pheromones", "cross_agent": True}
            ],
            _navigation_decisions=[
                {
                    "policy_type": "default_snn",
                    "next_node": 2,
                    "decision_diagnostics": {
                        "pheromone_changed_score_argmax": True,
                        "candidates": [
                            {
                                "node": 2,
                                "pheromone_feature": 1.0,
                                "pheromone_score": 8.0,
                                "pheromone_probability_effect": 0.2,
                            },
                            {
                                "node": 3,
                                "pheromone_feature": 0.0,
                                "pheromone_score": 0.0,
                                "pheromone_probability_effect": -0.2,
                            },
                        ],
                    },
                }
            ],
        )
        result = pheromone_metrics(sim)
        self.assertEqual(result["pheromone_reads"], 1)
        self.assertEqual(result["decisions_with_candidate_pheromone_diagnostics"], 1)
        self.assertEqual(result["pheromone_changed_score_argmax"], 1)
        self.assertEqual(
            result["mean_selected_candidate_probability_effect"], 0.2
        )

    def test_numeric_metric_aggregation_tracks_missing_values(self):
        records = [
            {
                "performance": {
                    "travel_time": {"mean_completed_agent_goal_time_steps": 10},
                    "makespan": {"team_completion_makespan_steps": 12},
                    "all_completed": True,
                },
                "optional": None,
            },
            {
                "performance": {
                    "travel_time": {"mean_completed_agent_goal_time_steps": 14},
                    "makespan": {"team_completion_makespan_steps": None},
                    "all_completed": False,
                },
                "optional": None,
            },
            {
                "performance": {
                    "travel_time": {"mean_completed_agent_goal_time_steps": None},
                    "makespan": {"team_completion_makespan_steps": 18},
                },
            },
        ]

        result = aggregate_numeric_metrics(records)

        travel = result[
            "performance.travel_time.mean_completed_agent_goal_time_steps"
        ]
        self.assertEqual(travel["mean"], 12.0)
        self.assertEqual(travel["std"], 2.0)
        self.assertEqual(travel["n"], 2)
        self.assertEqual(travel["total_runs"], 3)
        self.assertEqual(travel["missing_n"], 1)

        makespan = result[
            "performance.makespan.team_completion_makespan_steps"
        ]
        self.assertEqual(makespan["mean"], 15.0)
        self.assertEqual(makespan["n"], 2)
        self.assertEqual(makespan["missing_n"], 1)

        # Boolean flags are intentionally not averaged as numeric metrics.
        self.assertNotIn("performance.all_completed", result)

    def test_numeric_metric_aggregation_handles_empty_records(self):
        self.assertEqual(aggregate_numeric_metrics([]), {})

    def test_empty_simulation(self):
        sim = SimpleNamespace(
            agents=[],
            _hazard_event_records={},
            _hazard_discoveries=[],
            _shared_memory_reads=[],
            _navigation_decisions=[],
            _response={},
            bus=SimpleNamespace(stats=lambda: {}),
        )

        result = collective_metrics_from_sim(sim)

        self.assertEqual(result["coverage"]["agent_count"], 0)
        self.assertEqual(
            result["information"]["registered_hazard_events"], 0
        )


    def test_distance_totals_require_complete_population(self):
        agents = [
            SimpleNamespace(
                reached_goal=True, alive=True, distance_travelled_m=12.0,
                path=[1, 2],
            ),
            SimpleNamespace(
                reached_goal=False, alive=True, distance_travelled_m=None,
                path=[3],
            ),
        ]
        sim = SimpleNamespace(
            current_step=10,
            _goal_step={0: 8},
            _agent_finish_step={},
            _agent_finish_reason={},
            _agent_steps_by_agent={},
        )
        result = performance_metrics(sim, agents)["distance"]

        self.assertIsNone(result["total_travelled_m_all_navigators"])
        self.assertEqual(result["total_travelled_m_completed_navigators"], 12.0)
        self.assertEqual(result["agents_with_distance"], 1)
        self.assertEqual(result["agents_missing_distance"], [1])
        self.assertFalse(result["all_navigator_distances_complete"])
        self.assertTrue(result["completed_distance_complete"])

    def test_hazard_severity_means_use_accurate_denominators(self):
        sim = SimpleNamespace(
            _hazard_exposure={
                "severity_weighted_exposure_by_agent_by_step": {
                    "0": {"0": 0.0, "1": 0.5},
                    "1": {"0": 0.0, "1": 0.0},
                }
            }
        )
        result = hazard_exposure_metrics(sim, [])["severity_weighted_exposure"]

        self.assertEqual(result["observed_agent_step_samples"], 4)
        self.assertEqual(result["positive_severity_agent_step_samples"], 1)
        self.assertEqual(result["total"], 0.5)
        self.assertEqual(result["mean_per_observed_agent_step"], 0.125)
        self.assertEqual(result["mean_per_positive_severity_agent_step"], 0.5)
        self.assertNotIn("mean_per_exposed_agent_step", result)

    def test_message_novelty_ignores_unread_global_board_entries(self):
        sim = object.__new__(DynamicSwarmSimulation)
        agent = SimpleNamespace(
            current_node=10,
            plan=[],
            local_memory={"blocked_nodes": set(), "danger_nodes": {}},
        )
        sim.agents = [agent, SimpleNamespace()]
        sim.comm = SimpleNamespace(shared_memory=True)
        sim.shared_memory = {
            "blocked_nodes": {42},
            "danger_nodes": {42: 0.9},
        }
        sim.graph = SimpleNamespace(_adj={10: []})
        sim._local_message_read_keys = set()
        sim._local_message_reads = []
        sim._shared_memory_reads = []

        msg = SimpleNamespace(
            id=1,
            sender=1,
            sent_step=2,
            kind="hazard",
            payload={"blocked": [42], "danger": {42: 0.8}},
        )
        sim._record_local_message_reads(0, msg, 3)

        self.assertEqual(len(sim._local_message_reads), 2)
        self.assertTrue(all(r["new_information"] for r in sim._local_message_reads))

    def test_message_novelty_respects_prior_recipient_board_reads(self):
        sim = object.__new__(DynamicSwarmSimulation)
        agent = SimpleNamespace(
            current_node=10,
            plan=[],
            local_memory={"blocked_nodes": set(), "danger_nodes": {}},
        )
        sim.agents = [agent, SimpleNamespace()]
        sim.comm = SimpleNamespace(shared_memory=True)
        sim.shared_memory = {
            "blocked_nodes": {42},
            "danger_nodes": {42: 0.9},
        }
        sim.graph = SimpleNamespace(_adj={10: []})
        sim._local_message_read_keys = set()
        sim._local_message_reads = []
        sim._shared_memory_reads = [
            {
                "reader_agent": 0,
                "channel": "blocked_nodes",
                "node_id": 42,
                "observed_value": True,
            },
            {
                "reader_agent": 0,
                "channel": "danger_nodes",
                "node_id": 42,
                "observed_value": 0.9,
            },
        ]

        msg = SimpleNamespace(
            id=2,
            sender=1,
            sent_step=2,
            kind="hazard",
            payload={"blocked": [42], "danger": {42: 0.8}},
        )
        sim._record_local_message_reads(0, msg, 3)

        self.assertEqual(len(sim._local_message_reads), 2)
        self.assertTrue(all(not r["new_information"] for r in sim._local_message_reads))


if __name__ == "__main__":
    unittest.main()
