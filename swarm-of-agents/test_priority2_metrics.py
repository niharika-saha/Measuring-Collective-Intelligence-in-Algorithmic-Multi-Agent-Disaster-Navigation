"""Focused regression tests for Priority 2 metric semantics.

Run with:
    python -m pytest -q test_priority2_metrics.py
"""
from types import SimpleNamespace

from collective_metrics import (
    performance_metrics,
    hazard_exposure_metrics,
    coverage_metrics,
)


def _agent(completed=False, role="navigator", alive=True, path=None, visited=None):
    return SimpleNamespace(
        role=role,
        reached_goal=completed,
        alive=alive,
        path=list(path or []),
        visited=set(visited or ()),
        cell_counts={},
    )


def test_even_sample_median_is_conventional_median():
    agents = [_agent(True), _agent(True)]
    sim = SimpleNamespace(
        current_step=12,
        _goal_step={0: 5, 1: 12},
        _agent_finish_step={0: 5, 1: 12},
        _agent_finish_reason={0: "goal", 1: "goal"},
        _agent_steps_by_agent={0: 4, 1: 9},
        _path_distance=lambda path: 10.0 * max(0, len(path) - 1),
    )
    metrics = performance_metrics(sim, agents)
    assert metrics["travel_time"]["median_completed_agent_goal_time_steps"] == 6.5
    assert metrics["travel_time"]["completed_agent_goal_times"] == [4, 9]


def test_incomplete_team_does_not_report_timeout_as_team_makespan():
    agents = [_agent(True), _agent(False, alive=True)]
    sim = SimpleNamespace(
        current_step=100,
        _goal_step={0: 7},
        _agent_finish_step={0: 7, 1: 100},
        _agent_finish_reason={0: "goal", 1: "timeout"},
        _agent_steps_by_agent={0: 6, 1: 100},
        _path_distance=lambda path: 0.0,
    )
    metrics = performance_metrics(sim, agents)
    assert metrics["completed_navigators"] == 1
    assert metrics["unfinished_navigators"] == 1
    assert metrics["makespan"]["successful_agent_makespan_steps"] == 7
    assert metrics["makespan"]["team_completion_makespan_steps"] is None
    assert metrics["makespan"]["observed_simulation_horizon_steps"] == 100
    assert metrics["travel_time"]["completed_agent_goal_times"] == [6]


def test_distance_uses_simulator_path_distance_when_agent_counter_absent():
    agents = [_agent(True, path=[1, 2, 3])]
    sim = SimpleNamespace(
        current_step=3,
        _goal_step={0: 3},
        _agent_finish_step={0: 3},
        _agent_finish_reason={0: "goal"},
        _agent_steps_by_agent={0: 2},
        _path_distance=lambda path: 123.4,
    )
    metrics = performance_metrics(sim, agents)
    assert metrics["distance"]["total_travelled_m_all_navigators"] == 123.4
    assert metrics["distance"]["total_travelled_m_completed_navigators"] == 123.4


def test_coverage_can_report_explicit_population():
    agents = [
        _agent(role="navigator", visited={1, 2}),
        _agent(role="scout", visited={2, 3}),
    ]
    metrics = coverage_metrics(agents, indices=[0])
    assert metrics["agent_indices"] == [0]
    assert metrics["unique_nodes_covered"] == 2


def test_severity_weighted_exposure_is_aggregated_from_recorded_agent_steps():
    sim = SimpleNamespace(_hazard_exposure={
        "severity_weighted_exposure_by_agent_by_step": {
            "1": {"0": 0.4, "1": 0.2},
            "2": {"0": 0.6},
        }
    })
    metrics = hazard_exposure_metrics(sim, [])
    dose = metrics["severity_weighted_exposure"]
    assert abs(dose["total"] - 1.2) < 1e-12
    assert dose["per_agent"] == {"0": 1.0, "1": 0.2}
    assert dose["agent_step_samples"] == 3
