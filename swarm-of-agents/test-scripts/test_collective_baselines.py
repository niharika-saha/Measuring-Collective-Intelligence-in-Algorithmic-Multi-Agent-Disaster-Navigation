"""Tests for the paired collective-intelligence baseline harness."""
import pytest

import collective_baselines as baselines
from swarm_simulation_dynamic import DynamicSwarmSimulation
from types import SimpleNamespace


class _FakeEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeBus:
    def stats(self):
        return {
            "transmissions": 0,
            "deliveries": 0,
            "dropped": 0,
            "bytes_tx": 0,
            "mean_delay_steps": None,
            "by_channel": {},
        }


class _FakeAgent:
    def __init__(self, task):
        self.reached_goal = True
        self.death_reason = None
        self.path = [task["start"], task["goal"]]


class _FakeSimulation:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.agents = [
            _FakeAgent(task) for task in kwargs["tasks"][:kwargs["num_agents"]]
        ]
        self.bus = _FakeBus()
        self.current_step = 2
        self._goal_step = {i: 2 for i in range(len(self.agents))}
        self.__class__.calls.append(kwargs)

    def run(self, steps):
        self.steps = steps

    def sequential_task_metrics(self):
        return [
            {
                "task_id": task["task_id"],
                "start": task["start"],
                "goal": task["goal"],
                "success": True,
                "finish_reason": "goal",
                "finish_step": 2 * (i + 1),
                "agent_steps": 2,
                "path_distance_m": 100.0,
                "path_node_count": 2,
                "reposition_steps_before": 0 if i == 0 else 1,
                "reposition_distance_before_m": 0.0 if i == 0 else 10.0,
            }
            for i, task in enumerate(self.kwargs["sequential_task_queue"])
        ]

    def execution_metrics(self):
        n = len(self.agents)
        if self.kwargs.get("sequential_task_queue"):
            return {
                "agent_steps_total": 4,
                "agent_steps_by_agent": {"0": 4},
                "finish_step_by_agent": {"0": 4},
                "finish_reason_by_agent": {"0": "goal"},
                "makespan_steps": 4,
            }
        return {
            "agent_steps_total": n * 2,
            "agent_steps_by_agent": {str(i): 2 for i in range(n)},
            "finish_step_by_agent": {str(i): 2 for i in range(n)},
            "finish_reason_by_agent": {str(i): "goal" for i in range(n)},
            "makespan_steps": 2,
        }

    def _path_distance(self, path):
        return 210.0 if self.kwargs.get("sequential_task_queue") else 100.0

    def communication_instrumentation(self):
        return {
            "agents_receiving_teammate_information": 1,
            "shared_memory_route_relevant_cross_agent_retrievals": 1,
        }

    def hazard_response_metrics(self):
        return {
            "planner_replan_source_attribution": {
                "self_observation": 0,
                "teammate_message": 1,
                "shared_memory": 0,
            }
        }


def _install_fake_runner(monkeypatch):
    tasks = [
        {"task_id": "task_001", "start": 1, "goal": 11},
        {"task_id": "task_002", "start": 2, "goal": 12},
    ]
    monkeypatch.setattr(baselines.runner, "load_fresh_graph", lambda: object())
    monkeypatch.setattr(baselines.runner, "_paired_seed", lambda base, idx: base + idx)
    monkeypatch.setattr(
        baselines.runner,
        "generate_run_tasks",
        lambda **kwargs: [dict(task) for task in tasks],
    )
    monkeypatch.setattr(baselines.runner, "DynamicDisasterEngine", _FakeEngine)
    monkeypatch.setattr(baselines, "DynamicSwarmSimulation", _FakeSimulation)
    _FakeSimulation.calls = []
    return tasks


def test_three_arms_reuse_identical_task_manifest_and_agent_rng_streams(monkeypatch):
    tasks = _install_fake_runner(monkeypatch)

    result = baselines.run_collective_baseline_experiment(
        workload="independent",
        n_runs=1,
        num_agents=2,
        steps=10,
        base_seed=17,
    )

    run = result["runs"][0]
    expected_manifest = [
        {"task_id": t["task_id"], "start": t["start"], "goal": t["goal"]}
        for t in tasks
    ]
    assert run["task_manifest"] == expected_manifest
    for arm in run["arms"].values():
        assert [
            {k: row[k] for k in ("task_id", "start", "goal")}
            for row in arm["per_task"]
        ] == expected_manifest

    # Calls: one serial queue, then independent and coordinated parallel arms.
    calls = _FakeSimulation.calls
    assert [len(call["tasks"]) for call in calls] == [2, 2, 2]
    assert [call["num_agents"] for call in calls] == [1, 2, 2]
    assert calls[0]["sequential_task_queue"] == tasks
    assert calls[0]["agent_rngs"][0].getstate() == calls[1]["agent_rngs"][0].getstate()
    assert calls[1]["agent_rngs"][0].getstate() == calls[2]["agent_rngs"][0].getstate()
    assert calls[1]["agent_rngs"][1].getstate() == calls[2]["agent_rngs"][1].getstate()


def test_independent_and_coordinated_arms_differ_only_in_channels(monkeypatch):
    _install_fake_runner(monkeypatch)

    baselines.run_collective_baseline_experiment(
        workload="independent",
        n_runs=1,
        num_agents=2,
        steps=10,
        base_seed=17,
    )
    independent_call, coordinated_call = _FakeSimulation.calls[-2:]

    assert independent_call["roles"] == coordinated_call["roles"] == [
        "navigator", "navigator"
    ]
    assert independent_call["seed"] == coordinated_call["seed"]
    assert independent_call["tasks"] == coordinated_call["tasks"]
    assert independent_call["comm"].shared_memory is False
    assert independent_call["comm"].local_messaging is False
    assert independent_call["comm"].pheromones is False
    assert coordinated_call["comm"].shared_memory is True
    assert coordinated_call["comm"].local_messaging is True
    assert coordinated_call["comm"].pheromones is True


def test_serial_arm_has_same_total_agent_step_budget_as_parallel_arms(monkeypatch):
    _install_fake_runner(monkeypatch)

    result = baselines.run_collective_baseline_experiment(
        workload="independent",
        n_runs=1,
        num_agents=2,
        steps=10,
        base_seed=17,
    )
    arms = result["runs"][0]["arms"]
    assert {
        arm["agent_step_budget_total"] for arm in arms.values()
    } == {20}
    assert arms["individual_serial"]["agent_steps_total"] == 4
    assert arms["independent_multi_agent"]["agent_steps_total"] == 4
    assert arms["coordinated_swarm"]["agent_steps_total"] == 4


def test_workload_and_team_size_validation():
    with pytest.raises(ValueError, match="at least 2"):
        baselines.run_collective_baseline_experiment(
            workload="collaboration", num_agents=1, n_runs=1
        )
    with pytest.raises(ValueError, match="at least 2"):
        baselines.run_collective_baseline_experiment(
            workload="independent", num_agents=1, n_runs=1
        )
    with pytest.raises(ValueError, match="workload"):
        baselines.run_collective_baseline_experiment(
            workload="unknown", num_agents=2, n_runs=1
        )


def test_execution_metrics_report_work_separately_from_makespan():
    sim = DynamicSwarmSimulation.__new__(DynamicSwarmSimulation)
    sim.agents = [
        SimpleNamespace(role="navigator", reached_goal=True, death_reason=None),
        SimpleNamespace(role="navigator", reached_goal=False, death_reason="trapped"),
        SimpleNamespace(role="scout", reached_goal=False, death_reason=None),
    ]
    sim._agent_steps_by_agent = {0: 4, 1: 2}
    sim._agent_finish_step = {0: 4, 1: 2}
    sim._agent_finish_reason = {0: "goal", 1: "trapped"}
    sim.current_step = 4

    metrics = sim.execution_metrics()
    assert metrics["agent_steps_total"] == 6
    assert metrics["agent_steps_by_agent"] == {"0": 4, "1": 2}
    assert metrics["makespan_steps"] == 4
    assert metrics["finish_reason_by_agent"] == {
        "0": "goal", "1": "trapped"
    }


def test_sequential_task_queue_records_task_and_reposition_legs():
    sim = DynamicSwarmSimulation.__new__(DynamicSwarmSimulation)
    sim.sequential_task_queue = [
        {"task_id": "task_001", "start": 1, "goal": 2},
        {"task_id": "task_002", "start": 3, "goal": 4},
    ]
    sim.task_queues = [sim.sequential_task_queue]
    sim.sequential_task_results = []
    sim._sequential_task_state = {
        "task_index": 0,
        "next_task_index": None,
        "phase": "task",
        "path_start_index": 0,
        "phase_steps": 0,
        "reposition_path_start_index": None,
        "reposition_steps_before": 0,
        "reposition_distance_before_m": 0.0,
        "complete": False,
    }

    sim._task_queue_states = {
    0: sim._sequential_task_state
    }
    sim._path_distance = lambda path: max(0, len(path) - 1) * 10.0

    class Agent:
        def __init__(self):
            self.path = [1, 2]
            self.current_node = 2
            self.goal_node = 2
            self.reached_goal = True
            self.alive = True
            self.death_reason = None

        def set_goal(self, node):
            self.goal_node = node
            self.reached_goal = False

    agent = Agent()
    sim._advance_sequential_task_queue(0, agent, 1)
    assert sim._sequential_task_state["phase"] == "reposition"
    assert agent.goal_node == 3
    assert sim.sequential_task_results[0]["task_id"] == "task_001"
    assert sim.sequential_task_results[0]["path_distance_m"] == 10.0

    agent.path.append(3)
    agent.current_node = 3
    agent.reached_goal = True
    sim._advance_sequential_task_queue(0, agent, 2)
    assert sim._sequential_task_state["phase"] == "task"
    assert agent.goal_node == 4
    assert sim._sequential_task_state["reposition_distance_before_m"] == 10.0

    agent.path.append(4)
    agent.current_node = 4
    agent.reached_goal = True
    sim._advance_sequential_task_queue(0, agent, 3)
    assert sim._sequential_task_state["complete"] is True
    assert [row["task_id"] for row in sim.sequential_task_results] == [
        "task_001", "task_002"
    ]
    assert all(row["success"] for row in sim.sequential_task_results)


def test_collaboration_workload_uses_local_radio_and_exports_diagnostics(monkeypatch):
    tasks = _install_fake_runner(monkeypatch)
    captured = {}

    def fake_collaboration_scenario(**kwargs):
        captured.update(kwargs)
        return (
            [dict(task) for task in tasks],
            {"event_type": "dynamic_hazard_exposure", "trigger_step": 1},
            {
                "communication_mode": "local_radio",
                "leader_to_nearest_follower_distance_m": 200.0,
            },
        )

    monkeypatch.setattr(
        baselines.runner,
        "generate_collaboration_scenario",
        fake_collaboration_scenario,
    )

    result = baselines.run_collective_baseline_experiment(
        workload="collaboration",
        n_runs=1,
        num_agents=2,
        steps=10,
        base_seed=17,
    )

    assert captured["local_communication"] is True
    coordinated = result["runs"][0]["arms"]["coordinated_swarm"]
    assert coordinated["communication"]["instrumentation"][
        "agents_receiving_teammate_information"
    ] == 1
    assert coordinated["hazard_response"][
        "planner_replan_source_attribution"
    ]["teammate_message"] == 1
    assert result["experiment"]["communication_mode"] == "local_radio"
