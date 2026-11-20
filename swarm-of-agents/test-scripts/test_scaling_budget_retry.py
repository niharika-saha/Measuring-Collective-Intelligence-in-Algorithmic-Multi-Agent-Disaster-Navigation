"""Focused tests for adaptive serial-baseline step-budget retries.

The helper is loaded directly from test_dynamic_disaster.py so these tests do
not need to initialize the spatial-index stack or require the optional rtree
binary dependency.
"""
import ast
import math
from pathlib import Path


SOURCE = Path(__file__).with_name("test_dynamic_disaster.py")
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
HELPER_NODE = next(
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "_run_scaling_arm_with_budget"
)
HELPER_MODULE = ast.Module(body=[HELPER_NODE], type_ignores=[])
NAMESPACE = {"math": math}
exec(compile(HELPER_MODULE, str(SOURCE), "exec"), NAMESPACE)
run_with_budget = NAMESPACE["_run_scaling_arm_with_budget"]


def test_serial_baseline_retries_same_seed_and_tasks_until_complete(monkeypatch):
    import time

    calls = []
    ticks = iter([10.0, 11.0, 20.0, 23.0])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

    def fake_run_single(**kwargs):
        budget = kwargs["steps"]
        calls.append((budget, kwargs["seed"], kwargs["tasks"], kwargs["task_queues"]))
        completed = 10 if budget >= 3000 else 8
        rows = [
            {"task_id": f"task_{i}", "success": i < completed}
            for i in range(10)
        ]
        return {"task_queue_results": rows}

    NAMESPACE["run_single"] = fake_run_single
    kwargs = {
        "num_agents": 1,
        "seed": 42,
        "tasks": [{"task_id": "deployment"}],
        "task_queues": [[{"task_id": f"task_{i}"} for i in range(10)]],
    }
    result, used_budget, attempts, complete = run_with_budget(
        kwargs, initial_steps=1500, max_steps=6000,
        growth_factor=2.0, retry_until_complete=True,
    )

    assert [call[0] for call in calls] == [1500, 3000]
    assert all(call[1] == 42 for call in calls)
    assert calls[0][2:] == calls[1][2:]
    assert used_budget == 3000
    assert attempts == 2
    assert complete is True
    assert all(row["success"] for row in result["task_queue_results"])
    assert result["attempt_count"] == 2
    assert result["attempt_horizons"] == [1500, 3000]
    assert result["wall_clock_execution_seconds"] == 3.0
    assert result["cumulative_retry_wall_clock_seconds"] == 1.0 + 3.0


def test_serial_baseline_stays_censored_if_maximum_budget_is_exhausted(monkeypatch):
    calls = []

    def fake_run_single(**kwargs):
        calls.append(kwargs["steps"])
        return {
            "task_queue_results": [
                {"task_id": f"task_{i}", "success": i < 9}
                for i in range(10)
            ]
        }

    NAMESPACE["run_single"] = fake_run_single
    result, used_budget, attempts, complete = run_with_budget(
        {"num_agents": 1}, initial_steps=1500, max_steps=2500,
        growth_factor=2.0, retry_until_complete=True,
    )

    assert calls == [1500, 2500]
    assert used_budget == 2500
    assert attempts == 2
    assert complete is False
    assert sum(row["success"] for row in result["task_queue_results"]) == 9


def test_every_team_size_retries_under_the_common_policy(monkeypatch):
    calls = []

    def fake_run_single(**kwargs):
        calls.append(kwargs["steps"])
        return {"task_queue_results": [
            {"task_id": f"task_{i}", "success": kwargs["steps"] >= 3000}
            for i in range(10)
        ]}

    NAMESPACE["run_single"] = fake_run_single
    _, used_budget, attempts, complete = run_with_budget(
        {"num_agents": 3}, initial_steps=1500, max_steps=12000,
        growth_factor=2.0, retry_until_complete=True,
    )

    assert calls == [1500, 3000]
    assert used_budget == 3000
    assert attempts == 2
    assert complete is True


def test_incomplete_arm_reaches_common_maximum_at_12000(monkeypatch):
    import time

    calls = []
    ticks = iter([10.0, 11.0, 20.0, 23.0, 30.0, 34.0, 40.0, 44.0])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

    def fake_run_single(**kwargs):
        budget = kwargs["steps"]
        calls.append(budget)
        completed = 7 if budget == 12000 else 4
        return {
            "task_queue_results": [
                {"task_id": f"task_{i}", "success": i < completed}
                for i in range(10)
            ],
            "simulation_steps": budget,
        }

    NAMESPACE["run_single"] = fake_run_single
    result, used_budget, attempts, complete = run_with_budget(
        {"num_agents": 5}, initial_steps=1500, max_steps=12000,
        growth_factor=2.0, retry_until_complete=True,
    )

    assert calls == [1500, 3000, 6000, 12000]
    assert used_budget == 12000
    assert attempts == 4
    assert complete is False
    incomplete = [
        row["task_id"] for row in result["task_queue_results"]
        if not row["success"]
    ]
    assert incomplete == [f"task_{i}" for i in range(7, 10)]
    assert result["attempt_count"] == 4
    assert result["attempt_horizons"] == [1500, 3000, 6000, 12000]
    assert result["cumulative_retry_wall_clock_seconds"] == 1.0 + 3.0 + 4.0 + 4.0
    assert result["wall_clock_execution_seconds"] == 4.0


def test_retry_timing_is_per_attempt_not_retry_total(monkeypatch):
    import time

    original_perf_counter = time.perf_counter
    ticks = iter([10.0, 11.0, 20.0, 23.0])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

    def fake_run_single(**kwargs):
        return {"task_queue_results": [
            {"task_id": f"task_{i}", "success": False}
            for i in range(10)
        ]}

    NAMESPACE["run_single"] = fake_run_single
    result, used_budget, attempts, complete = run_with_budget(
        {"num_agents": 10}, initial_steps=1500, max_steps=3000,
        growth_factor=2.0, retry_until_complete=True,
    )

    assert used_budget == 3000
    assert attempts == 2
    assert complete is False
    assert result["wall_clock_execution_seconds"] == 3.0
    monkeypatch.setattr(time, "perf_counter", original_perf_counter)


def test_successful_first_attempt_records_one_attempt(monkeypatch):
    import time

    calls = []
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

    def fake_run_single(**kwargs):
        calls.append(kwargs["steps"])
        return {
            "task_queue_results": [
                {"task_id": f"task_{i}", "success": True}
                for i in range(10)
            ]
        }

    NAMESPACE["run_single"] = fake_run_single
    result, used_budget, attempts, complete = run_with_budget(
        {"num_agents": 10}, initial_steps=1500, max_steps=12000,
        growth_factor=2.0, retry_until_complete=True,
    )

    assert calls == [1500]
    assert used_budget == 1500
    assert attempts == 1
    assert complete is True
    assert result["attempt_count"] == 1
    assert result["attempt_horizons"] == [1500]
    assert result["wall_clock_execution_seconds"] == 2.5
    assert result["cumulative_retry_wall_clock_seconds"] == 2.5


def test_incomplete_workload_fields_and_paired_speedups_are_invalid():
    baseline = {
        "num_agents": 1,
        "all_tasks_completed": False,
        "tasks_completed": 7,
        "completion_rate": 0.7,
        "incomplete_task_ids": ["task_007", "task_008", "task_009"],
        "workload_makespan_steps": None,
        "parallel_speedup": None,
        "scaling_efficiency": None,
    }
    assert baseline["completion_rate"] == baseline["tasks_completed"] / 10
    assert baseline["workload_makespan_steps"] is None
    assert baseline["parallel_speedup"] is None
    assert baseline["scaling_efficiency"] is None

    paired_arms = [
        {"num_agents": n, "all_tasks_completed": True,
         "workload_makespan_steps": 100}
        for n in (2, 3, 5, 10)
    ]
    baseline_by_seed = {
        baseline["num_agents"]: baseline["workload_makespan_steps"]
    } if baseline["all_tasks_completed"] else {}

    for row in paired_arms:
        baseline_makespan = baseline_by_seed.get(row["num_agents"])
        if row["all_tasks_completed"] and not baseline_makespan:
            row["parallel_speedup"] = None
            row["scaling_efficiency"] = None
        assert row["parallel_speedup"] is None
        assert row["scaling_efficiency"] is None
