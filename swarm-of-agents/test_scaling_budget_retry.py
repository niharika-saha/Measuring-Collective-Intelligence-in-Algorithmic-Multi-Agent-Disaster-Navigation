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
    calls = []

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


def test_parallel_arm_runs_only_once_at_original_budget(monkeypatch):
    calls = []

    def fake_run_single(**kwargs):
        calls.append(kwargs["steps"])
        return {"task_queue_results": [
            {"task_id": f"task_{i}", "success": True}
            for i in range(10)
        ]}

    NAMESPACE["run_single"] = fake_run_single
    _, used_budget, attempts, complete = run_with_budget(
        {"num_agents": 3}, initial_steps=1500, max_steps=1500,
        growth_factor=2.0, retry_until_complete=False,
    )

    assert calls == [1500]
    assert used_budget == 1500
    assert attempts == 1
    assert complete is True
