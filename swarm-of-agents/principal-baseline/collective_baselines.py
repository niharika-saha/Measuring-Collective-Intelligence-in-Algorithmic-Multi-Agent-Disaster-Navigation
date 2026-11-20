"""
Paired individual / independent / coordinated baselines for the dynamic swarm.

Run from this directory:
    python collective_baselines.py --agents 10 --runs 5 --steps 200 \
        --workload collaboration --type earthquake --output collective_baselines.json

The individual arm uses one navigator to execute the complete task manifest
serially in a single continuously evolving disaster simulation. Between tasks,
it must travel from the previous goal to the next task's start. Independent and
coordinated arms execute the same manifest in parallel, one task per agent, in
their own matched copies of the same seeded environment. The collaboration
workload uses a local-radio convoy so teammate warnings can arrive before
followers independently sense the scheduled blockage.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import statistics
from dataclasses import asdict

from comms import CommConfig
from feasibility import agent_rng
from swarm_simulation_dynamic import DynamicSwarmSimulation
import test_dynamic_disaster as runner
from collective_metrics import collective_metrics_from_sim


def _planner_response_metrics(sim) -> dict:
    """Return planner diagnostics when the simulation exposes them.

    Some lightweight test doubles and older simulation adapters implement
    hazard_response_metrics() but not response_metrics(). In that case keep
    the baseline export schema stable with empty, explicitly neutral values.
    """
    method = getattr(sim, "response_metrics", None)
    if not callable(method):
        return {
            "planner_replan_source_attribution": {},
            "planner_replan_source_records": [],
            "planner_successful_replans": 0,
            "planner_route_changes_following_hazard_discovery": 0,
        }
    metrics = method() or {}
    return {
        "planner_replan_source_attribution": metrics.get(
            "planner_replan_source_attribution", {}
        ),
        "planner_replan_source_records": metrics.get(
            "planner_replan_source_records", []
        ),
        "planner_successful_replans": metrics.get(
            "planner_successful_replans", 0
        ),
        "planner_route_changes_following_hazard_discovery": metrics.get(
            "planner_route_changes_following_hazard_discovery", 0
        ),
    }


def _disabled_comm() -> CommConfig:
    return CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=False,
        sync_on_contact=False,
    )


def _enabled_comm() -> CommConfig:
    return CommConfig(
        shared_memory=True,
        local_messaging=True,
        pheromones=True,
        sync_on_contact=True,
    )


def _make_engine(graph, disaster_type: str, seed: int, exposure_event=None):
    common = dict(
        graph=graph,
        disaster_type=disaster_type,
        centre=runner.DISASTER_CENTRE,
        radius_m=runner.DISASTER_RADIUS_M,
        severity=runner.DISASTER_SEVERITY,
        seed=seed,
        wind_bearing_deg=runner.WIND_BEARING,
    )
    if exposure_event is not None:
        return runner.CollaborationHazardEngine(
            **common, exposure_event=exposure_event
        )
    return runner.DynamicDisasterEngine(**common)


def _new_simulation(
    *,
    tasks: list[dict],
    disaster_type: str,
    seed: int,
    steps: int,
    comm: CommConfig,
    agent_indices: list[int],
    exposure_event=None,
    num_agents: int | None = None,
    sequential_task_queue: list[dict] | None = None,
):
    graph = runner.load_fresh_graph()
    engine = _make_engine(graph, disaster_type, seed, exposure_event)
    # Match the exact per-agent stochastic streams used by the parallel arms.
    rngs = [agent_rng(seed, i) for i in agent_indices]
    actual_num_agents = len(agent_indices) if num_agents is None else num_agents
    with contextlib.redirect_stdout(io.StringIO()):
        sim = DynamicSwarmSimulation(
            graph=graph,
            num_agents=actual_num_agents,
            min_start_goal_distance=1000,
            max_start_goal_distance=4000,
            disaster_type=disaster_type,
            disaster_centre=runner.DISASTER_CENTRE,
            disaster_radius_m=runner.DISASTER_RADIUS_M,
            disaster_severity=runner.DISASTER_SEVERITY,
            wind_bearing_deg=runner.WIND_BEARING,
            seed=seed,
            tasks=tasks,
            require_feasible=True,
            comm=comm,
            roles=["navigator"] * actual_num_agents,
            engine=engine,
            agent_rngs=rngs,
            sequential_task_queue=sequential_task_queue,
        )
        sim.run(steps=steps)
    return sim


def _task_result(sim, task: dict, agent_index: int = 0) -> dict:
    agent = sim.agents[agent_index]
    execution = sim.execution_metrics()
    key = str(agent_index)
    return {
        "task_id": task.get("task_id", f"task_{agent_index + 1:03d}"),
        "start": task["start"],
        "goal": task["goal"],
        "success": bool(agent.reached_goal),
        "finish_reason": execution["finish_reason_by_agent"].get(key, "unknown"),
        "finish_step": execution["finish_step_by_agent"].get(key),
        "agent_steps": execution["agent_steps_by_agent"].get(key, 0),
        "path_distance_m": round(sim._path_distance(agent.path), 3),
        "path_node_count": len(agent.path),
        "goal_step": sim._goal_step.get(agent_index),
    }


def _summarise_task_results(
    tasks: list[dict],
    task_results: list[dict],
    *,
    agent_steps_total: int,
    makespan_steps: int,
    communication: dict,
    episode_count: int,
    max_agent_step_budget: int,
    total_distance_travelled_m: float | None = None,
) -> dict:
    n = len(tasks)
    successes = sum(bool(row["success"]) for row in task_results)
    attempted_results = [
        row for row in task_results
        if row.get("finish_reason") != "not_attempted"
    ]
    distances = [row["path_distance_m"] for row in task_results]
    attempted_distances = [
        row["path_distance_m"] for row in attempted_results
    ]
    successful_distances = [
        row["path_distance_m"] for row in task_results if row["success"]
    ]
    successful_finish_steps = [
        row["finish_step"] for row in task_results
        if row["success"] and row["finish_step"] is not None
    ]
    return {
        "tasks": n,
        "successful_tasks": successes,
        "failed_or_timed_out_tasks": sum(
            1 for row in task_results
            if not row["success"] and row.get("finish_reason") != "not_attempted"
        ),
        "not_completed_tasks": n - successes,
        "attempted_tasks": len(attempted_results),
        "not_attempted_tasks": n - len(attempted_results),
        "success_rate": successes / n if n else 0.0,
        "mean_distance_per_task_m": (
            statistics.fmean(attempted_distances)
            if attempted_distances else None
        ),
        "mean_distance_per_manifest_task_m": (
            statistics.fmean(distances) if distances else None
        ),
        "mean_successful_path_distance_m": (
            statistics.fmean(successful_distances)
            if successful_distances else None
        ),
        "total_distance_travelled_m": (
            sum(distances) if total_distance_travelled_m is None
            else float(total_distance_travelled_m)
        ),
        "reposition_distance_total_m": sum(
            float(row.get("reposition_distance_before_m", 0.0))
            for row in task_results
        ),
        "reposition_steps_total": sum(
            int(row.get("reposition_steps_before", 0))
            for row in task_results
        ),
        "mean_successful_completion_step": (
            statistics.fmean(successful_finish_steps)
            if successful_finish_steps else None
        ),
        "agent_steps_total": int(agent_steps_total),
        "max_agent_step_budget": int(max_agent_step_budget),
        "agent_step_budget_total": int(max_agent_step_budget),
        "agent_step_budget_utilisation": (
            agent_steps_total / max_agent_step_budget
            if max_agent_step_budget else 0.0
        ),
        "makespan_steps": int(makespan_steps),
        "episodes": int(episode_count),
        "communication": communication,
        "per_task": task_results,
    }




def _run_individual_arm(
    *,
    tasks: list[dict],
    disaster_type: str,
    seed: int,
    steps: int,
    exposure_event=None,
) -> dict:
    """One navigator executes the complete task queue in one evolving world."""
    total_budget = len(tasks) * steps

    sim = _new_simulation(
        tasks=tasks,
        disaster_type=disaster_type,
        seed=seed,
        steps=total_budget,
        comm=_disabled_comm(),
        agent_indices=[0],
        num_agents=1,
        sequential_task_queue=tasks,
        exposure_event=exposure_event,
    )

    task_results = sim.sequential_task_metrics()
    execution = sim.execution_metrics()
    total_distance = sim._path_distance(sim.agents[0].path)

    # Export the same event-level and planner diagnostics as the parallel arms.
    hazard_response = sim.hazard_response_metrics()
    planner_metrics = _planner_response_metrics(sim)

    # Preserve attribution already exported by hazard_response_metrics().
    # Lightweight adapters may expose event attribution there without a
    # separate response_metrics() method. Merge instead of overwriting it
    # with the neutral fallback from _planner_response_metrics().
    source_counts = {
        "self_observation": 0,
        "teammate_message": 0,
        "shared_memory": 0,
        "unattributed": 0,
    }
    source_counts.update(
        hazard_response.get("planner_replan_source_attribution", {}) or {}
    )
    planner_sources = (
        planner_metrics.get("planner_replan_source_attribution", {}) or {}
    )
    if any(int(value or 0) for value in planner_sources.values()):
        source_counts.update(planner_sources)
    hazard_response["planner_replan_source_attribution"] = source_counts
    hazard_response["planner_replan_source_records"] = (
        planner_metrics["planner_replan_source_records"]
    )
    hazard_response["planner_successful_replans"] = (
        planner_metrics["planner_successful_replans"]
    )
    hazard_response[
        "planner_route_changes_following_hazard_discovery"
    ] = planner_metrics[
        "planner_route_changes_following_hazard_discovery"
    ]

    summary = _summarise_task_results(
        tasks,
        task_results,
        agent_steps_total=execution["agent_steps_total"],
        makespan_steps=execution["makespan_steps"],
        communication={
            "transmissions": 0,
            "deliveries": 0,
            "dropped": 0,
            "bytes_tx": 0,
        },
        episode_count=1,
        max_agent_step_budget=total_budget,
        total_distance_travelled_m=total_distance,
    )

    summary["hazard_response"] = hazard_response
    summary["collective_intelligence"] = collective_metrics_from_sim(sim)
    return summary



def _run_parallel_arm(
    *,
    tasks: list[dict],
    disaster_type: str,
    seed: int,
    steps: int,
    comm: CommConfig,
    exposure_event=None,
) -> dict:
    sim = _new_simulation(
        tasks=tasks,
        disaster_type=disaster_type,
        seed=seed,
        steps=steps,
        comm=comm,
        agent_indices=list(range(len(tasks))),
        exposure_event=exposure_event,  # Critical: forward the event
    )
    task_results = [
        _task_result(sim, task, i) for i, task in enumerate(tasks)
    ]
    execution = sim.execution_metrics()
    bus_stats = sim.bus.stats()
    communication_instrumentation = sim.communication_instrumentation()
    hazard_response = sim.hazard_response_metrics()
    planner_metrics = _planner_response_metrics(sim)
    # Keep the source attribution beside the event-level response records in
    # each arm's JSON. The counters are based on successful route changes and
    # the first recorded awareness source, not every source seen in a time step.
    # Preserve attribution already exported by hazard_response_metrics().
    # Lightweight adapters may expose event attribution there without a
    # separate response_metrics() method. Merge instead of overwriting it
    # with the neutral fallback from _planner_response_metrics().
    source_counts = {
        "self_observation": 0,
        "teammate_message": 0,
        "shared_memory": 0,
        "unattributed": 0,
    }
    source_counts.update(
        hazard_response.get("planner_replan_source_attribution", {}) or {}
    )
    planner_sources = (
        planner_metrics.get("planner_replan_source_attribution", {}) or {}
    )
    if any(int(value or 0) for value in planner_sources.values()):
        source_counts.update(planner_sources)
    hazard_response["planner_replan_source_attribution"] = source_counts
    hazard_response["planner_replan_source_records"] = (
        planner_metrics["planner_replan_source_records"]
    )
    hazard_response["planner_successful_replans"] = (
        planner_metrics["planner_successful_replans"]
    )
    hazard_response[
        "planner_route_changes_following_hazard_discovery"
    ] = planner_metrics[
        "planner_route_changes_following_hazard_discovery"
    ]
    summary = _summarise_task_results(
        tasks,
        task_results,
        agent_steps_total=execution["agent_steps_total"],
        makespan_steps=execution["makespan_steps"],
        communication={
            "transmissions": bus_stats["transmissions"],
            "deliveries": bus_stats["deliveries"],
            "dropped": bus_stats["dropped"],
            "bytes_tx": bus_stats["bytes_tx"],
            "mean_delay_steps": bus_stats["mean_delay_steps"],
            "by_channel": bus_stats["by_channel"],
            "instrumentation": communication_instrumentation,
        },
        episode_count=1,
        max_agent_step_budget=len(tasks) * steps,
    )
    summary["hazard_response"] = hazard_response
    summary["collective_intelligence"] = collective_metrics_from_sim(sim)
    return summary


def _paired_delta(left: dict, right: dict, metric: str) -> float | None:
    a, b = left.get(metric), right.get(metric)
    if a is None or b is None:
        return None
    return float(b) - float(a)


def _mean_std(values: list[float]) -> dict:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {
        "mean": statistics.fmean(clean) if clean else None,
        "std": statistics.stdev(clean) if len(clean) > 1 else (0.0 if clean else None),
        "n": len(clean),
    }


def run_collective_baseline_experiment(
    *,
    disaster_type: str = "earthquake",
    workload: str = "collaboration",
    n_runs: int = 5,
    num_agents: int = 10,
    steps: int = 200,
    min_dist: float = 1000,
    max_dist: float = 4000,
    base_seed: int = 42,
    output_path: str | None = None,
) -> dict:
    """
    Paired A/B/C comparison:
      A: one agent executes the complete task queue serially in one evolving world;
      B: N homogeneous agents, all coordination channels disabled;
      C: N homogeneous agents, all coordination channels enabled.

    B vs C isolates the added effect of coordination. A vs B compares a
    single serial task executor with a parallel team under equal total
    agent-step budgets.
    """
    if workload not in {"independent", "collaboration"}:
        raise ValueError("workload must be 'independent' or 'collaboration'")
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    if num_agents < 2:
        raise ValueError(
            "collective baseline comparison requires at least 2 tasks/agents"
        )
    if steps < 1:
        raise ValueError("steps must be at least 1")

    paired_runs = []
    for run_index in range(n_runs):
        seed = runner._paired_seed(base_seed, run_index)
        task_graph = runner.load_fresh_graph()
        exposure_event = None
        scenario = None
        if workload == "collaboration":
            tasks, exposure_event, scenario = runner.generate_collaboration_scenario(
                graph=task_graph,
                num_agents=num_agents,
                disaster_type=disaster_type,
                seed=seed,
                min_dist=min_dist,
                max_dist=max_dist,
                # The previous long-range geometry placed the leader outside
                # radio range and yielded zero useful communication in the
                # controlled 10-agent pilot. Use a local-radio convoy: the
                # leader senses the scheduled blockage and can warn followers
                # before they independently reach their own sensing radius.
                local_communication=True,
            )
        else:
            tasks = runner.generate_run_tasks(
                graph=task_graph,
                num_agents=num_agents,
                disaster_type=disaster_type,
                seed=seed,
                workload="independent",
                min_dist=min_dist,
                max_dist=max_dist,
            )

        manifest = [
            {
                "task_id": task.get("task_id", f"task_{i + 1:03d}"),
                "start": task["start"],
                "goal": task["goal"],
            }
            for i, task in enumerate(tasks)
        ]

        individual = _run_individual_arm(
            tasks=tasks,
            disaster_type=disaster_type,
            seed=seed,
            steps=steps,
            exposure_event=exposure_event,
        )
        independent = _run_parallel_arm(
            tasks=tasks,
            disaster_type=disaster_type,
            seed=seed,
            steps=steps,
            comm=_disabled_comm(),
            exposure_event=exposure_event,
        )

        coordinated = _run_parallel_arm(
            tasks=tasks,
            disaster_type=disaster_type,
            seed=seed,
            steps=steps,
            comm=_enabled_comm(),
            exposure_event=exposure_event,
        )

        paired_runs.append({
            "run": run_index + 1,
            "seed": seed,
            "disaster_type": disaster_type,
            "workload": workload,
            "task_manifest": manifest,
            "exposure_event": exposure_event,
            "scenario": scenario,
            "arms": {
                "individual_serial": individual,
                "independent_multi_agent": independent,
                "coordinated_swarm": coordinated,
            },
            "paired_differences": {
                "parallelism_success_rate_delta_vs_individual": (
                    independent["success_rate"] - individual["success_rate"]
                ),
                "coordination_success_rate_delta_vs_independent": (
                    coordinated["success_rate"] - independent["success_rate"]
                ),
                "coordination_distance_delta_m_vs_independent": _paired_delta(
                    independent, coordinated, "mean_distance_per_task_m"
                ),
                "coordination_makespan_delta_steps_vs_independent": _paired_delta(
                    independent, coordinated, "makespan_steps"
                ),
                "coordination_agent_steps_delta_vs_independent": _paired_delta(
                    independent, coordinated, "agent_steps_total"
                ),
            },
        })

    metric_names = (
        "success_rate",
        "mean_distance_per_task_m",
        "mean_successful_path_distance_m",
        "total_distance_travelled_m",
        "reposition_distance_total_m",
        "reposition_steps_total",
        "mean_successful_completion_step",
        "agent_steps_total",
        "agent_step_budget_utilisation",
        "makespan_steps",
    )
    aggregate = {}
    for arm_name in (
        "individual_serial",
        "independent_multi_agent",
        "coordinated_swarm",
    ):
        aggregate[arm_name] = {
            metric: _mean_std([
                run["arms"][arm_name].get(metric)
                for run in paired_runs
            ])
            for metric in metric_names
        }

    difference_names = (
        "parallelism_success_rate_delta_vs_individual",
        "coordination_success_rate_delta_vs_independent",
        "coordination_distance_delta_m_vs_independent",
        "coordination_makespan_delta_steps_vs_independent",
        "coordination_agent_steps_delta_vs_independent",
    )
    paired_aggregate = {
        metric: _mean_std([
            run["paired_differences"].get(metric)
            for run in paired_runs
        ])
        for metric in difference_names
    }

    result = {
        "experiment": {
            "name": "paired_collective_baselines",
            "disaster_type": disaster_type,
            "workload": workload,
            "communication_mode": ("local_radio" if workload == "collaboration" else "disabled"),
            "n_runs": n_runs,
            "num_tasks": num_agents,
            "steps_per_task_budget": steps,
            "min_start_goal_distance_m": min_dist,
            "max_start_goal_distance_m": max_dist,
            "total_agent_step_budget_per_arm": num_agents * steps,
            "base_seed": base_seed,
            "disaster_centre": runner.DISASTER_CENTRE,
            "disaster_radius_m": runner.DISASTER_RADIUS_M,
            "disaster_severity": runner.DISASTER_SEVERITY,
            "wind_bearing_deg": runner.WIND_BEARING,
        },
        "design": {
            "communication_mode": ("local_radio" if workload == "collaboration" else "disabled"),
            "arms": {
                "individual_serial": {
                    "agents": 1,
                    "tasks": num_agents,
                    "execution": "serial_task_queue",
                    "coordination": False,
                    "description": (
                        "One navigator executes the complete task queue in order "
                        "inside one continuously evolving disaster simulation, "
                        "travelling between each completed task's goal and the "
                        "next task's start."
                    ),
                },
                "independent_multi_agent": {
                    "agents": num_agents,
                    "coordination": False,
                    "communication": asdict(_disabled_comm()),
                },
                "coordinated_swarm": {
                    "agents": num_agents,
                    "coordination": True,
                    "communication": asdict(_enabled_comm()),
                    "communication_mode": ("local_radio" if workload == "collaboration" else "general"),
                },
            },
            "navigation_policy": {
                "role": "navigator",
                "planner_enabled": True,
                "planner_weight_w_plan": 100.0,
                "specialized_roles": False,
            },
            "controls": [
                "Identical task manifest within each paired seed",
                "Identical disaster type, centre, radius, severity and engine seed",
                "Identical homogeneous navigator role and navigation policy",
                "Same deterministic seed policy; agent 0's RNG stream is shared between the individual arm and agent 0 in both multi-agent arms",
                "Equal total maximum agent-step budget (number of tasks multiplied by the per-agent parallel step cap)",
                "Independent and coordinated arms execute concurrently in matched environments",
                *([
                    "Collaboration workload uses local-radio geometry: leader/follower starts are within 500 m",
                    "The nearest follower begins at least three route hops from the scheduled hazard",
                ] if workload == "collaboration" else []),
                "The individual arm executes tasks serially in one environment and pays inter-task repositioning distance and steps",
            ],
            "individual_arm_execution": (
                "One agent begins at task 1's start, completes its route, travels to "
                "task 2's start, and continues through the manifest. The disaster "
                "continues evolving throughout this serial run; no task or disaster "
                "state is reset between tasks."
            ),
            "primary_causal_comparison": (
                "coordinated_swarm versus independent_multi_agent; both use the "
                "same number of agents, tasks, initial conditions, policy and budget."
            ),
        },
        "aggregate": aggregate,
        "paired_difference_aggregate": paired_aggregate,
        "runs": paired_runs,
    }

    if output_path:
        output_file = os.path.abspath(output_path)
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")

    return result

from dataclasses import replace


def run_pheromone_ablation_pair(
    *, tasks, disaster_type, seed, steps, exposure_event=None
):
    """
    Matched comparison:
      A: full coordinated swarm
      B: same swarm with pheromones disabled

    Both receive identical tasks, seed, disaster and step budget.
    """

    full_comm = _enabled_comm()
    no_pheromone_comm = replace(full_comm, pheromones=False)

    with_pheromones = _run_parallel_arm(
        tasks=tasks,
        disaster_type=disaster_type,
        seed=seed,
        steps=steps,
        comm=full_comm,
        exposure_event=exposure_event,
    )

    without_pheromones = _run_parallel_arm(
        tasks=tasks,
        disaster_type=disaster_type,
        seed=seed,
        steps=steps,
        comm=no_pheromone_comm,
        exposure_event=exposure_event,
    )

    metrics = (
        "success_rate",
        "mean_distance_per_task_m",
        "agent_steps_total",
        "makespan_steps",
    )

    return {
        "with_pheromones": with_pheromones,
        "without_pheromones": without_pheromones,
        "deltas_with_minus_without": {
            metric: (
                with_pheromones.get(metric)
                - without_pheromones.get(metric)
                if with_pheromones.get(metric) is not None
                and without_pheromones.get(metric) is not None
                else None
            )
            for metric in metrics
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Paired individual vs independent vs coordinated swarm baselines"
    )
    parser.add_argument("--type", choices=["earthquake", "flood", "fire", "compound"],
                        default="earthquake")
    parser.add_argument("--workload", choices=["independent", "collaboration"],
                        default="collaboration")
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-dist", type=float, default=1000)
    parser.add_argument("--max-dist", type=float, default=4000)
    parser.add_argument("--output", default="collective_baselines.json")
    args = parser.parse_args()

    result = run_collective_baseline_experiment(
        disaster_type=args.type,
        workload=args.workload,
        n_runs=args.runs,
        num_agents=args.agents,
        steps=args.steps,
        min_dist=args.min_dist,
        max_dist=args.max_dist,
        base_seed=args.seed,
        output_path=args.output,
    )
    print(f"Saved paired baseline results to {os.path.abspath(args.output)}")
    print("\nPaired aggregate:")
    for name, stats in result["paired_difference_aggregate"].items():
        print(f"  {name}: mean={stats['mean']} std={stats['std']} n={stats['n']}")

    if args.workload == "collaboration":
        print("\\nCoordination diagnostics (coordinated arm):")
        aggregate_sources = {
            "self_observation": 0,
            "teammate_message": 0,
            "shared_memory": 0,
            "unattributed": 0,
        }
        aggregate_records = []

        for run in result["runs"]:
            arm = run["arms"]["coordinated_swarm"]
            comm = arm.get("communication", {})
            inst = comm.get("instrumentation", {})
            response = arm.get("hazard_response", {})
            sources = response.get(
                "planner_replan_source_attribution", {}
            )
            records = response.get("planner_replan_source_records", [])
            aggregate_records.extend(records)

            for source in aggregate_sources:
                aggregate_sources[source] += int(
                    sources.get(source, 0)
                )

            print(
                f"  Run {run['run']} seed={run['seed']}: "
                f"tx={comm.get('transmissions', 0)}, "
                f"delivered={comm.get('deliveries', 0)}, "
                f"agents_receiving_teammate_info="
                f"{inst.get('agents_receiving_teammate_information', 0)}, "
                f"route_relevant_board_reads="
                f"{inst.get('shared_memory_route_relevant_cross_agent_retrievals', 0)}, "
                f"successful_hazard_route_changes={len(records)}, "
                f"message_replans={sources.get('teammate_message', 0)}, "
                f"shared_memory_replans={sources.get('shared_memory', 0)}, "
                f"self_replans={sources.get('self_observation', 0)}, "
                f"unattributed={sources.get('unattributed', 0)}"
            )

        print("\\nAggregate successful hazard-linked route changes:")
        for source, count in aggregate_sources.items():
            print(f"  {source}: {count}")
        print(f"  total: {len(aggregate_records)}")

        by_agent = {}
        for record in aggregate_records:
            agent_id = str(record.get("agent", "unknown"))
            source = record.get("awareness_source", "unattributed")
            by_agent.setdefault(agent_id, {
                key: 0 for key in aggregate_sources
            })
            by_agent[agent_id][source] = (
                by_agent[agent_id].get(source, 0) + 1
            )

        print("Per-agent successful route changes:")
        for agent_id in sorted(
            by_agent, key=lambda value: (
                int(value) if value.isdigit() else value
            )
        ):
            print(f"  Agent {agent_id}: {by_agent[agent_id]}")



if __name__ == "__main__":
    main()
