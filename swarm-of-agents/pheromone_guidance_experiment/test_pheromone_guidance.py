"""
Paired pheromone-guidance experiment with unique-node decisions and route-quality
metrics.

Two modes:
  --scenario-set legacy   Re-run the original five Tokyo fork scenarios.
  --scenario-set expanded Run legacy scenarios plus newly sampled scenarios
                           with different forks, branches, goals and disaster
                           placements/severities.

Each scenario/seed is paired: the ON and OFF arms use the same tasks, seed and
disaster configuration. Shared memory and local messaging remain disabled;
pheromones are the only enabled communication mechanism.

This is a mechanism and route-quality experiment, not a city-wide benchmark.
"""

import argparse
import contextlib
import heapq
import io
import json
import math
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_dynamic_disaster import load_fresh_graph
from comms import CommConfig
from swarm_simulation_dynamic import DynamicSwarmSimulation


LEGACY_SCENARIOS = [
    {
        "scenario_id": "legacy_1", "fork": 1041237562,
        "trail_branch": 1041177511, "alternative_branch": 1041237898,
        "goal": 1708899045, "reference_branch_route_m": 646.30,
        "alternative_route_m": 647.83,
    },
    {
        "scenario_id": "legacy_2", "fork": 587663864,
        "trail_branch": 6027742004, "alternative_branch": 6027742006,
        "goal": 1982999577, "reference_branch_route_m": 894.05,
        "alternative_route_m": 911.42,
    },
    {
        "scenario_id": "legacy_3", "fork": 1310455650,
        "trail_branch": 2087559739, "alternative_branch": 1310455869,
        "goal": 2087331588, "reference_branch_route_m": 908.20,
        "alternative_route_m": 920.41,
    },
    {
        "scenario_id": "legacy_4", "fork": 7161252077,
        "trail_branch": 1278061896, "alternative_branch": 1278062088,
        "goal": 1231567976, "reference_branch_route_m": 748.62,
        "alternative_route_m": 763.63,
    },
    {
        "scenario_id": "legacy_5", "fork": 1041794082,
        "trail_branch": 1041794061, "alternative_branch": 1041794209,
        "goal": 1042432713, "reference_branch_route_m": 724.70,
        "alternative_route_m": 724.93,
    },
]


class ScenarioInfeasibleError(RuntimeError):
    pass


def unique_neighbors(graph, node):
    return list(dict.fromkeys(nid for nid, _ in graph._adj.get(node, ())))


def edge_length(graph, u, v):
    lengths = [
        float(data.get("length", 1.0))
        for neighbour, data in graph._adj.get(u, ())
        if neighbour == v
    ]
    if not lengths:
        return None
    return min(lengths)


def shortest_distances_to_goal(graph, goal, max_distance=5000.0):
    """Dijkstra distances on currently passable edges, bounded for speed."""
    distances = {goal: 0.0}
    queue = [(0.0, goal)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances.get(node):
            continue
        if distance > max_distance:
            continue
        for neighbour in unique_neighbors(graph, node):
            if not graph.is_edge_passable(node, neighbour):
                continue
            edge_m = edge_length(graph, node, neighbour)
            if edge_m is None:
                continue
            candidate = distance + edge_m
            if candidate < distances.get(neighbour, float("inf")):
                distances[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    return distances


def path_distance_m(graph, path):
    total = 0.0
    for u, v in zip(path, path[1:]):
        length = edge_length(graph, u, v)
        if length is None:
            raise RuntimeError(f"Missing graph edge in executed path: {u} -> {v}")
        total += length
    return total


def generate_scenarios(graph, count=25, seed=2026):
    """Sample distinct feasible-looking forks/goals and paired hazard settings.

    Candidate scenarios are checked on the undamaged graph. The simulation's
    own post-disaster feasibility validation remains authoritative.
    """
    rng = random.Random(seed)
    node_ids = list(graph._nodes)
    if len(node_ids) < 100:
        raise RuntimeError("Graph is too small to generate varied scenarios")

    scenarios = []
    used_forks = set()
    used_pairs = set()
    goal_pool = rng.sample(node_ids, min(max(count * 3, 60), len(node_ids)))
    disaster_types = ["earthquake", "flood", "fire"]

    for goal in goal_pool:
        if len(scenarios) >= count:
            break
        distances = shortest_distances_to_goal(graph, goal, max_distance=3500.0)
        fork_candidates = [
            node for node in distances
            if node not in used_forks
            and 450.0 <= distances[node] <= 2600.0
            and len(unique_neighbors(graph, node)) >= 3
        ]
        rng.shuffle(fork_candidates)

        for fork in fork_candidates[:2500]:
            if len(scenarios) >= count:
                break
            branch_options = []
            for branch in unique_neighbors(graph, fork):
                if not graph.is_edge_passable(fork, branch):
                    continue
                branch_dist = distances.get(branch)
                edge_m = edge_length(graph, fork, branch)
                if branch_dist is None or edge_m is None:
                    continue
                total = edge_m + branch_dist
                if 400.0 <= total <= 2800.0:
                    branch_options.append((branch, total))

            if len(branch_options) < 2:
                continue

            pairs = []
            for i in range(len(branch_options)):
                for j in range(i + 1, len(branch_options)):
                    a, da = branch_options[i]
                    b, db = branch_options[j]
                    # Require two genuine, distinct first-hop alternatives.
                    if a != b and abs(da - db) >= 8.0:
                        pairs.append((a, da, b, db))
            if not pairs:
                continue

            a, da, b, db = rng.choice(pairs)
            if rng.random() < 0.5:
                trail, trail_d, alternative, alternative_d = a, da, b, db
            else:
                trail, trail_d, alternative, alternative_d = b, db, a, da

            pair_key = (fork, goal)
            if pair_key in used_pairs:
                continue

            node_data = graph._nodes[fork]
            lat, lon = float(node_data["lat"]), float(node_data["lon"])
            bearing = rng.uniform(0.0, 2.0 * math.pi)
            offset_m = rng.uniform(450.0, 950.0)
            centre_lat = lat + (math.sin(bearing) * offset_m / 111320.0)
            lon_scale = max(0.2, math.cos(math.radians(lat)))
            centre_lon = lon + (
                math.cos(bearing) * offset_m / (111320.0 * lon_scale)
            )

            scenario_id = f"generated_{len(scenarios) + 1:03d}"
            scenario = {
                "scenario_id": scenario_id,
                "fork": fork,
                "trail_branch": trail,
                "alternative_branch": alternative,
                "goal": goal,
                "reference_branch_route_m": round(trail_d, 3),
                "alternative_route_m": round(alternative_d, 3),
                "disaster_type": rng.choice(disaster_types),
                "disaster_centre": [round(centre_lat, 7), round(centre_lon, 7)],
                "disaster_radius_m": round(rng.uniform(100.0, 180.0), 1),
                "disaster_severity": round(rng.uniform(0.35, 0.65), 2),
                "wind_bearing_deg": round(rng.uniform(0.0, 360.0), 1),
                "scenario_generation_seed": seed,
            }
            scenarios.append(scenario)
            used_forks.add(fork)
            used_pairs.add(pair_key)
            break

    if len(scenarios) < count:
        raise RuntimeError(
            f"Could generate only {len(scenarios)}/{count} distinct scenarios. "
            "Try a smaller --generated-scenarios value."
        )
    return scenarios


def _decision_at_fork(sim, fork_node, follower_idx=1):
    for event in sim._navigation_decisions:
        if (
            event.get("agent") == follower_idx
            and event.get("from_node") == fork_node
            and event.get("policy_type") == "default_snn"
            and event.get("decision_diagnostics")
        ):
            diagnostic = event["decision_diagnostics"]
            return {
                "step": event.get("step"),
                "from_node": fork_node,
                "selected_node": event.get("next_node"),
                "diagnostics_available": True,
                **diagnostic,
            }
    return None


def run_arm(scenario, seed, steps, pheromones):
    """Run one paired arm and separate route quality from information transfer."""
    with contextlib.redirect_stdout(io.StringIO()):
        graph = load_fresh_graph()
        comm = CommConfig(
            shared_memory=False,
            local_messaging=False,
            pheromones=pheromones,
        )
        sim = DynamicSwarmSimulation(
            graph=graph,
            num_agents=2,
            seed=seed,
            tasks=[
                {"task_id": "leader", "start": scenario["trail_branch"],
                 "goal": scenario["goal"]},
                {"task_id": "follower", "start": scenario["fork"],
                 "goal": scenario["goal"]},
            ],
            require_feasible=True,
            comm=comm,
            roles=None,
            disaster_type=scenario.get("disaster_type", "earthquake"),
            disaster_centre=tuple(scenario.get("disaster_centre", (35.68, 139.76))),
            disaster_radius_m=scenario.get("disaster_radius_m", 2000.0),
            disaster_severity=scenario.get("disaster_severity", 0.7),
            wind_bearing_deg=scenario.get("wind_bearing_deg", 45.0),
        )

        # Reference shortest path is measured after the initial disaster and
        # before the agents move. Later dynamic events can make the realised
        # path longer; the reference is therefore explicitly an initial-state
        # benchmark, not an oracle for the evolving environment.
        initial_distances = shortest_distances_to_goal(
            sim.graph, scenario["goal"], max_distance=10000.0
        )
        initial_shortest_m = initial_distances.get(scenario["fork"])
        trail_branch_shortest_m = initial_distances.get(scenario["trail_branch"])
        alternative_branch_shortest_m = initial_distances.get(
            scenario["alternative_branch"]
        )
        sim.run(steps=steps)

    follower = sim.agents[1]
    follower_path = list(follower.path)
    distance_m = path_distance_m(sim.graph, follower_path)
    fork_decision = _decision_at_fork(sim, scenario["fork"], follower_idx=1)

    read_events = [
        event for event in getattr(sim, "_shared_memory_reads", [])
        if event.get("channel") == "pheromones"
        and event.get("reader_agent") == 1
        and event.get("cross_agent")
    ]
    retrieval_steps = [event["retrieval_step"] for event in read_events]
    retrieval_delays = [
        event["retrieval_delay_steps"] for event in read_events
        if event.get("retrieval_delay_steps") is not None
    ]

    if initial_shortest_m and initial_shortest_m > 0:
        route_efficiency_ratio = distance_m / initial_shortest_m
        route_excess_m = distance_m - initial_shortest_m
    else:
        route_efficiency_ratio = None
        route_excess_m = None

    return {
        "success": bool(follower.reached_goal),
        "death_reason": getattr(follower, "death_reason", None),
        "path_nodes": len(follower_path),
        "distance_m": round(distance_m, 3),
        "first_hop": follower_path[1] if len(follower_path) > 1 else None,
        "path": follower_path,
        "initial_shortest_distance_m": (
            round(initial_shortest_m, 3) if initial_shortest_m is not None else None
        ),
        "initial_shortest_trail_branch_to_goal_m": (
            round(trail_branch_shortest_m, 3)
            if trail_branch_shortest_m is not None else None
        ),
        "initial_shortest_alternative_branch_to_goal_m": (
            round(alternative_branch_shortest_m, 3)
            if alternative_branch_shortest_m is not None else None
        ),
        "route_efficiency_ratio_vs_initial_shortest": (
            round(route_efficiency_ratio, 4)
            if route_efficiency_ratio is not None else None
        ),
        "route_excess_vs_initial_shortest_m": (
            round(route_excess_m, 3) if route_excess_m is not None else None
        ),
        "cross_agent_pheromone_reads": len(read_events),
        "pheromone_information": {
            "cross_agent_read_count": len(read_events),
            "unique_teammate_nodes_read": len({
                event.get("node_id") for event in read_events
            }),
            "first_retrieval_step": min(retrieval_steps) if retrieval_steps else None,
            "mean_retrieval_delay_steps": (
                round(statistics.mean(retrieval_delays), 3)
                if retrieval_delays else None
            ),
            "route_relevant_read_count": sum(
                bool(event.get("route_relevant")) for event in read_events
            ),
        },
        "fork_decision": fork_decision,
        "navigation_decision_count": sum(
            event.get("agent") == 1
            and event.get("policy_type") == "default_snn"
            for event in sim._navigation_decisions
        ),
    }


def _candidate_probability(decision, node, key="probability"):
    if not decision:
        return None
    for candidate in decision.get("candidates", []):
        if candidate.get("node") == node:
            return candidate.get(key)
    return None


def _decision_information(record, scenario):
    on = record["pheromones_on"].get("fork_decision")
    off = record["pheromones_off"].get("fork_decision")
    trail = scenario["trail_branch"]
    alternative = scenario["alternative_branch"]
    trail_actual = _candidate_probability(on, trail, "probability")
    trail_cf = _candidate_probability(on, trail, "probability_without_pheromone")
    alternative_actual = _candidate_probability(on, alternative, "probability")
    return {
        "on_fork_decision_available": on is not None,
        "off_fork_decision_available": off is not None,
        "trail_probability_on": trail_actual,
        "trail_probability_without_pheromone_counterfactual": trail_cf,
        "trail_probability_effect": (
            trail_actual - trail_cf
            if trail_actual is not None and trail_cf is not None else None
        ),
        "alternative_probability_on": alternative_actual,
        "trail_is_best_scoring_on": (
            on.get("best_node") == trail if on else None
        ),
        "best_node_without_pheromone_on": (
            on.get("best_node_without_pheromone") if on else None
        ),
        "pheromone_changed_score_argmax": (
            on.get("pheromone_changed_score_argmax") if on else None
        ),
    }


def run_experiment(
    runs=5,
    base_seed=42,
    steps=120,
    output="pheromone_guidance_results.json",
    scenario_set="legacy",
    generated_scenarios=25,
    scenario_generation_seed=2026,
):
    graph_for_generation = None
    scenarios = list(LEGACY_SCENARIOS)
    if scenario_set == "expanded":
        graph_for_generation = load_fresh_graph()
        scenarios.extend(generate_scenarios(
            graph_for_generation,
            count=generated_scenarios,
            seed=scenario_generation_seed,
        ))
    elif scenario_set != "legacy":
        raise ValueError("scenario_set must be 'legacy' or 'expanded'")

    records = []
    rejected_scenarios = []
    for scenario in scenarios:
        for run_idx in range(runs):
            seed = base_seed + run_idx * 42
            try:
                on = run_arm(scenario, seed, steps, pheromones=True)
                off = run_arm(scenario, seed, steps, pheromones=False)
            except Exception as exc:
                # Paired design: reject the entire pair if either arm cannot
                # instantiate/run this generated scenario under the same setup.
                rejected_scenarios.append({
                    "scenario_id": scenario["scenario_id"],
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                print(
                    f"REJECTED scenario={scenario['scenario_id']} seed={seed}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            record = {
                "scenario_id": scenario["scenario_id"],
                "seed": seed,
                "trail_branch": scenario["trail_branch"],
                "alternative_branch": scenario["alternative_branch"],
                "pheromones_on": on,
                "pheromones_off": off,
                "paths_equal": on["path"] == off["path"],
                "on_followed_trail_first_hop": (
                    on["first_hop"] == scenario["trail_branch"]
                ),
                "off_followed_alternative_first_hop": (
                    off["first_hop"] == scenario["alternative_branch"]
                ),
            }
            record["information_vs_route_effect"] = _decision_information(
                record, scenario
            )
            records.append(record)

            print(
                f"scenario={scenario['scenario_id']} seed={seed} | "
                f"ON success={int(on['success'])} distance={on['distance_m']:.1f}m "
                f"trail_first={int(record['on_followed_trail_first_hop'])} | "
                f"OFF success={int(off['success'])} distance={off['distance_m']:.1f}m "
                f"alternative_first={int(record['off_followed_alternative_first_hop'])}"
            )

    if not records:
        raise RuntimeError("No paired runs completed successfully")

    on_distances = [r["pheromones_on"]["distance_m"] for r in records]
    off_distances = [r["pheromones_off"]["distance_m"] for r in records]
    on_successes = sum(r["pheromones_on"]["success"] for r in records)
    off_successes = sum(r["pheromones_off"]["success"] for r in records)
    on_trail_choices = sum(r["on_followed_trail_first_hop"] for r in records)
    off_alt_choices = sum(r["off_followed_alternative_first_hop"] for r in records)
    total_reads = sum(
        r["pheromones_on"]["pheromone_information"]["cross_agent_read_count"]
        for r in records
    )
    changed_paths = sum(not r["paths_equal"] for r in records)

    def mean_or_none(values):
        return round(statistics.mean(values), 4) if values else None

    on_efficiency = [
        r["pheromones_on"]["route_efficiency_ratio_vs_initial_shortest"]
        for r in records
        if r["pheromones_on"]["route_efficiency_ratio_vs_initial_shortest"] is not None
    ]
    off_efficiency = [
        r["pheromones_off"]["route_efficiency_ratio_vs_initial_shortest"]
        for r in records
        if r["pheromones_off"]["route_efficiency_ratio_vs_initial_shortest"] is not None
    ]
    on_trail_prob_effect = [
        r["information_vs_route_effect"]["trail_probability_effect"]
        for r in records
        if r["information_vs_route_effect"]["trail_probability_effect"] is not None
    ]

    def decision_metric(arm, metric):
        values = [
            run[arm]["fork_decision"].get(metric)
            for run in records
            if run[arm].get("fork_decision")
            and run[arm]["fork_decision"].get(metric) is not None
        ]
        return mean_or_none(values)

    decision_summary = {
        "on_fork_decisions_available": sum(
            bool(r["pheromones_on"].get("fork_decision")) for r in records
        ),
        "off_fork_decisions_available": sum(
            bool(r["pheromones_off"].get("fork_decision")) for r in records
        ),
        "on_mean_best_candidate_probability": decision_metric(
            "pheromones_on", "best_candidate_probability"
        ),
        "off_mean_best_candidate_probability": decision_metric(
            "pheromones_off", "best_candidate_probability"
        ),
        "on_mean_entropy_nats": decision_metric(
            "pheromones_on", "probability_entropy_nats"
        ),
        "off_mean_entropy_nats": decision_metric(
            "pheromones_off", "probability_entropy_nats"
        ),
        "on_mean_unique_node_score_gap": decision_metric(
            "pheromones_on", "score_gap_best_second"
        ),
        "off_mean_unique_node_score_gap": decision_metric(
            "pheromones_off", "score_gap_best_second"
        ),
        "on_mean_candidate_count": decision_metric(
            "pheromones_on", "candidate_count"
        ),
        "off_mean_candidate_count": decision_metric(
            "pheromones_off", "candidate_count"
        ),
        "mean_adjacency_duplicates_removed": mean_or_none([
            run["pheromones_on"]["fork_decision"].get(
                "adjacency_duplicate_count", 0
            )
            for run in records if run["pheromones_on"].get("fork_decision")
        ]),
    }

    scenario_ids = list(dict.fromkeys(r["scenario_id"] for r in records))
    per_scenario = []
    for scenario_id in scenario_ids:
        subset = [r for r in records if r["scenario_id"] == scenario_id]
        on_mean = statistics.mean(r["pheromones_on"]["distance_m"] for r in subset)
        off_mean = statistics.mean(r["pheromones_off"]["distance_m"] for r in subset)
        per_scenario.append({
            "scenario_id": scenario_id,
            "runs": len(subset),
            "pheromones_on_mean_distance_m": round(on_mean, 3),
            "pheromones_off_mean_distance_m": round(off_mean, 3),
            "distance_difference_on_minus_off_m": round(on_mean - off_mean, 3),
            "on_successes": sum(r["pheromones_on"]["success"] for r in subset),
            "off_successes": sum(r["pheromones_off"]["success"] for r in subset),
            "on_trail_first_hop_count": sum(r["on_followed_trail_first_hop"] for r in subset),
            "off_alternative_first_hop_count": sum(r["off_followed_alternative_first_hop"] for r in subset),
        })

    on_mean = statistics.mean(on_distances)
    off_mean = statistics.mean(off_distances)
    summary = {
        "design": {
            "navigation_policy": "default_snn_potential_field",
            "planner_assisted_navigation": False,
            "shared_memory": False,
            "local_messaging": False,
            "pheromones_are_only_enabled_communication_mechanism": True,
            "scenario_set": scenario_set,
            "scenario_count_requested": len(scenarios),
            "scenario_count_observed": len(scenario_ids),
            "runs_per_scenario_requested": runs,
            "paired_runs_completed": len(records),
            "paired_runs_rejected": len(rejected_scenarios),
            "steps": steps,
            "base_seed": base_seed,
            "scenario_generation_seed": (
                scenario_generation_seed if scenario_set == "expanded" else None
            ),
            "route_quality_reference": (
                "initial shortest path after t=0 disaster, before movement"
            ),
        },
        "aggregate": {
            "pheromones_on_successes": on_successes,
            "pheromones_off_successes": off_successes,
            "total_paired_runs": len(records),
            "pheromones_on_mean_follower_distance_m": round(on_mean, 3),
            "pheromones_off_mean_follower_distance_m": round(off_mean, 3),
            "distance_difference_on_minus_off_m": round(on_mean - off_mean, 3),
            "distance_difference_percent_of_off": (
                round(100.0 * (on_mean - off_mean) / off_mean, 3)
                if off_mean else None
            ),
            "on_followed_leader_trail_first_hop": on_trail_choices,
            "off_followed_alternative_first_hop": off_alt_choices,
            "mean_cross_agent_pheromone_reads": round(total_reads / len(records), 3),
            "paths_changed_between_arms": changed_paths,
            "on_mean_route_efficiency_ratio": mean_or_none(on_efficiency),
            "off_mean_route_efficiency_ratio": mean_or_none(off_efficiency),
            "mean_trail_probability_effect_from_pheromone": mean_or_none(on_trail_prob_effect),
            "decision_diagnostics": decision_summary,
        },
        "per_scenario": per_scenario,
        "scenario_definitions": scenarios,
        "rejected_pairs": rejected_scenarios,
        "runs": records,
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== PHEROMONE GUIDANCE SUMMARY ===")
    print(f"Scenario set: {scenario_set} | scenarios generated/requested: {len(scenarios)}")
    print(f"Paired runs completed: {len(records)} | rejected pairs: {len(rejected_scenarios)}")
    print(f"Success: ON {on_successes}/{len(records)} | OFF {off_successes}/{len(records)}")
    print(f"Mean distance: ON {on_mean:.2f}m | OFF {off_mean:.2f}m")
    print(f"Distance difference (ON - OFF): {on_mean - off_mean:+.2f}m")
    print(f"Trail first hop: ON {on_trail_choices}/{len(records)}")
    print(f"Alternative first hop: OFF {off_alt_choices}/{len(records)}")
    print(f"Mean cross-agent pheromone reads: {total_reads / len(records):.2f}")
    print(f"Mean route-efficiency ratio: ON {mean_or_none(on_efficiency)} | OFF {mean_or_none(off_efficiency)}")
    print(f"Mean pheromone effect on trail probability: {mean_or_none(on_trail_prob_effect)}")
    print(
        "Unique-node diagnostics: "
        f"ON best-p={decision_summary['on_mean_best_candidate_probability']} "
        f"entropy={decision_summary['on_mean_entropy_nats']} "
        f"gap={decision_summary['on_mean_unique_node_score_gap']} | "
        f"OFF best-p={decision_summary['off_mean_best_candidate_probability']} "
        f"entropy={decision_summary['off_mean_entropy_nats']} "
        f"gap={decision_summary['off_mean_unique_node_score_gap']}"
    )
    print(f"Saved: {output_path.resolve()}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paired pheromone-guidance experiment with route-quality diagnostics"
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--output", default="pheromone_guidance_results.json")
    parser.add_argument(
        "--scenario-set", choices=("legacy", "expanded"), default="legacy",
        help="legacy replays original five scenarios; expanded adds generated scenarios",
    )
    parser.add_argument("--generated-scenarios", type=int, default=25)
    parser.add_argument("--scenario-generation-seed", type=int, default=2026)
    args = parser.parse_args()
    run_experiment(
        runs=args.runs,
        base_seed=args.seed,
        steps=args.steps,
        output=args.output,
        scenario_set=args.scenario_set,
        generated_scenarios=args.generated_scenarios,
        scenario_generation_seed=args.scenario_generation_seed,
    )
