"""
test_dynamic_disaster.py
------------------------
Test harness for the dynamic disaster swarm experiments.

Modes
-----
1. run_single()
   One simulation with a chosen disaster type. Good for debugging and
   quick inspection of agent behaviour.

2. run_experiment()
   n_runs independent simulations of one disaster type. Aggregates
   success rate, path length, and genuine planner response events.
   Synthetic reroute and stall-counter adaptation metrics are intentionally
   not reported.

3. run_comparative_study()
   Runs run_experiment() for ALL four disaster types
   (earthquake, flood, fire, compound) under identical conditions and
   prints a side-by-side comparison table.
   This is the main experiment for the paper.

Usage
-----
    python test_dynamic_disaster.py
    python test_dynamic_disaster.py --mode single   --type fire      --agents 10 --steps 200
    python test_dynamic_disaster.py --mode experiment --type compound --runs 5
    python test_dynamic_disaster.py --mode compare  --agents 10 --steps 200 --runs 5
    python test_dynamic_disaster.py --mode paired --workload collaboration --agents 2 --steps 100 --runs 1 --seed 42
    python test_dynamic_disaster.py --mode delay --workload collaboration_local --agents 2 --steps 160 --runs 10 --seed 42 --delays 0 1 2 3 --output communication_delay_results.json
"""

import sys
import os
import random
import argparse
import math
import contextlib
import io
import statistics
import hashlib
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
# Prefer this experiment package over similarly named files in the repo root.
# This prevents tests from silently importing an older simulation module.
sys.path.insert(0, root_dir)
sys.path.insert(0, script_dir)

import json
from dataclasses import asdict
from collective_metrics import collective_metrics_from_sim, aggregate_numeric_metrics

from feasibility import (
    generate_tasks,
    generate_shared_shelter_tasks,
    passable_components, candidate_nodes, sample_feasible_task,
    validate_task, SpawnError,
)
from dynamic_disaster_engine import DisasterEvent, DynamicDisasterEngine
from comms import CommConfig, build_delay_matrix_configs

from hybrid_spatial_index import HybridTokyoGraph, load_cached
from swarm_simulation_dynamic import DynamicSwarmSimulation
from snn_core_v2 import haversine_distance

# ---------------------------------------------------------------------------
# Graph path — adjust to your local setup
# ---------------------------------------------------------------------------

def _find_graph_path() -> str:
    """TOKYO_GRAPH env var wins; otherwise look in the usual places."""
    candidates = [
        os.environ.get("TOKYO_GRAPH"),
        os.path.join(script_dir, "tokyo_graph.json"),
        os.path.join(script_dir, "..", "dataset", "tokyo_graph.json"),
        os.path.join(script_dir, "..", "dataset", "tokyo_graph", "tokyo_graph.json"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "tokyo_graph.json not found. Unzip dataset/tokyo_graph.zip next to this "
        "script or set the TOKYO_GRAPH environment variable.")

GRAPH_PATH = _find_graph_path()


def _paired_seed(base_seed, run_idx):
    """Derive paired-run seeds while retaining the historical 42-step spacing."""
    return (0 if base_seed is None else base_seed) + run_idx * 42

# ---------------------------------------------------------------------------
# Shared disaster parameters — same across all experiment modes so results
# are directly comparable
# ---------------------------------------------------------------------------
DISASTER_CENTRE   = (35.68, 139.76)
DISASTER_RADIUS_M = 2000
DISASTER_SEVERITY = 0.7
WIND_BEARING      = 45.0   # NE wind for fire scenarios


# ---------------------------------------------------------------------------
# Graph loader
# ---------------------------------------------------------------------------

def load_fresh_graph() -> "HybridTokyoGraph":
    """
    Undamaged graph for one run. The file is parsed and indexed once per
    process; each call returns an independent fork, so damage cannot leak
    between runs and there is no ~17 s reload per run.
    """
    return load_cached(GRAPH_PATH)


# ---------------------------------------------------------------------------
# Workload generation and communication configurations
# ---------------------------------------------------------------------------

DYNAMIC_EXPOSURE_STEP = 20
DYNAMIC_EXPOSURE_RADIUS_M = 250.0
DYNAMIC_EXPOSURE_SEVERITY = 0.9


def _generate_workload_tasks(
    graph,
    num_agents,
    seed,
    workload,
    min_dist,
    max_dist,
):
    if workload == "independent":
        return generate_tasks(
            graph,
            n=num_agents,
            seed=seed,
            min_dist=min_dist,
            max_dist=max_dist,
        )
    if workload == "shared_shelter":
        return generate_shared_shelter_tasks(
            graph,
            n=num_agents,
            seed=seed,
            min_dist=min_dist,
            max_dist=max_dist,
        )
    raise ValueError(
        f"Unknown base workload '{workload}'. "
        "Choose 'independent' or 'shared_shelter'."
    )


class DynamicHazardExposureEngine(DynamicDisasterEngine):
    """Existing disaster engine plus one deterministic post-start event."""

    def __init__(self, *args, exposure_event, **kwargs):
        super().__init__(*args, **kwargs)
        self.exposure_event = dict(exposure_event)
        self.exposure_event_applied = False

    def tick(self, step):
        events = super().tick(step)
        if (self.exposure_event_applied
                or step != self.exposure_event["trigger_step"]):
            return events

        before_damage = dict(self.graph._node_damage)
        before_edges = set(self.graph._disabled_edges)
        summary = self.graph.apply_disaster(
            self.exposure_event["disaster_type"],
            tuple(self.exposure_event["centre"]),
            self.exposure_event["radius_m"],
            self.exposure_event["severity"],
        )
        changed_nodes = [
            node_id for node_id, damage in self.graph._node_damage.items()
            if damage > before_damage.get(node_id, 0.0)
        ]
        changed_edges = len(self.graph._disabled_edges - before_edges)
        event = DisasterEvent(
            step=step,
            event_type="dynamic_hazard_exposure",
            centre=tuple(self.exposure_event["centre"]),
            radius_m=self.exposure_event["radius_m"],
            severity=self.exposure_event["severity"],
            nodes_newly_damaged=changed_nodes,
            edges_newly_disabled=changed_edges,
            description=(
                f"Reference-route exposure: {summary['nodes_affected']} nodes, "
                f"{changed_edges} new edges"
            ),
        )
        self.events.append(event)
        self.exposure_event_applied = True
        return events + [event]


class CollaborationHazardEngine(DynamicDisasterEngine):
    """Controlled collaboration event: initial disaster plus one scheduled event.

    Unlike the continuous dynamic-exposure engine, this workload suppresses
    subsequent background spread so communication effects can be isolated from
    unrelated later hazard growth.
    """

    def __init__(self, *args, exposure_event, **kwargs):
        super().__init__(*args, **kwargs)
        self.exposure_event = dict(exposure_event)
        self.exposure_event_applied = False

    def tick(self, step):
        if (self.exposure_event_applied
                or step != self.exposure_event["trigger_step"]):
            return []

        before_damage = dict(self.graph._node_damage)
        before_edges = set(self.graph._disabled_edges)
        summary = self.graph.apply_disaster(
            self.exposure_event["disaster_type"],
            tuple(self.exposure_event["centre"]),
            self.exposure_event["radius_m"],
            self.exposure_event["severity"],
        )
        changed_nodes = [
            node_id for node_id, damage in self.graph._node_damage.items()
            if damage > before_damage.get(node_id, 0.0)
        ]
        changed_edges = len(self.graph._disabled_edges - before_edges)
        event = DisasterEvent(
            step=step,
            event_type="dynamic_hazard_exposure",
            centre=tuple(self.exposure_event["centre"]),
            radius_m=self.exposure_event["radius_m"],
            severity=self.exposure_event["severity"],
            nodes_newly_damaged=changed_nodes,
            edges_newly_disabled=changed_edges,
            description=(
                f"Controlled collaboration hazard: "
                f"{summary['nodes_affected']} nodes, {changed_edges} new edges"
            ),
        )
        self.events.append(event)
        self.exposure_event_applied = True
        return [event]


def _add_task_ids(tasks):
    for i, task in enumerate(tasks):
        task["task_id"] = f"task_{i:03d}"
    return tasks


def generate_dynamic_hazard_exposure_scenario(
    graph,
    num_agents,
    disaster_type,
    seed,
    base_workload="independent",
    min_dist=1000,
    max_dist=4000,
):
    """Create feasible tasks and a reproducible event over a reference route."""
    initial_engine = DynamicDisasterEngine(
        graph=graph,
        disaster_type=disaster_type,
        centre=DISASTER_CENTRE,
        radius_m=DISASTER_RADIUS_M,
        severity=DISASTER_SEVERITY,
        seed=seed,
        wind_bearing_deg=WIND_BEARING,
    )
    initial_engine.initialise()
    tasks = _add_task_ids(_generate_workload_tasks(
        graph, num_agents, seed, base_workload, min_dist, max_dist))

    reference_routes = []
    for task in tasks:
        route = graph.shortest_path(task["start"], task["goal"], avoid_damaged=True)
        if not route:
            raise RuntimeError(
                f"No reference route for {task['task_id']} after initial disaster."
            )
        reference_routes.append(route)

    # Use the longest deterministic reference route so the event has a clear,
    # reproducible opportunity to affect navigation without forcing traversal.
    reference_index = max(
        range(len(reference_routes)),
        key=lambda i: (len(reference_routes[i]), -i),
    )
    reference_route = reference_routes[reference_index]
    route_index = len(reference_route) // 2
    anchor_node = reference_route[route_index]
    anchor = graph._nodes[anchor_node]
    exposure_event = {
        "event_type": "dynamic_hazard_exposure",
        "disaster_type": disaster_type,
        "trigger_step": DYNAMIC_EXPOSURE_STEP,
        "centre": (anchor["lat"], anchor["lon"]),
        "radius_m": DYNAMIC_EXPOSURE_RADIUS_M,
        "severity": DYNAMIC_EXPOSURE_SEVERITY,
        "reference_task_id": tasks[reference_index]["task_id"],
        "reference_route_index": route_index,
        "reference_route": list(reference_route),
    }
    return tasks, exposure_event

def _passable_shortest_path(graph, start, goal, threshold=0.6):
    """Deterministic distance-shortest route using the agent's passability rule."""
    import heapq

    if start == goal:
        return [start]
    distance = {start: 0.0}
    parent = {}
    heap = [(0.0, start)]
    while heap:
        cost, node = heapq.heappop(heap)
        if cost != distance.get(node):
            continue
        if node == goal:
            route = [goal]
            while route[-1] != start:
                route.append(parent[route[-1]])
            return list(reversed(route))
        for neighbour, edge in graph._adj.get(node, ()):
            if not graph.is_edge_passable(node, neighbour):
                continue
            if graph.get_node_damage(neighbour) > threshold:
                continue
            next_cost = cost + float(edge["length"])
            if next_cost < distance.get(neighbour, float("inf")):
                distance[neighbour] = next_cost
                parent[neighbour] = node
                heapq.heappush(heap, (next_cost, neighbour))
    return None


def build_collaboration_tasks_from_route(
    graph, reference_route, num_agents, lead_gap=None,
    min_leader_separation_m=600.0,
    max_leader_separation_m=None,
):
    """Build a convoy of feasible tasks with one leader and shared route suffix.

    Agent 0 starts ahead of the followers, one hop before the target hazard.
    Remaining agents start at distinct, progressively advanced positions on
    the same reference route. Every independently computed passable route must
    contain the target node and converge on the same shelter.
    """
    if num_agents < 2:
        raise ValueError("The collaboration workload requires at least two agents.")
    route = list(reference_route or ())
    lead_gap = int(lead_gap if lead_gap is not None else num_agents + 7)
    hazard_index = lead_gap + 1
    if len(route) <= hazard_index + 2:
        raise SpawnError(
            f"Reference route needs at least {hazard_index + 3} nodes; "
            f"received {len(route)}."
        )

    starts = [route[lead_gap]] + [route[i] for i in range(num_agents - 1)]
    if len(set(starts)) != num_agents:
        raise SpawnError("Collaboration starts are not distinct.")

    goal = route[-1]
    hazard_node = route[hazard_index]
    leader_data = graph._nodes[starts[0]]
    nearest_follower_data = graph._nodes[starts[-1]]
    leader_separation = haversine_distance(
        leader_data["lat"], leader_data["lon"],
        nearest_follower_data["lat"], nearest_follower_data["lon"],
    )
    if leader_separation <= min_leader_separation_m:
        raise SpawnError(
            "Leader is too close to the nearest follower for the requested "
            "communication-range scenario."
        )
    if (max_leader_separation_m is not None
            and leader_separation > max_leader_separation_m):
        raise SpawnError(
            "Leader is outside the maximum separation for the requested "
            "local-communication scenario."
        )

    comp = passable_components(graph)
    tasks = []
    follower_routes = []
    for i, start in enumerate(starts):
        info = validate_task(graph, start, goal, comp=comp)
        route_i = _passable_shortest_path(graph, start, goal)
        if info is None or not route_i or hazard_node not in route_i:
            raise SpawnError(
                f"Agent {i}'s feasible route does not contain the shared hazard node."
            )
        follower_routes.append(route_i)
        tasks.append({
            **info,
            "task_id": f"task_{i:03d}",
            "start": start,
            "goal": goal,
        })

    common_nodes = set(follower_routes[0])
    for route_i in follower_routes[1:]:
        common_nodes.intersection_update(route_i)
    if hazard_node not in common_nodes or goal not in common_nodes:
        raise SpawnError("The generated tasks do not share the hazard-to-shelter route.")

    return tasks, {
        "leader_agent": 0,
        "follower_agents": list(range(1, num_agents)),
        "shared_shelter": goal,
        "hazard_node": hazard_node,
        "hazard_route_index": hazard_index,
        "follower_hazard_hops_at_start": min(
            route_i.index(hazard_node) for route_i in follower_routes[1:]
        ),
        "leader_start": starts[0],
        "follower_starts": starts[1:],
        "leader_to_nearest_follower_distance_m": round(leader_separation, 3),
        "communication_radius_m": CommConfig().radius_m,
        "reference_route": route,
        "common_route_nodes": sorted(common_nodes),
        "common_route_node_count": len(common_nodes),
        "route_overlap_fraction": round(
            len(common_nodes) / max(len(r) for r in follower_routes), 4
        ),
        "event_trigger_step": 1,
        "navigation_policy": "planner_assisted_navigator",
    }


def generate_collaboration_scenario(
    graph, num_agents, disaster_type, seed,
    min_dist=1000, max_dist=4000, max_attempts=80,
    local_communication=False,
):
    """Generate a feasible convoy with either local-radio or long-range spacing.

    In local-radio mode the leader starts one hop before the scheduled hazard;
    all followers start upstream, with the nearest follower within radio range
    but at least three route hops from the hazard. This supports teams larger
    than two while preserving an information-transfer opportunity.
    """
    if num_agents < 2:
        raise ValueError("The collaboration workload requires at least two agents.")
    if local_communication and num_agents < 2:
        raise ValueError(
            "The local-communication collaboration workload requires at least "
            "two agents."
        )

    initial_engine = DynamicDisasterEngine(
        graph=graph,
        disaster_type=disaster_type,
        centre=DISASTER_CENTRE,
        radius_m=DISASTER_RADIUS_M,
        severity=DISASTER_SEVERITY,
        seed=seed,
        wind_bearing_deg=WIND_BEARING,
    )
    initial_engine.initialise()

    comp = passable_components(graph)
    cands = candidate_nodes(graph, comp)
    search_max_dist = (
        float(max_dist) if local_communication
        else max(float(max_dist), 8000.0)
    )
    # Local mode places the leader one route hop before the event and the
    # nearest follower two route hops behind the leader. With N agents, the
    # leader starts at route[N] and followers occupy route[0:N-1], so all
    # starts remain distinct and every follower is still upstream of the event.
    # Long-range mode preserves the original > radio-range shared-board design.
    lead_gap = num_agents if local_communication else num_agents + 7
    min_separation = (
        0.0 if local_communication else CommConfig().radius_m + 100.0
    )
    max_separation = CommConfig().radius_m if local_communication else None

    for attempt in range(max_attempts):
        rng = random.Random(f"{seed}:collaboration:{attempt}")
        try:
            base = sample_feasible_task(
                graph, rng, comp, cands,
                min_dist=min_dist,
                max_dist=search_max_dist,
                max_attempts=500,
            )
            reference_route = _passable_shortest_path(
                graph, base["start"], base["goal"]
            )
            tasks, scenario = build_collaboration_tasks_from_route(
                graph, reference_route, num_agents,
                lead_gap=(num_agents if local_communication else lead_gap),
                min_leader_separation_m=min_separation,
                max_leader_separation_m=max_separation,
            )
        except SpawnError:
            continue

        hazard_node = scenario["hazard_node"]
        hazard_data = graph._nodes[hazard_node]
        centre = (hazard_data["lat"], hazard_data["lon"])
        radius_m = 10.0
        severity = 0.9

        # Verify that this exact event blocks the target but leaves every
        # evacuation task feasible through an alternative route. Restore the
        # pre-event overlay before returning the scenario.
        before_damage = dict(graph._node_damage)
        before_edges = set(graph._disabled_edges)
        try:
            graph.apply_disaster(disaster_type, centre, radius_m, severity)
            target_blocked = graph.get_node_damage(hazard_node) > 0.6
            alternatives_exist = target_blocked and all(
                _passable_shortest_path(graph, task["start"], task["goal"])
                for task in tasks
            )
        finally:
            graph._node_damage = before_damage
            graph._disabled_edges = before_edges

        if not alternatives_exist:
            continue

        exposure_event = {
            "event_type": "dynamic_hazard_exposure",
            "disaster_type": disaster_type,
            "trigger_step": 1,
            "centre": centre,
            "radius_m": radius_m,
            "severity": severity,
            "reference_task_id": "task_001",
            "reference_route_index": scenario["hazard_route_index"],
            "reference_route": list(reference_route),
        }
        scenario.update({
            "workload": (
                "collaboration_local" if local_communication
                else "collaboration"
            ),
            "communication_mode": (
                "local_radio" if local_communication else "long_range_shared_board"
            ),
            "seed": seed,
            "disaster_type": disaster_type,
            "event": dict(exposure_event),
            "expected_blocked_node": hazard_node,
            "all_tasks_share_goal": True,
            "all_routes_include_hazard_before_event": True,
            "alternative_routes_verified_after_event": True,
        })
        return tasks, exposure_event, scenario

    if local_communication:
        geometry = (
            f"a leader/follower separation within {CommConfig().radius_m:.0f} m"
        )
    else:
        geometry = (
            f"a leader/follower separation above "
            f"{CommConfig().radius_m + 100:.0f} m"
        )
    raise SpawnError(
        "Could not generate a collaboration scenario with a shared passable "
        f"route, {geometry}, and a post-event alternative route. "
        "Try a different seed or wider distance limits."
    )


def generate_run_tasks(
    graph,
    num_agents,
    disaster_type,
    seed,
    workload="independent",
    min_dist=1000,
    max_dist=4000,
):
    """
    Generate tasks against the graph AFTER its initial t=0 disaster damage.

    The temporary engine applies the initial disaster. The actual simulation
    later resets and reapplies the same disaster using the same seed.

    This lets us generate feasible tasks before passing them into the
    simulation, while ensuring every experiment arm gets identical tasks.
    """

    # Apply the initial disaster to establish the correct passability state.
    temp_engine = DynamicDisasterEngine(
        graph=graph,
        disaster_type=disaster_type,
        centre=DISASTER_CENTRE,
        radius_m=DISASTER_RADIUS_M,
        severity=DISASTER_SEVERITY,
        seed=seed,
        wind_bearing_deg=WIND_BEARING,
    )
    temp_engine.initialise()

    return _add_task_ids(_generate_workload_tasks(
        graph, num_agents, seed, workload, min_dist, max_dist))


def get_comm_configs():
    """
    Communication configurations used for controlled ablation experiments.
    All configurations retain the same navigation policy and agent count.
    """

    return {
        "full": CommConfig(
            shared_memory=True,
            local_messaging=True,
            pheromones=True,
        ),

        "none": CommConfig(
            shared_memory=False,
            local_messaging=False,
            pheromones=False,
        ),

        "no_shared_memory": CommConfig(
            shared_memory=False,
            local_messaging=True,
            pheromones=True,
        ),

        "no_local_messaging": CommConfig(
            shared_memory=True,
            local_messaging=False,
            pheromones=True,
        ),

        "no_pheromones": CommConfig(
            shared_memory=True,
            local_messaging=True,
            pheromones=False,
        ),
    }


def get_channel_isolation_configs():
    """
    Four-arm 2x2 ablation of local messaging and shared memory.

    Pheromones are held ON in every arm so their contribution is controlled
    rather than confounded with the two message channels.
    """
    return {
        "both_channels": CommConfig(
            shared_memory=True, local_messaging=True, pheromones=True,
        ),
        "local_only": CommConfig(
            shared_memory=False, local_messaging=True, pheromones=True,
        ),
        "shared_memory_only": CommConfig(
            shared_memory=True, local_messaging=False, pheromones=True,
        ),
        "neither_message_channel": CommConfig(
            shared_memory=False, local_messaging=False, pheromones=True,
        ),
    }


def get_full_channel_factorial_configs():
    """
    Full 2^3 factorial ablation of the three independent information channels.

    Every combination of shared memory, local messaging, and pheromones is
    represented. This permits estimation of each channel's main effect and
    interactions without treating pheromones as an uncontrolled background.
    """
    return {
        "all_on": CommConfig(
            shared_memory=True, local_messaging=True, pheromones=True,
        ),
        "no_local_messaging": CommConfig(
            shared_memory=True, local_messaging=False, pheromones=True,
        ),
        "no_shared_memory": CommConfig(
            shared_memory=False, local_messaging=True, pheromones=True,
        ),
        "no_pheromones": CommConfig(
            shared_memory=True, local_messaging=True, pheromones=False,
        ),
        "shared_memory_only": CommConfig(
            shared_memory=True, local_messaging=False, pheromones=False,
        ),
        "local_messaging_only": CommConfig(
            shared_memory=False, local_messaging=True, pheromones=False,
        ),
        "pheromones_only": CommConfig(
            shared_memory=False, local_messaging=False, pheromones=True,
        ),
        "all_off": CommConfig(
            shared_memory=False, local_messaging=False, pheromones=False,
        ),
    }



def get_communication_dependent_configs():
    """Two-arm paired test isolating local hazard messaging.

    Shared memory and pheromones are disabled in both arms; local messaging
    is the only changed communication factor.
    """
    return {
        "local_messaging_enabled": CommConfig(
            shared_memory=False, local_messaging=True, pheromones=False,
        ),
        "local_messaging_disabled": CommConfig(
            shared_memory=False, local_messaging=False, pheromones=False,
        ),
    }


def log_experiment_setup(
    workload,
    seed,
    disaster_type,
    tasks,
    comm_name,
    comm_config,
):
    """Print the exact workload and communication setup for reproducibility."""

    print("\n--- EXPERIMENT CONFIGURATION ---")
    print(f"Workload       : {workload}")
    print(f"Disaster       : {disaster_type}")
    print(f"Seed           : {seed}")
    print(f"Communication  : {comm_name}")
    print(f"Comm settings  : {asdict(comm_config)}")
    print("Tasks:")

    for task in tasks:
        print(
            f"  {task['task_id']}: "
            f"{task['start']} -> {task['goal']}"
        )

    print("--------------------------------\n")


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def _summarise_collaboration_outcomes(sim, scenario):
    """Measure whether followers receive and act on the leader's hazard report."""
    target = scenario["hazard_node"]
    leader = scenario["leader_agent"]
    comm_data = sim.communication_instrumentation()
    discoveries = comm_data.get("hazard_discoveries", [])
    read_events = comm_data.get("shared_memory_read_events", [])
    follower_results = {}
    leader_self_detections = [
        event for event in discoveries
        if event["agent"] == leader
        and event["hazard_id"] == target
        and event["source"] == "self_observation"
    ]
    leader_self_detection_step = (
        min(event["step"] for event in leader_self_detections)
        if leader_self_detections else None
    )

    for follower in scenario["follower_agents"]:
        board_reads = [
            event for event in read_events
            if event["reader_agent"] == follower
            and event["discoverer_agent"] == leader
            and event["node_id"] == target
            and event["channel"] in {"blocked_nodes", "danger_nodes"}
        ]
        pheromone_reads = [
            event for event in read_events
            if event["reader_agent"] == follower
            and event["discoverer_agent"] == leader
            and event["channel"] == "pheromones"
        ]
        message_receipts = [
            event for event in discoveries
            if event["agent"] == follower
            and event["hazard_id"] == target
            and event["source"] == "teammate_message"
        ]
        self_detections = [
            event for event in discoveries
            if event["agent"] == follower
            and event["hazard_id"] == target
            and event["source"] == "self_observation"
        ]

        retrieval_steps = (
            [event["retrieval_step"] for event in board_reads]
            + [event["step"] for event in message_receipts]
        )
        retrieval_step = min(retrieval_steps) if retrieval_steps else None
        self_detection_step = (
            min(event["step"] for event in self_detections)
            if self_detections else None
        )
        retrieved_before_self_detection = (
            retrieval_step is not None
            and (
                self_detection_step is None
                or retrieval_step < self_detection_step
            )
        )

        first_pheromone_step = (
            min(event["retrieval_step"] for event in pheromone_reads)
            if pheromone_reads else None
        )
        pheromone_retrieved_before_self_detection = (
            first_pheromone_step is not None
            and (
                self_detection_step is None
                or first_pheromone_step < self_detection_step
            )
        )

        planner_events = list(getattr(sim.agents[follower], "plan_events", ()))
        teammate_replans = [
            event for event in planner_events
            if target in event.get("blocked_route_nodes", [])
            and event.get("blocked_route_sources", {}).get(target)
            in {"shared_memory", "msg"}
        ]
        early_route_changes = [
            event for event in teammate_replans
            if event.get("route_changed")
            and (
                self_detection_step is None
                or event["step"] < self_detection_step
            )
        ]
        divergence_events = [
            event for event in early_route_changes
            if event.get("first_divergence_hops_from_current") is not None
        ]

        follower_results[str(follower)] = {
            "shared_memory_retrievals": len(board_reads),
            "pheromone_reads_from_leader": len(pheromone_reads),
            "route_relevant_pheromone_reads_from_leader": sum(
                event["route_relevant"] for event in pheromone_reads
            ),
            "first_pheromone_retrieval_step": (
                min(event["retrieval_step"] for event in pheromone_reads)
                if pheromone_reads else None
            ),
            "local_message_receipts": len(message_receipts),
            "information_retrieval_step": retrieval_step,
            "self_detection_step": self_detection_step,
            # Explicit hazard reports only: board reads or hazard-message receipts.
            # Pheromone trail reads are tracked separately because a trail is not
            # itself a hazard report and must not inflate hazard-awareness counts.
            "retrieved_before_self_detection": retrieved_before_self_detection,
            "explicit_hazard_info_before_self_detection": retrieved_before_self_detection,
            "pheromone_retrieved_before_self_detection": (
                pheromone_retrieved_before_self_detection
            ),
            "teammate_informed_plan_replans": len(teammate_replans),
            "route_changed_before_self_detection": bool(early_route_changes),
            "planned_route_divergence_hops": (
                min(event["first_divergence_hops_from_current"]
                    for event in divergence_events)
                if divergence_events else None
            ),
            "immediate_next_hop_changed_before_self_detection": any(
                event.get("old_next") != event.get("new_next")
                for event in early_route_changes
            ),
            "self_detection_observed": self_detection_step is not None,
            "hazard_node_traversed": target in sim.agents[follower].path,
            "follower_reached_goal": bool(sim.agents[follower].reached_goal),
            "route_change_step": (
                min(event["step"] for event in early_route_changes)
                if early_route_changes else None
            ),
            "plan_change_events": teammate_replans,
        }

    explicit_early_receipts = sum(
        item["explicit_hazard_info_before_self_detection"]
        for item in follower_results.values()
    )
    event_on_reference_route = target in set(scenario.get("reference_route", ()))
    comm_config = getattr(sim, "comm", CommConfig())
    local_channel_enabled = bool(comm_config.local_messaging)
    shared_channel_enabled = bool(comm_config.shared_memory)
    manipulation_checks = {
        "hazard_node_on_reference_route": event_on_reference_route,
        "leader_self_observed_hazard": leader_self_detection_step is not None,
        "follower_received_explicit_info_before_self_detection": explicit_early_receipts > 0,
        "disabled_local_channel_produced_no_teammate_message": (
            True if local_channel_enabled else not any(
                item["local_message_receipts"] > 0
                for item in follower_results.values()
            )
        ),
        "disabled_shared_channel_produced_no_board_reads": (
            True if shared_channel_enabled else not any(
                item["shared_memory_retrievals"] > 0
                for item in follower_results.values()
            )
        ),
    }
    manipulation_checks["scenario_geometry_valid"] = (
        manipulation_checks["hazard_node_on_reference_route"]
        and manipulation_checks["leader_self_observed_hazard"]
    )
    manipulation_checks["communication_dependent_check_passed"] = (
        manipulation_checks["scenario_geometry_valid"]
        and (
            manipulation_checks["follower_received_explicit_info_before_self_detection"]
            if (local_channel_enabled or shared_channel_enabled)
            else manipulation_checks["disabled_local_channel_produced_no_teammate_message"]
            and manipulation_checks["disabled_shared_channel_produced_no_board_reads"]
        )
    )

    return {
        "leader_agent": leader,
        "leader_self_detection_step": leader_self_detection_step,
        "hazard_node": target,
        "manipulation_checks": manipulation_checks,
        "followers": follower_results,
        "followers_retrieved_before_self_detection": sum(
            item["retrieved_before_self_detection"]
            for item in follower_results.values()
        ),
        "followers_received_explicit_hazard_info_before_self_detection": sum(
            item["explicit_hazard_info_before_self_detection"]
            for item in follower_results.values()
        ),
        "followers_retrieved_pheromone_before_self_detection": sum(
            item["pheromone_retrieved_before_self_detection"]
            for item in follower_results.values()
        ),
        "total_teammate_pheromone_reads": sum(
            item["pheromone_reads_from_leader"]
            for item in follower_results.values()
        ),
        "followers_replanned_before_self_detection": sum(
            item["route_changed_before_self_detection"]
            for item in follower_results.values()
        ),
        "followers_changed_route_before_self_detection": sum(
            item["route_changed_before_self_detection"]
            for item in follower_results.values()
        ),
        "followers_changed_immediate_next_hop_before_self_detection": sum(
            item["immediate_next_hop_changed_before_self_detection"]
            for item in follower_results.values()
        ),
        "follower_count": len(follower_results),
    }


def run_single(
    disaster_type: str = "earthquake",
    num_agents: int = 10,
    steps: int = 200,
    min_dist: float = 1000,
    max_dist: float = 4000,
    seed: int = None,
    verbose: bool = True,
    workload: str = "independent",
    tasks: list = None,
    comm_config: CommConfig = None,
    comm_name: str = "default",
    exposure_event: dict = None,
    collaboration_scenario: dict = None,
    task_queues: list = None,
) -> dict:
    """
    Run one dynamic simulation.

    If tasks are supplied, they are reused exactly. This supports paired
    comparisons where only the communication configuration changes.
    """

    if workload not in (
        "independent", "shared_shelter", "dynamic_hazard_exposure",
        "collaboration", "collaboration_local", "communication_dependent", "scaling",
    ):
        raise ValueError(
            "workload must be 'independent', 'shared_shelter', "
            "'dynamic_hazard_exposure', 'collaboration', 'collaboration_local', or 'communication_dependent'"
        )

    if comm_config is None:
        comm_config = CommConfig()

    if workload == "dynamic_hazard_exposure" and steps <= DYNAMIC_EXPOSURE_STEP:
        raise ValueError(
            "dynamic_hazard_exposure requires steps greater than "
            f"{DYNAMIC_EXPOSURE_STEP} so the post-start event occurs."
        )

    print(f"\n{'#' * 60}")
    print(f"  DYNAMIC SIMULATION — {disaster_type.upper()}")
    print(
        f"  Workload: {workload} | Agents: {num_agents} | "
        f"Steps: {steps} | Seed: {seed}"
    )
    print(f"  Communication: {comm_name}")
    print(f"  Centre: {DISASTER_CENTRE}  Severity: {DISASTER_SEVERITY}  "
          f"Radius: {DISASTER_RADIUS_M}m")
    print(f"{'#' * 60}\n")

    graph = load_fresh_graph()

    # Scaling runs supply one initial task per agent plus explicit per-agent queues.
    if workload == "scaling":
        if task_queues is None or tasks is None:
            raise ValueError("scaling workload requires tasks and task_queues")
    # Generate tasks only when the caller has not supplied a paired task list.
    elif workload == "dynamic_hazard_exposure" and tasks is None:
        tasks, exposure_event = generate_dynamic_hazard_exposure_scenario(
            graph=graph,
            num_agents=num_agents,
            disaster_type=disaster_type,
            seed=seed,
            base_workload="independent",
            min_dist=min_dist,
            max_dist=max_dist,
        )
    elif workload in ("collaboration", "collaboration_local", "communication_dependent") and tasks is None:
        tasks, exposure_event, collaboration_scenario = (
            generate_collaboration_scenario(
                graph=graph,
                num_agents=num_agents,
                disaster_type=disaster_type,
                seed=seed,
                min_dist=min_dist,
                max_dist=max_dist,
                local_communication=(workload in ("collaboration_local", "communication_dependent")),
            )
        )
    elif tasks is None:
        tasks = generate_run_tasks(
            graph=graph,
            num_agents=num_agents,
            disaster_type=disaster_type,
            seed=seed,
            workload=workload,
            min_dist=min_dist,
            max_dist=max_dist,
        )

    if workload != "scaling" and len(tasks) != num_agents:
        raise ValueError(
            f"Expected {num_agents} tasks, received {len(tasks)}."
        )
    if workload == "scaling":
        if len(task_queues) != num_agents or any(not q for q in task_queues):
            raise ValueError("Scaling requires one non-empty queue per agent")
        if len([t for q in task_queues for t in q]) != 10:
            raise ValueError("Scaling queues must partition exactly 10 tasks")

    log_experiment_setup(
        workload=workload,
        seed=seed,
        disaster_type=disaster_type,
        tasks=tasks,
        comm_name=comm_name,
        comm_config=comm_config,
    )

    engine = None
    if workload in (
        "dynamic_hazard_exposure", "collaboration", "collaboration_local", "communication_dependent"
    ):
        if exposure_event is None:
            raise ValueError(
                f"{workload} requires an exposure_event when tasks are supplied"
            )
        if workload in ("collaboration", "collaboration_local", "communication_dependent") and collaboration_scenario is None:
            raise ValueError(
                "collaboration requires collaboration_scenario metadata "
                "when tasks are supplied"
            )
        engine_class = (
            CollaborationHazardEngine
            if workload in ("collaboration", "collaboration_local", "communication_dependent")
            else DynamicHazardExposureEngine
        )
        engine = engine_class(
            graph=graph,
            disaster_type=disaster_type,
            centre=DISASTER_CENTRE,
            radius_m=DISASTER_RADIUS_M,
            severity=DISASTER_SEVERITY,
            seed=seed,
            wind_bearing_deg=WIND_BEARING,
            exposure_event=exposure_event,
        )

    sim = DynamicSwarmSimulation(
        graph=graph,
        num_agents=num_agents,
        min_start_goal_distance=min_dist,
        max_start_goal_distance=max_dist,
        disaster_type=disaster_type,
        disaster_centre=DISASTER_CENTRE,
        disaster_radius_m=DISASTER_RADIUS_M,
        disaster_severity=DISASTER_SEVERITY,
        wind_bearing_deg=WIND_BEARING,
        seed=seed,
        tasks=tasks,
        task_queues=task_queues,
        require_feasible=True,
        comm=comm_config,
        engine=engine,
        roles=(
            ["navigator"] * num_agents
            if workload in ("collaboration", "collaboration_local", "communication_dependent", "scaling") else None
        ),
    )

    sim.run(steps=steps, quiet=(workload == "scaling" and not verbose))

    if verbose:
        sim.summary()

    # Preserve the existing top-level metric keys so that the old
    # aggregation code continues to work.
    result = sim.adaptation_metrics()

    # Add experiment metadata and communication measurements.
    result["workload"] = workload
    result["seed"] = seed
    result["disaster_type"] = disaster_type
    result["communication_name"] = comm_name
    result["communication_config"] = asdict(comm_config)
    manifest_tasks = (
        [task for queue in task_queues for task in queue]
        if task_queues is not None else tasks
    )
    result["task_manifest"] = [
        {
            "task_id": task.get("task_id", f"task_{i + 1:03d}"),
            "start": task["start"],
            "goal": task["goal"],
        }
        for i, task in enumerate(manifest_tasks)
    ]
    if task_queues is not None:
        result["task_queue_results"] = sim.sequential_task_metrics()
        result["task_queues"] = [
            [task.get("task_id") for task in queue] for queue in task_queues
        ]
    simulation_metrics = sim.metrics()
    result["simulation_steps"] = int(sim.current_step)
    result["communication_metrics"] = simulation_metrics["comms"]
    result["communication_instrumentation"] = simulation_metrics[
        "communication_instrumentation"
    ]
    result["info_propagation"] = sim.info_propagation()
    result["hazard_exposure"] = simulation_metrics["hazard_exposure"]
    result["response_metrics"] = simulation_metrics["response"]
    result["collective_metrics"] = collective_metrics_from_sim(sim)
    if exposure_event is not None:
        result["exposure_event"] = exposure_event
    if collaboration_scenario is not None:
        result["collaboration_scenario"] = collaboration_scenario
        result["collaboration_metrics"] = _summarise_collaboration_outcomes(
            sim, collaboration_scenario
        )
        result["planner_events_by_agent"] = {
            str(i): list(getattr(agent, "plan_events", ()))
            for i, agent in enumerate(sim.agents)
        }
        result["actual_paths_by_agent"] = {
            str(i): list(agent.path) for i, agent in enumerate(sim.agents)
        }

    return result



# ---------------------------------------------------------------------------
# Multi-run experiment (one disaster type)
# ---------------------------------------------------------------------------

def run_experiment(
    disaster_type:  str   = "earthquake",
    n_runs:         int   = 5,
    num_agents:     int   = 10,
    steps:          int   = 200,
    min_dist:       float = 1000,
    max_dist:       float = 4000,
    workload: str = "independent"
) -> dict:
    """
    Run n_runs independent simulations of one disaster type.
    Returns aggregate statistics dict.
    """
    print(f"\n{'#'*60}")
    print(f"EXPERIMENT: {disaster_type.upper()} — {n_runs} runs × {num_agents} agents × {steps} steps")
    print(f"{'#'*60}\n")

    results = []
    for i in range(n_runs):
        print(f"\n{'='*60}")
        print(f"RUN {i+1}/{n_runs}  (seed={i*42})")
        print(f"{'='*60}")
        m = run_single(
            disaster_type = disaster_type,
            num_agents    = num_agents,
            steps         = steps,
            min_dist      = min_dist,
            max_dist      = max_dist,
            seed          = i * 42,
            verbose       = False,
            workload      = workload,
        )
        results.append(m)
        print(f"  → success={m['reached_goal']}/{m['total_agents']} "
              f"({m['success_rate']:.0%})  "
              f"planner_invalidations={m['planner_invalidated_plan_events']}  "
              f"successful_replans={m['planner_successful_replans']}")

    # Aggregate
    def _mean(key):
        vals = [r[key] for r in results if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    def _std(key):
        vals = [r[key] for r in results if r[key] is not None]
        if len(vals) < 2:
            return 0.0
        mu = sum(vals) / len(vals)
        return round((sum((v-mu)**2 for v in vals) / len(vals)) ** 0.5, 3)

    agg = {
        "disaster_type":          disaster_type,
        "n_runs":                 n_runs,
        "num_agents":             num_agents,
        "steps":                  steps,
        "mean_success_rate":      _mean("success_rate"),
        "std_success_rate":       _std("success_rate"),
        "mean_planner_invalidations": _mean("planner_invalidated_plan_events"),
        "std_planner_invalidations":  _std("planner_invalidated_plan_events"),
        "mean_planner_successful_replans": _mean("planner_successful_replans"),
        "mean_path_length":       _mean("mean_path_length"),
        "mean_goal_step":         _mean("mean_goal_step"),
        "mean_disaster_events":   _mean("disaster_events"),
    }

    _print_experiment_summary(agg, results)
    return agg


def run_paired_communication_experiment(
    disaster_type="earthquake",
    workload="independent",
    n_runs=5,
    num_agents=10,
    steps=200,
    min_dist=1000,
    max_dist=4000,
    output_path=None,
    base_seed=None,
    comm_configs=None,
):
    """
    Compare communication configurations using paired experiments.

    For each run:
      1. Generate one task list.
      2. Reuse that exact task list for every communication configuration.
      3. Reuse the same disaster parameters and seed.
      4. Save all results and experiment metadata.

    This isolates communication configuration as the experimental variable.
    """

    if workload not in (
        "independent", "shared_shelter", "dynamic_hazard_exposure",
        "collaboration", "collaboration_local", "communication_dependent",
    ):
        raise ValueError(
            "workload must be 'independent', 'shared_shelter', "
            "'dynamic_hazard_exposure', 'collaboration', 'collaboration_local', or 'communication_dependent'"
        )

    if output_path is None:
        output_path = (
            (
                "collaboration_local_paired_results.json"
                if workload in ("collaboration_local", "communication_dependent")
                else "collaboration_paired_results.json"
            )
            if workload in ("collaboration", "collaboration_local", "communication_dependent")
            else "issue7_paired_results.json"
        )

    if workload == "communication_dependent":
        if num_agents != 2:
            raise ValueError(
                "communication_dependent requires exactly 2 agents "
                "(one observing leader and one follower)."
            )
        if comm_configs is None:
            comm_configs = get_communication_dependent_configs()

    comm_configs = dict(comm_configs or get_comm_configs())
    if not comm_configs:
        raise ValueError("At least one communication configuration is required")
    paired_results = []

    print("\n" + "=" * 75)
    print("PAIRED COMMUNICATION EXPERIMENT")
    print(f"Disaster: {disaster_type}")
    print(f"Workload: {workload}")
    print(f"Runs: {n_runs} | Agents: {num_agents} | Steps: {steps}")
    print(f"Configurations: {', '.join(comm_configs.keys())}")
    print("=" * 75)

    for run_idx in range(n_runs):
        seed = _paired_seed(base_seed, run_idx)

        print(f"\n{'#' * 75}")
        print(f"PAIRED RUN {run_idx + 1}/{n_runs} — seed={seed}")
        print(f"{'#' * 75}")

        # Generate tasks and, for the exposure workload, the event definition
        # ONCE for this run. Every arm receives the same objects' values.
        task_graph = load_fresh_graph()
        exposure_event = None
        collaboration_scenario = None
        if workload == "dynamic_hazard_exposure":
            shared_tasks, exposure_event = generate_dynamic_hazard_exposure_scenario(
                graph=task_graph,
                num_agents=num_agents,
                disaster_type=disaster_type,
                seed=seed,
                base_workload="independent",
                min_dist=min_dist,
                max_dist=max_dist,
            )
        elif workload in ("collaboration", "collaboration_local", "communication_dependent"):
            shared_tasks, exposure_event, collaboration_scenario = (
                generate_collaboration_scenario(
                    graph=task_graph,
                    num_agents=num_agents,
                    disaster_type=disaster_type,
                    seed=seed,
                    min_dist=min_dist,
                    max_dist=max_dist,
                    local_communication=(workload in ("collaboration_local", "communication_dependent")),
                )
            )
        else:
            shared_tasks = generate_run_tasks(
                graph=task_graph,
                num_agents=num_agents,
                disaster_type=disaster_type,
                seed=seed,
                workload=workload,
                min_dist=min_dist,
                max_dist=max_dist,
            )

        run_record = {
            "run": run_idx + 1,
            "seed": seed,
            "disaster_type": disaster_type,
            "workload": workload,
            "exposure_event": exposure_event,
            "collaboration_scenario": collaboration_scenario,
            "tasks": [
                {
                    "task_id": t["task_id"],
                    "start": t["start"],
                    "goal": t["goal"],
                }
                for t in shared_tasks
            ],
            "configurations": {},
        }

        for comm_name, comm_config in comm_configs.items():
            print(f"\n--- Configuration: {comm_name} ---")

            result = run_single(
                disaster_type=disaster_type,
                num_agents=num_agents,
                steps=steps,
                min_dist=min_dist,
                max_dist=max_dist,
                seed=seed,
                verbose=False,
                workload=workload,
                tasks=shared_tasks,
                comm_config=comm_config,
                comm_name=comm_name,
                exposure_event=exposure_event,
                collaboration_scenario=collaboration_scenario,
            )

            run_record["configurations"][comm_name] = result

            print(
                f"{comm_name:<22} "
                f"success={result['success_rate']:.1%} | "
                f"path={result['mean_path_length']} | "
                f"messages={result['communication_metrics']['transmissions']} | "
                f"bytes={result['communication_metrics']['bytes_tx']}"
                + (
                    f" | cross-agent={result['info_propagation']['cross_agent_retrievals']}"
                    f" | early-replans={result['collaboration_metrics']['followers_replanned_before_self_detection']}"
                    if workload in ("collaboration", "collaboration_local", "communication_dependent") else ""
                )
            )

        paired_results.append(run_record)

    # Aggregate mean and population standard deviation for each arm.
    print("\n" + "=" * 75)
    print("PAIRED EXPERIMENT SUMMARY")
    print("=" * 75)

    aggregate = {}
    aggregate_keys = [
        "success_rate",
        "mean_path_length",
        "total_distance_travelled_m",
        "communication_metrics.transmissions",
        "communication_metrics.bytes_tx",
        "hazard_exposure.agents_encountered_hazard",
        "response_metrics.route_invalidations",
        "response_metrics.hazard_event_metrics.mean_awareness_delay_all_agents_steps",
        "response_metrics.hazard_event_metrics.mean_response_delay_all_agents_steps",
        "response_metrics.hazard_event_metrics.successful_responses",
        "response_metrics.hazard_event_metrics.failed_responses",
        "response_metrics.hazard_event_metrics.total_hazard_events",
        "response_metrics.hazard_event_metrics.affected_agent_event_pairs",
        "response_metrics.hazard_event_metrics.affected_pairs_became_aware",
        "response_metrics.hazard_event_metrics.affected_pairs_responded",
        "response_metrics.hazard_event_metrics.response_records_outside_initial_affected_set",
        "response_metrics.hazard_event_metrics.successful_responses_among_affected_pairs",
        "response_metrics.hazard_event_metrics.successful_responses_outside_initial_affected_set",
        "collective_metrics.coverage.unique_nodes_covered",
        "collective_metrics.coverage.unique_cells_covered",
        "collective_metrics.coverage.redundant_agent_node_memberships",
        "collective_metrics.coverage.redundant_membership_fraction",
        "collective_metrics.coverage.mean_pairwise_node_jaccard",
        "collective_metrics.coverage.mean_pairwise_cell_jaccard",
        "collective_metrics.information.registered_hazard_events",
        "collective_metrics.information.agent_event_first_awareness_pairs",
        "collective_metrics.information.early_warning.comparisons_with_self_discovery",
        "collective_metrics.information.early_warning.censored_no_self_discovery",
        "collective_metrics.information.early_warning.mean_lead_steps_positive_means_teammate_was_earlier",
        "collective_metrics.information.information_retrieval.cross_agent_hazard_reads",
        "collective_metrics.information.information_retrieval.route_relevant_cross_agent_hazard_reads",
        "collective_metrics.information.information_retrieval.cross_agent_pheromone_reads",
        "collective_metrics.information.navigation_association_not_causation.decisions_with_preceding_teammate_or_board_information",
        "collective_metrics.information.navigation_association_not_causation.same_node_next_hop_changes_after_information",
        "collective_metrics.communication_efficiency.deliveries_per_transmission",
        "collective_metrics.communication_efficiency.bytes_per_delivery",
        "collective_metrics.communication_efficiency.mean_delivery_delay_steps",
        "collective_metrics.communication_efficiency.deliveries_per_attributed_planner_route_change",
        "collective_metrics.pheromone.decisions_with_candidate_pheromone_diagnostics",
        "collective_metrics.pheromone.decisions_with_comparable_no_pheromone_counterfactual",
        "collective_metrics.pheromone.pheromone_changed_score_argmax",
        "collective_metrics.pheromone.mean_selected_candidate_probability_effect",
        "collective_metrics.pheromone.mean_absolute_candidate_probability_effect",
        "communication_instrumentation.agents_receiving_teammate_information",
        "communication_instrumentation.shared_memory_cross_agent_retrievals",
        "communication_instrumentation.shared_memory_route_relevant_cross_agent_retrievals",
        "communication_instrumentation.pheromone_cross_agent_retrievals",
        "communication_instrumentation.pheromone_route_relevant_cross_agent_retrievals",
    ]
    if workload in ("collaboration", "collaboration_local", "communication_dependent"):
        aggregate_keys.extend([
            "collaboration_metrics.followers_retrieved_before_self_detection",
            "collaboration_metrics.followers_received_explicit_hazard_info_before_self_detection",
            "collaboration_metrics.followers_retrieved_pheromone_before_self_detection",
            "collaboration_metrics.followers_replanned_before_self_detection",
            "collaboration_metrics.followers_changed_immediate_next_hop_before_self_detection",
            "info_propagation.cross_agent_retrievals",
            "info_propagation.route_relevant_shared_memory_retrievals",
            "communication_instrumentation.pheromone_cross_agent_retrievals",
            "communication_instrumentation.pheromone_route_relevant_cross_agent_retrievals",
            "collaboration_metrics.total_teammate_pheromone_reads",
        ])

    def read_metric(result, dotted_key):
        """Read a dotted metric path without failing on absent nested fields."""
        value = result
        for part in dotted_key.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    for comm_name in comm_configs:
        arm_runs = [
            run["configurations"][comm_name]
            for run in paired_results
        ]
        arm_summary = {}

        # Preserve the established experiment-summary fields for compatibility.
        # Count valid observations and missing values explicitly; null is not zero.
        for key in aggregate_keys:
            values = [
                read_metric(run, key)
                for run in arm_runs
            ]
            numeric_values = [
                float(value) for value in values
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ]
            count = len(numeric_values)
            mean = sum(numeric_values) / count if count else None
            variance = (
                sum((value - mean) ** 2 for value in numeric_values) / count
                if count else None
            )
            arm_summary[key] = {
                "mean": round(mean, 6) if mean is not None else None,
                "std": round(variance ** 0.5, 6) if variance is not None else None,
                "n": count,
                "total_runs": len(arm_runs),
                "missing_n": len(arm_runs) - count,
            }

        # Automatically aggregate every numeric scalar emitted by the Priority 2
        # schema. This prevents new/previously omitted metrics (travel time,
        # makespan, throughput, detailed hazard exposure, agent work, etc.) from
        # disappearing from paired summaries when the schema grows.
        priority2_records = [
            {
                "collective_metrics": run.get("collective_metrics", {}),
                "communication_metrics": run.get("communication_metrics", {}),
            }
            for run in arm_runs
        ]
        arm_summary.update(aggregate_numeric_metrics(priority2_records))

        aggregate[comm_name] = arm_summary
        success = arm_summary["success_rate"]["mean"]
        print(
            f"{comm_name:<22} mean success rate = "
            f"{success:.1%}" if success is not None else
            f"{comm_name:<22} mean success rate = n/a"
        )

    # Save complete paired results, including task manifests and settings.
    if output_path:
        output_file = os.path.abspath(output_path)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment": {
                        "disaster_type": disaster_type,
                        "workload": workload,
                        "n_runs": n_runs,
                        "num_agents": num_agents,
                        "steps": steps,
                        "base_seed": base_seed,
                        "disaster_centre": DISASTER_CENTRE,
                        "disaster_radius_m": DISASTER_RADIUS_M,
                        "disaster_severity": DISASTER_SEVERITY,
                        "wind_bearing": WIND_BEARING,
                    },
                    "communication_configurations": {
                        name: asdict(cfg)
                        for name, cfg in comm_configs.items()
                    },
                    "comparison_design": {
                        "paired_by_seed_and_task_manifest": True,
                        "channels_varied": [
                            channel for channel in (
                                "local_messaging", "shared_memory", "pheromones"
                            )
                            if len({
                                getattr(cfg, channel)
                                for cfg in comm_configs.values()
                            }) > 1
                        ],
                        "pheromones_held_constant": (
                            len({cfg.pheromones for cfg in comm_configs.values()}) == 1
                        ),
                        "full_2x2x2_factorial": (
                            len(comm_configs) == 8
                            and {
                                (cfg.shared_memory, cfg.local_messaging, cfg.pheromones)
                                for cfg in comm_configs.values()
                            } == {
                                (sm, lm, ph)
                                for sm in (False, True)
                                for lm in (False, True)
                                for ph in (False, True)
                            }
                        ),
                        "note": (
                            "Each arm reuses the same seed, task manifest, disaster "
                            "parameters, and navigation policy within a paired run."
                        ),
                    },
                    "aggregate": aggregate,
                    "runs": paired_results,
                },
                f,
                indent=2,
                default=str,
            )

        print(f"\nSaved paired results to: {output_file}")

    return paired_results


def _print_experiment_summary(agg: dict, results: list):
    print(f"\n{'#'*60}")
    print(f"AGGREGATE — {agg['disaster_type'].upper()} "
          f"({agg['n_runs']} runs × {agg['num_agents']} agents)")
    print(f"{'#'*60}")
    sr_pct = round((agg['mean_success_rate'] or 0) * 100, 1)
    sr_std = round((agg['std_success_rate'] or 0) * 100, 1)
    print(f"  Success rate     : {sr_pct:.1f}% ± {sr_std:.1f}%")
    print("  Planner invalidations: "
          f"{agg['mean_planner_invalidations']:.2f} "
          f"± {agg['std_planner_invalidations']:.2f} per run")
    print("  Successful replans   : "
          f"{agg['mean_planner_successful_replans']:.2f} per run")
    print("  Default SNN reroutes : not applicable (no persistent planned route)")
    if agg['mean_path_length']:
        print(f"  Mean path length : {agg['mean_path_length']:.1f} steps")
    if agg['mean_goal_step']:
        print(f"  Mean goal step   : {agg['mean_goal_step']:.1f}")
    print(f"  Disaster events  : {agg['mean_disaster_events']:.1f} per run (avg)")

    # Per-run table
    print(f"\n  {'Run':<5} {'Success':<10} {'PlanInvalid':<13} {'Replans':<10} {'PathLen':<10} {'DisEvents'}")
    print("  " + "-"*60)
    for i, r in enumerate(results):
        sr   = f"{r['success_rate']:.0%}"
        invalidations = str(r["planner_invalidated_plan_events"])
        replans = str(r["planner_successful_replans"])
        pl   = f"{r['mean_path_length']:.1f}" if r['mean_path_length'] else "-"
        devs = str(r['disaster_events'])
        print(f"  {i+1:<5} {sr:<10} {invalidations:<13} {replans:<10} {pl:<10} {devs}")
    print(f"{'#'*60}\n")


# ---------------------------------------------------------------------------
# Comparative study — all four disaster types
# ---------------------------------------------------------------------------

def run_comparative_study(
    n_runs:     int = 5,
    num_agents: int = 10,
    steps:      int = 200,
    min_dist:   float = 1000,
    max_dist:   float = 4000,
    workload: str = "independent"
) -> dict:
    """
    Run all four disaster types under identical conditions and print a
    side-by-side comparison table for the research paper.

    Returns dict keyed by disaster_type → aggregate metrics dict.
    """
    disaster_types = ["earthquake", "flood", "fire", "compound"]
    all_agg = {}

    for dtype in disaster_types:
        agg = run_experiment(
            disaster_type = dtype,
            n_runs        = n_runs,
            num_agents    = num_agents,
            steps         = steps,
            min_dist      = min_dist,
            max_dist      = max_dist,
            workload      = workload,
        )
        all_agg[dtype] = agg

    # Side-by-side comparison table
    print("\n" + "="*80)
    print("  COMPARATIVE STUDY — SNN ADAPTATION ACROSS DISASTER TYPES")
    print("="*80)
    print(f"\n  Conditions: {n_runs} runs × {num_agents} agents × {steps} steps")
    print(f"  Disaster: severity={DISASTER_SEVERITY}, radius={DISASTER_RADIUS_M}m, "
          f"centre={DISASTER_CENTRE}")
    print()

    # Header
    col_w = 14
    header = (f"  {'Metric':<28}" +
              "".join(f"{dt.upper():<{col_w}}" for dt in disaster_types))
    print(header)
    print("  " + "-" * (28 + col_w * len(disaster_types)))

    def row(label, key, fmt="{:.1f}", pct=False):
        vals = []
        for dt in disaster_types:
            v = all_agg[dt].get(key)
            if v is None:
                vals.append("-".center(col_w))
            elif pct:
                vals.append(f"{v*100:.1f}%".ljust(col_w))
            else:
                vals.append(fmt.format(v).ljust(col_w))
        print(f"  {label:<28}" + "".join(vals))

    row("Success rate",        "mean_success_rate",    pct=True)
    row("Std (success)",       "std_success_rate",     pct=True)
    row("Planner invalidations/run", "mean_planner_invalidations", "{:.2f}")
    row("Successful replans/run", "mean_planner_successful_replans", "{:.2f}")
    row("Mean path length",    "mean_path_length",     "{:.1f}")
    row("Mean goal step",      "mean_goal_step",       "{:.1f}")
    row("Disaster events/run", "mean_disaster_events", "{:.1f}")

    print("  " + "-" * (28 + col_w * len(disaster_types)))
    print("\n  Interpretation guide:")
    print("    Success rate    : fraction of agents that reached their goal")
    print("    Planner invalidations: actual invalidated-plan events emitted by planner agents")
    print("    Successful replans   : invalidated plans for which the planner found a replacement")
    print("    Default SNN policy   : greedy one-hop decisions; planned-route invalidation is not applicable")
    print("    Path length     : mean steps taken by successful agents")
    print("    Goal step       : mean simulation step at which goal was reached")
    print("    Disaster events : total events fired (aftershocks/flood expansions/etc.)")
    print("=" * 80 + "\n")

    return all_agg


def run_communication_delay_experiment(
    disaster_type="earthquake",
    workload="collaboration_local",
    n_runs=5,
    num_agents=2,
    steps=160,
    min_dist=1000,
    max_dist=4000,
    base_seed=42,
    delays=(0, 1, 2, 3),
    output_path="communication_delay_results.json",
):
    """
    Paired latency sweep. For each seed, generate one task/event scenario and
    reuse it for every delay. Both local radio and shared-board delivery use
    the selected delay; all other communication settings remain enabled.
    """
    delays = tuple(int(d) for d in delays)
    if not delays or any(d < 0 for d in delays) or len(set(delays)) != len(delays):
        raise ValueError("delays must be unique non-negative integers")

    results = []
    print("\\n" + "=" * 76)
    print("PAIRED COMMUNICATION-DELAY EXPERIMENT")
    print(f"Disaster={disaster_type} | workload={workload} | agents={num_agents}")
    print(f"Runs={n_runs} | steps={steps} | delays={delays}")
    print("Both local messages and shared-memory posts receive the same delay.")
    print("=" * 76)

    for run_idx in range(n_runs):
        seed = _paired_seed(base_seed, run_idx)
        task_graph = load_fresh_graph()
        exposure_event = None
        collaboration_scenario = None

        if workload == "dynamic_hazard_exposure":
            shared_tasks, exposure_event = generate_dynamic_hazard_exposure_scenario(
                graph=task_graph, num_agents=num_agents,
                disaster_type=disaster_type, seed=seed,
                base_workload="independent", min_dist=min_dist, max_dist=max_dist,
            )
        elif workload in ("collaboration", "collaboration_local", "communication_dependent"):
            shared_tasks, exposure_event, collaboration_scenario = (
                generate_collaboration_scenario(
                    graph=task_graph, num_agents=num_agents,
                    disaster_type=disaster_type, seed=seed,
                    min_dist=min_dist, max_dist=max_dist,
                    local_communication=(workload in ("collaboration_local", "communication_dependent")),
                )
            )
        else:
            shared_tasks = generate_run_tasks(
                graph=task_graph, num_agents=num_agents,
                disaster_type=disaster_type, seed=seed,
                workload=workload, min_dist=min_dist, max_dist=max_dist,
            )

        run_record = {
            "run": run_idx + 1,
            "seed": seed,
            "workload": workload,
            "tasks": [
                {"task_id": t.get("task_id"), "start": t["start"], "goal": t["goal"]}
                for t in shared_tasks
            ],
            "delay_conditions": {},
        }

        for delay in delays:
            cfg = CommConfig(
                shared_memory=True,
                local_messaging=True,
                pheromones=True,
                delay_steps=delay,
                board_delay_steps=delay,
            )
            print(f"\\nRun {run_idx + 1}/{n_runs}, seed={seed}, delay={delay}")
            result = run_single(
                disaster_type=disaster_type,
                num_agents=num_agents,
                steps=steps,
                min_dist=min_dist,
                max_dist=max_dist,
                seed=seed,
                verbose=False,
                workload=workload,
                tasks=shared_tasks,
                comm_config=cfg,
                comm_name=f"delay_{delay}",
                exposure_event=exposure_event,
                collaboration_scenario=collaboration_scenario,
            )
            run_record["delay_conditions"][str(delay)] = result
            response = result.get("response_metrics", {})
            hazard_response = response.get("hazard_event_metrics", {})
            comm_metrics = result.get("communication_metrics", {})
            print(
                f"  success={result.get('success_rate')} | "
                f"awareness_delay={hazard_response.get('mean_awareness_delay_steps')} | "
                f"response_delay={hazard_response.get('mean_response_delay_steps')} | "
                f"tx/rx={comm_metrics.get('transmissions')}/{comm_metrics.get('deliveries')} | "
                f"pending={comm_metrics.get('pending_deliveries', comm_metrics.get('in_flight'))}"
            )

        results.append(run_record)

    # Summarise only metrics that exist in each run. Preserve null when no
    # qualifying awareness/response was observed; do not convert it to zero.
    metric_paths = {
        "success_rate": ("success_rate",),
        "mean_awareness_delay_steps": (
            "response_metrics", "hazard_event_metrics", "mean_awareness_delay_steps"
        ),
        "mean_response_delay_steps": (
            "response_metrics", "hazard_event_metrics", "mean_response_delay_steps"
        ),
        "successful_responses": (
            "response_metrics", "hazard_event_metrics", "successful_responses"
        ),
        "transmissions": ("communication_metrics", "transmissions"),
        "deliveries": ("communication_metrics", "deliveries"),
        "dropped": ("communication_metrics", "dropped"),
        "pending_deliveries": ("communication_metrics", "pending_deliveries"),
        "mean_message_delay_steps": ("communication_metrics", "mean_delay_steps"),
    }

    def nested_get(obj, path):
        for key in path:
            if not isinstance(obj, dict):
                return None
            obj = obj.get(key)
        return obj

    aggregate = {}
    for delay in delays:
        per_metric = {}
        for label, path in metric_paths.items():
            vals = []
            for run in results:
                value = nested_get(run["delay_conditions"][str(delay)], path)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    vals.append(float(value))
            per_metric[label] = {
                "n": len(vals),
                "mean": round(sum(vals) / len(vals), 4) if vals else None,
            }
        aggregate[str(delay)] = per_metric

    output = {
        "experiment": {
            "name": "paired_communication_delay_sweep",
            "disaster_type": disaster_type,
            "workload": workload,
            "n_runs": n_runs,
            "num_agents": num_agents,
            "steps": steps,
            "base_seed": base_seed,
            "delays": list(delays),
            "delay_applied_to": ["local_messaging", "shared_memory_board"],
            "fixed_channels": {
                "shared_memory": True, "local_messaging": True, "pheromones": True
            },
        },
        "aggregate_by_delay": aggregate,
        "runs": results,
    }
    if output_path:
        output_file = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        print(f"\\nSaved communication-delay results to: {output_file}")

    return output


# ---------------------------------------------------------------------------
# Independent local-radio / shared-board delay matrix
# ---------------------------------------------------------------------------

def run_independent_channel_delay_experiment(
    disaster_type="earthquake",
    workload="collaboration_local",
    n_runs=5,
    num_agents=2,
    steps=160,
    min_dist=1000,
    max_dist=4000,
    base_seed=42,
    local_delays=(0, 1, 2, 3),
    board_delays=(0, 1, 2, 3),
    output_path="issue5_independent_channel_delay_matrix.json",
):
    """Paired Cartesian sweep varying local and board latency independently.

    Each seed reuses exactly one task/event manifest across every delay pair.
    Shared memory, local messaging and pheromones stay enabled; only the two
    latency parameters vary. This separates channel latency effects that the
    legacy diagonal delay sweep confounded.
      
       """
    if workload not in ("collaboration", "collaboration_local", "communication_dependent"):
        raise ValueError("independent channel delay matrix requires a collaboration workload")
    configs = build_delay_matrix_configs(local_delays, board_delays)
    if workload == "communication_dependent" and num_agents != 2:
        raise ValueError("communication_dependent requires exactly two agents")

    results = []
    print("\n" + "=" * 76)
    print("INDEPENDENT COMMUNICATION-CHANNEL DELAY MATRIX")
    print(f"Disaster={disaster_type} | workload={workload} | agents={num_agents}")
    print(f"Runs={n_runs} | steps={steps} | conditions={len(configs)}")
    print("Local radio delay and shared-board delay vary independently.")
    print("=" * 76)

    for run_idx in range(n_runs):
        seed = _paired_seed(base_seed, run_idx)
        task_graph = load_fresh_graph()
        shared_tasks, exposure_event, collaboration_scenario = generate_collaboration_scenario(
            graph=task_graph,
            num_agents=num_agents,
            disaster_type=disaster_type,
            seed=seed,
            min_dist=min_dist,
            max_dist=max_dist,
            local_communication=(workload in ("collaboration_local", "communication_dependent")),
        )
        run_record = {
            "run": run_idx + 1,
            "seed": seed,
            "workload": workload,
            "task_manifest": [
                {"task_id": t.get("task_id"), "start": t["start"], "goal": t["goal"]}
                for t in shared_tasks
            ],
            "exposure_event": exposure_event,
            "collaboration_scenario": collaboration_scenario,
            "conditions": {},
        }
        for name, cfg in configs.items():
            print(f"\nRun {run_idx + 1}/{n_runs}, seed={seed}, {name}")
            result = run_single(
                disaster_type=disaster_type,
                num_agents=num_agents,
                steps=steps,
                min_dist=min_dist,
                max_dist=max_dist,
                seed=seed,
                verbose=False,
                workload=workload,
                tasks=shared_tasks,
                comm_config=cfg,
                comm_name=name,
                exposure_event=exposure_event,
                collaboration_scenario=collaboration_scenario,
            )
            run_record["conditions"][name] = result
        results.append(run_record)

    metric_paths = {
        "success_rate": ("success_rate",),
        "mean_distance_travelled_m": ("mean_distance_travelled_m",),
        "mean_awareness_delay_all_agents_steps": (
            "response_metrics", "hazard_event_metrics", "mean_awareness_delay_all_agents_steps"
        ),
        "mean_response_delay_all_agents_steps": (
            "response_metrics", "hazard_event_metrics", "mean_response_delay_all_agents_steps"
        ),
        "successful_responses": (
            "response_metrics", "hazard_event_metrics", "successful_responses"
        ),
        "transmissions": ("communication_metrics", "transmissions"),
        "deliveries": ("communication_metrics", "deliveries"),
        "dropped": ("communication_metrics", "dropped"),
        "pending_deliveries": ("communication_metrics", "pending_deliveries"),
        "late_deliveries": ("communication_metrics", "late_deliveries"),
        "bytes_tx": ("communication_metrics", "bytes_tx"),
        "mean_message_delay_steps": ("communication_metrics", "mean_delay_steps"),
        "followers_received_explicit_hazard_info_before_self_detection": (
            "collaboration_metrics", "followers_received_explicit_hazard_info_before_self_detection"
        ),
        "followers_changed_route_before_self_detection": (
            "collaboration_metrics", "followers_changed_route_before_self_detection"
        ),
    }

    def nested_get(obj, path):
        for key in path:
            if not isinstance(obj, dict):
                return None
            obj = obj.get(key)
        return obj

    aggregates = {}
    baseline_name = "local_0_board_0" if "local_0_board_0" in configs else next(iter(configs))
    paired_differences = {}
    for name in configs:
        metric_summary = {}
        for metric, path in metric_paths.items():
            values = []
            for run in results:
                value = nested_get(run["conditions"][name], path)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(float(value))
            if values:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1) if len(values) > 1 else 0.0
                metric_summary[metric] = {
                    "n": len(values), "mean": round(mean, 4),
                    "sd": round(variance ** 0.5, 4),
                }
            else:
                metric_summary[metric] = {"n": 0, "mean": None, "sd": None}
        aggregates[name] = metric_summary

        if name != baseline_name:
            paired_differences[name] = {}
            for metric, path in metric_paths.items():
                diffs = []
                for run in results:
                    a = nested_get(run["conditions"][name], path)
                    b = nested_get(run["conditions"][baseline_name], path)
                    if (isinstance(a, (int, float)) and not isinstance(a, bool)
                            and isinstance(b, (int, float)) and not isinstance(b, bool)):
                        diffs.append(float(a) - float(b))
                paired_differences[name][metric] = {
                    "baseline": baseline_name,
                    "n_pairs": len(diffs),
                    "mean_difference": round(sum(diffs) / len(diffs), 4) if diffs else None,
                    "differences": [round(v, 4) for v in diffs],
                }

    output = {
        "experiment": {
            "name": "independent_local_and_board_delay_matrix",
            "disaster_type": disaster_type,
            "workload": workload,
            "n_runs": n_runs,
            "num_agents": num_agents,
            "steps": steps,
            "base_seed": base_seed,
            "local_delays": list(local_delays),
            "board_delays": list(board_delays),
            "condition_count": len(configs),
            "fixed_channels": {"shared_memory": True, "local_messaging": True, "pheromones": True},
            "paired_task_manifest_per_seed": True,
            "baseline_condition": baseline_name,
        },
        "aggregate_by_condition": aggregates,
        "paired_differences_vs_baseline": paired_differences,
        "runs": results,
    }
    if output_path:
        output_file = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved independent channel-delay matrix to: {output_file}")
    return output



# ---------------------------------------------------------------------------
# Fixed-workload scaling experiment
# ---------------------------------------------------------------------------

def _partition_fixed_tasks(tasks, num_agents):
    """Deterministic round-robin partition; every task is assigned exactly once."""
    if not 1 <= num_agents <= len(tasks):
        raise ValueError("num_agents must be between 1 and the fixed task count")
    queues = [list(tasks[i::num_agents]) for i in range(num_agents)]
    if any(not queue for queue in queues):
        raise AssertionError("Task partition unexpectedly produced an empty queue")
    flattened = [task["task_id"] for queue in queues for task in queue]
    expected = [task["task_id"] for task in tasks]
    if sorted(flattened) != sorted(expected) or len(flattened) != len(set(flattened)):
        raise AssertionError("Task queues must partition the fixed manifest exactly once")
    return queues


def _mean_sd(values):
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": round(statistics.mean(clean), 4),
        "std": round(statistics.stdev(clean), 4) if len(clean) > 1 else 0.0,
        "n": len(clean),
    }


def _run_scaling_arm_with_budget(
    run_kwargs,
    initial_steps,
    max_steps,
    growth_factor=2.0,
    retry_until_complete=False,
):
    """Run one scaling arm with a common adaptive horizon policy.

    Every retry restarts from step 1 with the same seed and exact task queues.
    This avoids treating a step-budget timeout as a completed serial workload.
    """
    if initial_steps < 1:
        raise ValueError("initial_steps must be at least 1")
    if max_steps < initial_steps:
        raise ValueError("max_steps must be >= initial_steps")
    if growth_factor <= 1.0:
        raise ValueError("growth_factor must be greater than 1.0")

    budget = int(initial_steps)
    attempts = 0
    cumulative_wall_clock_seconds = 0.0
    attempt_horizons = []
    while True:
        attempts += 1
        attempt_horizons.append(int(budget))
        print(
            f"[scaling] disaster={run_kwargs.get('disaster_type')} "
            f"seed={run_kwargs.get('seed')} team_size={run_kwargs.get('num_agents')} "
            f"budget={budget} retry={attempts}",
            file=__import__("sys").stderr,
            flush=True,
        )
        started = __import__("time").perf_counter()
        result = run_single(**dict(run_kwargs, steps=budget))
        attempt_wall_clock_seconds = __import__("time").perf_counter() - started
        cumulative_wall_clock_seconds += attempt_wall_clock_seconds
        result["wall_clock_execution_seconds"] = attempt_wall_clock_seconds
        result["cumulative_retry_wall_clock_seconds"] = cumulative_wall_clock_seconds
        result["attempt_count"] = attempts
        result["attempt_horizons"] = list(attempt_horizons)
        result["wall_clock_timing_boundary"] = (
            "run_single call including graph load, initialization, simulation, and result extraction"
        )
        task_rows = result.get("task_queue_results", [])
        all_complete = (
            len(task_rows) == 10
            and all(bool(row.get("success")) for row in task_rows)
        )
        if not retry_until_complete or all_complete or budget >= max_steps:
            return result, budget, attempts, all_complete

        next_budget = max(budget + 1, int(math.ceil(budget * growth_factor)))
        budget = min(int(max_steps), next_budget)


def run_scaling_experiment(
    disaster_type="earthquake",
    steps=500,
    runs=20,
    seed=42,
    min_dist=1000,
    max_dist=4000,
    agent_counts=(1, 2, 3, 5, 10),
    verbose=False,
    serial_max_steps=12000,
    serial_step_growth_factor=2.0,
):
    """
    Compare team sizes on the same ten-task manifest for every seed.

    Each run generates one disaster-feasible task manifest. The exact same
    manifest and disaster seed are reused for all team sizes. Agents execute
    their assigned tasks serially; travel between task endpoints and the next
    task's start is part of the recorded path and elapsed workload time.
    """
    task_count = 10
    agent_counts = list(agent_counts)
    if len(agent_counts) != len(set(agent_counts)):
        raise ValueError("agent_counts must not contain duplicates")
    if not agent_counts or any(n < 1 or n > task_count for n in agent_counts):
        raise ValueError("agent_counts must be selected from 1..10")
    if 1 not in agent_counts:
        raise ValueError("agent_counts must include 1 as the serial baseline")
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if serial_max_steps < steps:
        raise ValueError("serial_max_steps must be >= the initial scaling step budget")
    if serial_step_growth_factor <= 1.0:
        raise ValueError("serial_step_growth_factor must be greater than 1.0")

    seeds = [int(seed) + i for i in range(runs)]
    raw = {n: [] for n in agent_counts}

    for run_index, run_seed in enumerate(seeds):
        # Generate the fixed ten-task manifest once for this seed.
        manifest_graph = load_fresh_graph()
        manifest = generate_run_tasks(
            graph=manifest_graph,
            num_agents=task_count,
            disaster_type=disaster_type,
            seed=run_seed,
            workload="independent",
            min_dist=min_dist,
            max_dist=max_dist,
        )
        manifest = [dict(task) for task in manifest]
        for task_index, task in enumerate(manifest):
            task.setdefault("task_id", f"task_{task_index + 1:03d}")

        for num_agents in agent_counts:
            queues = _partition_fixed_tasks(manifest, num_agents)
            # Every team size deploys from the same common base node. Agents
            # assigned tasks elsewhere must navigate to their first task start;
            # that initial deployment leg is recorded as repositioning cost.
            deployment_base = manifest[0]["start"]
            initial_tasks = []
            for agent_idx, queue in enumerate(queues):
                first = queue[0]
                initial_goal = (
                    first["goal"] if first["start"] == deployment_base
                    else first["start"]
                )
                initial_tasks.append({
                    "task_id": f"deployment_agent_{agent_idx}",
                    "start": deployment_base,
                    "goal": initial_goal,
                })
            # All arms start from the same disaster seed and task manifest.
            # Keep the configured default communication channels consistent.
            run_kwargs = {
                "disaster_type": disaster_type,
                "num_agents": num_agents,
                "min_dist": min_dist,
                "max_dist": max_dist,
                "seed": run_seed,
                "verbose": verbose,
                "workload": "scaling",
                "tasks": initial_tasks,
                "task_queues": queues,
                "comm_config": CommConfig(),
                "comm_name": "default",
            }
            # The serial arm must finish the entire fixed workload before it
            # can serve as a valid speedup denominator. If the initial budget
            # is insufficient, replay the same seed + queues with a larger
            # horizon. Parallel teams retain the original configured budget.
            # Every team size uses the same deterministic horizon/retry policy.
            with contextlib.redirect_stdout(io.StringIO()):
                result, used_budget, budget_attempts, helper_all_complete = (
                    _run_scaling_arm_with_budget(
                        run_kwargs=run_kwargs,
                        initial_steps=steps,
                        max_steps=serial_max_steps,
                        growth_factor=serial_step_growth_factor,
                        retry_until_complete=True,
                    )
                )

            task_rows = result.get("task_queue_results", [])
            task_rows_by_id = {row["task_id"]: row for row in task_rows}
            if set(task_rows_by_id) != {task["task_id"] for task in manifest}:
                raise AssertionError(
                    f"Run {run_seed}, {num_agents} agents: task results do not "
                    "cover the fixed manifest exactly once"
                )
            completed = sum(bool(row["success"]) for row in task_rows)
            all_complete = completed == task_count
            elapsed_observed = int(result.get("simulation_steps", steps))
            workload_makespan = (
                max(row["finish_step"] for row in task_rows)
                if all_complete else None
            )
            total_agent_busy_steps = sum(
                int(row.get("agent_steps", 0))
                + int(row.get("reposition_steps_before", 0))
                for row in task_rows
            )
            denominator = num_agents * max(1, elapsed_observed)
            utilization = min(1.0, total_agent_busy_steps / denominator)
            task_manifest = [
                dict(task)
                for task in manifest
            ]
            task_manifest_hash = hashlib.sha256(
                json.dumps(
                    task_manifest, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            hazard_config = {
                "disaster_type": disaster_type,
                "centre": DISASTER_CENTRE,
                "radius_m": DISASTER_RADIUS_M,
                "severity": DISASTER_SEVERITY,
                "wind_bearing_deg": WIND_BEARING,
                "seed": run_seed,
            }
            hazard_config_hash = hashlib.sha256(
                json.dumps(
                    hazard_config, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            incomplete_tasks = [
                row["task_id"] for row in task_rows
                if not row.get("success")
            ]

            exposure = result.get("hazard_exposure", {})
            hazard_encounters = (
                int(exposure.get("actual_node_encounters") or 0)
                + int(exposure.get("actual_edge_encounters") or 0)
            )
            total_distance_m = result.get("total_distance_travelled_m")
            hazard_sensing_agent_steps = sum(
                len(indices) for indices in
                exposure.get("agents_with_hazard_in_sensing_neighborhood_by_step", {}).values()
            )
            dynamic_hazard_sensing_agent_steps = sum(
                len(indices) for indices in
                exposure.get(
                    "agents_with_dynamic_hazard_in_sensing_neighborhood_by_step", {}
                ).values()
            )
            row = {
                "seed": run_seed,
                "num_agents": num_agents,
                "task_count": task_count,
                "task_manifest": task_manifest,
                "task_manifest_hash": task_manifest_hash,
                "hazard_config": hazard_config,
                "hazard_config_hash": hazard_config_hash,
                "algorithm_config_id": "homogeneous_navigator_default_comm_v1",
                "tasks_completed": completed,
                "completion_rate": completed / task_count,
                "all_tasks_completed": all_complete,
                "incomplete_task_ids": incomplete_tasks,
                "observed_simulation_steps": elapsed_observed,
                "configured_initial_step_budget": int(steps),
                "actual_step_budget": int(used_budget),
                "serial_budget_attempts": int(budget_attempts),
                "serial_budget_extended": bool(used_budget > steps),
                "budget_extended": bool(used_budget > steps),
                "budget_policy_id": "common_geometric_retry_1500_3000_6000_12000",
                "maximum_step_budget": int(serial_max_steps),
                "workload_makespan_steps": workload_makespan,
                "throughput_tasks_per_step": (
                    completed / workload_makespan if all_complete and workload_makespan
                    else completed / elapsed_observed if elapsed_observed else 0.0
                ),
                "total_distance_travelled_m": total_distance_m,
                "hazard_encounters_total": hazard_encounters,
                "hazard_encounters_per_km": (
                    hazard_encounters / (total_distance_m / 1000.0)
                    if total_distance_m and total_distance_m > 0 else None
                ),
                "hazard_encounters_per_1000_agent_steps": (
                    hazard_encounters / denominator * 1000.0
                    if denominator else None
                ),
                "hazard_sensing_agent_steps": hazard_sensing_agent_steps,
                "dynamic_hazard_sensing_agent_steps": dynamic_hazard_sensing_agent_steps,
                "agent_busy_steps": total_agent_busy_steps,
                "agent_utilization": utilization,
                "actual_node_encounters": exposure.get("actual_node_encounters"),
                "actual_edge_encounters": exposure.get("actual_edge_encounters"),
                "agents_encountered_hazard": exposure.get("agents_encountered_hazard"),
                "agents_with_hazard_in_sensing_neighborhood": exposure.get(
                    "agents_with_hazard_in_sensing_neighborhood"
                ),
                "traversed_route_nodes_affected_by_damage": exposure.get(
                    "traversed_route_nodes_affected_by_damage"
                ),
                "communication_metrics": result.get("communication_metrics"),
                "planner_stats_by_agent": result.get("planner_stats_by_agent"),
                "performance": result.get("performance"),
                "task_results": task_rows,
                "task_queues": result.get("task_queues"),
                "deployment_base_node": deployment_base,
                "wall_clock_execution_seconds": result.get("wall_clock_execution_seconds"),
                "wall_clock_timing_boundary": result.get("wall_clock_timing_boundary"),
                "cumulative_retry_wall_clock_seconds": result.get("cumulative_retry_wall_clock_seconds"),
                "attempt_count": result.get("attempt_count"),
                "attempt_horizons": result.get("attempt_horizons"),
            }
            raw[num_agents].append(row)

        print(
            f"Scaling seed {run_seed}: "
            + ", ".join(
                f"N={n}: {raw[n][-1]['tasks_completed']}/10"
                for n in agent_counts
            )
        )

    # Per-seed speedup is calculated only when both the serial baseline and
    # the corresponding parallel arm completed all ten tasks.
    baseline_by_seed = {
        row["seed"]: row["workload_makespan_steps"]
        for row in raw[1] if row["all_tasks_completed"]
    }
    for n in agent_counts:
        for row in raw[n]:
            baseline = baseline_by_seed.get(row["seed"])
            if n == 1:
                row["parallel_speedup"] = 1.0 if row["all_tasks_completed"] else None
                row["scaling_efficiency"] = 1.0 if row["all_tasks_completed"] else None
            elif baseline and row["all_tasks_completed"] and row["workload_makespan_steps"]:
                speedup = baseline / row["workload_makespan_steps"]
                row["parallel_speedup"] = speedup
                row["scaling_efficiency"] = speedup / n
            else:
                row["parallel_speedup"] = None
                row["scaling_efficiency"] = None

    metrics = [
        "completion_rate", "observed_simulation_steps",
        "workload_makespan_steps", "throughput_tasks_per_step",
        "total_distance_travelled_m", "agent_utilization",
        "actual_node_encounters", "actual_edge_encounters",
        "hazard_encounters_total", "hazard_encounters_per_km",
        "hazard_encounters_per_1000_agent_steps",
        "hazard_sensing_agent_steps", "dynamic_hazard_sensing_agent_steps",
        "agents_encountered_hazard",
        "agents_with_hazard_in_sensing_neighborhood",
        "traversed_route_nodes_affected_by_damage",
        "parallel_speedup", "scaling_efficiency",
    ]
    summary = {}
    for n in agent_counts:
        summary[str(n)] = {
            "num_agents": n,
            "runs": len(raw[n]),
            "metrics": {
                metric: _mean_sd([row.get(metric) for row in raw[n]])
                for metric in metrics
            },
        }

    return {
        "experiment": "fixed_workload_multi_agent_scaling",
        "schema_version": "1.2",
        "disaster_type": disaster_type,
        "fixed_task_count": task_count,
        "agent_counts": agent_counts,
        "runs": runs,
        "seeds": seeds,
        "simulation_step_budget": steps,
        "serial_baseline_max_step_budget": int(serial_max_steps),
        "serial_baseline_budget_policy": (
            "Replay every team size from the same initial horizon with geometrically "
            "increasing horizons 1500, 3000, 6000, 12000 until all ten tasks "
            "complete or the common maximum is reached."
        ),
        "serial_step_growth_factor": float(serial_step_growth_factor),
        "task_assignment": (
            "Deterministic round-robin partition of the same ten-task manifest; "
            "each agent executes its queue serially."
        ),
        "repositioning_cost_included": True,
        "communication_configuration": asdict(CommConfig()),
        "summary_by_agent_count": summary,
        "raw_runs_by_agent_count": {str(n): raw[n] for n in agent_counts},
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dynamic disaster SNN experiment runner")
    p.add_argument(
    "--mode",
    choices=["single", "experiment", "compare", "paired", "delay", "channel_delays", "channels", "factorial", "communication_dependent", "scaling"],
    default="single",
    )
    p.add_argument(
    "--workload",
    choices=[
        "independent", "shared_shelter", "dynamic_hazard_exposure",
        "collaboration", "collaboration_local", "communication_dependent",
    ],
    default="independent",
    help="Task workload design",
    )
    p.add_argument("--type",   choices=["earthquake", "flood", "fire", "compound"],
                   default="earthquake",   help="Disaster type (single/experiment mode)")
    p.add_argument("--agents", type=int, default=10,   help="Number of agents")
    p.add_argument("--steps",  type=int, default=200,  help="Max simulation steps")
    p.add_argument("--runs",   type=int, default=5,    help="Runs per disaster type (experiment/compare; scaling uses the same count)")
    p.add_argument("--scaling-agents", type=int, nargs="+", default=[1, 2, 3, 5, 10],
                   help="Team sizes for --mode scaling")
    p.add_argument("--scaling-steps", type=int, default=1500,
                   help="Initial per-run step budget for --mode scaling")
    p.add_argument("--scaling-max-steps", type=int, default=12000,
                   help="Maximum adaptive step budget for the one-agent serial baseline")
    p.add_argument("--scaling-step-growth-factor", type=float, default=2.0,
                   help="Geometric budget multiplier for serial-baseline retries (>1)")
    p.add_argument("--seed",   type=int, default=None, help="RNG seed (single/paired mode)")
    p.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path (single and paired modes)",
    )
    p.add_argument(
        "--delays", type=int, nargs="+", default=[0, 1, 2, 3],
        help="Same delay for local messages and board posts in --mode delay",
    )
    p.add_argument(
        "--local-delays", type=int, nargs="+", default=[0, 1, 2, 3],
        help="Independent local-radio delays for --mode channel_delays",
    )
    p.add_argument(
        "--board-delays", type=int, nargs="+", default=[0, 1, 2, 3],
        help="Independent shared-board delays for --mode channel_delays",
    )
    p.add_argument("--min-dist", type=float, default=1000)
    p.add_argument("--max-dist", type=float, default=4000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "scaling":
        scaling_result = run_scaling_experiment(
            disaster_type=args.type,
            steps=args.scaling_steps,
            runs=args.runs,
            seed=args.seed if args.seed is not None else 42,
            min_dist=args.min_dist,
            max_dist=args.max_dist,
            agent_counts=args.scaling_agents,
            serial_max_steps=args.scaling_max_steps,
            serial_step_growth_factor=args.scaling_step_growth_factor,
        )
        output_path = os.path.abspath(args.output or "scaling_experiment_results.json")
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(scaling_result, output_file, indent=2, ensure_ascii=False)
        print(f"\nSaved scaling experiment results to: {output_path}")
        print("\nSUMMARY (mean ± SD; n = valid runs)")
        print(
            "Serial baseline: initial budget="
            f"{args.scaling_steps}, maximum budget={args.scaling_max_steps}, "
            f"growth factor={args.scaling_step_growth_factor}"
        )
        for count, data in scaling_result["summary_by_agent_count"].items():
            m = data["metrics"]
            print(
                f"N={count:>2}: completion={m['completion_rate']['mean']} "
                f"makespan={m['workload_makespan_steps']['mean']} "
                f"speedup={m['parallel_speedup']['mean']} "
                f"efficiency={m['scaling_efficiency']['mean']} "
                f"distance_m={m['total_distance_travelled_m']['mean']}"
            )

    elif args.mode == "single":
        single_result = run_single(
            disaster_type = args.type,
            num_agents    = args.agents,
            steps         = args.steps,
            min_dist      = args.min_dist,
            max_dist      = args.max_dist,
            seed          = args.seed,
            workload      = args.workload,
        )
        if args.output:
            output_path = os.path.abspath(args.output)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            def _json_fallback(value):
                # Keep structured metrics serializable without discarding data.
                if isinstance(value, (set, frozenset)):
                    return sorted(value, key=str)
                if hasattr(value, "item"):
                    try:
                        return value.item()
                    except (ValueError, TypeError):
                        pass
                if hasattr(value, "__dict__"):
                    return value.__dict__
                return repr(value)

            with open(output_path, "w", encoding="utf-8") as output_file:
                json.dump(
                    single_result,
                    output_file,
                    indent=2,
                    ensure_ascii=False,
                    default=_json_fallback,
                )
                output_file.write("\\n")
            print(f"\\n[JSON] Single-run results saved to: {output_path}")
    elif args.mode == "experiment":
        run_experiment(
            disaster_type = args.type,
            n_runs        = args.runs,
            num_agents    = args.agents,
            steps         = args.steps,
            min_dist      = args.min_dist,
            max_dist      = args.max_dist,
            workload      = args.workload,
        )
    elif args.mode == "compare":
        run_comparative_study(
            n_runs     = args.runs,
            num_agents = args.agents,
            steps      = args.steps,
            min_dist   = args.min_dist,
            max_dist   = args.max_dist,
            workload   = args.workload,
        )
    elif args.mode == "paired":
        run_paired_communication_experiment(
            disaster_type = args.type,
            workload      = args.workload,
            n_runs        = args.runs,
            num_agents    = args.agents,
            steps         = args.steps,
            min_dist      = args.min_dist,
            max_dist      = args.max_dist,
            base_seed     = args.seed,
            output_path   = args.output,
        )
    elif args.mode == "channel_delays":
        run_independent_channel_delay_experiment(
            disaster_type=args.type,
            workload=args.workload,
            n_runs=args.runs,
            num_agents=args.agents,
            steps=args.steps,
            min_dist=args.min_dist,
            max_dist=args.max_dist,
            base_seed=args.seed,
            local_delays=args.local_delays,
            board_delays=args.board_delays,
            output_path=args.output or "issue5_independent_channel_delay_matrix.json",
        )
    elif args.mode == "channels":
        run_paired_communication_experiment(
            disaster_type=args.type,
            workload=args.workload,
            n_runs=args.runs,
            num_agents=args.agents,
            steps=args.steps,
            min_dist=args.min_dist,
            max_dist=args.max_dist,
            base_seed=args.seed,
            output_path=args.output or "issue5_channel_isolation_results.json",
            comm_configs=get_channel_isolation_configs(),
        )
    elif args.mode == "communication_dependent":
        run_paired_communication_experiment(
            disaster_type=args.type,
            workload="communication_dependent",
            n_runs=args.runs,
            num_agents=args.agents,
            steps=args.steps,
            min_dist=args.min_dist,
            max_dist=args.max_dist,
            base_seed=args.seed,
            output_path=args.output or "issue5_communication_dependent.json",
            comm_configs=get_communication_dependent_configs(),
        )
    elif args.mode == "factorial":
        run_paired_communication_experiment(
            disaster_type=args.type,
            workload=args.workload,
            n_runs=args.runs,
            num_agents=args.agents,
            steps=args.steps,
            min_dist=args.min_dist,
            max_dist=args.max_dist,
            base_seed=args.seed,
            output_path=args.output or "issue5_full_channel_factorial.json",
            comm_configs=get_full_channel_factorial_configs(),
        )
    elif args.mode == "delay":
        run_communication_delay_experiment(
            disaster_type=args.type,
            workload=args.workload,
            n_runs=args.runs,
            num_agents=args.agents,
            steps=args.steps,
            min_dist=args.min_dist,
            max_dist=args.max_dist,
            base_seed=args.seed,
            delays=args.delays,
            output_path=args.output or "communication_delay_results.json",
        )


def test_channel_isolation_configs_form_controlled_two_by_two_ablation():
    configs = get_channel_isolation_configs()
    assert set(configs) == {
        "both_channels", "local_only", "shared_memory_only",
        "neither_message_channel",
    }
    assert all(cfg.pheromones is True for cfg in configs.values())
    assert configs["both_channels"].shared_memory is True
    assert configs["both_channels"].local_messaging is True
    assert configs["local_only"].shared_memory is False
    assert configs["local_only"].local_messaging is True
    assert configs["shared_memory_only"].shared_memory is True
    assert configs["shared_memory_only"].local_messaging is False
    assert configs["neither_message_channel"].shared_memory is False
    assert configs["neither_message_channel"].local_messaging is False


def test_full_channel_factorial_covers_all_eight_combinations():
    configs = get_full_channel_factorial_configs()
    assert len(configs) == 8
    observed = {
        (cfg.shared_memory, cfg.local_messaging, cfg.pheromones)
        for cfg in configs.values()
    }
    expected = {
        (sm, lm, ph)
        for sm in (False, True)
        for lm in (False, True)
        for ph in (False, True)
    }
    assert observed == expected


def test_collaboration_summary_reports_successful_communication_manipulation_check():
    class FakeAgent:
        plan_events = []
        path = [10, 99, 20]
        reached_goal = True

    class FakeSim:
        comm = CommConfig(shared_memory=False, local_messaging=True, pheromones=False)
        agents = [FakeAgent(), FakeAgent()]
        def communication_instrumentation(self):
            return {
                "hazard_discoveries": [
                    {"agent": 0, "hazard_id": 99, "source": "self_observation", "step": 3},
                    {"agent": 1, "hazard_id": 99, "source": "teammate_message", "step": 4},
                ],
                "shared_memory_read_events": [],
            }

    scenario = {
        "hazard_node": 99,
        "leader_agent": 0,
        "follower_agents": [1],
        "reference_route": [10, 99, 20],
    }
    result = _summarise_collaboration_outcomes(FakeSim(), scenario)
    assert result["manipulation_checks"]["leader_self_observed_hazard"] is True
    assert result["manipulation_checks"]["follower_received_explicit_info_before_self_detection"] is True
    assert result["manipulation_checks"]["communication_dependent_check_passed"] is True


def test_collaboration_summary_flags_failed_information_propagation_manipulation():
    class FakeAgent:
        plan_events = []
        path = [10, 99, 20]
        reached_goal = True

    class FakeSim:
        comm = CommConfig(shared_memory=False, local_messaging=True, pheromones=False)
        agents = [FakeAgent(), FakeAgent()]
        def communication_instrumentation(self):
            return {
                "hazard_discoveries": [
                    {"agent": 0, "hazard_id": 99, "source": "self_observation", "step": 3},
                ],
                "shared_memory_read_events": [],
            }

    scenario = {
        "hazard_node": 99,
        "leader_agent": 0,
        "follower_agents": [1],
        "reference_route": [10, 99, 20],
    }
    result = _summarise_collaboration_outcomes(FakeSim(), scenario)
    assert result["manipulation_checks"]["scenario_geometry_valid"] is True
    assert result["manipulation_checks"]["follower_received_explicit_info_before_self_detection"] is False
    assert result["manipulation_checks"]["communication_dependent_check_passed"] is False


def test_collaboration_metrics_keep_pheromone_separate_from_hazard_awareness():
    # The metric contract is tested through a minimal fake simulation. A
    # pheromone read alone must not count as explicit hazard information.
    class FakeAgent:
        plan_events = []
        path = []
        reached_goal = True

    class FakeSimulation:
        agents = [FakeAgent(), FakeAgent()]

        def communication_instrumentation(self):
            return {
                "hazard_discoveries": [],
                "shared_memory_read_events": [
                    {
                        "reader_agent": 1,
                        "discoverer_agent": 0,
                        "node_id": 999,
                        "channel": "pheromones",
                        "retrieval_step": 2,
                        "step": 2,
                        "route_relevant": True,
                        "cross_agent": True,
                    }
                ],
            }

    scenario = {
        "hazard_node": 999,
        "leader_agent": 0,
        "follower_agents": [1],
    }
    result = _summarise_collaboration_outcomes(FakeSimulation(), scenario)
    follower = result["followers"]["1"]
    assert follower["retrieved_before_self_detection"] is False
    assert follower["explicit_hazard_info_before_self_detection"] is False
    assert follower["pheromone_retrieved_before_self_detection"] is True
    assert result["followers_received_explicit_hazard_info_before_self_detection"] == 0
    assert result["followers_retrieved_pheromone_before_self_detection"] == 1


def test_paired_seed_propagation():
    """An explicit CLI seed is used for the first paired run."""
    assert _paired_seed(42, 0) == 42
    assert _paired_seed(42, 1) == 84
    assert _paired_seed(None, 0) == 0


def test_dynamic_exposure_scenario_is_reproducible_and_route_anchored():
    graph_a = load_fresh_graph()
    tasks_a, event_a = generate_dynamic_hazard_exposure_scenario(
        graph_a, 2, "earthquake", 42, min_dist=1000, max_dist=4000
    )
    graph_b = load_fresh_graph()
    tasks_b, event_b = generate_dynamic_hazard_exposure_scenario(
        graph_b, 2, "earthquake", 42, min_dist=1000, max_dist=4000
    )

    assert tasks_a == tasks_b
    assert event_a == event_b
    anchor = event_a["reference_route"][event_a["reference_route_index"]]
    anchor_node = graph_a._nodes[anchor]
    assert event_a["centre"] == (anchor_node["lat"], anchor_node["lon"])


def test_dynamic_exposure_engine_schedules_event_after_start():
    graph = load_fresh_graph()
    _, event = generate_dynamic_hazard_exposure_scenario(
        graph, 2, "earthquake", 42, min_dist=1000, max_dist=4000
    )
    engine = DynamicHazardExposureEngine(
        graph=graph,
        disaster_type="earthquake",
        centre=DISASTER_CENTRE,
        radius_m=DISASTER_RADIUS_M,
        severity=DISASTER_SEVERITY,
        seed=42,
        wind_bearing_deg=WIND_BEARING,
        exposure_event=event,
    )
    engine.initialise()
    events = []
    for step in range(1, DYNAMIC_EXPOSURE_STEP + 1):
        events.extend(engine.tick(step))

    exposure_events = [
        item for item in events if item.event_type == "dynamic_hazard_exposure"
    ]
    assert len(exposure_events) == 1
    assert exposure_events[0].step == DYNAMIC_EXPOSURE_STEP
    assert engine.exposure_event_applied


def test_dynamic_exposure_run_records_event_and_exposure_metrics():
    result = run_single(
        disaster_type="earthquake",
        workload="dynamic_hazard_exposure",
        num_agents=2,
        steps=DYNAMIC_EXPOSURE_STEP + 5,
        min_dist=1000,
        max_dist=4000,
        seed=42,
        verbose=False,
    )

    assert result["exposure_event"]["trigger_step"] == DYNAMIC_EXPOSURE_STEP
    assert result["hazard_exposure"]["exposure_event_applied"]
    assert result["hazard_exposure"]["damaged_nodes"] > 0
    assert result["hazard_exposure"]["disabled_edges"] > 0
    assert result["communication_instrumentation"]["sensing_events"] > 0


def test_dynamic_exposure_instrumentation_uses_real_observations_and_channels():
    from test_comms_roles import make

    observed = make(n=6, steps=40, comm=CommConfig(local_messaging=False))
    observed_metrics = observed.communication_instrumentation()
    assert observed_metrics["sensing_events"] > 0
    assert observed_metrics["observation_events"] > 0
    assert observed_metrics["board_successful_deliveries"] > 0

    disabled = make(
        n=6,
        steps=40,
        comm=CommConfig(
            shared_memory=False,
            local_messaging=False,
            pheromones=False,
        ),
    )
    disabled_metrics = disabled.communication_instrumentation()
    assert disabled_metrics["board_successful_deliveries"] == 0
    assert disabled_metrics["local_message_successful_deliveries"] == 0
    assert not disabled.shared_memory["pheromones"]


def test_paired_dynamic_exposure_runs_share_manifest_and_event():
    paired = run_paired_communication_experiment(
        disaster_type="earthquake",
        workload="dynamic_hazard_exposure",
        n_runs=1,
        num_agents=2,
        steps=DYNAMIC_EXPOSURE_STEP + 5,
        min_dist=1000,
        max_dist=4000,
        output_path=None,
        base_seed=42,
    )
    run = paired[0]
    configurations = run["configurations"].values()
    manifests = [result["task_manifest"] for result in configurations]
    events = [result["exposure_event"] for result in run["configurations"].values()]
    assert run["seed"] == 42
    assert all(manifest == manifests[0] for manifest in manifests)
    assert all(event == events[0] for event in events)
    assert run["exposure_event"] == events[0]


def test_temporal_node_encounter_excludes_pre_damage_traversal():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    node = sim.agents[0].current_node
    sim._dynamic_damage_first_step[node] = 2
    sim._record_temporal_encounters(
        0, sim.agents[0].current_node, node, 1)
    assert node not in sim._hazard_exposure["actual_node_encounters"]
    sim._record_temporal_encounters(
        0, sim.agents[0].current_node, node, 2)
    assert node in sim._hazard_exposure["actual_node_encounters"]


def test_temporal_edge_encounter_excludes_pre_disable_traversal():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    start = sim.agents[0].current_node
    end = sim.graph._adj[start][0][0]
    edge = frozenset({start, end})
    sim._dynamic_edge_first_step[edge] = 2
    sim.graph._disabled_edges.add(edge)
    sim._record_temporal_encounters(0, start, end, 1)
    assert edge not in sim._hazard_exposure["actual_edge_encounters"]
    sim._record_temporal_encounters(0, start, end, 2)
    assert edge in sim._hazard_exposure["actual_edge_encounters"]


def test_hazard_discovery_attribution_is_hazard_and_time_specific():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    sim._record_hazard_discoveries(0, "teammate_message", 3, {101})
    sim._last_decision_step[0] = 2
    assert sim._preceding_discovery_sources(0, 3) == {"teammate_message"}
    sim._last_decision_step[0] = 3
    assert sim._preceding_discovery_sources(0, 4) == set()


def test_damage_metrics_count_positive_damage_only():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    node = sim.agents[0].current_node
    sim.graph._node_damage[node] = 0.0
    assert node not in sim._damaged_nodes()
    sim.graph._node_damage[node] = 0.2
    assert node in sim._damaged_nodes()


def test_occupying_agent_counts_dynamic_node_encounter_immediately():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    node = sim.agents[0].current_node
    sim._record_occupying_node_encounters({node}, 3)
    assert node in sim._hazard_exposure["actual_node_encounters"]
    assert 0 in sim._hazard_exposure["agents_encountered_hazard"]


def test_disabled_edge_attempt_is_not_successful_traversal():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    start = sim.agents[0].current_node
    end = sim.graph._adj[start][0][0]
    edge = frozenset({start, end})
    sim._dynamic_edge_first_step[edge] = 2
    sim.graph._disabled_edges.add(edge)
    sim._record_default_navigation_decision(0, start, end, 2)
    assert edge in sim._hazard_exposure[
        "actual_edge_attempts_after_disablement"
    ]
    assert edge not in sim._hazard_exposure[
        "successful_edge_traversals_after_disablement"
    ]


def test_edge_disablement_backfills_prior_successful_traversal():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    start = sim.agents[0].current_node
    end = sim.graph._adj[start][0][0]
    edge = frozenset({start, end})
    sim._agent_edge_traversal_steps[0].append((edge, 1))
    sim._record_pre_disablement_traversals({edge}, 2)
    assert edge in sim._hazard_exposure[
        "successful_edge_traversals_before_disablement"
    ]
    assert edge not in sim._hazard_exposure["actual_edge_encounters"]


def test_shared_memory_availability_is_not_retrieval():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    sim.board_visible[123] = 1
    sim.shared_memory["blocked_nodes"].add(123)
    metrics = sim.communication_instrumentation()
    assert metrics["shared_memory_available_nodes"] == 1
    assert metrics["shared_memory_retrievals_recorded"] == 0
    assert metrics["shared_memory_retrievals_instrumented"] is True
    assert not any(
        record["source"] == "shared_memory"
        for record in metrics["hazard_discoveries"]
    )


def test_navigation_decisions_compare_same_current_node_only():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    sim._record_default_navigation_decision(0, 1, 2, 1)
    sim._record_default_navigation_decision(0, 2, 3, 2)
    sim._record_default_navigation_decision(0, 1, 4, 3)
    decisions = sim._navigation_decisions
    assert decisions[1]["changed_from_previous_decision_at_same_node"] is False
    assert decisions[2]["changed_from_previous_decision_at_same_node"] is True


def test_unrelated_discovery_is_not_response_relevant():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    current = sim.agents[0].current_node
    sim._record_hazard_discoveries(0, "teammate_message", 1, {999999})
    assert sim._preceding_discoveries(0, 2, current, None) == []


def test_historical_path_nodes_do_not_make_discoveries_relevant():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    agent = sim.agents[0]
    current = agent.current_node
    historical_only = 999999
    agent.path.append(historical_only)
    sim._record_hazard_discoveries(
        0, "teammate_message", 1, {historical_only}
    )

    assert sim._preceding_discoveries(
        0, 2, current_node=current, next_node=None
    ) == []


def test_planner_discovery_snapshot_survives_observer_timestamp_update():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    agent = sim.agents[0]
    current = agent.current_node
    adjacent = sim.graph._adj[current][0][0]
    sim._last_decision_step[0] = 3
    sim._record_hazard_discoveries(
        0, "teammate_message", 4, {adjacent}
    )
    # This is the snapshot taken immediately before the planner decision.
    snapshot = sim._preceding_discoveries(
        0,
        4,
        current_node=current,
        last_decision_step=3,
        plan=tuple(agent.plan or ()),
    )
    assert {item["source"] for item in snapshot} == {"teammate_message"}

    # The observer runs during agent.step() and advances the timestamp.
    sim._record_default_navigation_decision(
        0, current, adjacent, 4, policy_type="planner_assisted"
    )
    assert sim._preceding_discoveries(
        0, 4, current_node=current, next_node=adjacent
    ) == []

    sources = sim._planner_replan_sources(
        0, 4, current, snapshot, ()
    )
    assert sources == {"teammate_message"}


def test_navigation_decision_metrics_separate_default_and_planner_policies():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    a0, a1 = sim.agents
    a0.use_planner = False
    a1.use_planner = True
    n0 = sim.graph._adj[a0.current_node][0][0]
    n1 = sim.graph._adj[a1.current_node][0][0]

    sim._record_default_navigation_decision(
        0, a0.current_node, n0, 1
    )
    sim._record_default_navigation_decision(
        1, a1.current_node, n1, 1
    )

    response = sim.response_metrics()
    assert len(response["default_policy_navigation_decisions"]) == 1
    assert response["default_policy_navigation_decisions"][0]["agent"] == 0
    assert len(response["planner_assisted_navigation_decisions"]) == 1
    assert response["planner_assisted_navigation_decisions"][0]["agent"] == 1


def test_near_goal_rescue_is_recorded_as_its_own_policy_decision():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    agent = sim.agents[0]
    agent._last_dist_to_goal = 200.0
    agent._steps_without_progress = 10
    old_node = agent.current_node

    sim._rescue_near_goal_stuck(agent, agent_idx=0, step=7)

    rescue_records = [
        item for item in sim._navigation_decisions
        if item["policy_type"] == "near_goal_rescue"
    ]
    assert len(rescue_records) == 1
    assert rescue_records[0]["from_node"] == old_node
    assert rescue_records[0]["next_node"] == agent.current_node
    assert rescue_records[0]["step"] == 7


def test_damage_occupancy_and_sensing_thresholds_are_distinct():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    agent = sim.agents[0]
    node = agent.current_node

    # Any positive damage counts as physical occupancy, but 0.2 is below
    # the agent's >0.3 danger-observation threshold.
    sim._dynamic_damage_nodes = {node}
    sim.graph._node_damage[node] = 0.2
    sim._record_occupying_node_encounters({node}, 1)
    sim._record_hazard_exposure(1)
    exposure = sim.hazard_exposure_metrics()

    assert exposure["nearest_initial_hazard_distance_m_by_step"]["1"]["0"] == 0
    assert exposure["nearest_dynamic_hazard_distance_m_by_step"]["1"]["1"] == 0

    # At 0.4 the node crosses the danger-observation threshold, but remains
    # passable because the impassability threshold is >0.6.
    sim.graph._node_damage[node] = 0.4
    sim._record_hazard_exposure(2)
    exposure = sim.hazard_exposure_metrics()
    assert 0 in exposure[
        "agents_with_dynamic_hazard_in_sensing_neighborhood_by_step"
    ]["2"]
    assert sim.graph.get_node_damage(node) <= agent.BLOCK_DAMAGE_THRESHOLD
    assert exposure["damage_threshold_semantics"]["danger_observation"] == "> 0.3"
    assert exposure["damage_threshold_semantics"]["impassable_node"] == "> 0.6"



def test_info_propagation_does_not_count_global_board_visibility_as_retrieval():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    sim.detection_log[123] = (1, 0)
    sim.board_visible[123] = 2
    sim.shared_memory["blocked_nodes"].add(123)

    metrics = sim.info_propagation()
    assert metrics["agent_hazard_pairs"] == 1
    assert metrics["cross_agent_retrievals"] == 0
    assert metrics["retrievals_via_local_messages"] == 0
    assert metrics["retrievals_via_shared_memory"] == 0
    assert metrics["shared_memory_retrievals_instrumented"] is True
    assert metrics["shared_memory_available_agent_hazard_pairs"] == 1
    assert metrics["fraction_informed"] == 0.0


def test_info_propagation_counts_actual_teammate_message_once():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    sim.detection_log[123] = (1, 0)
    sim.board_visible[123] = 2
    sim.shared_memory["blocked_nodes"].add(123)
    sim.agents[1].learned_at[123] = (3, "msg")

    metrics = sim.info_propagation()
    assert metrics["cross_agent_retrievals"] == 1
    assert metrics["retrievals_via_local_messages"] == 1
    assert metrics["retrievals_via_shared_memory"] == 0
    assert metrics["shared_memory_available_agent_hazard_pairs"] == 1
    assert metrics["fraction_informed"] == 1.0
    assert metrics["mean_delay_steps"] == 1.0
    assert metrics["mean_message_retrieval_delay_steps"] == 2.0



def test_shared_memory_successful_blocked_node_lookup_is_recorded():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    reader = sim.agents[1]
    node = reader.current_node
    sim.shared_memory["blocked_nodes"].add(node)
    sim._board_publications[node] = {
        "blocked_nodes": {"step": 2, "discoverer": 0}
    }
    sim.current_step = 4

    assert node in reader.shared_memory["blocked_nodes"]

    records = sim.communication_instrumentation()["shared_memory_read_events"]
    assert len(records) == 1
    record = records[0]
    assert record["reader_agent"] == 1
    assert record["discoverer_agent"] == 0
    assert record["node_id"] == node
    assert record["channel"] == "blocked_nodes"
    assert record["cross_agent"] is True
    assert record["route_relevant"] is True
    assert record["publication_step"] == 2
    assert record["retrieval_step"] == 4
    assert record["retrieval_delay_steps"] == 2


def test_shared_memory_successful_danger_lookup_is_recorded_once_per_step():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    reader = sim.agents[1]
    node = reader.current_node
    sim.shared_memory["danger_nodes"][node] = 0.7
    sim._board_publications[node] = {
        "danger_nodes": {"step": 3, "discoverer": 0}
    }
    sim.current_step = 5

    assert reader.shared_memory["danger_nodes"].get(node, 0.0) == 0.7
    assert reader.shared_memory["danger_nodes"].get(node, 0.0) == 0.7

    records = sim.communication_instrumentation()["shared_memory_read_events"]
    assert len(records) == 1
    assert records[0]["channel"] == "danger_nodes"
    assert records[0]["cross_agent"] is True
    assert records[0]["retrieval_delay_steps"] == 2


def test_shared_memory_reads_by_original_discoverer_are_not_cross_agent():
    from test_comms_roles import NoEngine, make

    sim = make(n=2, steps=0, engine_cls=NoEngine)
    reader = sim.agents[0]
    node = reader.current_node
    sim.shared_memory["blocked_nodes"].add(node)
    sim._board_publications[node] = {
        "blocked_nodes": {"step": 1, "discoverer": 0}
    }

    assert node in reader.shared_memory["blocked_nodes"]
    metrics = sim.communication_instrumentation()
    assert metrics["shared_memory_retrievals_recorded"] == 0
    assert metrics["shared_memory_retrievals_instrumented"] is True
    assert len(metrics["shared_memory_read_events"]) == 1
    assert metrics["shared_memory_read_events"][0]["cross_agent"] is False



def test_collaboration_task_builder_creates_shared_hazard_to_shelter_route():
    from test_comms_roles import GridGraph

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks, scenario = build_collaboration_tasks_from_route(
        graph, route, num_agents=2
    )

    assert tasks[0]["goal"] == tasks[1]["goal"] == route[-1]
    assert tasks[0]["start"] == route[9]
    assert tasks[1]["start"] == route[0]
    assert scenario["hazard_node"] == route[10]
    assert scenario["route_overlap_fraction"] > 0.5
    assert scenario["leader_to_nearest_follower_distance_m"] > 500.0
    assert scenario["hazard_node"] in scenario["common_route_nodes"]


def test_collaboration_shared_memory_reaches_follower_before_self_detection():
    from test_comms_roles import GridGraph, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from dynamic_disaster_engine import DisasterEvent
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    def make_sim(shared_memory):
        graph = GridGraph()
        route = [graph.node(10, col) for col in range(30)]
        tasks, scenario = build_collaboration_tasks_from_route(
            graph, route, num_agents=2
        )
        hazard = scenario["hazard_node"]
        node_data = graph._nodes[hazard]

        class OneNodeHazard:
            disaster_type = "scripted"

            def __init__(self, g):
                self.graph = g
                self.events = []

            def initialise(self):
                self.graph.reset_disaster()
                return DisasterEvent(
                    0, "initial", (node_data["lat"], node_data["lon"]),
                    0, 0, [], 0, "initial state",
                )

            def tick(self, step):
                if step != 1:
                    return []
                self.graph._node_damage[hazard] = 0.9
                event = DisasterEvent(
                    0, "dynamic_hazard_exposure",
                    (node_data["lat"], node_data["lon"]),
                    10.0, 0.9, [hazard], 0,
                    "single blocked route node",
                )
                self.events.append(event)
                return [event]

            def summary(self):
                return {"disaster_type": "scripted", "total_events": len(self.events)}

        cfg = CommConfig(
            shared_memory=shared_memory,
            local_messaging=False,
            pheromones=False,
            radius_m=500.0,
        )
        sim = quiet(
            DynamicSwarmSimulation,
            graph,
            num_agents=2,
            seed=7,
            tasks=tasks,
            require_feasible=True,
            comm=cfg,
            roles=["navigator", "navigator"],
            engine=OneNodeHazard(graph),
        )
        quiet(sim.run, 4)
        return sim, scenario

    informed, scenario = make_sim(shared_memory=True)
    target = scenario["hazard_node"]
    follower = informed.agents[1]

    reads = [
        event for event in informed.communication_instrumentation()[
            "shared_memory_read_events"
        ]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["node_id"] == target
        and event["channel"] == "blocked_nodes"
    ]
    assert reads, "Follower never actually read the leader's shared hazard report."
    assert reads[0]["route_relevant"] is True
    assert reads[0]["retrieval_step"] < 4

    teammate_replans = [
        event for event in follower.plan_events
        if target in event.get("blocked_route_nodes", [])
        and event.get("blocked_route_sources", {}).get(target) == "shared_memory"
    ]
    assert teammate_replans, "Follower did not invalidate its known route from shared information."
    assert teammate_replans[0]["route_changed"] is True
    assert target not in follower.path

    self_detection_steps = [
        event["step"]
        for event in informed.communication_instrumentation()["hazard_discoveries"]
        if event["agent"] == 1
        and event["hazard_id"] == target
        and event["source"] == "self_observation"
    ]
    assert not self_detection_steps or reads[0]["retrieval_step"] < min(self_detection_steps)

    uninformed, _ = make_sim(shared_memory=False)
    uninformed_follower = uninformed.agents[1]
    assert not [
        event for event in uninformed_follower.plan_events
        if target in event.get("blocked_route_nodes", [])
        and event.get("blocked_route_sources", {}).get(target) == "shared_memory"
    ]
    assert not [
        event for event in uninformed.communication_instrumentation()[
            "shared_memory_read_events"
        ]
        if event["reader_agent"] == 1 and event["node_id"] == target
    ]



def test_leading_agent_pheromone_trail_is_retrieved_by_shared_shelter_follower():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks, _ = build_collaboration_tasks_from_route(graph, route, num_agents=2)
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north


def test_teammate_hazard_message_changes_immediate_next_hop_before_self_sensing():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent
    from comms import Message

    graph = GridGraph()
    start = graph.node(10, 10)
    warned_node = graph.node(10, 11)
    goal = graph.node(10, 20)
    agent = SNNAgent(graph, start, goal)
    agent.use_planner = True
    agent.plan = agent._astar_known()
    original_next = agent.plan[1]
    assert original_next == warned_node

    # The warning arrives before the hazard materialises locally. The graph
    # remains undamaged, so this decision is caused by teammate information,
    # not by the agent's own sensor or a ground-truth graph mutation.
    agent.receive(Message(
        id=0,
        kind="hazard",
        sender=0,
        sent_step=0,
        payload={"blocked": (warned_node,), "danger": {}},
    ))
    agent.step()

    assert agent.current_node != original_next
    assert agent.plan[1] == agent.current_node
    assert warned_node not in agent.plan
    assert agent.stats["blocked_learned_msg"] == 1
    assert agent.stats["blocked_learned_self"] == 0
    replan = agent.plan_events[-1]
    assert replan["route_changed"] is True
    assert replan["first_divergence_hops_from_current"] == 0
    assert replan["blocked_route_sources"][warned_node] == "msg"


def test_pheromones_are_deposited_incrementally_before_goal_completion():
    from test_comms_roles import GridGraph, NoEngine, quiet
    from comms import CommConfig, build_delay_matrix_configs
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = GridGraph()
    route = [graph.node(10, col) for col in range(30)]
    tasks = [
        {"start": route[1], "goal": route[-1]},
        {"start": route[0], "goal": route[-1]},
    ]
    cfg = CommConfig(
        shared_memory=False,
        local_messaging=False,
        pheromones=True,
        radius_m=500.0,
    )
    sim = quiet(
        DynamicSwarmSimulation,
        graph,
        num_agents=2,
        seed=17,
        tasks=tasks,
        require_feasible=True,
        comm=cfg,
        engine=NoEngine(graph),
    )
    # Initial occupied nodes are marked at t=0; the leader's first newly
    # visited node is deposited during movement, not at goal completion.
    assert sim._pheromone_publications[route[1]][0]["step"] == 0
    quiet(sim.run, 1)
    assert not sim.agents[0].reached_goal
    deposited = [
        (node, item)
        for node, publications in sim._pheromone_publications.items()
        for item in publications
        if item["agent"] == 0 and item["step"] == 1
    ]
    assert deposited, "Leader did not publish a trail during movement."
    reads = [
        event for event in sim.communication_instrumentation()["pheromone_read_events"]
        if event["reader_agent"] == 1
        and event["discoverer_agent"] == 0
        and event["retrieval_step"] == 1
    ]
    assert reads, "Follower did not read a teammate trail before goal completion."


def test_default_snn_policy_can_change_next_hop_due_to_teammate_pheromone():
    from test_comms_roles import GridGraph
    from agent_snn_v3 import SNNAgent

    def choose(pheromones):
        graph = GridGraph()
        current = graph.node(10, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        north = graph.node(9, 10)
        # Put the goal at the current coordinate (but keep a distinct graph
        # node) so the default policy takes its deterministic near-goal branch.
        # North and south are symmetric candidates; the pheromone feature is
        # the only difference between their scores.
        goal = graph.node(10, 20)
        graph._nodes[goal]["lat"] = graph._nodes[current]["lat"]
        graph._nodes[goal]["lon"] = graph._nodes[current]["lon"]
        agent = SNNAgent(graph, current, goal, rng=__import__("random").Random(1))
        agent.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes": {},
            "pheromones": dict(pheromones),
        }
        agent.step()
        return agent.current_node, north

    without_trail, north = choose({})
    with_teammate_trail, north = choose({north: 1.0})
    assert without_trail != north
    assert with_teammate_trail == north
