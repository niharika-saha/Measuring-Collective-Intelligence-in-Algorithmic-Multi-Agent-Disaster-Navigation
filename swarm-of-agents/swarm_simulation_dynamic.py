"""
swarm_simulation_dynamic.py
----------------------------
Dynamic-environment swarm simulation.

Extends SwarmSimulation v3 with per-step calls to DynamicDisasterEngine,
giving agents a genuinely changing environment to navigate.

Key differences from swarm_simulation_v3.py
--------------------------------------------
  Per-step disaster  DynamicDisasterEngine.tick() is called BEFORE agents
                     move each step. Disaster ground truth is retained by
                     the simulator for evaluation, but is not injected into
                     agent state as an automatic reroute instruction.

  Response metrics  Planner-assisted agents retain their genuine plan_events.
                     The default greedy SNN-inspired policy is measured through
                     its actual navigation decisions; no planned-route
                     invalidation or stall-based adaptation metric is invented.

  Event log          self.disaster_log records every DisasterEvent with the
                     step it fired — useful for correlating with agent
                     behaviour in the evaluator.

Usage
-----
    from hybrid_spatial_index import HybridTokyoGraph
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = HybridTokyoGraph("dataset/tokyo_graph.json")
    graph.build_index()

    sim = DynamicSwarmSimulation(
        graph,
        num_agents=10,
        disaster_type="earthquake",       # or "flood" | "fire" | "compound"
        disaster_centre=(35.68, 139.76),
        disaster_radius_m=2000,
        disaster_severity=0.7,
        seed=42,
    )
    sim.run(steps=200)
    sim.summary()
"""

import random
import math
import sys
import time
from scipy.spatial import cKDTree
from agent_snn_v3 import SNNAgent, haversine_distance
from dynamic_disaster_engine import DynamicDisasterEngine
from feasibility import (agent_rng, spawn_rng, passable_components,
                         candidate_nodes, sample_feasible_task, validate_task,
                         SpawnError)
from comms import CommConfig, MessageBus, BOARD
from roles import (Role, RoleSpec, ROLE_SPECS, HOMOGENEOUS_SPEC, CoordinationConfig,
                   Coordinator, Allocator, resolve_roles)


class _InstrumentedMessageBus(MessageBus):
    """MessageBus that records existing send calls without changing them."""

    def __init__(self, cfg, instrumentation):
        super().__init__(cfg)
        self._instrumentation = instrumentation

    def send(self, sender, kind, payload, step, recipients, delay=None):
        recipients = list(recipients)
        if kind == "board":
            self._instrumentation["board_publication_attempts"] += 1
        else:
            self._instrumentation["local_message_publication_attempts"] += 1

        message = super().send(sender, kind, payload, step, recipients, delay)
        if message is not None:
            if kind == "board":
                self._instrumentation["board_successful_transmissions"] += 1
            else:
                self._instrumentation["local_message_successful_transmissions"] += 1
        return message


class _TrackedSharedSet:
    """Read-through shared-board set that records successful per-agent lookups."""

    def __init__(self, backing, on_read, channel):
        self._backing = backing
        self._on_read = on_read
        self._channel = channel
        self._seen_reads = set()
        self._seen_step = None

    def set_step(self, step):
        if step != self._seen_step:
            self._seen_step = step
            self._seen_reads.clear()

    def __contains__(self, item):
        found = item in self._backing
        if found:
            key = (self._seen_step, item)
            if key not in self._seen_reads:
                self._seen_reads.add(key)
                self._on_read(self._channel, item, "contains")
        return found

    def __iter__(self):
        return iter(self._backing)

    def __len__(self):
        return len(self._backing)

    def add(self, item):
        self._backing.add(item)

    def discard(self, item):
        self._backing.discard(item)

    def __repr__(self):
        return repr(self._backing)



class _TrackedSharedDict:
    """Read-through shared-board mapping with per-step read deduplication."""

    def __init__(self, backing, on_read, channel, step_getter=None):
        self._backing = backing
        self._on_read = on_read
        self._channel = channel
        self._step_getter = step_getter
        self._seen_step = None
        self._seen_keys = set()

    def _record_read_once(self, key, operation):
        if self._step_getter is None:
            self._on_read(self._channel, key, operation)
            return

        step = self._step_getter()

        if step != self._seen_step:
            self._seen_step = step
            self._seen_keys.clear()

        if key in self._seen_keys:
            return

        self._seen_keys.add(key)
        self._on_read(self._channel, key, operation)

    def get(self, key, default=None):
        found = key in self._backing
        if found:
            self._record_read_once(key, "get")
        return self._backing.get(key, default)

    def __getitem__(self, key):
        value = self._backing[key]
        self._record_read_once(key, "getitem")
        return value

    def __contains__(self, key):
        found = key in self._backing
        if found:
            self._record_read_once(key, "contains")
        return found

    def __iter__(self):
        return iter(self._backing)

    def __len__(self):
        return len(self._backing)

    def items(self):
        return self._backing.items()

    def keys(self):
        return self._backing.keys()

    def __repr__(self):
        return repr(self._backing)




class DynamicSwarmSimulation:
    """
    Multi-agent SNN swarm with a temporally evolving disaster environment.
    """

    def __init__(
        self,
        graph,
        num_agents:            int   = 10,
        min_start_goal_distance: float = 1000,
        max_start_goal_distance: float = 4000,
        # Disaster engine parameters
        disaster_type:         str   = "earthquake",
        disaster_centre:       tuple = (35.68, 139.76),
        disaster_radius_m:     float = 2000.0,
        disaster_severity:     float = 0.7,
        wind_bearing_deg:      float = 45.0,
        seed:                  int   = None,
        # Task control
        tasks:                 list  = None,   # [(start, goal) | {"start","goal"}, ...]
        require_feasible:      bool  = True,
        max_optimal_hops:      int   = None,
        spawn_attempts:        int   = 2000,
        # Communication and roles
        comm:                  CommConfig = None,
        roles                  = None,    # None | "default" | {role: n} | [role, ...]
        role_specs:            dict  = None,   # override ROLE_SPECS entries
        coordination:          CoordinationConfig = None,
        shelters:              list  = None,   # shared goal pool (ALLOCATOR role)
        shelter_capacity:      int   = None,
        alert_centre:          tuple = None,   # publicly announced epicentre
        engine                 = None,         # inject a custom engine (tests)
        agent_rngs:            list  = None,   # optional matched per-agent RNG streams
        sequential_task_queue: list  = None,   # legacy one-agent serial queue
        task_queues: list = None,              # one ordered task queue per navigator
    ):
        self.graph      = graph
        self.num_agents = num_agents
        self.min_start_goal_distance = min_start_goal_distance
        self.max_start_goal_distance = max_start_goal_distance

        # No global seeding: every agent owns a private stream derived from
        # (seed, agent_index), and the engine owns its own, so results do not
        # depend on swarm size, agent order, or anything else using `random`.
        self.seed             = seed
        if agent_rngs is not None and len(agent_rngs) != num_agents:
            raise ValueError(
                f"agent_rngs has {len(agent_rngs)} entries, need {num_agents}"
            )
        self.agent_rngs = list(agent_rngs) if agent_rngs is not None else None
        if sequential_task_queue is not None and task_queues is not None:
            raise ValueError("Use task_queues or sequential_task_queue, not both")
        legacy_sequential = sequential_task_queue is not None
        if legacy_sequential:
            if num_agents != 1:
                raise ValueError("sequential_task_queue requires num_agents=1")
            if tasks is None or len(tasks) != len(sequential_task_queue):
                raise ValueError("Legacy sequential task manifest must match its queue")
            if not all(
                isinstance(a, dict) and a.get("start") == b.get("start")
                and a.get("goal") == b.get("goal")
                for a, b in zip(tasks, sequential_task_queue)
            ):
                raise ValueError("Legacy task manifest must match sequential_task_queue")
            task_queues = [list(sequential_task_queue)]
        if task_queues is not None:
            if tasks is None:
                raise ValueError("task_queues requires an explicit initial tasks list")
            if not isinstance(task_queues, (list, tuple)) or len(task_queues) != num_agents:
                raise ValueError("task_queues must contain exactly one queue per agent")
            if any(not isinstance(q, (list, tuple)) or not q for q in task_queues):
                raise ValueError("Every agent task queue must be a non-empty task list")
            if not all(isinstance(task, dict) and {"start", "goal"} <= set(task)
                       for q in task_queues for task in q):
                raise ValueError("Queued task entries need start and goal")
            initial_tasks = [tasks[0]] if legacy_sequential else tasks
            def _initial_matches_queue(i):
                initial = initial_tasks[i]
                first = task_queues[i][0]
                if not isinstance(initial, dict):
                    return False
                # Either start the first task directly, or begin at a common
                # deployment node and route to that task's start first.
                direct = (
                    initial.get("start") == first["start"]
                    and initial.get("goal") == first["goal"]
                )
                deployment = initial.get("goal") == first["start"]
                return direct or deployment

            if len(initial_tasks) != num_agents or not all(
                _initial_matches_queue(i) for i in range(num_agents)
            ):
                raise ValueError(
                    "Each initial task must either match its first queued task "
                    "or route to that task's start as a deployment leg"
                )
            flat_ids = [task.get("task_id") for q in task_queues for task in q
                        if task.get("task_id") is not None]
            if len(flat_ids) != len(set(flat_ids)):
                raise ValueError("Task IDs must be unique across task_queues")
        self.task_queues = [list(q) for q in task_queues] if task_queues is not None else None
        self.sequential_task_queue = (
            list(sequential_task_queue) if sequential_task_queue is not None else None
        )
        self._task_queue_states = {}
        self._sequential_task_state = None
        self.sequential_task_results = []
        self.require_feasible = require_feasible
        self.max_optimal_hops = max_optimal_hops
        self.spawn_attempts   = spawn_attempts
        self._given_tasks     = tasks
        self.tasks: list      = []     # task info per agent (start, goal, optimal_m, ...)

        # ---- Disaster engine ----
        self.engine = engine or DynamicDisasterEngine(
            graph            = graph,
            disaster_type    = disaster_type,
            centre           = disaster_centre,
            radius_m         = disaster_radius_m,
            severity         = disaster_severity,
            seed             = seed,
            wind_bearing_deg = wind_bearing_deg,
        )

        # ---- Communication, roles, coordination ----
        self.comm         = comm or CommConfig()
        # Runtime trace for controlled hazard-exposure experiments. This is
        # diagnostic-only: it records what the engine changed and what the
        # designated agent could sense, without injecting hazard knowledge.
        self._runtime_hazard_diagnostics = []

        self._comm_instrumentation = {
            "observation_publication_calls": 0,
            "observation_events": 0,
            "agents_with_non_empty_observations": set(),
            "newly_observed_blocked_nodes": 0,
            "newly_observed_danger_nodes": 0,
            "observations_with_new_information": 0,
            "board_publication_attempts": 0,
            "board_successful_transmissions": 0,
            "board_successful_deliveries": 0,
            "local_message_publication_attempts": 0,
            "local_message_successful_transmissions": 0,
            "local_message_successful_deliveries": 0,
        }
        self.bus          = _InstrumentedMessageBus(
            self.comm, self._comm_instrumentation)
        self.coord_cfg    = coordination or CoordinationConfig()
        self.coordinator  = Coordinator(self, self.coord_cfg)
        self.alert_centre = alert_centre if alert_centre is not None else disaster_centre
        self.role_list    = resolve_roles(roles, num_agents, with_allocator=bool(shelters))
        self._specialised = roles is not None
        self._role_specs  = {**ROLE_SPECS, **{Role(k): v for k, v in (role_specs or {}).items()}}
        self.allocator    = None
        self._shelters    = list(shelters) if shelters else None
        self._shelter_capacity = shelter_capacity
        self.cands        = []
        self.board_visible: dict = {}     # blocked node -> step it became readable
        self._board_publications: dict = {}  # node -> channel -> publication metadata
        self._shared_memory_reads: list[dict] = []
        self._local_message_reads: list[dict] = []
        self._local_message_read_keys: set = set()
        self._shared_memory_read_keys: set = set()
        self._pheromone_publications: dict = {}
        # Per-agent path cursor: only newly visited path entries are deposited.
        self._pheromone_deposited_path_lengths: dict[int, int] = {}
        self._board_rep_blocked: dict = {}
        self._board_rep_danger:  dict = {}
        self._radio_rep_blocked: dict = {}
        self._radio_rep_danger:  dict = {}
        self._observed_blocked: dict = {}
        self._observed_danger: dict = {}
        self._contacts: set = set()
        self._initial_damage_nodes: set = set()
        self._initial_disabled_edges: set = set()
        self._dynamic_damage_nodes: set = set()
        self._dynamic_disabled_edges: set = set()
        self._dynamic_damage_first_step: dict = {}
        self._dynamic_edge_first_step: dict = {}
        self._hazard_exposure = {
            "nearest_initial_hazard_distance_m_by_step": {},
            "nearest_dynamic_hazard_distance_m_by_step": {},
            "agents_with_hazard_in_sensing_neighborhood_by_step": {},
            "agents_with_dynamic_hazard_in_sensing_neighborhood_by_step": {},
            # Per-agent, per-step maximum node-damage severity in the
            # current-node/one-hop neighbourhood (proximity dose proxy).
            "severity_weighted_exposure_by_agent_by_step": {},
            "dynamic_severity_weighted_exposure_by_agent_by_step": {},
            "agents_encountered_hazard": set(),
            "agents_occupying_dynamically_damaged_nodes": set(),
            "actual_node_encounters": set(),
            "actual_edge_encounters": set(),
            "actual_edge_attempts_after_disablement": set(),
            "successful_edge_traversals_before_disablement": set(),
            "successful_edge_traversals_after_disablement": set(),
            "traversed_route_nodes_affected_by_damage": set(),
            "traversed_edges_that_became_disabled": set(),
        }
        self._hazard_discoveries: list[dict] = []
        self._hazard_discovery_keys: set = set()
        # Nearest-hazard diagnostics use cached spherical KD-trees instead of
        # scanning every damaged node for every agent at every simulation step.
        # Keep only the latest tree for each hazard class to bound memory.
        self._hazard_distance_index_cache = {
            "initial": None,
            "dynamic": None,
        }
        # Event-level response instrumentation. Ground-truth events are
        # registered separately from agent awareness; registration never
        # changes agent knowledge or navigation state.
        self._hazard_event_records: dict[str, dict] = {}
        self._event_ids_by_node: dict = {}
        self._event_ids_by_edge: dict = {}
        self._event_counter = 0
        self._agent_event_awareness: dict[tuple[int, str], dict] = {}
        self._agent_event_responses: dict[tuple[int, str], dict] = {}
        self._agent_node_entry_steps: dict[int, list] = {}
        self._agent_edge_traversal_steps: dict[int, list] = {}
        self._last_decision_step: dict = {}
        self._last_decision_by_node: dict = {}
        self._navigation_decisions: list[dict] = []
        self._response = {
            "planner_successful_replans": 0,
            "planner_route_changes_following_hazard_discovery": 0,
            # Counts successful, event-linked planner route changes by the
            # agent's FIRST recorded awareness source for that event.
            "planner_replan_source_attribution": {
                "self_observation": 0,
                "teammate_message": 0,
                "shared_memory": 0,
                "unattributed": 0,
            },
            "planner_replan_source_records": [],
        }

        # ---- Shared memory ----
        self.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes":  {},
            "pheromones":    {},
        }

        self.PHEROMONE_DECAY   = 0.97
        self.PHEROMONE_DEPOSIT = 0.5

        # ---- Disaster and adaptation logs ----
        self.disaster_log: list = []          # [(step, DisasterEvent), ...]
        # node -> (step, agent_idx) of the FIRST time any agent sensed it blocked.
        # Shared memory only ever contains nodes that appear here.
        self.detection_log: dict = {}
        self.current_step: int = 0

        self.agents   = []
        self._goal_step: dict = {}
        # Count actual agent decision steps, not makespan * team size. These
        # counters support fair serial-vs-parallel workload accounting.
        self._agent_steps_by_agent: dict[int, int] = {}
        self._agent_finish_step: dict[int, int] = {}
        self._agent_finish_reason: dict[int, str] = {}
        self.coordination_log = self.coordinator.log

        # Apply t=0 mainshock, THEN spawn agents so spawn distances
        # reflect the post-disaster graph passability.
        init_ev = self.engine.initialise()
        self.disaster_log.append((0, init_ev))
        self._initial_damage_nodes = self._damaged_nodes()
        self._initial_disabled_edges = set(self.graph._disabled_edges)
        print(f"\n{init_ev}")

        self._spawn_agents()
        if self.task_queues is not None:
            for agent_idx, queue in enumerate(self.task_queues):
                first = queue[0]
                agent = self.agents[agent_idx]
                direct_start = (
                    agent.current_node == first["start"]
                    and agent.goal_node == first["goal"]
                )
                deployment_start = agent.goal_node == first["start"]
                if not (direct_start or deployment_start):
                    raise ValueError(
                        f"Agent {agent_idx} must start its first task directly "
                        "or navigate to its first task's start node"
                    )
                self._task_queue_states[agent_idx] = {
                    "task_index": 0,
                    "next_task_index": None if direct_start else 0,
                    "phase": "task" if direct_start else "reposition",
                    "path_start_index": 0,
                    "phase_steps": 0,
                    "reposition_path_start_index": None if direct_start else 0,
                    "reposition_steps_before": 0,
                    "reposition_distance_before_m": 0.0, "complete": False,
                }
            if num_agents == 1:
                self._sequential_task_state = self._task_queue_states[0]

    # -----------------------------------------------------------------------
    # Spawn (feasible tasks, per-agent seeded)
    # -----------------------------------------------------------------------

    def _spec(self, role):
        return self._role_specs[role] if self._specialised else HOMOGENEOUS_SPEC

    def _apply_spec(self, agent, spec):
        agent.role          = spec.role.value
        agent.sense_hops    = spec.sense_hops
        agent.stationary    = spec.stationary
        agent.use_planner   = spec.use_planner
        agent.use_lookahead = spec.use_lookahead
        agent.radius_mult   = spec.radius_mult
        agent.snn.apply_weights(spec.weights)

    def _make_view(self, agent_idx):
        """Agent-specific read-through views; successful keyed reads are logged."""
        sm = self.shared_memory
        c = self.comm
        if c.shared_memory:
            blocked = _TrackedSharedSet(
                sm["blocked_nodes"],
                lambda channel, node, operation:
                    self._record_shared_memory_read(
                        agent_idx, channel, node, operation),
                "blocked_nodes",
            )
            danger = _TrackedSharedDict(
                sm["danger_nodes"],
                lambda channel, node, operation:
                    self._record_shared_memory_read(
                        agent_idx, channel, node, operation),
                "danger_nodes",
                step_getter=lambda: self.current_step,
            )
        else:
            blocked, danger = frozenset(), {}
        pheromones = (
            _TrackedSharedDict(
                sm["pheromones"],
                lambda channel, node, operation:
                    self._record_shared_memory_read(
                        agent_idx, channel, node, operation),
                "pheromones",
                step_getter=lambda: self.current_step,
            )
            if c.pheromones else {}
        )
        return {
            "blocked_nodes": blocked,
            "danger_nodes": danger,
            "pheromones": pheromones,
        }

    def _set_shared_view_step(self, step):
        for agent in self.agents:
            shared = getattr(agent, "shared_memory", None) or {}
            blocked = shared.get("blocked_nodes")
            if isinstance(blocked, _TrackedSharedSet):
                blocked.set_step(step)

    def _shared_read_route_relevant(self, agent, node_id):
        """Whether a board lookup concerns the reader's immediate route context."""
        current = agent.current_node
        if node_id == current:
            return True
        if any(n == node_id for n, _ in self.graph._adj.get(current, ())):
            return True
        plan = agent.plan
        if plan:
            plan = list(plan)
            if current in plan and node_id in plan[plan.index(current):]:
                return True
        return False

    def _record_local_message_reads(self, recipient, msg, step):
        """Record hazard information actually merged from a teammate message.

        Records each blocked/danger node in a delivered hazard payload, including
        whether it was novel to the recipient and relevant to its current route.
        Delivery is treated as information acquisition, not proof of causal use.
        """
        if msg.kind == "assign":
            return
        agent = self.agents[recipient]
        payload = msg.payload or {}
        for node_id in payload.get("blocked", ()):
            key = (recipient, msg.id, "blocked_nodes", node_id)
            if key in self._local_message_read_keys:
                continue
            self._local_message_read_keys.add(key)
            # A node being present on the global board is not evidence that
            # this recipient has read it. Count it as previously known only if
            # it is in local memory or this recipient has an earlier successful
            # shared-board lookup for the same node.
            previously_read_from_board = any(
                record.get("reader_agent") == recipient
                and record.get("channel") == "blocked_nodes"
                and record.get("node_id") == node_id
                for record in self._shared_memory_reads
            )
            already_known = (
                node_id in agent.local_memory["blocked_nodes"]
                or previously_read_from_board
            )
            self._local_message_reads.append({
                "reader_agent": recipient,
                "discoverer_agent": msg.sender,
                "node_id": node_id,
                "channel": "teammate_message",
                "message_kind": msg.kind,
                "message_id": msg.id,
                "publication_step": msg.sent_step,
                "retrieval_step": int(step),
                "retrieval_delay_steps": max(0, int(step) - int(msg.sent_step)),
                "cross_agent": msg.sender != recipient,
                "route_relevant": self._shared_read_route_relevant(agent, node_id),
                "new_information": not already_known,
            })
        for node_id, severity in payload.get("danger", {}).items():
            key = (recipient, msg.id, "danger_nodes", node_id)
            if key in self._local_message_read_keys:
                continue
            self._local_message_read_keys.add(key)
            known = float(agent.local_memory["danger_nodes"].get(node_id, 0.0))
            # Include only shared-board values this recipient actually read
            # before this message. Global board availability alone is not
            # recipient awareness.
            for prior_read in self._shared_memory_reads:
                if (
                    prior_read.get("reader_agent") == recipient
                    and prior_read.get("channel") == "danger_nodes"
                    and prior_read.get("node_id") == node_id
                ):
                    observed_value = prior_read.get("observed_value")
                    if observed_value is not None:
                        try:
                            known = max(known, float(observed_value))
                        except (TypeError, ValueError):
                            pass
            self._local_message_reads.append({
                "reader_agent": recipient,
                "discoverer_agent": msg.sender,
                "node_id": node_id,
                "channel": "teammate_message",
                "message_kind": msg.kind,
                "message_id": msg.id,
                "publication_step": msg.sent_step,
                "retrieval_step": int(step),
                "retrieval_delay_steps": max(0, int(step) - int(msg.sent_step)),
                "cross_agent": msg.sender != recipient,
                "route_relevant": self._shared_read_route_relevant(agent, node_id),
                "new_information": float(severity) > float(known) + 1e-12,
            })

    def _record_shared_memory_read(self, reader_idx, channel, node_id, operation):
        """Record a successful, individual board lookup (not mere board visibility)."""
        agent = self.agents[reader_idx]
        if channel == "pheromones":
            publications = self._pheromone_publications.get(node_id, ())
            if not publications:
                return
            teammate_publications = [
                item for item in publications if item["agent"] != reader_idx
            ]
            publication = min(
                teammate_publications or list(publications),
                key=lambda item: item["step"],
            )
            detector = publication["agent"]
            publication_step = publication["step"]
        else:
            publication = self._board_publications.get(node_id, {}).get(channel)
            if publication is None:
                return
            detector = publication.get("discoverer")
            publication_step = publication.get("step")

        # Deduplicate repeated feature reads in the same agent-step. Keep the
        # first concrete lookup and preserve subsequent steps as separate reads.
        step = int(self.current_step)
        key = (reader_idx, channel, node_id, step)
        if key in self._shared_memory_read_keys:
            return
        self._shared_memory_read_keys.add(key)
        cross_agent = detector is not None and detector != reader_idx
        if channel == "blocked_nodes":
            observed_value = node_id in self.shared_memory["blocked_nodes"]
        elif channel == "danger_nodes":
            observed_value = self.shared_memory["danger_nodes"].get(node_id)
        elif channel == "pheromones":
            observed_value = self.shared_memory["pheromones"].get(node_id)
        else:
            observed_value = None

        self._shared_memory_reads.append({
            "reader_agent": reader_idx,
            "discoverer_agent": detector,
            "node_id": node_id,
            "channel": channel,
            "operation": operation,
            "observed_value": observed_value,
            "publication_step": publication_step,
            "retrieval_step": step,
            "retrieval_delay_steps": (
                max(0, step - publication_step)
                if publication_step is not None else None
            ),
            "cross_agent": cross_agent,
            "route_relevant": self._shared_read_route_relevant(agent, node_id),
            "reader_current_node": agent.current_node,
        })
        # Board visibility is not awareness. Only a successful read of a
        # teammate-published hazard node records shared-memory awareness.
        if cross_agent and channel in {"blocked_nodes", "danger_nodes"}:
            self._record_hazard_discoveries(
                reader_idx, "shared_memory", step, {node_id}
            )

    def _sample_shelter_task(self, i, comp, cands):
        rng = spawn_rng(self.seed, i)
        min_d = 0.25 * self.min_start_goal_distance
        for _ in range(self.spawn_attempts):
            s = rng.choice(cands)
            g = self.allocator.pick(self.graph, s, comp=comp,
                                    max_dist=self.max_start_goal_distance)
            if g is None or g == s or self._distance(s, g) < min_d:
                continue
            info = validate_task(self.graph, s, g, comp, self.max_optimal_hops)
            if info is not None:
                return info
        raise SpawnError(f"No feasible shelter task for agent {i}")

    def _spawn_agents(self):
        """
        Create agents with feasible tasks.

        Feasible = start and goal intact after the t=0 damage, connected by a
        passable route (same rules the agent obeys), and optionally within
        `max_optimal_hops`. Pass `tasks=` to reuse identical tasks across
        experiment arms; otherwise task i is drawn from spawn_rng(seed, i).
        With `shelters=`, navigators are assigned a shelter by the allocator
        (nearest with spare capacity); hubs are placed at the centroid of the
        navigators' starts; scouts get their first waypoint.
        """
        comp = passable_components(self.graph)
        cands = candidate_nodes(self.graph, comp)
        self.cands = cands
        if self.task_queues is not None and self.require_feasible:
            for agent_idx, queue in enumerate(self.task_queues):
                for task_index, task in enumerate(queue):
                    info = validate_task(
                        self.graph, task["start"], task["goal"], comp,
                        self.max_optimal_hops,
                    )
                    if info is None:
                        raise ValueError(
                            f"Agent {agent_idx} queued task {task_index} "
                            f"({task['start']} -> {task['goal']}) is infeasible at t=0."
                        )
        given = self._given_tasks
        if given is not None and len(given) < self.num_agents:
            raise ValueError(f"tasks has {len(given)} entries, need {self.num_agents}")

        n_nav = sum(r == Role.NAVIGATOR for r in self.role_list)
        if self._shelters:
            cap = self._shelter_capacity or (-(-n_nav // len(self._shelters)) + 1)
            self.allocator = Allocator(self._shelters, cap)

        for i in range(self.num_agents):
            role = self.role_list[i]
            if given is not None:
                t = given[i]
                s_node, g_node = (t["start"], t["goal"]) if isinstance(t, dict) else t
                info = validate_task(self.graph, s_node, g_node, comp,
                                     self.max_optimal_hops)
                if info is None:
                    if self.require_feasible:
                        raise ValueError(
                            f"Task {i} ({s_node} -> {g_node}) is infeasible in this "
                            f"scenario at t=0. Pass require_feasible=False to allow it.")
                    info = {"start": s_node, "goal": g_node, "feasible": False}
            elif self.allocator is not None and role == Role.NAVIGATOR:
                info = self._sample_shelter_task(i, comp, cands)
            else:
                info = sample_feasible_task(
                    self.graph, spawn_rng(self.seed, i), comp, cands,
                    self.min_start_goal_distance, self.max_start_goal_distance,
                    self.max_optimal_hops, self.spawn_attempts)
            info.setdefault("feasible", True)

            rng = (
                self.agent_rngs[i]
                if self.agent_rngs is not None
                else agent_rng(self.seed, i)
            )
            agent = SNNAgent(self.graph, info["start"], info["goal"],
                             rng=rng, role=role.value)
            agent.decision_observer = (
                lambda current, next_node, step, diagnostics=None, agent_idx=i:
                self._record_default_navigation_decision(
                    agent_idx,
                    current,
                    next_node,
                    step,
                    decision_diagnostics=diagnostics,
                )
            )
            self._apply_spec(agent, self._spec(role))
            agent.shared_memory = self._make_view(i)
            agent.task_info = info
            self.agents.append(agent)
            self.tasks.append(info)

            if self.allocator is not None and role == Role.NAVIGATOR \
                    and info["goal"] in self.allocator.shelters:
                self.allocator.load[info["goal"]] += 1

            self._agent_node_entry_steps[i] = [(info["start"], 0)]
            self._agent_edge_traversal_steps[i] = []
            self._agent_steps_by_agent[i] = 0
            self._last_decision_step[i] = 0
            self._last_decision_by_node[i] = {}
            self._board_rep_blocked[i] = set()
            self._board_rep_danger[i]  = {}
            self._radio_rep_blocked[i] = set()
            self._radio_rep_danger[i]  = {}

            opt = info.get("optimal_m")
            opt_s = f"{opt:.0f}m route" if opt is not None else "route n/a"
            print(f"Agent {i:2d} [{role.value:<9}] | Start: {info['start']} → "
                  f"Goal: {info['goal']} | "
                  f"{info.get('straight_m', float('nan')):.0f}m straight, {opt_s}")

        self._place_hubs(cands)
        for i, a in enumerate(self.agents):
            if a.role == Role.SCOUT.value:
                self.coordinator.scout_waypoint(i)
            if self.comm.pheromones and a.alive and not a.stationary:
                # An agent marks its initial occupied node at t=0. Subsequent
                # trail deposits happen only when it enters a new path node.
                self._deposit_pheromones(a, agent_idx=i)

    def _place_hubs(self, cands):
        hubs = [i for i, a in enumerate(self.agents) if a.stationary]
        if not hubs:
            return
        ref = [a for a in self.agents if a.role == Role.NAVIGATOR.value] or self.agents
        clat = sum(self.graph._nodes[a.current_node]["lat"] for a in ref) / len(ref)
        clon = sum(self.graph._nodes[a.current_node]["lon"] for a in ref) / len(ref)
        node = min(cands, key=lambda n: (self.graph._nodes[n]["lat"] - clat) ** 2
                                        + (self.graph._nodes[n]["lon"] - clon) ** 2)
        for i in hubs:
            a = self.agents[i]
            a.current_node = node
            a.path = [node]
            a.visited = {node}
            a.visit_counts = {node: 1}

    @staticmethod
    def _set_sequential_task_goal(agent, node):
        """Change a queued task waypoint without carrying stale cycle state."""
        agent.set_goal(node)
        agent.prev_node = None
        for attribute in ("blacklist", "position_history"):
            state = getattr(agent, attribute, None)
            if hasattr(state, "clear"):
                state.clear()

    def _queue_for_agent(self, agent_idx):
        return self.task_queues[agent_idx]

    def _record_sequential_task_result(
        self, task_index, agent, step, success, finish_reason,
        path_start_index, phase_steps, reposition_steps=0,
        reposition_distance_m=0.0, agent_idx=0,
    ):
        task = self._queue_for_agent(agent_idx)[task_index]
        segment = agent.path[path_start_index:]
        self.sequential_task_results.append({
            "task_id": task.get("task_id", f"agent{agent_idx}_task_{task_index + 1:03d}"),
            "agent_index": int(agent_idx),
            "start": task["start"], "goal": task["goal"],
            "success": bool(success), "finish_reason": finish_reason,
            "finish_step": int(step), "agent_steps": int(phase_steps),
            "path_distance_m": round(self._path_distance(segment), 3),
            "path_node_count": len(segment),
            "reposition_steps_before": int(reposition_steps),
            "reposition_distance_before_m": round(float(reposition_distance_m), 3),
        })

    def _mark_sequential_tasks_unattempted(self, start_index, step, agent_idx=0):
        queue = self._queue_for_agent(agent_idx)
        for task_index in range(start_index, len(queue)):
            task = queue[task_index]
            self.sequential_task_results.append({
                "task_id": task.get("task_id", f"agent{agent_idx}_task_{task_index + 1:03d}"),
                "agent_index": int(agent_idx),
                "start": task["start"], "goal": task["goal"],
                "success": False, "finish_reason": "not_attempted",
                "finish_step": int(step), "agent_steps": 0,
                "path_distance_m": 0.0, "path_node_count": 0,
                "reposition_steps_before": 0,
                "reposition_distance_before_m": 0.0,
            })

    def _advance_sequential_task_queue(self, agent_idx, agent, step):
        """Advance this agent's independent task queue, including reposition legs."""
        state = self._task_queue_states.get(agent_idx)
        if state is None or state["complete"]:
            return
        queue = self._queue_for_agent(agent_idx)
        state["phase_steps"] += 1

        if not agent.alive:
            if state["phase"] == "task":
                self._record_sequential_task_result(
                    state["task_index"], agent, step, False,
                    agent.death_reason or "failed", state["path_start_index"],
                    state["phase_steps"], state["reposition_steps_before"],
                    state["reposition_distance_before_m"], agent_idx,
                )
                next_unfinished = state["task_index"] + 1
            else:
                next_index = state["next_task_index"]
                reposition_segment = agent.path[state["reposition_path_start_index"]:]
                self._record_sequential_task_result(
                    next_index, agent, step, False, "reposition_failed",
                    len(agent.path), 0, state["phase_steps"],
                    self._path_distance(reposition_segment), agent_idx,
                )
                next_unfinished = next_index + 1
            self._mark_sequential_tasks_unattempted(next_unfinished, step, agent_idx)
            state["complete"] = True
            return

        if not agent.reached_goal:
            return

        if state["phase"] == "task":
            task_index = state["task_index"]
            self._record_sequential_task_result(
                task_index, agent, step, True, "goal", state["path_start_index"],
                state["phase_steps"], state["reposition_steps_before"],
                state["reposition_distance_before_m"], agent_idx,
            )
            next_index = task_index + 1
            if next_index >= len(queue):
                state["complete"] = True
                return

            next_task = queue[next_index]
            state["next_task_index"] = next_index
            state["phase_steps"] = 0
            state["reposition_path_start_index"] = len(agent.path) - 1
            if agent.current_node == next_task["start"]:
                state.update({
                    "phase": "task", "task_index": next_index,
                    "path_start_index": len(agent.path) - 1,
                    "reposition_steps_before": 0,
                    "reposition_distance_before_m": 0.0,
                })
                self._set_sequential_task_goal(agent, next_task["goal"])
            else:
                state["phase"] = "reposition"
                self._set_sequential_task_goal(agent, next_task["start"])
            return

        next_index = state["next_task_index"]
        reposition_segment = agent.path[state["reposition_path_start_index"]:]
        state["reposition_steps_before"] = state["phase_steps"]
        state["reposition_distance_before_m"] = self._path_distance(reposition_segment)
        state.update({
            "task_index": next_index, "phase": "task", "phase_steps": 0,
            "path_start_index": len(agent.path) - 1, "next_task_index": None,
        })
        self._set_sequential_task_goal(agent, queue[next_index]["goal"])

    def _finalize_sequential_task_queue(self):
        for agent_idx, state in self._task_queue_states.items():
            if state is None or state["complete"]:
                continue
            agent = self.agents[agent_idx]
            step = self.current_step
            if state["phase"] == "task":
                self._record_sequential_task_result(
                    state["task_index"], agent, step, False, "timeout",
                    state["path_start_index"], state["phase_steps"],
                    state["reposition_steps_before"],
                    state["reposition_distance_before_m"], agent_idx,
                )
                next_unfinished = state["task_index"] + 1
            else:
                next_index = state["next_task_index"]
                reposition_segment = agent.path[state["reposition_path_start_index"]:]
                self._record_sequential_task_result(
                    next_index, agent, step, False, "reposition_timeout",
                    len(agent.path), 0, state["phase_steps"],
                    self._path_distance(reposition_segment), agent_idx,
                )
                next_unfinished = next_index + 1
            self._mark_sequential_tasks_unattempted(next_unfinished, step, agent_idx)
            state["complete"] = True

    def sequential_task_metrics(self) -> list[dict]:
        """Return task-level results for all serial/parallel task queues."""
        return [dict(row) for row in self.sequential_task_results]

    # -----------------------------------------------------------------------
    # Main run loop
    # -----------------------------------------------------------------------

    def run(self, steps: int = 200, quiet: bool = False):
        nav = Role.NAVIGATOR.value
        self._runtime_performance = {
            "started_at": time.perf_counter(),
            "flood_update_seconds": 0.0,
            "flood_update_calls": 0,
            "progress_reports": [],
        }
        for t in range(1, steps + 1):
            if not quiet:
                print(f"\n{'='*60}")
                print(f"STEP {t}")
                print(f"{'='*60}")
            self.current_step = t
            self._set_shared_view_step(t)
            for a in self.agents:
                a.clock = t

            # 1. Advance disaster — BEFORE agents move
            before_damage = dict(self.graph._node_damage)
            before_edges = set(self.graph._disabled_edges)
            hazard_started_at = time.perf_counter()
            new_events = self.engine.tick(t)
            self._runtime_performance["flood_update_seconds"] += (
                time.perf_counter() - hazard_started_at
            )
            self._runtime_performance["flood_update_calls"] += 1
            changed_nodes = {
                node_id for node_id, damage in self.graph._node_damage.items()
                if damage > before_damage.get(node_id, 0.0)
            }
            changed_edges = self.graph._disabled_edges - before_edges
            for node_id in changed_nodes:
                self._dynamic_damage_nodes.add(node_id)
                self._dynamic_damage_first_step.setdefault(node_id, t)
            for edge_id in changed_edges:
                self._dynamic_disabled_edges.add(edge_id)
                self._dynamic_edge_first_step.setdefault(edge_id, t)
            self._record_occupying_node_encounters(changed_nodes, t)
            self._record_pre_disablement_traversals(changed_edges, t)
            self._register_hazard_events(new_events, t, changed_nodes, changed_edges)

            # Capture the controlled-event target immediately after the engine
            # ticks and before any agent moves. This distinguishes an event
            # application/graph-sharing problem from a sensing/publication bug.
            exposure = getattr(self.engine, "exposure_event", None)
            if exposure and t == exposure.get("trigger_step") and not any(
                row.get("step") == t for row in self._runtime_hazard_diagnostics
            ):
                route = exposure.get("reference_route") or ()
                route_index = exposure.get("reference_route_index")
                target = (
                    route[route_index]
                    if isinstance(route_index, int) and 0 <= route_index < len(route)
                    else None
                )
                task_id = exposure.get("reference_task_id")

                # The reference route can belong to an upstream follower even
                # when the convoy leader is the agent immediately beside the
                # event node. Select the agent with a real one-hop sensing
                # opportunity first; use task_id only as a fallback.
                adjacent_candidates = []
                for i, candidate in enumerate(self.agents):
                    try:
                        candidate_neighbors = list(candidate.get_all_neighbors())
                    except Exception:
                        candidate_neighbors = []
                    if target is not None and target in candidate_neighbors:
                        adjacent_candidates.append((i, candidate_neighbors))

                if adjacent_candidates:
                    target_agent_idx, neighbors = adjacent_candidates[0]
                    target_agent_selection_reason = "one_hop_adjacent_to_event_target"
                else:
                    target_agent_idx = next((
                        i for i, a in enumerate(self.agents)
                        if str(getattr(a, "task_info", {}).get("task_id")) == str(task_id)
                    ), None)
                    target_agent = (
                        self.agents[target_agent_idx]
                        if target_agent_idx is not None else None
                    )
                    neighbors = (
                        list(target_agent.get_all_neighbors())
                        if target_agent is not None else []
                    )
                    target_agent_selection_reason = "reference_task_id_fallback"

                target_agent = (
                    self.agents[target_agent_idx]
                    if target_agent_idx is not None else None
                )
                self._runtime_hazard_diagnostics.append({
                    "step": t,
                    "event_applied": bool(getattr(self.engine, "exposure_event_applied", False)),
                    "engine_graph_is_simulation_graph": (
                        getattr(self.engine, "graph", None) is self.graph
                    ),
                    "target_node": target,
                    "target_damage_after_tick": (
                        self.graph.get_node_damage(target) if target is not None else None
                    ),
                    "target_agent_index": target_agent_idx,
                    "target_agent_selection_reason": target_agent_selection_reason,
                    "target_agent_task_id": (
                        getattr(target_agent, "task_info", {}).get("task_id")
                        if target_agent is not None else None
                    ),
                    "event_reference_task_id": task_id,
                    "target_agent_graph_is_simulation_graph": (
                        getattr(target_agent, "graph", None) is self.graph
                        if target_agent is not None else None
                    ),
                    "target_agent_current_node_before_move": (
                        target_agent.current_node if target_agent is not None else None
                    ),
                    "target_is_one_hop_neighbor_before_move": (
                        target in neighbors if target is not None and target_agent is not None else None
                    ),
                    "target_agent_observation_after_step": None,
                })
            self._record_hazard_exposure(t)
            for ev in new_events:
                self.disaster_log.append((t, ev))
                if not quiet:
                    print(f"  ⚡ DISASTER: {ev}")

            # 2. Disaster ground truth is not pushed into agent state.
            #    Agents learn through their own sensing and configured
            #    communication channels during agent.step().

            # 3. Adaptive coordination (reads knowledge only; sends via the bus)
            self.coordinator.allocator_step(t)

            # 4. Move agents
            for i, agent in enumerate(self.agents):
                if not agent.alive:
                    continue
                if agent.reached_goal:
                    if agent.role == Role.SCOUT.value:
                        self.coordinator.scout_waypoint(i)     # next patrol leg
                    else:
                        continue

                is_nav = agent.role == nav
                if is_nav:
                    self._rescue_near_goal_stuck(agent, agent_idx=i, step=t)

                prev_node = agent.current_node
                previous_plan = tuple(agent.plan or ())
                previous_plan_events = len(agent.plan_events)

                # Snapshot knowledge before SNNAgent's observer advances the
                # decision timestamp. Include the active route only, not the
                # agent's historical trajectory.
                decision_discovery_floor = self._last_decision_step.get(i, 0)
                decision_plan_snapshot = tuple(agent.plan or ())
                decision_discoveries = self._preceding_discoveries(
                    i,
                    t,
                    current_node=prev_node,
                    last_decision_step=decision_discovery_floor,
                    plan=decision_plan_snapshot,
                )

                self._agent_steps_by_agent[i] = (
                    self._agent_steps_by_agent.get(i, 0) + 1
                )
                agent.step()
                for diagnostic in self._runtime_hazard_diagnostics:
                    if (diagnostic.get("step") == t
                            and diagnostic.get("target_agent_index") == i
                            and diagnostic.get("target_agent_observation_after_step") is None):
                        obs = getattr(agent, "last_observation", {}) or {}
                        target = diagnostic.get("target_node")
                        diagnostic["target_agent_observation_after_step"] = {
                            "blocked_nodes": sorted(
                                (str(n) for n in obs.get("blocked_nodes", ()))
                            ),
                            "danger_nodes": sorted(
                                (str(n) for n in obs.get("danger", {}).keys())
                            ),
                            "target_in_blocked": target in obs.get("blocked_nodes", ()),
                            "target_in_danger": target in obs.get("danger", {}),
                            "target_damage_after_agent_step": (
                                self.graph.get_node_damage(target)
                                if target is not None else None
                            ),
                            "agent_current_node_after_step": agent.current_node,
                        }
                if i in self._task_queue_states:
                    self._advance_sequential_task_queue(i, agent, t)
                if agent.reached_goal:
                    self._agent_finish_step.setdefault(i, t)
                    self._agent_finish_reason.setdefault(i, "goal")
                elif not agent.alive:
                    self._agent_finish_step.setdefault(i, t)
                    self._agent_finish_reason.setdefault(
                        i, agent.death_reason or "failed"
                    )

                # Sensing occurs inside agent.step() before its navigation
                # decision. Preserve those exact pre-movement observations for
                # planner attribution; do not use the post-move current node.
                sensed = agent.last_observation
                self._record_hazard_discoveries(
                    i,
                    "self_observation",
                    t,
                    set(sensed.get("blocked_nodes", ()))
                    | set(sensed.get("danger", {})),
                )

                if agent.current_node != prev_node:
                    self._agent_node_entry_steps[i].append(
                        (agent.current_node, t))
                    self._agent_edge_traversal_steps[i].append(
                        (frozenset({prev_node, agent.current_node}), t))
                    self._record_temporal_encounters(
                        i, prev_node, agent.current_node, t)

                for plan_event in agent.plan_events[previous_plan_events:]:
                    if plan_event.get("reason") != "invalidated":
                        continue

                    # Link the invalidated route to registered hazard events
                    # only when this agent had actually become aware of the
                    # event. Ground truth alone is not evidence of awareness.
                    blocked_nodes = set(
                        plan_event.get("blocked_route_nodes", ())
                    )
                    matched_event_ids = {
                        event_id
                        for node_id in blocked_nodes
                        for event_id in self._event_ids_by_node.get(node_id, ())
                        if (i, event_id) in self._agent_event_awareness
                    }

                    successful_route_change = bool(
                        plan_event.get("ok")
                        and plan_event.get("route_changed")
                    )

                    for event_id in sorted(matched_event_ids):
                        awareness = self._agent_event_awareness.get(
                            (i, event_id), {}
                        )
                        first_source = awareness.get(
                            "awareness_source", "unattributed"
                        )
                        if first_source not in self._response[
                            "planner_replan_source_attribution"
                        ]:
                            first_source = "unattributed"

                        # _record_event_response preserves the first response
                        # per agent/event pair. Only attribute a replan if
                        # this call actually records that first response.
                        self._record_event_response(
                            i,
                            event_id,
                            t,
                            "planner_replan",
                            successful_route_change,
                            {
                                "plan_found": bool(plan_event.get("ok")),
                                "route_changed": bool(
                                    plan_event.get("route_changed")
                                ),
                                "blocked_route_nodes": sorted(
                                    blocked_nodes, key=str
                                ),
                                "causal_trigger": "known_blocked_route_node",
                                "awareness_source": first_source,
                                "awareness_step": awareness.get(
                                    "awareness_step"
                                ),
                            },
                        )

                        # Planner-replan attribution is separate from the
                        # event's first response record: a prior default-policy
                        # response must not hide a later genuine route change.
                        self._record_planner_replan_source(
                            i, event_id, t, awareness,
                            successful_route_change,
                        )

                    if plan_event.get("ok"):
                        self._response["planner_successful_replans"] += 1
                        if tuple(agent.plan or ()) != previous_plan:
                            self._response[
                                "planner_route_changes_following_hazard_discovery"
                            ] += 1

                if agent.current_node != prev_node:
                    cd  = self.graph._nodes[agent.current_node]
                    gd  = self.graph._nodes[agent.goal_node]
                    dtg = haversine_distance(cd['lat'], cd['lon'], gd['lat'], gd['lon'])
                    status = "ALIVE"
                    if not agent.alive:      status = "DEAD"
                    elif agent.reached_goal: status = "GOAL!"
                    if not quiet:
                        print(f"  Agent {i:2d} ({agent.role[:3]}): {prev_node} → "
                              f"{agent.current_node} | DTG: {dtg:6.0f}m | {status}")

                self._publish_observation(i, agent)

                if is_nav and agent.reached_goal:
                    self._goal_step[i] = t

            # Deposit every agent's newly occupied path nodes only after all
            # agents have made this step's decision. This avoids an agent-order
            # advantage where later agents could consume a trail laid earlier
            # in the same sequential movement loop.
            if self.comm.pheromones:
                for i, agent in enumerate(self.agents):
                    if agent.alive and not agent.stationary:
                        self._deposit_pheromones(agent, agent_idx=i)

            # 5. Communication phase: send, then deliver everything due
            self._comms_phase(t)
            if self.comm.pheromones:
                self._evaporate_pheromones()

            active_after = sum(1 for a in self.agents
                               if a.role == nav and a.alive and not a.reached_goal)
            if not quiet:
                print(f"\n  Active navigators: {active_after}")
            if t == 1 or t % 50 == 0 or active_after == 0:
                planner_attempts = sum(
                    int(agent.stats.get("plan_attempts", 0))
                    for agent in self.agents
                )
                planner_expansions = sum(
                    int(agent.stats.get("plan_expansions_total", 0))
                    for agent in self.agents
                )
                planner_failures = sum(
                    int(agent.stats.get("plan_failures", 0))
                    for agent in self.agents
                )
                report = {
                    "disaster_type": getattr(self.engine, "disaster_type", None),
                    "seed": self.seed,
                    "team_size": self.num_agents,
                    "step": t,
                    "active_agents": active_after,
                    "completed_agents": sum(
                        1 for agent in self.agents if agent.reached_goal
                    ),
                    "planning_attempts": planner_attempts,
                    "astar_expansions": planner_expansions,
                    "failed_planning_attempts": planner_failures,
                    "shared_memory_read_callbacks": len(self._shared_memory_reads),
                    "flood_update_seconds": self._runtime_performance[
                        "flood_update_seconds"
                    ],
                }
                self._runtime_performance["progress_reports"].append(report)
                print(
                    "[progress] "
                    f"step={t}/{steps} active={active_after} "
                    f"completed={report['completed_agents']} "
                    f"plans={planner_attempts} expansions={planner_expansions} "
                    f"plan_failures={planner_failures} "
                    f"shared_reads={report['shared_memory_read_callbacks']} "
                    f"flood_seconds={report['flood_update_seconds']:.3f}",
                    file=sys.stderr,
                    flush=True,
                )
            if active_after == 0 and any(a.role == nav for a in self.agents):
                if not quiet:
                    print("\n  All navigators finished.")
                break

        # Finalize any unfinished serial task queue before marking the agent
        # itself as timed out.
        self._finalize_sequential_task_queue()

        # Agents still active at the budget boundary are right-censored as
        # timeouts; record the actual number of decision steps they consumed.
        for i, agent in enumerate(self.agents):
            if agent.role != nav:
                continue
            if i not in self._agent_finish_step:
                self._agent_finish_step[i] = self.current_step
                self._agent_finish_reason[i] = (
                    "timeout" if agent.alive and not agent.reached_goal
                    else ("goal" if agent.reached_goal else (agent.death_reason or "failed"))
                )
        self._runtime_performance["elapsed_seconds"] = (
            time.perf_counter() - self._runtime_performance["started_at"]
        )

    # -----------------------------------------------------------------------
    # Dynamic disaster integration
    # -----------------------------------------------------------------------

    def _own_observation(self, agent):
        """(blocked set, danger dict) this agent physically sensed this step,
        including the node it stands on. Nothing it was merely told."""
        obs = agent.last_observation
        blocked = set(obs["blocked_nodes"])
        danger = dict(obs["danger"])
        node = agent.current_node
        hazard = agent.graph.get_node_damage(node)
        if hazard > agent.BLOCK_DAMAGE_THRESHOLD:
            blocked.add(node)
        if hazard > 0.0:
            danger[node] = max(hazard, danger.get(node, 0.0))
        return blocked, danger

    def _publish_observation(self, idx, agent):
        """
        Record first detections and post what `agent` sensed to the shared board
        (if enabled). Only the delta since its last post is sent, through the bus,
        so board latency and traffic are both modelled.
        """
        self._comm_instrumentation["observation_publication_calls"] += 1
        blocked, danger = self._own_observation(agent)
        observed_blocked = self._observed_blocked.setdefault(idx, set())
        observed_danger = self._observed_danger.setdefault(idx, set())
        new_observed_blocked = blocked - observed_blocked
        new_observed_danger = set(danger) - observed_danger
        observed_blocked.update(blocked)
        observed_danger.update(danger)
        if blocked or danger:
            self._comm_instrumentation["observation_events"] += 1
            self._comm_instrumentation["agents_with_non_empty_observations"].add(idx)
        self._comm_instrumentation["newly_observed_blocked_nodes"] += len(
            new_observed_blocked)
        self._comm_instrumentation["newly_observed_danger_nodes"] += len(
            new_observed_danger)
        if new_observed_blocked or new_observed_danger:
            self._comm_instrumentation["observations_with_new_information"] += 1
            self._record_hazard_discoveries(
                idx, "self_observation", self.current_step,
                new_observed_blocked | new_observed_danger,
            )
        for nid in blocked:
            self.detection_log.setdefault(nid, (self.current_step, idx))

        if not self.comm.shared_memory:
            return
        new_b = blocked - self._board_rep_blocked[idx]
        rep_d = self._board_rep_danger[idx]
        new_d = {n: d for n, d in danger.items() if d > rep_d.get(n, 0.0) + 0.05}
        if not new_b and not new_d:
            return
        self.bus.send(idx, "board", {"blocked": tuple(new_b), "danger": new_d},
                      self.current_step, [BOARD], delay=self.comm.board_delay_steps)
        self._board_rep_blocked[idx] |= new_b
        rep_d.update(new_d)

    # -----------------------------------------------------------------------
    # Communication phase
    # -----------------------------------------------------------------------

    def _range_m(self, idx):
        return self.comm.radius_m * getattr(self.agents[idx], "radius_mult", 1.0)

    def _in_range(self, idx):
        """Alive agents that can exchange messages with `idx` right now."""
        a = self.graph._nodes[self.agents[idx].current_node]
        ri = self._range_m(idx)
        out = []
        for j, b in enumerate(self.agents):
            if j == idx or not b.alive:
                continue
            bn = self.graph._nodes[b.current_node]
            d = haversine_distance(a['lat'], a['lon'], bn['lat'], bn['lon'])
            if d <= max(ri, self._range_m(j)):
                out.append(j)
        return out

    def _comms_phase(self, t):
        c = self.comm
        if c.local_messaging:
            self.coordinator.assessor_step(t)

            # hazard deltas: own-sensed information only (no re-broadcast of
            # other agents' reports; relaying is the assessor's job)
            for i, a in enumerate(self.agents):
                if not a.alive or a.stationary:
                    continue
                blocked, danger = self._own_observation(a)
                new_b = blocked - self._radio_rep_blocked[i]
                rep_d = self._radio_rep_danger[i]
                new_d = {n: d for n, d in danger.items() if d > rep_d.get(n, 0.0) + 0.05}
                if not new_b and not new_d:
                    continue
                msg = self.bus.send(i, "hazard",
                                    {"blocked": tuple(new_b), "danger": new_d},
                                    t, self._in_range(i))
                if msg is not None:       # nobody in range: keep it for later
                    self._radio_rep_blocked[i] |= new_b
                    rep_d.update(new_d)

            if c.sync_on_contact:
                self._sync_contacts(t)

        for recipient, msg in self.bus.deliver(t, is_alive=lambda r: self.agents[r].alive):
            self._apply_delivery(recipient, msg, t)

    def _sync_contacts(self, t):
        """When two agents come into range, each sends the other everything it
        knows. This is what lets knowledge travel with agents (store-carry-forward)."""
        pairs = set()
        for i, a in enumerate(self.agents):
            if not a.alive:
                continue
            for j in self._in_range(i):
                if j > i:
                    pairs.add((i, j))
        for i, j in sorted(pairs - self._contacts):
            for src, dst in ((i, j), (j, i)):
                a = self.agents[src]
                if a.role == Role.ALLOCATOR.value:
                    continue
                blocked = a.local_memory["blocked_nodes"]
                danger = a.local_memory["danger_nodes"]
                if blocked or danger:
                    self.bus.send(src, "sync",
                                  {"blocked": tuple(blocked), "danger": dict(danger)},
                                  t, [dst])
        self._contacts = pairs

    def _apply_delivery(self, recipient, msg, t):
        if recipient == BOARD:
            self._comm_instrumentation["board_successful_deliveries"] += 1
            sm = self.shared_memory
            for nid in msg.payload.get("blocked", ()):
                if nid not in sm["blocked_nodes"]:
                    sm["blocked_nodes"].add(nid)
                    self.board_visible[nid] = t
                    self._board_publications.setdefault(nid, {})[
                        "blocked_nodes"
                    ] = {
                        "step": t,
                        "discoverer": msg.sender,
                    }
            for nid, dmg in msg.payload.get("danger", {}).items():
                if dmg > sm["danger_nodes"].get(nid, 0.0):
                    sm["danger_nodes"][nid] = dmg
                    self._board_publications.setdefault(nid, {})[
                        "danger_nodes"
                    ] = {
                        "step": t,
                        "discoverer": msg.sender,
                    }
        elif msg.kind == "assign":
            self._comm_instrumentation["local_message_successful_deliveries"] += 1
            self.coordinator.on_assign(recipient, msg)
        else:
            self._comm_instrumentation["local_message_successful_deliveries"] += 1
            if msg.payload.get("blocked") or msg.payload.get("danger"):
                self._record_local_message_reads(recipient, msg, t)
                self._record_hazard_discoveries(
                    recipient,
                    "teammate_message",
                    t,
                    set(msg.payload.get("blocked", ()))
                    | set(msg.payload.get("danger", {})),
                )
            self.agents[recipient].receive(msg)

    def _record_hazard_discoveries(self, agent_idx, source, step, node_ids):
        for node_id in node_ids:
            event_ids = sorted(self._event_ids_by_node.get(node_id, ()))
            key = (agent_idx, "node", node_id, source)
            if key in self._hazard_discovery_keys:
                # The same physical node may become relevant to a later
                # disaster event. Preserve the original discovery timestamp,
                # but attach newly registered event IDs and record awareness
                # for the new event at the current observation time.
                for existing in self._hazard_discoveries:
                    if (
                        existing["agent"] == agent_idx
                        and existing["hazard_type"] == "node"
                        and existing["hazard_id"] == node_id
                        and existing["source"] == source
                    ):
                        existing["event_ids"] = sorted(
                            set(existing.get("event_ids", ())) | set(event_ids)
                        )
                        break
            else:
                self._hazard_discovery_keys.add(key)
                self._hazard_discoveries.append({
                    "agent": agent_idx,
                    "hazard_type": "node",
                    "hazard_id": node_id,
                    "source": source,
                    "step": int(step),
                    "event_ids": event_ids,
                })
            for event_id in event_ids:
                self._record_event_awareness(
                    agent_idx, event_id, source, int(step), node_id
                )

    def _register_hazard_events(self, events, step, changed_nodes=(), changed_edges=()):
        """Register ground-truth events without exposing them to agents.

        DisasterEvent currently reports newly damaged node IDs but only an
        edge count. Exact edge attribution is possible only when one event
        occurred in the tick; for multiple simultaneous events the changed
        edges remain step-level and are deliberately not guessed per event.
        """
        events = list(events or ())
        changed_edges = set(changed_edges or ())
        for event_index, event in enumerate(events):
            self._event_counter += 1
            event_id = f"hazard_{self._event_counter:06d}"
            event_nodes = set(getattr(event, "nodes_newly_damaged", ()) or ())
            if not event_nodes and len(events) == 1:
                event_nodes = set(changed_nodes or ())
            reported_edge_ids = getattr(
                event, "edges_newly_disabled_ids", None
            )
            if reported_edge_ids:
                event_edges = {
                    frozenset(edge_id) for edge_id in reported_edge_ids
                }
                edge_attribution = "event_provided_edge_ids"
            elif len(events) == 1:
                event_edges = set(changed_edges)
                edge_attribution = "single_event_graph_diff"
            else:
                event_edges = set()
                edge_attribution = (
                    "unavailable_for_simultaneous_events_without_edge_ids"
                )
            current_affected = []
            route_affected = []
            locally_exposed = []
            for agent_idx, agent in enumerate(self.agents):
                current = agent.current_node
                neighbours = {
                    node for node, _ in self.graph._adj.get(current, ())
                } if current is not None else set()
                local_nodes = set(neighbours)
                if current is not None:
                    local_nodes.add(current)
                local_edges = {
                    frozenset((current, neighbour))
                    for neighbour in neighbours
                } if current is not None else set()
                if event_nodes & local_nodes or event_edges & local_edges:
                    locally_exposed.append(agent_idx)

                plan = list(getattr(agent, "plan", ()) or ())
                if current in plan:
                    active_plan = set(plan[plan.index(current):])
                else:
                    active_plan = set()
                route_hit = bool(event_nodes & active_plan)
                if event_edges and active_plan:
                    route_edges = {
                        frozenset((u, v))
                        for u, v in zip(plan[plan.index(current):],
                                        plan[plan.index(current) + 1:])
                    } if current in plan else set()
                    route_hit = route_hit or bool(event_edges & route_edges)
                if route_hit:
                    route_affected.append(agent_idx)
                if route_hit or agent_idx in locally_exposed:
                    current_affected.append(agent_idx)

            record = {
                "event_id": event_id,
                "event_type": getattr(event, "event_type", "unknown"),
                "event_step": int(getattr(event, "step", step)),
                "centre": list(getattr(event, "centre", ()) or ()),
                "radius_m": float(getattr(event, "radius_m", 0.0) or 0.0),
                "severity": float(getattr(event, "severity", 0.0) or 0.0),
                "affected_nodes": sorted(event_nodes, key=str),
                "affected_edges": [
                    sorted(edge, key=str) for edge in sorted(
                        event_edges, key=lambda e: tuple(sorted(map(str, e)))
                    )
                ],
                "edge_attribution": edge_attribution,
                "affected_agents": sorted(current_affected),
                "route_affected_agents": sorted(route_affected),
                "locally_exposed_agents": sorted(locally_exposed),
                "awareness": {},
                "responses": {},
            }
            self._hazard_event_records[event_id] = record
            for node_id in event_nodes:
                self._event_ids_by_node.setdefault(node_id, set()).add(event_id)
            for edge_id in event_edges:
                self._event_ids_by_edge.setdefault(edge_id, set()).add(event_id)
                for endpoint in edge_id:
                    self._event_ids_by_node.setdefault(endpoint, set()).add(event_id)

    def _record_event_awareness(self, agent_idx, event_id, source, step, hazard_id):
        """Keep the first actual observation/retrieval time for an agent-event."""
        key = (agent_idx, event_id)
        if key in self._agent_event_awareness:
            return
        event = self._hazard_event_records.get(event_id)
        if event is None:
            return
        awareness = {
            "agent": int(agent_idx),
            "event_id": event_id,
            "event_step": event["event_step"],
            "awareness_step": int(step),
            "awareness_source": source,
            "awareness_delay_steps": max(0, int(step) - event["event_step"]),
            "hazard_ids": [hazard_id],
        }
        self._agent_event_awareness[key] = awareness
        event["awareness"][str(agent_idx)] = awareness

    def _record_planner_replan_source(
        self, agent_idx, event_id, step, awareness, successful_route_change
    ):
        """Attribute a successful hazard-linked route change to first awareness."""
        if not successful_route_change:
            return None

        source = str(
            (awareness or {}).get("awareness_source", "unattributed")
        )
        counts = self._response["planner_replan_source_attribution"]
        if source not in counts:
            source = "unattributed"

        awareness_step = (awareness or {}).get("awareness_step")
        record = {
            "agent": int(agent_idx),
            "event_id": event_id,
            "step": int(step),
            "awareness_source": source,
            "awareness_step": awareness_step,
            "response_delay_steps": (
                int(step) - int(awareness_step)
                if awareness_step is not None else None
            ),
            "plan_found": True,
            "route_changed": True,
            "causal_trigger": "known_blocked_route_node",
        }
        counts[source] += 1
        self._response.setdefault(
            "planner_replan_source_records", []
        ).append(record)
        return record

    def _record_event_response(self, agent_idx, event_id, step, response_type,
                               successful, details=None):
        """Record the first qualifying response for an agent-event pair."""
        key = (agent_idx, event_id)
        if key in self._agent_event_responses:
            return None
        awareness = self._agent_event_awareness.get(key)
        if awareness is None or int(step) < awareness["awareness_step"]:
            return None
        response = {
            "agent": int(agent_idx),
            "event_id": event_id,
            "response_step": int(step),
            "response_type": response_type,
            "successful": bool(successful),
            "response_delay_steps": int(step) - awareness["awareness_step"],
            "details": dict(details or {}),
        }
        self._agent_event_responses[key] = response
        event = self._hazard_event_records.get(event_id)
        if event is not None:
            event["responses"][str(agent_idx)] = response
        return response

    def _record_default_policy_event_responses(self, agent_idx, step, next_node,
                                               preceding_discoveries):
        """Record a safe next-hop choice after awareness; not causal proof."""
        event_ids = {
            event_id
            for discovery in preceding_discoveries
            for event_id in discovery.get("event_ids", ())
        }
        for event_id in sorted(event_ids):
            event = self._hazard_event_records.get(event_id)
            if event is None:
                continue
            hazard_nodes = set(event["affected_nodes"])
            hazard_edges = {
                frozenset(edge) for edge in event.get("affected_edges", ())
            }
            current = self.agents[agent_idx].current_node
            selected_edge = frozenset((current, next_node)) if next_node is not None else None
            safe_choice = (
                next_node not in hazard_nodes
                and (selected_edge is None or selected_edge not in hazard_edges)
            )
            self._record_event_response(
                agent_idx, event_id, step,
                "default_policy_safe_navigation_action",
                safe_choice,
                {
                    "selected_next_node": next_node,
                    "avoided_affected_nodes": safe_choice,
                    "causal_route_change_established": False,
                },
            )

    def _preceding_discoveries(
        self,
        agent_idx,
        step,
        current_node=None,
        next_node=None,
        *,
        last_decision_step=None,
        plan=None,
    ):
        """Return recent, locally relevant discoveries for one decision.

        Only the active planner route from the current position is considered.
        The agent's historical ``path`` is deliberately excluded: a hazard on
        a node visited long ago is not evidence that it affected this decision.
        ``last_decision_step`` and ``plan`` can be supplied to snapshot the
        decision context before the decision observer advances its timestamp.
        """
        if last_decision_step is None:
            last_decision_step = self._last_decision_step.get(agent_idx, 0)

        agent = self.agents[agent_idx]
        active_plan = agent.plan if plan is None else plan
        route_nodes = set()
        if active_plan:
            active_plan = list(active_plan)
            if current_node in active_plan:
                route_nodes.update(active_plan[active_plan.index(current_node):])
            else:
                # An off-plan agent has no active route segment to attribute.
                route_nodes.clear()

        adjacent_nodes = (
            {node_id for node_id, _ in self.graph._adj.get(current_node, ())}
            if current_node is not None else set()
        )
        relevant_nodes = set(route_nodes) | adjacent_nodes
        if current_node is not None:
            relevant_nodes.add(current_node)
        if next_node is not None:
            relevant_nodes.add(next_node)

        return [
            record for record in self._hazard_discoveries
            if record["agent"] == agent_idx
            and last_decision_step < record["step"] <= step
            and (
                current_node is None
                or record["hazard_id"] in relevant_nodes
            )
        ]

    def _preceding_discovery_sources(self, agent_idx, step, current_node=None,
                                     next_node=None, **kwargs):
        return {
            record["source"] for record in self._preceding_discoveries(
                agent_idx, step, current_node, next_node, **kwargs)
        }

    def _record_default_navigation_decision(
        self, agent_idx, from_node, to_node, step, policy_type=None, decision_diagnostics=None,
    ):
        """Record a pre-movement choice, retaining its actual policy label."""
        agent = self.agents[agent_idx]
        if policy_type is None:
            policy_type = (
                "planner_assisted" if agent.use_planner else "default_snn"
            )

        self._record_edge_attempt(agent_idx, from_node, to_node, step)
        if policy_type != "near_goal_rescue":
            # SNNAgent.sense() runs before its observer callback. Record that
            # observation now, while it is genuinely available to the choice.
            sensed = agent.last_observation
            self._record_hazard_discoveries(
                agent_idx,
                "self_observation",
                step,
                set(sensed.get("blocked_nodes", ()))
                | set(sensed.get("danger", {})),
            )
        preceding = self._preceding_discoveries(
            agent_idx, step, from_node, to_node
        )

        # A same-node next-hop comparison is meaningful only for the default
        # SNN policy. Planner and rescue choices are recorded separately.
        previous = None
        changed = None
        if policy_type == "default_snn":
            previous = self._last_decision_by_node[agent_idx].get(from_node)
            changed = previous is not None and previous != to_node
            self._last_decision_by_node[agent_idx][from_node] = to_node

        self._navigation_decisions.append({
            "agent": agent_idx,
            "role": agent.role,
            "policy_type": policy_type,
            "step": step,
            "from_node": from_node,
            "next_node": to_node,
            # The SNN policy exposes its actual candidate scores and
            # selection probabilities. Keep these alongside the observed
            # movement so experiments can distinguish score dominance from
            # RNG effects. Planner/rescue decisions have no SNN distribution.
            "decision_diagnostics": (
                {
                    **decision_diagnostics,
                    "candidates": [
                        dict(candidate)
                        for candidate in decision_diagnostics.get("candidates", [])
                    ],
                }
                if (
                    policy_type == "default_snn"
                    and isinstance(decision_diagnostics, dict)
                )
                else None
            ),
            "changed_from_previous_decision_at_same_node": changed,
            "preceding_discoveries": preceding,
            "preceding_sources": sorted(
                {record["source"] for record in preceding}
            ),
        })
        if policy_type == "default_snn":
            self._record_default_policy_event_responses(
                agent_idx, step, to_node, preceding
            )
        # Rescue is an external movement override, not the agent's SNN
        # decision. Do not consume the discovery window before agent.step()
        # makes its own decision later in this same simulation tick.
        if policy_type != "near_goal_rescue":
            self._last_decision_step[agent_idx] = max(
                step, self._last_decision_step.get(agent_idx, 0)
            )

    def _planner_replan_sources(
        self,
        agent_idx,
        step,
        current_node,
        decision_discoveries,
        same_step_self_discoveries=(),
    ):
        """Sources relevant to a planner decision, using its pre-step context."""
        relevant_nodes = {current_node}
        relevant_nodes.update(
            node_id for node_id, _ in self.graph._adj.get(current_node, ())
        )
        agent = self.agents[agent_idx]
        active_plan = list(agent.plan or ())
        if current_node in active_plan:
            relevant_nodes.update(active_plan[active_plan.index(current_node):])

        sources = {
            record["source"]
            for record in decision_discoveries
            if record["step"] <= step
        }
        sources.update(
            record["source"]
            for record in same_step_self_discoveries
            if record["step"] == step
            and record["hazard_id"] in relevant_nodes
        )
        return sources

    def _record_occupying_node_encounters(self, changed_nodes, step):
        for agent_idx, agent in enumerate(self.agents):
            if agent.current_node in changed_nodes:
                self._hazard_exposure["actual_node_encounters"].add(
                    agent.current_node)
                self._hazard_exposure["agents_encountered_hazard"].add(
                    agent_idx)
                self._hazard_exposure[
                    "agents_occupying_dynamically_damaged_nodes"
                ].add(agent_idx)

    def _record_pre_disablement_traversals(self, changed_edges, step):
        for entries in self._agent_edge_traversal_steps.values():
            for edge_id, traversal_step in entries:
                if edge_id in changed_edges and traversal_step < step:
                    self._hazard_exposure[
                        "successful_edge_traversals_before_disablement"
                    ].add(edge_id)

    def _damaged_nodes(self):
        return {
            node_id for node_id, damage in self.graph._node_damage.items()
            if damage > 0.0
        }

    @staticmethod
    def _latlon_to_unit_sphere(lat, lon):
        """Map latitude/longitude to a unit-sphere XYZ coordinate."""
        lat_r = math.radians(float(lat))
        lon_r = math.radians(float(lon))
        cos_lat = math.cos(lat_r)
        return (cos_lat * math.cos(lon_r),
                cos_lat * math.sin(lon_r),
                math.sin(lat_r))

    def _nearest_hazard_distance(self, agent_node, hazard_nodes, cache_name):
        """Exact great-circle nearest distance using a cached 3-D KD-tree.

        A unit-sphere chord-distance query has the same nearest-neighbour
        ordering as great-circle distance. Converting the returned chord to
        an arc gives the distance in metres without a Python scan over every
        hazard node for every agent and timestep.
        """
        if not hazard_nodes:
            return None

        cache = self._hazard_distance_index_cache.get(cache_name)
        node_key = frozenset(hazard_nodes)
        if cache is None or cache[0] != node_key:
            node_ids = tuple(hazard_nodes)
            xyz = [self._latlon_to_unit_sphere(
                self.graph._nodes[node_id]["lat"],
                self.graph._nodes[node_id]["lon"],
            ) for node_id in node_ids]
            tree = cKDTree(xyz)
            cache = (node_key, node_ids, tree)
            self._hazard_distance_index_cache[cache_name] = cache

        lat = self.graph._nodes[agent_node]["lat"]
        lon = self.graph._nodes[agent_node]["lon"]
        point = self._latlon_to_unit_sphere(lat, lon)
        chord_distance, _ = cache[2].query(point, k=1)
        chord_distance = min(2.0, max(0.0, float(chord_distance)))
        return 6_371_000.0 * (2.0 * math.asin(chord_distance / 2.0))

    def _record_hazard_exposure(self, step):
        """Record ground-truth exposure diagnostics without changing agents."""
        initial_hazard_nodes = {
            node_id for node_id in self._initial_damage_nodes
            if self.graph.get_node_damage(node_id) > 0.3
        }
        dynamic_hazard_nodes = {
            node_id for node_id in self._dynamic_damage_nodes
            if self.graph.get_node_damage(node_id) > 0.3
        }
        nearest_initial = {}
        nearest_dynamic = {}
        local_exposure = set()
        dynamic_local_exposure = set()
        severity_by_agent = {}
        dynamic_severity_by_agent = {}
        for idx, agent in enumerate(self.agents):
            nearest_initial[str(idx)] = self._nearest_hazard_distance(
                agent.current_node, initial_hazard_nodes, "initial")
            nearest_dynamic[str(idx)] = self._nearest_hazard_distance(
                agent.current_node, dynamic_hazard_nodes, "dynamic")

            neighbours = agent.get_all_neighbors()
            neighbourhood = {agent.current_node, *neighbours}
            local_severity = max(
                (float(self.graph.get_node_damage(node_id) or 0.0)
                 for node_id in neighbourhood),
                default=0.0,
            )
            dynamic_severity = max(
                (float(self.graph.get_node_damage(node_id) or 0.0)
                 for node_id in neighbourhood
                 if node_id in self._dynamic_damage_nodes),
                default=0.0,
            )
            severity_by_agent[str(idx)] = local_severity
            dynamic_severity_by_agent[str(idx)] = dynamic_severity

            has_local_hazard = (
                self.graph.get_node_damage(agent.current_node) > 0.3
                or any(self.graph.get_node_damage(node_id) > 0.3
                       for node_id in neighbours)
                or any(not self.graph.is_edge_passable(agent.current_node, node_id)
                       for node_id in neighbours)
            )
            if has_local_hazard:
                local_exposure.add(idx)
            has_dynamic_local_hazard = (
                (agent.current_node in self._dynamic_damage_nodes
                 and self.graph.get_node_damage(agent.current_node) > 0.3)
                or any(node_id in self._dynamic_damage_nodes
                       and self.graph.get_node_damage(node_id) > 0.3
                       for node_id in neighbours)
                or any(edge_id in self._dynamic_disabled_edges
                       for edge_id in (
                           frozenset({agent.current_node, node_id})
                           for node_id in neighbours
                       ))
            )
            if has_dynamic_local_hazard:
                dynamic_local_exposure.add(idx)

        self._hazard_exposure[
            "nearest_initial_hazard_distance_m_by_step"
        ][str(step)] = nearest_initial
        self._hazard_exposure[
            "nearest_dynamic_hazard_distance_m_by_step"
        ][str(step)] = nearest_dynamic
        self._hazard_exposure[
            "agents_with_hazard_in_sensing_neighborhood_by_step"
        ][str(step)] = sorted(local_exposure)
        self._hazard_exposure[
            "agents_with_dynamic_hazard_in_sensing_neighborhood_by_step"
        ][str(step)] = sorted(dynamic_local_exposure)
        self._hazard_exposure[
            "severity_weighted_exposure_by_agent_by_step"
        ][str(step)] = severity_by_agent
        self._hazard_exposure[
            "dynamic_severity_weighted_exposure_by_agent_by_step"
        ][str(step)] = dynamic_severity_by_agent

    def _record_temporal_encounters(self, agent_idx, from_node, to_node, step):
        damage_step = self._dynamic_damage_first_step.get(to_node)
        if damage_step is not None and step >= damage_step:
            self._hazard_exposure["actual_node_encounters"].add(to_node)
            self._hazard_exposure["agents_encountered_hazard"].add(agent_idx)
            self._hazard_exposure[
                "agents_occupying_dynamically_damaged_nodes"
            ].add(agent_idx)

        edge_id = frozenset({from_node, to_node})
        disabled_step = self._dynamic_edge_first_step.get(edge_id)
        if disabled_step is None:
            return
        if step >= disabled_step:
            self._hazard_exposure["actual_edge_encounters"].add(edge_id)
            self._hazard_exposure[
                "successful_edge_traversals_after_disablement"
            ].add(edge_id)
            self._hazard_exposure["agents_encountered_hazard"].add(agent_idx)
        else:
            self._hazard_exposure[
                "successful_edge_traversals_before_disablement"
            ].add(edge_id)

    def _record_edge_attempt(self, agent_idx, from_node, next_node, step):
        if next_node is None:
            return
        edge_id = frozenset({from_node, next_node})
        disabled_step = self._dynamic_edge_first_step.get(edge_id)
        if disabled_step is not None and step >= disabled_step:
            if not self.graph.is_edge_passable(from_node, next_node):
                self._hazard_exposure[
                    "actual_edge_attempts_after_disablement"
                ].add(edge_id)

    # -----------------------------------------------------------------------
    # Near-goal stuck rescue (from SwarmSimulation v3)
    # -----------------------------------------------------------------------

    def _rescue_near_goal_stuck(self, agent, agent_idx=None, step=None):
        if agent._last_dist_to_goal is None:
            return
        if agent._last_dist_to_goal > 300:
            return
        if agent._steps_without_progress < 10:
            return

        neighbors = agent.get_neighbors()
        if not neighbors:
            return

        goal_data = agent.graph._nodes[agent.goal_node]
        agent._taboo_reset()
        agent.blacklist.clear()
        agent.snn.reset_momentum()
        agent._steps_without_progress = 0

        best = min(
            neighbors,
            key=lambda n: haversine_distance(
                agent.graph._nodes[n]['lat'], agent.graph._nodes[n]['lon'],
                goal_data['lat'], goal_data['lon']
            )
        )
        if agent_idx is None:
            agent_idx = next(
                (idx for idx, candidate in enumerate(self.agents)
                 if candidate is agent),
                None,
            )
        if step is None:
            step = self.current_step
        if agent_idx is not None:
            self._record_default_navigation_decision(
                agent_idx,
                agent.current_node,
                best,
                step,
                policy_type="near_goal_rescue",
            )
        agent._execute_move(best,
                            agent.graph._nodes[agent.current_node],
                            goal_data)

    # -----------------------------------------------------------------------
    # Shared memory helpers (from SwarmSimulation v3)
    # -----------------------------------------------------------------------

    def _deposit_pheromones(self, agent, agent_idx=None):
        """Deposit trail on newly occupied path nodes, with source provenance.

        This is an online trail: each path entry is deposited once when it is
        first appended, instead of waiting for goal completion and replaying
        the entire completed path. A repeated visit is a new path entry and
        can strengthen the field again, while provenance retains the first
        publication by each contributor for timing attribution.
        """
        if agent_idx is None:
            agent_idx = self.agents.index(agent)
        path = list(agent.path)
        previous_len = self._pheromone_deposited_path_lengths.get(agent_idx, 0)
        if previous_len > len(path):
            previous_len = len(path)
        if previous_len == len(path):
            return

        ph = self.shared_memory["pheromones"]
        quality = max(0.5, 1.0 - len(path) / 400.0)
        strength = self.PHEROMONE_DEPOSIT * quality
        for path_index in range(previous_len, len(path)):
            node = path[path_index]
            if strength <= 0.0:
                continue
            ph[node] = min(1.0, ph.get(node, 0.0) + strength)
            contributors = self._pheromone_publications.setdefault(node, [])
            if not any(item["agent"] == agent_idx for item in contributors):
                contributors.append({
                    "agent": agent_idx,
                    "step": int(self.current_step),
                    "path_index": int(path_index),
                    "strength": float(strength),
                })
        self._pheromone_deposited_path_lengths[agent_idx] = len(path)

    def _evaporate_pheromones(self):
        # In place: agents hold a view onto this same dict.
        ph = self.shared_memory["pheromones"]
        for k in list(ph):
            v = ph[k] * self.PHEROMONE_DECAY
            if v > 0.01:
                ph[k] = v
            else:
                del ph[k]

    def _distance(self, n1, n2):
        a, b = self.graph._nodes[n1], self.graph._nodes[n2]
        return haversine_distance(a['lat'], a['lon'], b['lat'], b['lon'])

    def _edge_distance(self, n1, n2):
        """Return the graph edge distance, with a geometric fallback."""
        for neighbour, edge in self.graph._adj.get(n1, ()):
            if neighbour == n2:
                return float(edge.get("length", self._distance(n1, n2)))
        return self._distance(n1, n2)

    def _path_distance(self, path):
        if len(path) < 2:
            return 0.0
        return sum(self._edge_distance(u, v) for u, v in zip(path, path[1:]))

    def communication_instrumentation(self):
        """Return JSON-safe counters for existing observation/message paths."""
        data = dict(self._comm_instrumentation)
        # Backward-compatible alias; this is a publication-hook count, not a
        # count of calls to Agent.sense().
        data["sensing_events"] = data["observation_publication_calls"]
        data["agents_with_non_empty_observations"] = len(
            data["agents_with_non_empty_observations"])
        data["own_blocked_observations"] = sum(
            a.stats["blocked_learned_self"] for a in self.agents)
        data["cross_agent_blocked_retrievals"] = sum(
            a.stats["blocked_learned_msg"] for a in self.agents)
        data["shared_memory_nodes_available"] = len(self.board_visible)
        data["agents_with_self_observations"] = len({
            record["agent"] for record in self._hazard_discoveries
            if record["source"] == "self_observation"
        })
        data["agents_receiving_teammate_information"] = len({
            record["agent"] for record in self._hazard_discoveries
            if record["source"] == "teammate_message"
        })
        data["shared_memory_available_nodes"] = len(self.board_visible)
        cross_agent_reads = [
            record for record in self._shared_memory_reads
            if record["cross_agent"] and record["channel"] in {
                "blocked_nodes", "danger_nodes"
            }
        ]
        relevant_cross_agent_reads = [
            record for record in cross_agent_reads if record["route_relevant"]
        ]
        data["shared_memory_retrievals_recorded"] = len(cross_agent_reads)
        data["shared_memory_retrievals_instrumented"] = True
        data["shared_memory_read_events"] = list(self._shared_memory_reads)
        local_message_reads = [
            record for record in self._local_message_reads
            if record["cross_agent"]
        ]
        data["local_message_hazard_read_events"] = list(self._local_message_reads)
        data["local_message_hazard_information_reads"] = len(local_message_reads)
        data["local_message_route_relevant_hazard_reads"] = sum(
            1 for record in local_message_reads if record["route_relevant"]
        )
        data["local_message_new_hazard_information_reads"] = sum(
            1 for record in local_message_reads if record["new_information"]
        )
        data["shared_memory_cross_agent_retrievals"] = len(cross_agent_reads)
        data["shared_memory_route_relevant_cross_agent_retrievals"] = len(
            relevant_cross_agent_reads
        )
        data["shared_memory_retrieved_agent_node_pairs"] = len({
            (record["reader_agent"], record["node_id"])
            for record in cross_agent_reads
        })
        pheromone_reads = [
            record for record in self._shared_memory_reads
            if record["channel"] == "pheromones"
        ]
        data["pheromone_read_events"] = list(pheromone_reads)
        data["pheromone_cross_agent_retrievals"] = sum(
            1 for record in pheromone_reads if record["cross_agent"]
        )
        data["pheromone_route_relevant_cross_agent_retrievals"] = sum(
            1 for record in pheromone_reads
            if record["cross_agent"] and record["route_relevant"]
        )
        data["agents_receiving_teammate_pheromones"] = len({
            record["reader_agent"] for record in pheromone_reads
            if record["cross_agent"]
        })
        data["pheromone_nodes_with_provenance"] = len(self._pheromone_publications)
        data["hazard_discoveries"] = list(self._hazard_discoveries)
        data["runtime_hazard_diagnostics"] = list(
            getattr(self, "_runtime_hazard_diagnostics", [])
        )
        return data

    def hazard_exposure_metrics(self):
        """Return ground-truth exposure diagnostics as JSON-safe values."""
        for agent_idx, agent in enumerate(self.agents):
            self._hazard_exposure[
                "traversed_route_nodes_affected_by_damage"
            ].update(set(agent.path) & self._dynamic_damage_nodes)
            self._hazard_exposure[
                "traversed_edges_that_became_disabled"
            ].update({
                edge_id for edge_id, _ in self._agent_edge_traversal_steps[
                    agent_idx
                ] if edge_id in self._dynamic_disabled_edges
            })

        return {
            "damage_threshold_semantics": {
                "physical_damage_occupancy": "> 0.0",
                "danger_observation": "> 0.3",
                "impassable_node": "> 0.6",
                "agent_fatal_damage": "> 0.9",
                "note": "These are distinct metrics; occupying a node with "
                        "positive damage does not necessarily mean the agent "
                        "observed danger or entered an impassable node.",
            },
            "damaged_nodes": len(self._damaged_nodes()),
            "disabled_edges": len(self.graph._disabled_edges),
            "dynamic_damaged_nodes": len(self._dynamic_damage_nodes),
            "dynamic_disabled_edges": len(self._dynamic_disabled_edges),
            "nearest_initial_hazard_distance_m_by_step": self._hazard_exposure[
                "nearest_initial_hazard_distance_m_by_step"
            ],
            "nearest_dynamic_hazard_distance_m_by_step": self._hazard_exposure[
                "nearest_dynamic_hazard_distance_m_by_step"
            ],
            "agents_with_hazard_in_sensing_neighborhood_by_step":
                self._hazard_exposure[
                    "agents_with_hazard_in_sensing_neighborhood_by_step"
                ],
            "agents_with_dynamic_hazard_in_sensing_neighborhood_by_step":
                self._hazard_exposure[
                    "agents_with_dynamic_hazard_in_sensing_neighborhood_by_step"
                ],
            "agents_with_hazard_in_sensing_neighborhood": len({
                idx for indices in self._hazard_exposure[
                    "agents_with_hazard_in_sensing_neighborhood_by_step"
                ].values() for idx in indices
            }),
            "agents_encountered_hazard": len(
                self._hazard_exposure["agents_encountered_hazard"]
            ),
            "agents_occupying_dynamically_damaged_nodes": len(
                self._hazard_exposure[
                    "agents_occupying_dynamically_damaged_nodes"
                ]
            ),
            "agents_with_dynamic_hazard_in_sensing_neighborhood": len({
                idx for indices in self._hazard_exposure[
                    "agents_with_dynamic_hazard_in_sensing_neighborhood_by_step"
                ].values() for idx in indices
            }),
            "traversed_route_nodes_affected_by_damage": len(
                self._hazard_exposure[
                    "traversed_route_nodes_affected_by_damage"
                ]
            ),
            "route_overlap_nodes_affected_by_dynamic_damage": len(
                self._hazard_exposure[
                    "traversed_route_nodes_affected_by_damage"
                ]
            ),
            "actual_node_encounters": len(
                self._hazard_exposure["actual_node_encounters"]
            ),
            "actual_edge_encounters": len(
                self._hazard_exposure["actual_edge_encounters"]
            ),
            "actual_edge_attempts_after_disablement": len(
                self._hazard_exposure[
                    "actual_edge_attempts_after_disablement"
                ]
            ),
            "successful_edge_traversals_before_disablement": len(
                self._hazard_exposure[
                    "successful_edge_traversals_before_disablement"
                ]
            ),
            "successful_edge_traversals_after_disablement": len(
                self._hazard_exposure[
                    "successful_edge_traversals_after_disablement"
                ]
            ),
            "traversed_edges_that_became_disabled": len(
                self._hazard_exposure[
                    "traversed_edges_that_became_disabled"
                ]
            ),
            "route_overlap_edges_affected_by_dynamic_disablement": len(
                self._hazard_exposure[
                    "traversed_edges_that_became_disabled"
                ]
            ),
            "exposure_event": getattr(self.engine, "exposure_event", None),
            "exposure_event_applied": bool(
                getattr(self.engine, "exposure_event_applied", False)
            ),
            "dynamic_damage_first_step": dict(self._dynamic_damage_first_step),
            "dynamic_edge_first_step": {
                str(edge_id): step for edge_id, step
                in self._dynamic_edge_first_step.items()
            },
            "node_entry_steps": {
                str(agent_idx): entries
                for agent_idx, entries in self._agent_node_entry_steps.items()
            },
            "edge_traversal_steps": {
                str(agent_idx): [
                    (str(edge_id), step) for edge_id, step in entries
                ] for agent_idx, entries
                in self._agent_edge_traversal_steps.items()
            },
        }

    def hazard_response_metrics(self):
        """Event-based awareness and response metrics with auditable raw records."""
        # Some unit tests construct the simulation with __new__() to isolate
        # metric behaviour. In that case __init__ has not created the event
        # registries; treat an uninitialised registry as an empty one.
        events = list(getattr(self, "_hazard_event_records", {}).values())
        awareness = list(getattr(self, "_agent_event_awareness", {}).values())
        responses = list(getattr(self, "_agent_event_responses", {}).values())
        affected_pairs_set = {
            (int(agent_idx), event["event_id"])
            for event in events for agent_idx in event["affected_agents"]
        }
        affected_pairs = len(affected_pairs_set)
        affected_awareness = [
            item for item in awareness
            if (item["agent"], item["event_id"]) in affected_pairs_set
        ]
        affected_responses = [
            item for item in responses
            if (item["agent"], item["event_id"]) in affected_pairs_set
        ]
        successful_responses = sum(bool(item["successful"]) for item in responses)
        affected_successful_responses = sum(
            bool(item["successful"]) for item in affected_responses
        )
        outside_affected_responses = [
            item for item in responses
            if (item["agent"], item["event_id"]) not in affected_pairs_set
        ]
        outside_affected_successful_responses = sum(
            bool(item["successful"]) for item in outside_affected_responses
        )
        # Awareness latency is reported at the same explicit population
        # scopes as response latency. The legacy mean remains scoped to the
        # initially affected agent-event pairs for backward compatibility.
        def _awareness_delays(records):
            return [
                float(item["awareness_delay_steps"])
                for item in records
                if item.get("awareness_delay_steps") is not None
            ]

        all_awareness_delays = _awareness_delays(awareness)
        affected_awareness_delays = _awareness_delays(affected_awareness)
        outside_affected_awareness = [
            item for item in awareness
            if (item["agent"], item["event_id"]) not in affected_pairs_set
        ]
        outside_awareness_delays = _awareness_delays(
            outside_affected_awareness
        )

        awareness_by_source = {}
        for item in awareness:
            source = str(item.get("awareness_source", item.get("source", "unknown")))
            bucket = awareness_by_source.setdefault(source, {
                "count": 0, "delay_sum_steps": 0.0, "delay_count": 0,
            })
            bucket["count"] += 1
            delay = item.get("awareness_delay_steps")
            if delay is not None:
                bucket["delay_sum_steps"] += float(delay)
                bucket["delay_count"] += 1
        for bucket in awareness_by_source.values():
            bucket["mean_delay_steps"] = (
                round(bucket["delay_sum_steps"] / bucket["delay_count"], 3)
                if bucket["delay_count"] else None
            )
            # Keep the denominator explicit; zero and no observations differ.
            bucket["delay_sum_steps"] = round(bucket["delay_sum_steps"], 3)

        # Response latency is reported at three explicit scopes.  The
        # all-agent aggregate includes agents who learn about an event later
        # through messaging/shared memory, not only those initially exposed.
        def _response_delays(records):
            return [
                float(item["response_delay_steps"])
                for item in records
                if item.get("response_delay_steps") is not None
            ]

        all_response_delays = _response_delays(responses)
        affected_response_delays = _response_delays(affected_responses)
        outside_response_delays = _response_delays(outside_affected_responses)

        def _mean_delay(delays):
            return round(sum(delays) / len(delays), 3) if delays else None

        planner_responses = [item for item in responses
                             if item["response_type"] == "planner_replan"]
        default_responses = [item for item in responses
                             if item["response_type"] ==
                             "default_policy_safe_navigation_action"]
        return {
            "total_hazard_events": len(events),
            "affected_agent_event_pairs": affected_pairs,
            # These counts are agent-event pairs, not unique agents.
            "affected_pairs_became_aware": len(affected_awareness),
            "agents_became_aware": len(affected_awareness),  # legacy alias
            "total_agent_event_awareness_records": len(awareness),
            "awareness_outside_initial_affected_set": (
                len(awareness) - len(affected_awareness)
            ),
            "awareness_coverage": (
                round(len(affected_awareness) / affected_pairs, 4)
                if affected_pairs else None
            ),
            "mean_awareness_delay_steps": (
                round(
                    sum(affected_awareness_delays) /
                    len(affected_awareness_delays), 3
                ) if affected_awareness_delays else None
            ),
            "mean_awareness_delay_all_agents_steps": (
                round(
                    sum(all_awareness_delays) / len(all_awareness_delays), 3
                ) if all_awareness_delays else None
            ),
            "mean_awareness_delay_affected_pairs_steps": (
                round(
                    sum(affected_awareness_delays) /
                    len(affected_awareness_delays), 3
                ) if affected_awareness_delays else None
            ),
            "mean_awareness_delay_outside_initial_affected_set_steps": (
                round(
                    sum(outside_awareness_delays) /
                    len(outside_awareness_delays), 3
                ) if outside_awareness_delays else None
            ),
            "awareness_delay_record_counts": {
                "all_agents": len(all_awareness_delays),
                "initially_affected_pairs": len(affected_awareness_delays),
                "outside_initial_affected_set": len(outside_awareness_delays),
            },
            "awareness_source_counts": {
                source: values["count"]
                for source, values in awareness_by_source.items()
            },
            "awareness_by_source": awareness_by_source,
            # Keep the legacy key, but make its scope explicit in new keys.
            "affected_pairs_responded": len(affected_responses),
            "agents_responded": len(affected_responses),  # legacy alias
            "response_records_outside_initial_affected_set": len(
                outside_affected_responses
            ),
            "successful_responses_among_affected_pairs": (
                affected_successful_responses
            ),
            "successful_responses_outside_initial_affected_set": (
                outside_affected_successful_responses
            ),
            "total_agent_event_response_records": len(responses),
            "response_coverage_of_affected_pairs": (
                round(len(affected_responses) / affected_pairs, 4)
                if affected_pairs else None
            ),
            # Generic/legacy aggregate now covers every recorded response.
            # Explicitly scoped metrics below prevent population ambiguity.
            "mean_response_delay_steps": _mean_delay(all_response_delays),
            "mean_response_delay_all_agents_steps": _mean_delay(
                all_response_delays
            ),
            "mean_response_delay_affected_pairs_steps": _mean_delay(
                affected_response_delays
            ),
            "mean_response_delay_outside_initial_affected_set_steps": (
                _mean_delay(outside_response_delays)
            ),
            "response_delay_record_counts": {
                "all_agents": len(all_response_delays),
                "initially_affected_pairs": len(affected_response_delays),
                "outside_initial_affected_set": len(outside_response_delays),
            },
            "successful_responses": successful_responses,
            "failed_responses": len(responses) - successful_responses,
            "planner_replan_responses": len(planner_responses),
            "planner_successful_replan_responses": sum(
                item["successful"] for item in planner_responses
            ),
            "default_policy_safe_navigation_actions": len(default_responses),
            "default_policy_safe_actions": sum(
                item["successful"] for item in default_responses
            ),
            "event_records": events,
            "awareness_records": awareness,
            "response_records": responses,
            "definitions": {
                "affected_agent": (
                    "At event time, the event intersects the agent's active "
                    "planner route or its current node/local adjacent nodes."
                ),
                "response_scope": (
                    "Affected-pair response coverage counts only agents in the "
                    "initial affected_agent set for each event. Responses from "
                    "agents who learn later through communication or shared "
                    "memory are counted separately as outside-initial-affected-set "
                    "responses; they still count in total successful_responses."
                ),
                "response_delay_scope": (
                    "mean_response_delay_steps and "
                    "mean_response_delay_all_agents_steps average all recorded "
                    "responses. The affected_pairs and outside_initial_affected_set "
                    "keys report those populations separately. Delay is measured "
                    "from that agent's first recorded awareness of the event."
                ),
                "awareness_delay_scope": (
                    "mean_awareness_delay_steps remains the legacy mean for "
                    "initially affected agent-event pairs. The explicit "
                    "all_agents and outside_initial_affected_set fields include "
                    "later-informed agents. Delays are averaged only over raw "
                    "records with a non-null awareness_delay_steps value."
                ),
                "awareness_source_counts": (
                    "Counts first-awareness records by their recorded source "
                    "(for example self_observation, teammate_message, or "
                    "shared_memory). Counts are records, not unique agents."
                ),
                "awareness": (
                    "First recorded self-observation, teammate-message delivery, "
                    "or successful cross-agent shared-memory hazard lookup "
                    "containing a node attributed to the event."
                ),
                "planner_success": (
                    "A known event hazard invalidates the active route, a "
                    "replacement plan is found, and the active route changes."
                ),
                "default_policy_response": (
                    "A safe next-hop choice after event awareness. This is "
                    "descriptive and does not establish causal route change."
                ),
                "edge_attribution": (
                    "The engine reports exact edge IDs. Older/custom event "
                    "providers may fall back to a single-event graph diff; "
                    "ambiguous simultaneous-event attribution is omitted."
                ),
            },
        }

    def response_metrics(self):
        """Return genuine planner events and event-linked awareness/response."""
        return {
            # Count only actual planner events emitted by the agent itself.
            # The default greedy policy has no persistent route to invalidate.
            "planner_invalidated_plan_events": sum(
                1
                for agent in self.agents
                for event in getattr(agent, "plan_events", ())
                if event.get("reason") == "invalidated"
            ),
            "planner_successful_replans": self._response[
                "planner_successful_replans"
            ],
            "planner_route_changes_following_hazard_discovery": self._response[
                "planner_route_changes_following_hazard_discovery"
            ],
            "planner_replan_source_attribution": dict(
                self._response["planner_replan_source_attribution"]
            ),
            "planner_replan_source_records": list(
                self._response.get("planner_replan_source_records", ())
            ),
            "planner_plan_events_by_agent": {
                str(i): list(getattr(agent, "plan_events", ()))
                for i, agent in enumerate(self.agents)
                if getattr(agent, "plan_events", ())
            },
            "hazard_event_metrics": self.hazard_response_metrics(),
            "navigation_decisions": list(self._navigation_decisions),
            "default_policy_navigation_decisions": [
                record for record in self._navigation_decisions
                if record["policy_type"] == "default_snn"
                and record["role"] == Role.NAVIGATOR.value
            ],
            "planner_assisted_navigation_decisions": [
                record for record in self._navigation_decisions
                if record["policy_type"] == "planner_assisted"
                and record["role"] == Role.NAVIGATOR.value
            ],
            "near_goal_rescue_decisions": [
                record for record in self._navigation_decisions
                if record["policy_type"] == "near_goal_rescue"
            ],
            "default_policy_route_change_causality": "not established; "
            "next-hop changes are descriptive only",
        }

    # -----------------------------------------------------------------------
    # Summary — includes adaptation metrics
    # -----------------------------------------------------------------------

    def navigators(self):
        return [a for a in self.agents if a.role == Role.NAVIGATOR.value]

    def summary(self):
        navs    = self.navigators()
        total   = len(navs)
        reached = sum(a.reached_goal for a in navs)
        alive   = sum(a.alive and not a.reached_goal for a in navs)
        dead    = sum(not a.alive and not a.reached_goal for a in navs)

        print("\n" + "=" * 60)
        print("DYNAMIC SWARM SUMMARY")
        print("=" * 60)
        print(f"Total navigators: {total}  (swarm size {len(self.agents)})")
        print(f"Reached goal  : {reached} ({100*reached/total:.1f}%)")
        print(f"Still alive   : {alive}   ({100*alive/total:.1f}%)")
        print(f"Dead          : {dead}   ({100*dead/total:.1f}%)")
        print("=" * 60)

        # Disaster progression
        print("\nDISASTER PROGRESSION:")
        eng_summary = self.engine.summary()
        for k, v in eng_summary.items():
            print(f"  {k:<30}: {v}")

        print(f"\n  Total disaster events logged : {len(self.disaster_log)}")
        event_types: dict = {}
        for _, ev in self.disaster_log:
            event_types[ev.event_type] = event_types.get(ev.event_type, 0) + 1
        for etype, cnt in sorted(event_types.items()):
            print(f"    {etype:<22}: {cnt}")

        # Report genuine planner events only. The default greedy policy has
        # no persistent route, so it must not be assigned synthetic reroutes
        # or stall-counter-based adaptation lag.
        planner_events = [
            event
            for agent in self.agents
            for event in getattr(agent, "plan_events", ())
            if event.get("reason") == "invalidated"
        ]
        planner_successes = sum(bool(event.get("ok")) for event in planner_events)
        planner_route_changes = sum(
            bool(event.get("ok") and event.get("route_changed"))
            for event in planner_events
        )
        print("\nPLANNER RESPONSE EVENTS:")
        print(f"  Actual plan invalidations    : {len(planner_events)}")
        print(f"  Successful replans           : {planner_successes}")
        print(f"  Replans changing active route: {planner_route_changes}")
        print("  Default SNN policy reroutes  : not applicable (greedy one-hop policy)")
        event_metrics = self.hazard_response_metrics()
        print("\nEVENT-BASED HAZARD RESPONSE:")
        print(f"  Hazard events                : {event_metrics['total_hazard_events']}")
        print(f"  Affected agent-event pairs   : {event_metrics['affected_agent_event_pairs']}")
        print(f"  Affected pairs aware         : {event_metrics['agents_became_aware']}")
        print(f"  Mean awareness delay         : {event_metrics['mean_awareness_delay_steps']}")
        print(f"  Affected pairs responded     : {event_metrics['affected_pairs_responded']}")
        print(f"  Responses outside initial set: {event_metrics['response_records_outside_initial_affected_set']}")
        print(f"  Successful affected responses: {event_metrics['successful_responses_among_affected_pairs']}")
        print(f"  Successful outside responses : {event_metrics['successful_responses_outside_initial_affected_set']}")
        print(f"  Mean all-agent response delay: {event_metrics['mean_response_delay_all_agents_steps']}")
        print(f"  Mean affected response delay  : {event_metrics['mean_response_delay_affected_pairs_steps']}")
        print(f"  Mean outside response delay   : {event_metrics['mean_response_delay_outside_initial_affected_set_steps']}")
        print(f"  Total successful responses   : {event_metrics['successful_responses']}")
        print("  Default-policy safe choices  : descriptive, not causal")

        if reached > 0:
            path_lens  = [len(a.path) for a in navs if a.reached_goal]
            goal_steps = list(self._goal_step.values())
            print("\nSUCCESSFUL AGENTS:")
            print(f"  Path length : min={min(path_lens)}  max={max(path_lens)}  "
                  f"mean={sum(path_lens)/len(path_lens):.1f}")
            if goal_steps:
                print(f"  Goal step   : first={min(goal_steps)}  last={max(goal_steps)}  "
                      f"mean={sum(goal_steps)/len(goal_steps):.1f}")
            print("\n  Per-agent breakdown:")
            for i, agent in enumerate(self.agents):
                if agent.role == Role.NAVIGATOR.value and agent.reached_goal:
                    step     = self._goal_step.get(i, "?")
                    bl = len(agent.blacklist)
                    print(f"    Agent {i:2d}: {len(agent.path):4d} steps | "
                          f"goal@{step} | blacklisted={bl}")

        ph_nodes = len(self.shared_memory["pheromones"])
        print(f"\nPheromone trail: {ph_nodes} nodes marked at run end")
        cs = self.bus.stats()
        mean_delay = cs["mean_delay_steps"]
        delay_label = f"{mean_delay:.2f}" if mean_delay is not None else "n/a"
        print(f"Communication  : {cs['transmissions']} tx, {cs['deliveries']} rx, "
              f"{cs['bytes_tx']} B tx, mean delay {delay_label} steps")
        if self.coordination_log:
            print(f"Coordination   : {len(self.coordination_log)} events "
                  f"({self.coord_cfg.mode})")
        print("=" * 60)

    def execution_metrics(self) -> dict:
        """Return actual per-agent work and elapsed simulation makespan."""
        nav_idx = [
            i for i, agent in enumerate(self.agents)
            if agent.role == Role.NAVIGATOR.value
        ]
        steps_by_agent = {
            str(i): int(self._agent_steps_by_agent.get(i, 0))
            for i in nav_idx
        }
        finish_by_agent = {
            str(i): int(self._agent_finish_step.get(i, self.current_step))
            for i in nav_idx
        }
        reason_by_agent = {
            str(i): self._agent_finish_reason.get(
                i,
                "goal" if self.agents[i].reached_goal else (
                    self.agents[i].death_reason or "unfinished"
                ),
            )
            for i in nav_idx
        }
        return {
            "agent_steps_total": sum(steps_by_agent.values()),
            "agent_steps_by_agent": steps_by_agent,
            "finish_step_by_agent": finish_by_agent,
            "finish_reason_by_agent": reason_by_agent,
            "makespan_steps": max(finish_by_agent.values(), default=0),
        }

    def adaptation_metrics(self) -> dict:
        """
        Machine-readable navigation and genuine planner-response metrics.
        Navigators only: scouts and hubs are support roles, not tasks to succeed at.
        No synthetic reroute or stall-counter adaptation metrics are reported.
        """
        nav_idx = [i for i, a in enumerate(self.agents)
                   if a.role == Role.NAVIGATOR.value]
        total = len(nav_idx)
        reached_idx = [i for i in nav_idx if self.agents[i].reached_goal]
        planner_invalidations = sum(
            1
            for i in nav_idx
            for event in getattr(self.agents[i], "plan_events", ())
            if event.get("reason") == "invalidated"
        )
        planner_successful_replans = sum(
            1
            for i in nav_idx
            for event in getattr(self.agents[i], "plan_events", ())
            if event.get("reason") == "invalidated" and event.get("ok")
        )
        planner_route_changes = sum(
            1
            for i in nav_idx
            for event in getattr(self.agents[i], "plan_events", ())
            if event.get("reason") == "invalidated"
            and event.get("ok")
            and event.get("route_changed")
        )
        distance_by_agent = {
            str(i): round(self._path_distance(self.agents[i].path), 3)
            for i in nav_idx
        }
        total_distance = round(sum(distance_by_agent.values()), 3)
        planner_stats = {
            str(i): {
                key: value
                for key, value in self.agents[i].stats.items()
                if key.startswith("plan_") or key == "replans"
            }
            for i in nav_idx
        }

        return {
            "total_agents":          total,
            "reached_goal":          len(reached_idx),
            "success_rate":          round(len(reached_idx) / total, 3) if total else 0,
            "planner_invalidated_plan_events": planner_invalidations,
            "planner_successful_replans": planner_successful_replans,
            "planner_route_changes": planner_route_changes,
            "mean_path_length": round(
                sum(len(self.agents[i].path) for i in reached_idx) / len(reached_idx),
                1,
            ) if reached_idx else None,
            "total_distance_travelled_m": total_distance,
            "mean_distance_travelled_m": round(
                total_distance / total if total else 0.0, 3
            ),
            "distance_travelled_m_by_agent": distance_by_agent,
            "planner_stats_by_agent": planner_stats,
            "performance": dict(getattr(self, "_runtime_performance", {})),
            "mean_goal_step": round(
                sum(self._goal_step[i] for i in reached_idx if i in self._goal_step)
                / len(reached_idx),
                1,
            ) if reached_idx else None,
            "disaster_events": len(self.disaster_log),
            "disaster_summary": self.engine.summary(),
        }

    # -----------------------------------------------------------------------
    # Communication / role metrics
    # -----------------------------------------------------------------------

    def info_propagation(self) -> dict:
        """
        Measure cross-agent hazard knowledge without treating global board
        availability as proof that every agent retrieved the information.

        ``learned_at`` is populated when an agent receives a teammate message,
        so it is evidence of an actual message-based retrieval. The shared
        board is exposed through a common view; individual reads are not
        instrumented. Board visibility is therefore reported as availability,
        not retrieval.

        ``mean_delay_steps`` is retained as the earliest communication-channel
        availability delay (message receipt or board publication) for
        compatibility. ``mean_message_retrieval_delay_steps`` measures only
        confirmed teammate-message receipt.
        """
        availability_delays = []
        message_delays = []
        board_delays = []
        pairs = 0
        message_informed = 0
        board_available_pairs = 0

        for node_id, (detected_step, detector) in self.detection_log.items():
            for recipient_idx, agent in enumerate(self.agents):
                if recipient_idx == detector or not agent.alive:
                    continue

                pairs += 1
                learned = agent.learned_at.get(node_id)
                channel_times = []

                # Count only an actual teammate-message learning record.
                if (
                    learned is not None
                    and len(learned) >= 2
                    and learned[1] == "msg"
                ):
                    message_informed += 1
                    message_step = learned[0]
                    message_delays.append(max(0, message_step - detected_step))
                    channel_times.append(message_step)

                # Board availability is not proof of an individual read.
                if (
                    self.comm.shared_memory
                    and node_id in self.board_visible
                    and self.board_visible[node_id] >= detected_step
                ):
                    board_step = self.board_visible[node_id]
                    board_available_pairs += 1
                    board_delays.append(max(0, board_step - detected_step))
                    channel_times.append(board_step)

                if channel_times:
                    availability_delays.append(
                        max(0, min(channel_times) - detected_step)
                    )

        message_pairs = {
            (recipient_idx, node_id)
            for node_id, (detected_step, detector) in self.detection_log.items()
            for recipient_idx, agent in enumerate(self.agents)
            if recipient_idx != detector and agent.alive
            and (
                (agent.learned_at.get(node_id) is not None)
                and len(agent.learned_at[node_id]) >= 2
                and agent.learned_at[node_id][1] == "msg"
            )
        }
        shared_pairs = {
            (r["reader_agent"], r["node_id"])
            for r in self._shared_memory_reads
            if r["cross_agent"] and r["channel"] in {
                "blocked_nodes", "danger_nodes"
            }
        }
        shared_detected_pairs = {
            pair for pair in shared_pairs if pair[1] in self.detection_log
        }
        informed_pairs = message_pairs | shared_detected_pairs
        return {
            "hazards_detected": len(self.detection_log),
            "agent_hazard_pairs": pairs,
            "fraction_informed": (
                round(len(informed_pairs) / pairs, 4) if pairs else None
            ),
            "fraction_informed_via_local_messages": (
                round(message_informed / pairs, 4) if pairs else None
            ),
            "fraction_with_information_available": (
                round(len(availability_delays) / pairs, 4) if pairs else None
            ),
            "mean_delay_steps": (
                round(sum(availability_delays) / len(availability_delays), 3)
                if availability_delays else None
            ),
            "mean_message_retrieval_delay_steps": (
                round(sum(message_delays) / len(message_delays), 3)
                if message_delays else None
            ),
            "mean_shared_memory_availability_delay_steps": (
                round(sum(board_delays) / len(board_delays), 3)
                if board_delays else None
            ),
            "cross_agent_retrievals": len(message_pairs | shared_pairs),
            "retrievals_via_local_messages": len(message_pairs),
            "retrievals_via_shared_memory": len(shared_pairs),
            "route_relevant_shared_memory_retrievals": len({
                (r["reader_agent"], r["node_id"])
                for r in self._shared_memory_reads
                if r["cross_agent"] and r["route_relevant"]
                and r["channel"] in {"blocked_nodes", "danger_nodes"}
            }),
            "shared_memory_retrievals_instrumented": True,
            "shared_memory_available_agent_hazard_pairs": board_available_pairs,
        }

    def contributions(self) -> list:
        """Per-agent contribution record (inputs to per-agent contribution analysis)."""
        stats = self.bus.stats()
        first = {}
        for _, (_, idx) in self.detection_log.items():
            first[idx] = first.get(idx, 0) + 1
        out = []
        for i, a in enumerate(self.agents):
            out.append({
                "agent": i, "role": a.role, "alive": a.alive,
                "reached_goal": a.reached_goal if a.role == Role.NAVIGATOR.value else None,
                "first_detections": first.get(i, 0),
                "tx": stats["tx_by_sender"].get(i, 0),
                "bytes_tx": stats["bytes_by_sender"].get(i, 0),
                "msgs_received": a.stats["msgs_received"],
                "blocked_learned_self": a.stats["blocked_learned_self"],
                "blocked_learned_msg": a.stats["blocked_learned_msg"],
                "nodes_visited": len(a.visited),
                "cells_covered": len(a.cell_counts),
                "path_len": len(a.path),
                "replans": a.stats["replans"],
                "plan_invalidations": a.stats["plan_invalidations"],
                "goal_changes": a.stats["goal_changes"],
            })
        return out

    def metrics(self) -> dict:
        from dataclasses import asdict
        from collections import Counter
        return {
            "config": {
                "comm": asdict(self.comm),
                "roles": dict(Counter(a.role for a in self.agents)),
                "specialised_roles": self._specialised,
                "coordination": self.coord_cfg.mode,
                "shelters": len(self._shelters) if self._shelters else 0,
            },
            "task": self.adaptation_metrics(),
            "comms": self.bus.stats(),
            "communication_instrumentation": self.communication_instrumentation(),
            "hazard_exposure": self.hazard_exposure_metrics(),
            "response": self.response_metrics(),
            "info_propagation": self.info_propagation(),
            "coordination_events": list(self.coordination_log),
            "contributions": self.contributions(),
        }
