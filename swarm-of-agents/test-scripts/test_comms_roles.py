"""
test_comms_roles.py
-------------------
Assertion-based tests for fixes 5 (communication layer) and 6 (roles and
adaptive coordination). Runs on a synthetic grid city with a scripted wall, so
it needs neither the Tokyo graph nor rtree/scipy and finishes in seconds.

    python test_comms_roles.py        # or: pytest test_comms_roles.py
"""

import contextlib
import io
import statistics
import sys
import traceback

from comms import (CommConfig, MessageBus, BOARD, payload_bytes, HEADER_BYTES,
                   build_delay_matrix_configs)
from dynamic_disaster_engine import DisasterEvent
from roles import (Role, CoordinationConfig, resolve_roles, default_role_counts,
                   ROLE_SPECS)
from swarm_simulation_dynamic import DynamicSwarmSimulation
import agent_snn_v3 as A

W = 30
STEP_LAT, STEP_LON = 0.0009, 0.0011          # ~100 m per hop


# --------------------------------------------------------------------------
# Synthetic city + scripted disaster
# --------------------------------------------------------------------------

class GridGraph:
    """W x W street grid, ~100 m edges. Same surface the swarm uses on Tokyo."""
    def __init__(self, w=W):
        self.w = w
        self._nodes = {r * w + c: {"lat": 35.0 + r * STEP_LAT, "lon": 139.0 + c * STEP_LON}
                       for r in range(w) for c in range(w)}
        self._adj = {n: [] for n in self._nodes}
        for r in range(w):
            for c in range(w):
                n = r * w + c
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < w and 0 <= cc < w:
                        self._adj[n].append((rr * w + cc, {"length": 100.0}))
        self._node_damage = {}
        self._disabled_edges = set()

    def node(self, r, c): return r * self.w + c
    def is_edge_passable(self, u, v): return frozenset({u, v}) not in self._disabled_edges
    def get_node_damage(self, n): return self._node_damage.get(n, 0.0)
    def reset_disaster(self): self._node_damage.clear(); self._disabled_edges.clear()


class WallEngine:
    """Wall down column `col`, rows [r0, r1), appearing at step `at`."""
    disaster_type = "scripted"

    def __init__(self, graph, col=15, r0=5, r1=25, at=3, dmg=0.8):
        self.graph, self.col, self.r0, self.r1, self.at, self.dmg = graph, col, r0, r1, at, dmg
        self.centre = (35.0 + 15 * STEP_LAT, 139.0 + col * STEP_LON)
        self.events = []

    def initialise(self):
        self.graph.reset_disaster()
        return DisasterEvent(0, "init", self.centre, 0, 0, [], 0, "none")

    def tick(self, step):
        if step != self.at:
            return []
        nodes = [self.graph.node(r, self.col) for r in range(self.r0, self.r1)]
        for n in nodes:
            self.graph._node_damage[n] = self.dmg
        ev = DisasterEvent(step, "wall", self.centre, 0, self.dmg, nodes, 0, "wall")
        self.events.append(ev)
        return [ev]

    def summary(self):
        return {"disaster_type": "scripted", "total_events": len(self.events)}


class NoEngine(WallEngine):
    def tick(self, step): return []


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def make(n=6, seed=1, steps=0, engine_cls=WallEngine, **kw):
    g = GridGraph()
    kw.setdefault("min_start_goal_distance", 600)
    kw.setdefault("max_start_goal_distance", 2500)
    sim = quiet(DynamicSwarmSimulation, g, num_agents=n, seed=seed,
                engine=engine_cls(g), **kw)
    if steps:
        quiet(sim.run, steps)
    return sim


def paths(sim):
    return [list(a.path) for a in sim.agents]


# --------------------------------------------------------------------------
# FIX 5 — message bus
# --------------------------------------------------------------------------

def test_bus_counts_transmissions_and_deliveries_separately():
    bus = MessageBus(CommConfig())
    payload = {"blocked": (1, 2, 3), "danger": {4: 0.5}}
    bus.send(0, "hazard", payload, step=1, recipients=[1, 2, 3])
    list(bus.deliver(1))
    s = bus.stats()
    size = payload_bytes(payload)
    assert size == HEADER_BYTES + 3 * 8 + 12
    assert s["transmissions"] == 1 and s["deliveries"] == 3
    assert s["bytes_tx"] == size and s["bytes_rx"] == 3 * size


def test_bus_send_to_nobody_costs_nothing():
    bus = MessageBus(CommConfig())
    assert bus.send(0, "hazard", {"blocked": (1,)}, 1, []) is None
    assert bus.stats()["transmissions"] == 0


def test_bus_validates_invalid_send_even_when_no_recipients():
    import pytest
    bus = MessageBus(CommConfig())
    with pytest.raises(ValueError):
        bus.send(0, "hazard", {}, -1, [])
    with pytest.raises(ValueError):
        bus.send(0, "hazard", {}, 0, [True])
    with pytest.raises(ValueError):
        bus.send(-1, "hazard", {}, 0, [])


def test_bus_reports_channel_specific_cost_and_latency():
    bus = MessageBus(CommConfig(delay_steps=1, board_delay_steps=2))
    bus.send(0, "hazard", {"blocked": (1,)}, 0, [1])
    bus.send(0, "board", {"blocked": (2,)}, 0, [BOARD], delay=2)
    bus.deliver(1)
    bus.deliver(2)
    channels = bus.stats()["by_channel"]
    assert channels["local_messaging"]["transmissions"] == 1
    assert channels["local_messaging"]["deliveries"] == 1
    assert channels["local_messaging"]["mean_delay_steps"] == 1
    assert channels["shared_memory_board"]["transmissions"] == 1
    assert channels["shared_memory_board"]["deliveries"] == 1
    assert channels["shared_memory_board"]["mean_delay_steps"] == 2


def test_bus_delay_is_exact():
    bus = MessageBus(CommConfig(delay_steps=3))
    bus.send(0, "hazard", {"blocked": (1,)}, step=5, recipients=[1])
    assert bus.deliver(5) == [] and bus.deliver(7) == []
    got = bus.deliver(8)
    assert len(got) == 1 and got[0][0] == 1
    assert bus.stats()["mean_delay_steps"] == 3






def test_independent_delay_matrix_covers_cartesian_product():
    configs = build_delay_matrix_configs([0, 2], [1, 3, 5])
    assert len(configs) == 6
    assert set(configs) == {
        "local_0_board_1", "local_0_board_3", "local_0_board_5",
        "local_2_board_1", "local_2_board_3", "local_2_board_5",
    }
    assert configs["local_2_board_5"].delay_steps == 2
    assert configs["local_2_board_5"].board_delay_steps == 5
    assert all(c.shared_memory and c.local_messaging and c.pheromones
               for c in configs.values())


def test_independent_delay_matrix_rejects_bad_inputs():
    import pytest
    for args in (([], [0]), ([0], []), ([0, -1], [0]), ([True], [0]), ([0, 0], [0])):
        with pytest.raises(ValueError):
            build_delay_matrix_configs(*args)


def test_comm_config_rejects_invalid_radius_and_delay_values():
    import math
    import pytest
    for radius in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            CommConfig(radius_m=radius)
    for delay in (-1, True, 1.5):
        with pytest.raises(ValueError):
            CommConfig(delay_steps=delay)
    for delay in (-1, True, 1.5):
        with pytest.raises(ValueError):
            CommConfig(board_delay_steps=delay)


def test_message_bus_deduplicates_recipients_and_snapshots_payload():
    bus = MessageBus(CommConfig())
    payload = {"blocked": [1, 2]}
    bus.send(0, "hazard", payload, 1, [1, 1, 2])
    payload["blocked"].append(3)
    delivered = bus.deliver(1)
    assert [recipient for recipient, _ in delivered] == [1, 2]
    assert all(msg.payload["blocked"] == [1, 2] for _, msg in delivered)
    stats = bus.stats()
    assert stats["transmissions"] == 1
    assert stats["deliveries"] == 2
    assert stats["rx_by_recipient"] == {1: 1, 2: 1}


def test_message_bus_reports_pending_by_kind_and_recipient():
    bus = MessageBus(CommConfig(delay_steps=3))
    bus.send(0, "hazard", {"blocked": [1]}, 2, [1, 2])
    bus.send(1, "sync", {"blocked": [3]}, 2, [2])
    stats = bus.stats()
    assert stats["pending_deliveries"] == 3
    assert stats["pending_by_kind"] == {"hazard": 2, "sync": 1}
    assert stats["pending_by_recipient"] == {1: 1, 2: 2}
    assert stats["by_kind"]["hazard"]["pending"] == 2


def test_message_bus_reports_late_delivery_separately_from_scheduled_delay():
    bus = MessageBus(CommConfig(delay_steps=1))
    bus.send(0, "hazard", {"blocked": [1]}, 2, [1])
    bus.deliver(5)
    stats = bus.stats()
    assert stats["mean_scheduled_delay_steps"] == 1
    assert stats["mean_delay_steps"] == 3
    assert stats["late_deliveries"] == 1




def test_message_bus_uses_null_mean_delay_when_nothing_was_delivered():
    bus = MessageBus(CommConfig(delay_steps=2))
    bus.send(0, "hazard", {"blocked": [1]}, 1, [2])
    # Recipient is dead when the delivery becomes due.
    assert bus.deliver(3, is_alive=lambda _recipient: False) == []
    stats = bus.stats()
    assert stats["deliveries"] == 0
    assert stats["mean_delay_steps"] is None
    assert stats["mean_scheduled_delay_steps"] is None
    assert stats["dropped"] == 1


def test_message_bus_rejects_invalid_send_and_delivery_steps():
    import pytest
    bus = MessageBus(CommConfig())
    for step in (-1, True, 1.5):
        with pytest.raises(ValueError):
            bus.send(0, "hazard", {}, step, [1])
        with pytest.raises(ValueError):
            bus.deliver(step)


def test_bus_never_delivers_before_each_configured_delay():
    for delay in (0, 1, 2, 3):
        bus = MessageBus(CommConfig(delay_steps=delay))
        sent_step = 7
        bus.send(0, "hazard", {"blocked": (11,)}, sent_step, [1])
        if delay > 0:
            assert bus.deliver(sent_step + delay - 1) == []
            assert bus.pending() == 1
        delivered = bus.deliver(sent_step + delay)
        assert len(delivered) == 1
        assert bus.pending() == 0
        assert bus.stats()["mean_scheduled_delay_steps"] == delay
        assert bus.stats()["mean_delay_steps"] == delay


def test_bus_drops_messages_to_dead_recipients():
    bus = MessageBus(CommConfig())
    bus.send(0, "hazard", {"blocked": (1,)}, 1, [1, 2])
    got = bus.deliver(1, is_alive=lambda r: r != 2)
    assert [r for r, _ in got] == [1]
    assert bus.stats()["dropped"] == 1 and bus.stats()["deliveries"] == 1


def test_board_uses_its_own_delay():
    bus = MessageBus(CommConfig(delay_steps=0, board_delay_steps=4))
    bus.send(0, "board", {"blocked": (9,)}, 2, [BOARD], delay=4)
    assert bus.deliver(5) == [] and len(bus.deliver(6)) == 1


# --------------------------------------------------------------------------
# FIX 5 — radius / link rule
# --------------------------------------------------------------------------

def test_radius_controls_who_is_in_range():
    sim = make(n=2, steps=0, comm=CommConfig(radius_m=500))
    a, b = sim.agents
    a.current_node, b.current_node = sim.graph.node(0, 0), sim.graph.node(0, 4)   # ~440 m
    assert sim._in_range(0) == [1]
    b.current_node = sim.graph.node(0, 8)                                           # ~880 m
    assert sim._in_range(0) == []
    sim.comm = CommConfig(radius_m=1000)
    assert sim._in_range(0) == [1]


def test_radius_boundary_is_inclusive_and_just_outside_is_excluded():
    sim = make(n=2, steps=0, comm=CommConfig(radius_m=500))
    a, b = sim.agents
    a.current_node = sim.graph.node(0, 0)
    b.current_node = sim.graph.node(0, 4)
    exact_distance = sim._distance(a.current_node, b.current_node)
    sim.comm = CommConfig(radius_m=exact_distance)
    assert sim._in_range(0) == [1]
    sim.comm = CommConfig(radius_m=exact_distance - 0.01)
    assert sim._in_range(0) == []


def test_dead_agent_is_never_a_local_radio_recipient():
    sim = make(n=2, steps=0, comm=CommConfig(radius_m=500))
    a, b = sim.agents
    a.current_node = sim.graph.node(0, 0)
    b.current_node = sim.graph.node(0, 1)
    b.alive = False
    assert sim._in_range(0) == []


def test_hub_range_uses_larger_of_the_two_radii():
    sim = make(n=6, roles={"navigator": 4, "scout": 1, "assessor": 1}, steps=0)
    hub = next(i for i, a in enumerate(sim.agents) if a.role == "assessor")
    nav = 0
    sim.agents[hub].current_node = sim.graph.node(0, 0)
    sim.agents[nav].current_node = sim.graph.node(0, 12)                            # ~1.3 km
    assert hub in sim._in_range(nav) and nav in sim._in_range(hub)                  # 3 x 500 m
    sim.agents[nav].current_node = sim.graph.node(0, 20)                            # ~2.2 km
    assert hub not in sim._in_range(nav)


# --------------------------------------------------------------------------
# FIX 5 — switches are independent and actually switch things off
# --------------------------------------------------------------------------

def test_all_channels_off_means_agents_learn_only_by_sensing():
    cfg = CommConfig(shared_memory=False, local_messaging=False, pheromones=False)
    sim = make(n=6, steps=40, comm=cfg)
    assert sim.bus.stats()["transmissions"] == 0
    assert not sim.shared_memory["blocked_nodes"] and not sim.shared_memory["pheromones"]
    assert all(a.stats["blocked_learned_msg"] == 0 for a in sim.agents)
    assert any(a.stats["blocked_learned_self"] > 0 for a in sim.agents)   # control: sensing works


def test_shared_memory_switch_is_separate_from_local_messaging():
    sim = make(n=6, steps=40, comm=CommConfig(shared_memory=False, pheromones=False))
    kinds = sim.bus.stats()["by_kind"]
    assert "board" not in kinds and not sim.shared_memory["blocked_nodes"]
    assert "hazard" in kinds or "sync" in kinds


def test_local_messaging_switch_is_separate_from_shared_memory():
    sim = make(n=6, steps=40, comm=CommConfig(local_messaging=False))
    kinds = sim.bus.stats()["by_kind"]
    assert set(kinds) == {"board"} and sim.shared_memory["blocked_nodes"]


def test_pheromone_switch():
    on  = make(n=6, steps=60, comm=CommConfig(pheromones=True))
    off = make(n=6, steps=60, comm=CommConfig(pheromones=False))
    assert any(a.reached_goal for a in on.agents), "need arrivals for the control"
    assert len(on.shared_memory["pheromones"]) > 0
    assert len(off.shared_memory["pheromones"]) == 0
    assert all(not a.shared_memory["pheromones"] for a in off.agents)


def test_disabled_channel_is_invisible_to_agents():
    sim = make(n=2, steps=0, comm=CommConfig(shared_memory=False))
    sim.shared_memory["blocked_nodes"].add(123)          # even if something wrote to it
    assert not sim.agents[0].knows_blocked(123)
    sim2 = make(n=2, steps=0, comm=CommConfig(shared_memory=True))
    sim2.shared_memory["blocked_nodes"].add(123)
    assert sim2.agents[0].knows_blocked(123)


def test_board_delay_delays_visibility():
    s0 = make(n=6, steps=30, comm=CommConfig(board_delay_steps=0, local_messaging=False))
    s5 = make(n=6, steps=30, comm=CommConfig(board_delay_steps=5, local_messaging=False))
    shared = [n for n in s0.board_visible if n in s5.board_visible]
    assert shared
    assert all(s5.board_visible[n] >= s0.board_visible[n] + 5 - 1 for n in shared)
    assert s5.info_propagation()["mean_delay_steps"] > s0.info_propagation()["mean_delay_steps"]


# --------------------------------------------------------------------------
# FIX 5 — information is only used if it was obtained
# --------------------------------------------------------------------------

def test_shared_danger_is_now_read_by_agents():
    sim = make(n=1, steps=0, comm=CommConfig())
    ag = sim.agents[0]
    node = sim.graph.node(10, 10)
    assert ag.known_danger(node) == 0.0
    sim.shared_memory["danger_nodes"][node] = 0.5
    assert ag.known_danger(node) == 0.5


def test_dead_end_feature_comes_from_knowledge_not_ground_truth():
    sim = make(n=1, steps=0, engine_cls=NoEngine, comm=CommConfig())
    ag = sim.agents[0]
    g = sim.graph
    ag.current_node = g.node(10, 10)
    cand = g.node(10, 11)
    exits = [m for m, _ in g._adj[cand] if m != ag.current_node]
    for m in exits:                                  # truly destroyed, but agent not told
        g._node_damage[m] = 0.9
    assert ag._exit_risk(cand) == (0.0, False)
    sim.shared_memory["blocked_nodes"].update(exits)  # now it is on the board
    risk, dead = ag._exit_risk(cand)
    assert dead and risk == 1.0
    ag.shared_memory = {"blocked_nodes": frozenset(), "danger_nodes": {}, "pheromones": {}}
    assert ag._exit_risk(cand) == (0.0, False)        # board switched off for this agent


def test_local_message_reaches_neighbour_but_not_distant_agent():
    sim = make(n=3, steps=0, engine_cls=NoEngine,
               comm=CommConfig(shared_memory=False, sync_on_contact=False))
    a, b, c = sim.agents
    g = sim.graph
    a.current_node, b.current_node, c.current_node = g.node(0, 0), g.node(0, 3), g.node(0, 20)
    hazard = g.node(1, 0)
    g._node_damage[hazard] = 0.9
    a.clock = 1
    a.sense()
    quiet(sim._comms_phase, 1)
    assert hazard in b.local_memory["blocked_nodes"] and b.learned_at[hazard][1] == "msg"
    assert hazard not in c.local_memory["blocked_nodes"]


def test_sync_on_contact_transfers_old_knowledge_once():
    sim = make(n=2, steps=0, engine_cls=NoEngine,
               comm=CommConfig(shared_memory=False, sync_on_contact=True))
    a, b = sim.agents
    g = sim.graph
    a.current_node, b.current_node = g.node(0, 0), g.node(0, 25)
    a.local_memory["blocked_nodes"].add(g.node(5, 5))
    quiet(sim._comms_phase, 1)
    assert g.node(5, 5) not in b.local_memory["blocked_nodes"]        # out of range
    b.current_node = g.node(0, 2)                                     # they meet
    quiet(sim._comms_phase, 2)
    assert g.node(5, 5) in b.local_memory["blocked_nodes"]
    before = sim.bus.stats()["by_kind"].get("sync", {}).get("tx", 0)
    quiet(sim._comms_phase, 3)                                        # still together
    assert sim.bus.stats()["by_kind"].get("sync", {}).get("tx", 0) == before


# --------------------------------------------------------------------------
# FIX 5 — safety / reproducibility regressions
# --------------------------------------------------------------------------

def test_agents_never_traverse_a_disabled_edge_or_destroyed_node():
    g = GridGraph()
    bad = []
    orig = A.SNNAgent._execute_move
    def spy(self, nxt, cp, gp):
        if not g.is_edge_passable(self.current_node, nxt) or g.get_node_damage(nxt) > 0.6:
            bad.append((self.current_node, nxt))
        return orig(self, nxt, cp, gp)
    A.SNNAgent._execute_move = spy
    try:
        sim = quiet(DynamicSwarmSimulation, g, num_agents=8, seed=2, engine=WallEngine(g),
                    min_start_goal_distance=600, max_start_goal_distance=2500,
                    roles="default")
        quiet(sim.run, 80)
    finally:
        A.SNNAgent._execute_move = orig
    assert not bad, bad[:3]


def test_same_seed_gives_identical_paths_and_traffic():
    kw = dict(n=8, steps=60, roles="default", coordination=CoordinationConfig(mode="adaptive"))
    a, b = make(seed=5, **kw), make(seed=5, **kw)
    assert paths(a) == paths(b)
    assert a.bus.stats()["transmissions"] == b.bus.stats()["transmissions"]
    assert a.bus.stats()["bytes_tx"] == b.bus.stats()["bytes_tx"]
    assert paths(make(seed=6, **kw)) != paths(a)


# --------------------------------------------------------------------------
# FIX 6 — role assignment
# --------------------------------------------------------------------------

def test_role_resolution():
    assert resolve_roles(None, 4) == [Role.NAVIGATOR] * 4
    c = default_role_counts(10)
    assert sum(c.values()) == 10 and c[Role.SCOUT] == 2 and c[Role.ASSESSOR] == 1
    assert c[Role.ALLOCATOR] == 0 and default_role_counts(10, True)[Role.ALLOCATOR] == 1
    assert resolve_roles({"navigator": 2, "scout": 1}, 3) == \
        [Role.NAVIGATOR, Role.NAVIGATOR, Role.SCOUT]
    for bad in ({"navigator": 2}, ["navigator"] * 3):
        try:
            resolve_roles(bad, 4); assert False, "should have raised"
        except ValueError:
            pass


def test_navigator_tasks_do_not_depend_on_support_roles():
    base = make(n=4, seed=9, roles={"navigator": 4})
    more = make(n=7, seed=9, roles={"navigator": 4, "scout": 2, "assessor": 1})
    assert [t["start"] for t in base.tasks[:4]] == [t["start"] for t in more.tasks[:4]]
    assert [t["goal"] for t in base.tasks[:4]] == [t["goal"] for t in more.tasks[:4]]


def test_specs_are_applied_to_agents():
    sim = make(n=8, roles={"navigator": 5, "scout": 2, "assessor": 1}, steps=0)
    by = {}
    for a in sim.agents:
        by.setdefault(a.role, []).append(a)
    assert all(a.use_planner and a.sense_hops == 1 for a in by["navigator"])
    assert all(a.sense_hops == 2 and not a.stationary for a in by["scout"])
    assert by["scout"][0].snn.w_explore == ROLE_SPECS[Role.SCOUT].weights["w_explore"]
    hub = by["assessor"][0]
    assert hub.stationary and sim._range_m(sim.agents.index(hub)) == 1500.0
    assert by["navigator"][0].snn.w_explore != by["scout"][0].snn.w_explore


def test_homogeneous_baseline_has_no_planner_and_no_specialisation():
    sim = make(n=4, steps=0)
    assert all(a.role == "navigator" and not a.use_planner and not a.stationary
               for a in sim.agents)


def test_hub_is_stationary_and_sits_among_the_navigators():
    sim = make(n=7, roles={"navigator": 5, "scout": 1, "assessor": 1}, steps=40)
    hub = next(a for a in sim.agents if a.role == "assessor")
    assert len(set(hub.path)) == 1
    nav_lat = [sim.graph._nodes[a.path[0]]["lat"] for a in sim.agents if a.role == "navigator"]
    assert min(nav_lat) - 0.01 <= sim.graph._nodes[hub.path[0]]["lat"] <= max(nav_lat) + 0.01


def test_scouts_keep_patrolling_and_success_counts_navigators_only():
    sim = make(n=7, roles={"navigator": 5, "scout": 2}, steps=120)
    scouts = [a for a in sim.agents if a.role == "scout"]
    assert all(a.alive and len(a.path) > 30 for a in scouts)
    assert all(a.stats["goal_changes"] >= 1 for a in scouts)
    m = sim.adaptation_metrics()
    assert m["total_agents"] == 5
    assert m["reached_goal"] == sum(a.reached_goal for a in sim.agents if a.role == "navigator")


def test_scouts_cover_more_ground_than_navigators_of_same_age():
    sim = make(n=8, roles={"navigator": 5, "scout": 3}, steps=50, seed=3)
    cov = lambda role: statistics.mean(len(a.cell_counts) for a in sim.agents if a.role == role)
    assert cov("scout") > cov("navigator")


# --------------------------------------------------------------------------
# FIX 6 — planner (the Routing role)
# --------------------------------------------------------------------------

def test_planner_routes_around_known_hazards_and_replans_on_new_knowledge():
    sim = make(n=1, steps=0, engine_cls=NoEngine, roles={"navigator": 1})
    ag, g = sim.agents[0], sim.graph
    ag.current_node, ag.goal_node = g.node(15, 5), g.node(15, 25)
    ag.plan = None
    ag._ensure_plan()
    assert ag.plan[0] == ag.current_node and ag.plan[-1] == ag.goal_node
    assert len(ag.plan) == 21                                   # straight line, 20 hops
    wall = [g.node(r, 15) for r in range(5, 25)]
    for n in wall:
        ag._learn_blocked(n, "msg")                             # told, not sensed
    ag._ensure_plan()
    assert not (set(ag.plan) & set(wall)), "plan crosses a known-blocked node"
    assert len(ag.plan) > 21                                    # detour around the wall
    assert ag.stats["plan_invalidations"] == 1 and ag.plan_events[-1]["reason"] == "invalidated"
    n_replans = ag.stats["replans"]
    ag._ensure_plan()
    assert ag.stats["replans"] == n_replans                     # no pointless replans


def test_planner_ignores_hazards_it_has_not_been_told_about():
    sim = make(n=1, steps=0, engine_cls=NoEngine, roles={"navigator": 1})
    ag, g = sim.agents[0], sim.graph
    ag.current_node, ag.goal_node = g.node(15, 5), g.node(15, 25)
    for r in range(5, 25):
        g._node_damage[g.node(r, 15)] = 0.9                     # truth, unknown to agent
    ag.plan = None
    ag._ensure_plan()
    assert len(ag.plan) == 21


def test_planner_gives_up_cleanly_when_goal_is_known_blocked():
    sim = make(n=1, steps=0, engine_cls=NoEngine, roles={"navigator": 1})
    ag, g = sim.agents[0], sim.graph
    ag.current_node, ag.goal_node = g.node(15, 5), g.node(15, 25)
    ag._learn_blocked(ag.goal_node, "msg")
    ag.plan = None
    ag._ensure_plan()
    assert ag.plan is None and ag.plan_events[-1]["ok"] is False


# --------------------------------------------------------------------------
# FIX 6 — assessor / allocator / adaptive coordination
# --------------------------------------------------------------------------

def _hub_sim(mode, **kw):
    return make(n=6, seed=4, engine_cls=NoEngine,
                roles={"navigator": 4, "scout": 1, "assessor": 1},
                comm=CommConfig(shared_memory=False, sync_on_contact=False, **kw),
                coordination=CoordinationConfig(mode=mode, advisory_period=5))


def test_assessor_relays_beyond_a_navigators_own_radius():
    sim = _hub_sim("static")
    g = sim.graph
    hub = next(i for i, a in enumerate(sim.agents) if a.role == "assessor")
    far = 0
    sim.agents[hub].current_node = g.node(0, 0)
    sim.agents[far].current_node = g.node(0, 12)                # 1.3 km: inside hub range only
    sim.agents[hub].local_memory["blocked_nodes"].add(g.node(20, 20))
    assert g.node(20, 20) not in sim.agents[far].local_memory["blocked_nodes"]
    quiet(sim._comms_phase, 5)                                   # period boundary
    assert g.node(20, 20) in sim.agents[far].local_memory["blocked_nodes"]
    assert sim.bus.stats()["by_kind"]["advisory"]["tx"] == 1


def test_static_assessor_waits_for_schedule_adaptive_does_not():
    for mode, expect_at_3 in (("static", 0), ("adaptive", 1)):
        sim = _hub_sim(mode)
        hub = next(a for a in sim.agents if a.role == "assessor")
        sim.agents[0].current_node = hub.current_node
        hub.local_memory["blocked_nodes"].add(77)
        quiet(sim._comms_phase, 3)
        got = sim.bus.stats()["by_kind"].get("advisory", {}).get("tx", 0)
        assert got == expect_at_3, (mode, got)


def _shelter_sim(mode, n=9):
    g = GridGraph()
    shelters = [g.node(2, 2), g.node(27, 27), g.node(2, 27)]
    sim = quiet(DynamicSwarmSimulation, g, num_agents=n, seed=7, engine=NoEngine(g),
                min_start_goal_distance=600, max_start_goal_distance=4000,
                roles={"navigator": n - 2, "assessor": 1, "allocator": 1},
                shelters=shelters, shelter_capacity=4,
                comm=CommConfig(shared_memory=False, sync_on_contact=False),
                coordination=CoordinationConfig(mode=mode, period=5))
    return sim


def test_allocator_assigns_navigators_to_shelters_within_capacity():
    sim = _shelter_sim("static")
    goals = [a.goal_node for a in sim.agents if a.role == "navigator"]
    assert all(gl in sim.allocator.shelters for gl in goals)
    assert max(goals.count(s) for s in set(goals)) <= sim.allocator.capacity
    assert sum(sim.allocator.load.values()) == len(goals)


def _block_goal_and_tick(sim):
    hub_idx = next(i for i, a in enumerate(sim.agents) if a.role == "allocator")
    nav_idx = next(i for i, a in enumerate(sim.agents) if a.role == "navigator")
    nav, hub = sim.agents[nav_idx], sim.agents[hub_idx]
    nav.current_node = hub.current_node                          # in range
    old = nav.goal_node
    hub.local_memory["blocked_nodes"].add(old)                   # allocator learns it is blocked
    sim.current_step = 5
    sim.coordinator.allocator_step(5)
    quiet(sim._comms_phase, 5)
    return nav_idx, nav, old


def test_adaptive_allocator_reassigns_when_goal_becomes_known_blocked():
    sim = _shelter_sim("adaptive")
    load_before = sum(sim.allocator.load.values())
    _, nav, old = _block_goal_and_tick(sim)
    assert nav.goal_node != old and nav.goal_node in sim.allocator.shelters
    assert nav.stats["goal_changes"] == 1 and nav.task_info["reassigned"]
    assert sim.bus.stats()["by_kind"]["assign"]["tx"] == 1
    assert sum(sim.allocator.load.values()) == load_before        # moved, not leaked
    assert any(e["type"] == "reassign" for e in sim.coordination_log)


def test_static_allocator_never_reassigns():
    sim = _shelter_sim("static")
    _, nav, old = _block_goal_and_tick(sim)
    assert nav.goal_node == old and not sim.coordination_log


def test_adaptive_scouts_head_for_the_hazard_static_scouts_do_not():
    def mean_dist(mode):
        sim = make(n=6, seed=2, engine_cls=NoEngine, roles={"navigator": 4, "scout": 2},
                   coordination=CoordinationConfig(mode=mode))
        scout_idx = next(i for i, a in enumerate(sim.agents) if a.role == "scout")
        sc, g = sim.agents[scout_idx], sim.graph
        sc.current_node = g.node(15, 5)
        for r in range(26, 30):                                   # known hazard, far corner
            sc._learn_blocked(g.node(r, 28), "msg")
        d = []
        for _ in range(60):
            sim.coordinator.scout_waypoint(scout_idx)
            n = g._nodes[sc.goal_node]
            d.append(((n["lat"] - (35.0 + 27 * STEP_LAT)) ** 2 +
                      (n["lon"] - (139.0 + 28 * STEP_LON)) ** 2) ** 0.5)
        return statistics.mean(d)
    assert mean_dist("adaptive") < mean_dist("static")


def test_coordination_events_only_exist_in_adaptive_mode():
    kw = dict(n=8, steps=60, seed=3, roles={"navigator": 5, "scout": 2, "assessor": 1})
    assert not make(coordination=CoordinationConfig(mode="static"), **kw).coordination_log
    assert make(coordination=CoordinationConfig(mode="adaptive"), **kw).coordination_log


def test_invalid_coordination_mode_is_rejected():
    try:
        CoordinationConfig(mode="clever"); assert False
    except ValueError:
        pass


def test_awareness_metrics_report_all_agents_and_sources_separately():
    sim = make(n=2, steps=0, engine_cls=NoEngine)
    sim._hazard_event_records = {
        "hazard_1": {"event_id": "hazard_1", "affected_agents": [0]}
    }
    sim._agent_event_awareness = {
        (0, "hazard_1"): {
            "agent": 0, "event_id": "hazard_1",
            "awareness_delay_steps": 0, "source": "self_observation",
        },
        (1, "hazard_1"): {
            "agent": 1, "event_id": "hazard_1",
            "awareness_delay_steps": 3, "source": "teammate_message",
        },
    }
    sim._agent_event_responses = {}
    metrics = sim.hazard_response_metrics()
    assert metrics["mean_awareness_delay_steps"] == 0.0
    assert metrics["mean_awareness_delay_affected_pairs_steps"] == 0.0
    assert metrics["mean_awareness_delay_all_agents_steps"] == 1.5
    assert metrics["mean_awareness_delay_outside_initial_affected_set_steps"] == 3.0
    assert metrics["awareness_delay_record_counts"] == {
        "all_agents": 2,
        "initially_affected_pairs": 1,
        "outside_initial_affected_set": 1,
    }
    assert metrics["awareness_source_counts"] == {
        "self_observation": 1,
        "teammate_message": 1,
    }
    assert metrics["awareness_by_source"]["self_observation"] == {
        "count": 1, "delay_sum_steps": 0.0, "delay_count": 1,
        "mean_delay_steps": 0.0,
    }
    assert metrics["awareness_by_source"]["teammate_message"]["count"] == 1
    assert metrics["awareness_by_source"]["teammate_message"]["mean_delay_steps"] == 3.0


# --------------------------------------------------------------------------
# FIX 6 — metrics plumbing
# --------------------------------------------------------------------------

def test_metrics_have_per_agent_contributions_and_propagation():
    sim = make(n=8, steps=60, seed=3, roles="default",
               coordination=CoordinationConfig(mode="adaptive"))
    m = sim.metrics()
    assert len(m["contributions"]) == 8
    assert sum(c["first_detections"] for c in m["contributions"]) == len(sim.detection_log)
    assert sum(c["tx"] for c in m["contributions"]) == m["comms"]["transmissions"]
    ip = m["info_propagation"]
    assert ip["hazards_detected"] > 0 and 0 < ip["fraction_informed"] <= 1
    assert m["config"]["roles"] == {"navigator": 5, "scout": 2, "assessor": 1}


def test_communication_improves_hazard_awareness():
    on  = make(n=8, steps=40, seed=3, comm=CommConfig())
    off = make(n=8, steps=40, seed=3,
               comm=CommConfig(shared_memory=False, local_messaging=False, pheromones=False))
    assert on.info_propagation()["fraction_informed"] > off.info_propagation()["fraction_informed"]


# --------------------------------------------------------------------------

def main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc(limit=3)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()



def test_non_scored_cycle_escape_does_not_reuse_previous_snn_diagnostics():
    """A movement outside score_neighbor must not inherit the prior tick's trace."""
    graph = GridGraph()
    current = graph.node(10, 10)
    goal = graph.node(20, 20)
    agent = A.SNNAgent(
        graph, current, goal,
        rng=__import__("random").Random(7),
    )
    agent.snn.last_decision = {
        "selection_method": "softmax_far",
        "candidates": [{"node": 999, "pheromone_score": 123.0}],
    }
    agent._detect_cycle = lambda: (True, 2)
    observed = []
    agent.decision_observer = (
        lambda from_node, to_node, step:
        observed.append(agent.snn.last_decision)
    )

    agent.step()

    assert len(observed) == 1
    assert observed[0] is None
