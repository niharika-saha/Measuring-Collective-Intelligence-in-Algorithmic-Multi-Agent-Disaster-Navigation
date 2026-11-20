"""Regression tests for dynamic fire-spread edge reporting.

These tests intentionally use a tiny in-memory graph so they do not require
the Tokyo dataset or the optional R-tree dependency.
"""
import random

from dynamic_disaster_engine import DynamicDisasterEngine


class _TinyGraph:
    def __init__(self):
        self._nodes = {
            1: {"lat": 35.0, "lon": 139.0},
            2: {"lat": 35.0, "lon": 139.0},
        }
        self._adj = {1: [(2, {})], 2: [(1, {})]}
        self._node_damage = {}
        self._disabled_edges = set()

    def get_fire_stations(self):
        return []


def test_fire_spread_reports_newly_disabled_edge_ids():
    graph = _TinyGraph()
    engine = DynamicDisasterEngine(
        graph,
        disaster_type="fire",
        centre=(35.0, 139.0),
        radius_m=1000.0,
        seed=0,
    )
    engine._fire_nodes = {1}
    engine._burned_nodes = {1}
    engine.FIRE_SPREAD_BASE = 1.0
    engine._rng = random.Random(0)

    events = engine._tick_fire(step=1)

    assert len(events) == 1
    event = events[0]
    expected_edge = frozenset({1, 2})
    assert event.nodes_newly_damaged == [2]
    assert event.edges_newly_disabled == 1
    assert event.edges_newly_disabled_ids == [expected_edge]
    assert graph._disabled_edges == {expected_edge}
