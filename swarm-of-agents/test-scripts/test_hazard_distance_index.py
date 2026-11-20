"""Regression tests for fast, exact nearest-hazard distance diagnostics."""

import math

from swarm_simulation_dynamic import DynamicSwarmSimulation


class _Graph:
    def __init__(self):
        self._nodes = {
            "agent": {"lat": 35.6800, "lon": 139.7600},
            "near": {"lat": 35.6810, "lon": 139.7600},
            "far": {"lat": 35.7000, "lon": 139.7600},
            "other": {"lat": 35.6800, "lon": 139.7700},
        }


def _simulation():
    sim = DynamicSwarmSimulation.__new__(DynamicSwarmSimulation)
    sim.graph = _Graph()
    sim._hazard_distance_index_cache = {"initial": None, "dynamic": None}
    return sim


def test_nearest_hazard_distance_matches_great_circle_distance():
    sim = _simulation()
    measured = sim._nearest_hazard_distance("agent", {"near", "far"}, "dynamic")

    # At this latitude, the north/south separation is approximately 111 m.
    expected = 6_371_000.0 * math.radians(0.001)
    assert abs(measured - expected) < 0.05


def test_nearest_hazard_index_is_reused_when_hazard_set_is_unchanged():
    sim = _simulation()
    hazards = {"near", "far"}
    sim._nearest_hazard_distance("agent", hazards, "dynamic")
    first_tree = sim._hazard_distance_index_cache["dynamic"][2]

    sim._nearest_hazard_distance("agent", set(hazards), "dynamic")
    second_tree = sim._hazard_distance_index_cache["dynamic"][2]

    assert second_tree is first_tree


def test_nearest_hazard_index_rebuilds_when_hazard_set_changes():
    sim = _simulation()
    sim._nearest_hazard_distance("agent", {"near", "far"}, "dynamic")
    first_tree = sim._hazard_distance_index_cache["dynamic"][2]

    sim._nearest_hazard_distance("agent", {"other"}, "dynamic")
    second_tree = sim._hazard_distance_index_cache["dynamic"][2]

    assert second_tree is not first_tree
    measured = sim._nearest_hazard_distance("agent", {"other"}, "dynamic")
    assert measured > 800.0


def test_empty_hazard_set_returns_none_without_building_index():
    sim = _simulation()
    assert sim._nearest_hazard_distance("agent", set(), "dynamic") is None
    assert sim._hazard_distance_index_cache["dynamic"] is None
