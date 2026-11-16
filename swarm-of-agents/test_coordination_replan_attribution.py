"""Focused regression tests for coordination replan-source attribution."""

from swarm_simulation_dynamic import DynamicSwarmSimulation


def _bare_sim():
    sim = DynamicSwarmSimulation.__new__(DynamicSwarmSimulation)
    sim._response = {
        "planner_successful_replans": 0,
        "planner_route_changes_following_hazard_discovery": 0,
        "planner_replan_source_attribution": {
            "self_observation": 0,
            "teammate_message": 0,
            "shared_memory": 0,
            "unattributed": 0,
        },
        "planner_replan_source_records": [],
    }
    sim._agent_event_awareness = {}
    sim._agent_event_responses = {}
    sim._hazard_event_records = {}
    return sim


def test_successful_route_change_uses_first_teammate_awareness():
    sim = _bare_sim()
    awareness = {
        "agent": 2,
        "event_id": "event_1",
        "awareness_step": 4,
        "awareness_source": "teammate_message",
    }

    record = sim._record_planner_replan_source(
        2, "event_1", 7, awareness, True
    )

    assert record["awareness_source"] == "teammate_message"
    assert record["response_delay_steps"] == 3
    assert sim._response["planner_replan_source_attribution"][
        "teammate_message"
    ] == 1
    assert len(sim._response["planner_replan_source_records"]) == 1


def test_successful_route_change_uses_first_shared_memory_awareness():
    sim = _bare_sim()
    awareness = {
        "agent": 3,
        "event_id": "event_2",
        "awareness_step": 5,
        "awareness_source": "shared_memory",
    }

    sim._record_planner_replan_source(
        3, "event_2", 8, awareness, True
    )

    counts = sim._response["planner_replan_source_attribution"]
    assert counts["shared_memory"] == 1
    assert counts["self_observation"] == 0
    assert counts["teammate_message"] == 0


def test_unsuccessful_or_unchanged_plan_is_not_attributed():
    sim = _bare_sim()
    awareness = {
        "agent": 1,
        "event_id": "event_3",
        "awareness_step": 2,
        "awareness_source": "teammate_message",
    }

    assert sim._record_planner_replan_source(
        1, "event_3", 4, awareness, False
    ) is None
    assert sum(
        sim._response["planner_replan_source_attribution"].values()
    ) == 0
    assert sim._response["planner_replan_source_records"] == []


def test_prior_default_response_does_not_hide_planner_attribution():
    sim = _bare_sim()
    sim._agent_event_awareness[(0, "event_4")] = {
        "agent": 0,
        "event_id": "event_4",
        "awareness_step": 1,
        "awareness_source": "teammate_message",
    }
    sim._agent_event_responses[(0, "event_4")] = {
        "agent": 0,
        "event_id": "event_4",
        "response_type": "default_policy_safe_navigation_action",
        "successful": True,
    }
    sim._hazard_event_records["event_4"] = {"responses": {}}

    # The first-response record remains unchanged, but the independently
    # observed planner route change is still attributed.
    assert sim._record_event_response(
        0, "event_4", 6, "planner_replan", True,
        {"plan_found": True, "route_changed": True},
    ) is None
    sim._record_planner_replan_source(
        0, "event_4", 6,
        sim._agent_event_awareness[(0, "event_4")],
        True,
    )

    assert sim._response["planner_replan_source_attribution"][
        "teammate_message"
    ] == 1
