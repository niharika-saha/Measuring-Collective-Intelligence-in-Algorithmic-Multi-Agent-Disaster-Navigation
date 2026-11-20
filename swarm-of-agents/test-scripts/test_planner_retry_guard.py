from agent_snn_v3 import SNNAgent


class TinyGraph:
    def __init__(self):
        self._nodes = {
            1: {"lat": 0.0, "lon": 0.0},
            2: {"lat": 0.0, "lon": 0.0005},
            3: {"lat": 0.0, "lon": 0.0010},
        }
        edge = {"length": 100.0}
        self._adj = {
            1: [(2, edge)],
            2: [(1, edge), (3, edge)],
            3: [(2, edge)],
        }


def make_agent():
    return SNNAgent(TinyGraph(), 1, 3)


def test_astar_finds_path_and_builds_constant_time_plan_index():
    agent = make_agent()
    route = agent._astar_known()
    assert route == [1, 2, 3]
    agent.plan = route
    agent._plan_index = {node: i for i, node in enumerate(route)}
    assert agent._plan_next() == 2
    agent.current_node = 2
    assert agent._plan_next() == 3
    assert agent.stats["plan_attempts"] == 1
    assert agent.stats["plan_successes"] == 1


def test_valid_plan_is_not_rescanned_when_knowledge_is_unchanged():
    agent = make_agent()

    agent.clock = 0
    agent._ensure_plan()
    assert agent.stats["plan_attempts"] == 1

    agent.current_node = 2
    agent.clock = 1
    agent._ensure_plan()

    assert agent.stats["plan_attempts"] == 1
    assert agent._plan_next() == 3


def test_failed_plan_is_not_repeated_twice_in_same_step_or_every_step():
    agent = make_agent()
    calls = []

    def failed_search():
        calls.append((agent.clock, agent.current_node))
        agent._last_plan_failure_reason = "expansion_limit"
        agent.stats["plan_attempts"] += 1
        agent.stats["plan_failures"] += 1
        return None

    agent._astar_known = failed_search

    agent.clock = 0
    agent._ensure_plan()
    agent._ensure_plan()  # second build_features call in the same step
    assert len(calls) == 1

    agent.clock = 1
    agent._ensure_plan()
    assert len(calls) == 1

    agent.clock = agent.PLAN_RETRY_INTERVAL
    agent._ensure_plan()
    assert len(calls) == 2
    assert agent.stats["plan_retry_skips"] >= 2


def test_new_local_knowledge_allows_immediate_retry():
    agent = make_agent()
    calls = []

    def failed_search():
        calls.append(agent.clock)
        agent._last_plan_failure_reason = "expansion_limit"
        agent.stats["plan_attempts"] += 1
        agent.stats["plan_failures"] += 1
        return None

    agent._astar_known = failed_search
    agent.clock = 0
    agent._ensure_plan()
    assert calls == [0]

    # New knowledge changes the planner's input, so it need not wait for the
    # normal retry interval.
    agent.clock = 1
    agent._learn_blocked(99, "self")
    agent._ensure_plan()
    assert calls == [0, 1]


def test_goal_change_clears_failed_plan_backoff():
    agent = make_agent()
    calls = []

    def failed_search():
        calls.append(agent.goal_node)
        agent._last_plan_failure_reason = "expansion_limit"
        agent.stats["plan_attempts"] += 1
        agent.stats["plan_failures"] += 1
        return None

    agent._astar_known = failed_search
    agent.clock = 0
    agent._ensure_plan()
    agent.clock = 1
    agent.set_goal(2)
    agent._ensure_plan()
    assert calls == [3, 2]


def test_known_blocked_goal_is_not_retried_until_goal_changes():
    agent = make_agent()
    calls = []

    def blocked_goal_search():
        calls.append(agent.goal_node)
        agent._last_plan_failure_reason = "goal_blocked"
        agent.stats["plan_attempts"] += 1
        agent.stats["plan_failures"] += 1
        return None

    agent._astar_known = blocked_goal_search
    agent._learn_blocked(3, "self")
    agent._ensure_plan()
    for step in (1, 10, 100):
        agent.clock = step
        agent._ensure_plan()
    assert calls == [3]
