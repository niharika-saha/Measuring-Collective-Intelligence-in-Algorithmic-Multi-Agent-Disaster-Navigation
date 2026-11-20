from types import SimpleNamespace

from swarm_simulation_dynamic import DynamicSwarmSimulation


class FakeAgent:
    def __init__(self, path, current_node, goal_node):
        self.path = list(path)
        self.current_node = current_node
        self.goal_node = goal_node
        self.alive = True
        self.reached_goal = True
        self.death_reason = None

    def set_goal(self, node):
        self.goal_node = node
        self.reached_goal = False


def _queue_sim():
    sim = DynamicSwarmSimulation.__new__(DynamicSwarmSimulation)
    sim.task_queues = [[
        {"task_id": "t1", "start": "a", "goal": "b"},
        {"task_id": "t2", "start": "c", "goal": "d"},
    ], [
        {"task_id": "t3", "start": "a", "goal": "e"},
    ]]
    sim._path_distance = lambda path: max(0, len(path) - 1) * 10.0
    sim.sequential_task_results = []
    sim._task_queue_states = {
        0: {
            "task_index": 0, "next_task_index": None, "phase": "task",
            "path_start_index": 0, "phase_steps": 1,
            "reposition_path_start_index": None,
            "reposition_steps_before": 0,
            "reposition_distance_before_m": 0.0, "complete": False,
        },
        1: {
            "task_index": 0, "next_task_index": None, "phase": "task",
            "path_start_index": 0, "phase_steps": 1,
            "reposition_path_start_index": None,
            "reposition_steps_before": 0,
            "reposition_distance_before_m": 0.0, "complete": False,
        },
    }
    sim.agents = [
        FakeAgent(["a", "b"], "b", "b"),
        FakeAgent(["a", "e"], "e", "e"),
    ]
    return sim


def test_each_agent_advances_its_own_queue_and_records_reposition_cost():
    sim = _queue_sim()

    # Agent 0 completes task 1; its next task starts elsewhere, so it repositions.
    sim._advance_sequential_task_queue(0, sim.agents[0], 1)
    state = sim._task_queue_states[0]
    assert state["phase"] == "reposition"
    assert state["next_task_index"] == 1
    assert sim.agents[0].goal_node == "c"
    assert sim.sequential_task_results[0]["task_id"] == "t1"

    # Complete the reposition leg. The queue should then start task 2.
    sim.agents[0].path.append("c")
    sim.agents[0].current_node = "c"
    sim.agents[0].reached_goal = True
    sim._advance_sequential_task_queue(0, sim.agents[0], 2)
    state = sim._task_queue_states[0]
    assert state["phase"] == "task"
    assert state["task_index"] == 1
    assert sim.agents[0].goal_node == "d"
    assert state["reposition_steps_before"] == 1
    assert state["reposition_distance_before_m"] == 10.0

    # Agent 1 independently completes its one-task queue.
    sim._advance_sequential_task_queue(1, sim.agents[1], 1)
    assert sim._task_queue_states[1]["complete"] is True
    assert sim.sequential_task_results[-1]["task_id"] == "t3"
    assert sim.sequential_task_results[-1]["agent_index"] == 1


def test_initial_deployment_reposition_is_recorded_as_queue_reposition():
    sim = DynamicSwarmSimulation.__new__(DynamicSwarmSimulation)
    sim.task_queues = [[{"task_id": "t1", "start": "b", "goal": "c"}]]
    sim._path_distance = lambda path: max(0, len(path) - 1) * 5.0
    sim.sequential_task_results = []
    sim._task_queue_states = {
        0: {
            "task_index": 0, "next_task_index": 0, "phase": "reposition",
            "path_start_index": 0, "phase_steps": 2,
            "reposition_path_start_index": 0,
            "reposition_steps_before": 0,
            "reposition_distance_before_m": 0.0, "complete": False,
        }
    }
    agent = FakeAgent(["a", "b"], "b", "b")
    sim.agents = [agent]

    sim._advance_sequential_task_queue(0, agent, 2)
    state = sim._task_queue_states[0]
    assert state["phase"] == "task"
    assert state["task_index"] == 0
    assert state["reposition_steps_before"] == 3
    assert state["reposition_distance_before_m"] == 5.0
    assert agent.goal_node == "c"
