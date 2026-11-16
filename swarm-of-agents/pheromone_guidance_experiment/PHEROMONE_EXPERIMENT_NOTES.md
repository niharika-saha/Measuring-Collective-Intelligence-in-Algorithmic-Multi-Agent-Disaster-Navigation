# Pheromone-guided navigation experiment

## What this experiment tests

This is a controlled route-choice experiment on the Tokyo OSM road graph. It tests whether a teammate's pheromone trail affects the follower's **executed next-hop decisions** under the default SNN-inspired potential-field policy.

It is deliberately separate from the earlier `collaboration` workload, which uses planner-assisted navigation.

## Experimental controls

- Two agents, one shared destination.
- The leader starts on one branch of a fork and deposits pheromone there.
- The follower starts at the fork and makes its own next-hop decision.
- `roles=None` is essential: it selects the homogeneous default agent specification (`use_planner=False`).
- Shared memory and local messaging are disabled in both arms.
- The only difference between paired arms is `pheromones=True` versus `pheromones=False`.
- Same scenario, seed, tasks, disaster initialization, and step limit within each pair.
- Five distinct Tokyo fork scenarios; five paired seeds per scenario by default (25 paired runs).

## How to run

Copy `test_pheromone_guidance.py` into the same folder as the project's `test_dynamic_disaster.py`, `swarm_simulation_dynamic.py`, `comms.py`, and graph dataset.

From PowerShell in that folder:

```powershell
python -m py_compile test_pheromone_guidance.py
python test_pheromone_guidance.py --runs 5 --seed 42 --steps 120 --output pheromone_guidance_results.json
```

The graph loader reuses `load_fresh_graph()` from `test_dynamic_disaster.py`, so it uses that file's existing `TOKYO_GRAPH`/dataset path handling.

## Important interpretation

The junctions are controlled diagnostic scenarios, not a random sample of all Tokyo intersections. They were selected to provide two feasible, nearly equivalent routes and a meaningful fork decision. Therefore:

- A changed first hop demonstrates that the pheromone signal affects navigation decisions.
- A shorter path in a particular scenario is evidence of a benefit in that scenario.
- A longer path in another scenario is evidence that trail following can reinforce a less efficient choice.
- This experiment does **not** by itself establish a city-wide evacuation benefit or improved disaster survival.

The correct paper claim must follow the aggregate and per-scenario results. Do not tune the pheromone weight or select only favourable runs after seeing the results.
