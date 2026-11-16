# Controlled hazard / coordination diagnostic patch

Copy the four Python files in `swarm-of-agents/` over the files with the same
names in your repository.

## Important correction
The controlled collaboration exposure event was previously supplied only to
the serial individual arm. The independent and coordinated multi-agent arms
were created without that event. This patch passes the exact same
`exposure_event` to both parallel arms.

## Diagnostic correction
The runtime trace now chooses the agent actually adjacent to the event target
before falling back to the event's reference task ID. In the seed-42 scenario,
the reference route belongs to an upstream follower, but the leader is the
agent positioned one hop from the target.

## Files
- `collective_baselines.py`
- `swarm_simulation_dynamic.py`
- `agent_snn_v3.py` (occupied-node sensing fix retained)
- `test_dynamic_disaster.py` (regression test retained)

## Run from `swarm-of-agents`
```powershell
python -m pytest test_dynamic_disaster.py -k "agent_senses_damage_on_occupied_node"
python collective_baselines.py --type earthquake --workload collaboration --agents 10 --runs 1 --steps 200 --seed 42 --output collective_baselines_debug_fixed.json
```

The run should print an `event_trace` for both `independent_multi_agent` and
`coordinated_swarm`. Check that `applied=True`, the engine and agent share the
simulation graph, the target node has damage above 0.6, and the selected agent
is adjacent to the target. Then check whether `detected_blocked=True` or
`detected_danger=True`.

Do not launch a multi-seed experiment until this single-seed trace confirms
that the same controlled event was applied in all three arms. The patch does
not guarantee that swarm will outperform the independent baseline; it makes
the comparison valid and makes the causal pathway observable.
