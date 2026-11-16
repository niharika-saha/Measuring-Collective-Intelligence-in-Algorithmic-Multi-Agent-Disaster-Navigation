# Issue 2 — Stage A: Remove invalid measurements

## Files
- `swarm_simulation_dynamic.py` — modified
- `test_dynamic_disaster.py` — modified
- `agent_snn_v3.py` — included unchanged; its real `plan_events` history is preserved
- `dynamic_disaster_engine.py` — included unchanged

## Changes
1. Removed `_invalidate_blocked_paths()` and its invocation from the simulation loop.
2. Removed the synthetic reroute counters and the stall-counter-based `adaptation_lag`.
3. Kept planner-generated `agent.plan_events`; response metrics now count actual `reason == "invalidated"` events and retain the raw event history per agent.
4. Updated summary and experiment aggregation output to report genuine planner invalidations and successful replans instead of fabricated greedy-policy reroutes.
5. Added regression tests verifying that disaster ground truth does not call an automatic path invalidation routine, the old stall counters are absent, and actual planner event history remains intact.

## Important scope note
This is Stage A only. It does not yet implement the Stage B event registry, per-agent hazard awareness timestamps, response timestamps, or event-to-agent attribution. The default SNN-inspired policy remains greedy one-hop navigation and therefore has no persistent planned route to invalidate.

## Validation
Run from `swarm-of-agents`:
```powershell
python -m py_compile agent_snn_v3.py swarm_simulation_dynamic.py dynamic_disaster_engine.py test_dynamic_disaster.py
pytest -q test_comms_roles.py test_dynamic_disaster.py
```
