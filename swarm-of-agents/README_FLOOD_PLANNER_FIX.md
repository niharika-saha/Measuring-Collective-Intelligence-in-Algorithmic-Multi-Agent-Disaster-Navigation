# Flood planner performance fix

## Files

- `agent_snn_v3.py` — replacement for the existing agent implementation.
- `test_planner_retry_guard.py` — focused regression tests for planner retry behavior.

## What this changes

1. Prevents repeated expensive failed A* searches from running twice in the same agent step.
2. Adds a 10-agent-step retry interval after a failed bounded search.
3. Allows an immediate retry when locally known hazard/blockage information changes or the goal changes.
4. Caches heuristic distances within each A* search.
5. Avoids repeated helper-method dispatch for local/shared blocked and danger lookups inside A*.
6. Maintains an index for the active plan so finding the next node does not repeatedly scan the route.
7. Records planner attempts, successes, failures, expanded nodes, retry skips and planning time.

## What this does NOT change

- The 40,000-node A* expansion limit.
- Hazard generation, flood severity, or hazard propagation.
- The SNN scoring policy or movement fallback when no planner route is available.
- Simulation step budgets, task manifest, random seeds, or scaling metrics.
- Existing decision-observer callback behavior.

A failed planner search still falls back to the existing local SNN policy. It is not treated as a successful route.

## Installation

1. Make a backup of your current `swarm-of-agents/agent_snn_v3.py`.
2. Extract this ZIP.
3. Replace only `swarm-of-agents/agent_snn_v3.py` with the included file.
4. Do not replace `snn_core_v2.py` or any other project file.

## Tests

The five focused tests cover:
- successful A* planning and plan-index lookup;
- preventing duplicate searches in the same step and throttling repeated failures;
- immediate retry after new local knowledge;
- immediate retry after a goal change;
- no repeated retry when the goal is known blocked.

These tests passed against the uploaded `agent_snn_v3.py` snapshot (5 passed).

## After installation

Rerun a short flood scaling smoke test first, for example:

```powershell
python test_dynamic_disaster.py --mode scaling --type flood --runs 1 --seed 42 --scaling-steps 1500 --scaling-max-steps 12000 --output scaling_flood_smoke.json
```

Check the resulting JSON and console output before starting the full 20-run experiment.

Note: the retry interval limits repeated failed searches; it does not guarantee that flood planning will be fast. Use the new `plan_attempts`, `plan_expansions_total`, `plan_retry_skips`, and `plan_time_seconds_total` statistics if your runner serializes agent stats. If it does not, a follow-up instrumentation change to the runner may be needed to expose them in the output.
