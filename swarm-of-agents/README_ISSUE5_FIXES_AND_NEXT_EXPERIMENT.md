# Issue 5 — source-count fix and communication-dependent experiment

This patch is based on the latest files supplied for the project.

## Changes

1. **Awareness source-count fix** in `swarm_simulation_dynamic.py`:
   - Raw awareness records store the key `awareness_source`.
   - The aggregate previously read `source`, causing valid records to be counted as `unknown`.
   - The aggregate now reads `awareness_source`, with a legacy `source` fallback.
   - Existing test coverage in `test_comms_roles.py` asserts that `self_observation` and `teammate_message` are counted separately.

2. **New `communication_dependent` workload** in `test_dynamic_disaster.py`:
   - Reuses the controlled two-agent local-radio collaboration geometry.
   - Requires exactly two agents: one leader/observer and one follower.
   - Runs paired configurations with only local messaging changed:
     - `local_messaging_enabled`
     - `local_messaging_disabled`
   - Shared memory and pheromones are disabled in both arms, isolating local messaging.
   - Reuses the same tasks, event, and seed for both arms.
   - Adds focused tests for the two-arm configuration and the two-agent requirement.

## Important scientific qualification

The workload is designed to create a leader/follower hazard-information propagation opportunity: the event is placed on their shared reference route, the leader is positioned immediately before the event, and the follower starts farther behind. However, do not assume the manipulation succeeded merely because the run completed. Inspect the resulting raw `response_metrics.hazard_event_metrics.awareness_records` and `collaboration_metrics`:

- The leader should have a `self_observation` record for the event.
- In the messaging-enabled arm, the follower should receive explicit hazard information before its own observation (e.g. `teammate_message`).
- In the messaging-disabled arm, the follower should not receive explicit hazard information before self-observation.
- The route should be invalidated and a replacement route successfully planned if the event intersects the follower's active route.

If these conditions do not occur, report the experiment as a failed manipulation check, not as evidence for or against communication benefit. This is a controlled candidate scenario, not a guarantee that every seed will satisfy all checks.

## Commands (run from the repository root)

Compile:
```powershell
python -m py_compile swarm_simulation_dynamic.py test_dynamic_disaster.py comms.py agent_snn_v3.py dynamic_disaster_engine.py hybrid_spatial_index.py
```

Regression tests:
```powershell
pytest -q test_comms_roles.py test_dynamic_disaster.py
```

Smoke test (one paired seed = two simulations):
```powershell
python test_dynamic_disaster.py --mode communication_dependent --agents 2 --steps 160 --runs 1 --seed 42 --output issue5_communication_dependent_smoke.json
```

Main experiment (recommend 20–30 paired seeds if runtime permits):
```powershell
python test_dynamic_disaster.py --mode communication_dependent --agents 2 --steps 160 --runs 20 --seed 42 --output issue5_communication_dependent.json
```

## Expected outputs

The JSON contains paired run records, shared task manifests, event definitions, per-configuration communication metrics, hazard awareness/response records, and aggregate comparisons. Preserve the JSON under a new filename; do not overwrite the original factorial result.

## Dependencies

The ZIP intentionally includes the six requested/current project modules plus the updated experiment runner. It does not replace unrelated project modules such as `roles.py`, graph data/cache files, or Python dependencies. Run tests in the full repository environment where those dependencies are installed.
