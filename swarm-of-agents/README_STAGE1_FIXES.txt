# Stage 1 metric fixes

These files are patched from the exact repository snapshot supplied in the conversation.
Replace the matching files in your `swarm-of-agents` directory.

## Files to copy
- `collective_metrics.py`
- `swarm_simulation_dynamic.py`
- `test_collective_metrics.py`

## Changes

1. **Message novelty is recipient-specific.**
   A hazard existing on the global shared board no longer counts as already known by every agent. A blocked node is considered previously known only if it is in that recipient's local memory or the recipient has a recorded earlier successful board read. For danger values, prior observed board values are recorded and compared with the incoming message.

2. **Distance totals require complete populations.**
   The all-navigator total is `null` if any navigator lacks a valid distance. The completed-only total is independently reported only if every completed navigator has a valid distance. Per-agent distances remain available, and missing agent indices are listed.

3. **Hazard severity mean names/denominators are explicit.**
   `mean_per_observed_agent_step` includes zero-severity observations. `mean_per_positive_severity_agent_step` excludes them. The generic `agent_step_samples` field is retained as a backward-compatible alias; the misleading `mean_per_exposed_agent_step` field is removed.

4. **Metric schema version** is incremented from 1.4 to 1.5.

## Validation performed
- `python -m py_compile collective_metrics.py swarm_simulation_dynamic.py test_collective_metrics.py` — passed.
- `python -m pytest -q test_collective_metrics.py test_priority2_metrics.py` — 21 passed.
- `test_collective_baselines.py` could not be collected in this environment because the optional `rtree` dependency is missing (`ModuleNotFoundError: No module named 'rtree'`). Run that test in the user's normal project environment.

After copying the files, run in PowerShell from `swarm-of-agents`:

```powershell
python -m py_compile collective_metrics.py swarm_simulation_dynamic.py
python -m pytest -q test_collective_metrics.py test_priority2_metrics.py
python -m pytest -q test_collective_baselines.py
```

If those pass, rerun only the one-run Stage 1 smoke test with a new output filename:

```powershell
python collective_baselines.py --type earthquake --workload collaboration --agents 10 --steps 200 --runs 1 --seed 42 --output collective_metrics_smoke_stage1_fixed.json
```

Do not rerun the larger experiment batches yet.
