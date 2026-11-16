# Patch addendum — fresh per-step decision diagnostics

## Change
`agent_snn_v3.py` now clears `self.snn.last_decision` at the beginning of each
active `SNNAgent.step()`. Only a score-based call to `SNNDecision.decide()`
repopulates it during that tick.

This prevents the simulation observer from attaching the previous tick's
candidate scores/pheromone counterfactual to a current movement that bypasses
the scorer (for example a cycle escape, planner-following move, stationary
agent, or deterministic fallback). Those movements now correctly carry
`decision_diagnostics: null` instead of stale diagnostics.

## Regression test
`test_comms_roles.py` adds
`test_non_scored_cycle_escape_does_not_reuse_previous_snn_diagnostics`,
which seeds a prior scorer trace, forces a cycle-escape movement, and verifies
that the observer sees no current scorer trace.

## Validation in the patch environment
- `py_compile` for the modified agent and related simulation/metrics/test files: passed.
- `python test_comms_roles.py`: 53/53 passed.
- `python -m unittest -v test_collective_metrics.py`: 10/10 passed.
- `pytest -q test_dynamic_disaster.py`: collection blocked because `rtree` is
  not installed in this environment. This is an environment dependency issue;
  run the full suite on the Windows project environment with the repository's
  requirements installed.

This is an instrumentation correctness fix only. It does not change agent
movement, random-number consumption, communication, or experiment outcomes.
