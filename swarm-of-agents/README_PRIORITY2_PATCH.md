# Priority 2 metrics correction patch

## Files
- `collective_metrics.py`: consolidated from the newer uploaded schema (1.1) and corrected to schema 1.2.
- `swarm_simulation_dynamic.py`: adds per-agent/per-step local maximum node-damage severity history.
- `test_dynamic_disaster.py`: included unchanged; it already imports `collective_metrics_from_sim` from `collective_metrics.py`.
- `test_priority2_metrics.py`: focused regression tests for the corrected metric semantics.

## Install
Back up the three existing repository files first. Then replace:
1. `collective_metrics.py`
2. `swarm_simulation_dynamic.py`

Keep your existing `test_dynamic_disaster.py` (the copy here is included only to document/check the integration point). Add `test_priority2_metrics.py` to your test directory.

The test harness already imports the canonical module name, so no import edit is required.

## Metric semantics
- Travel time: completed navigators' actual agent decision-step counts; global completion timestamps are separate.
- Makespan: team completion makespan is null unless every navigator reaches its goal. Successful-agent makespan and observed horizon are reported separately.
- Distance: computed from the simulator's path-distance function when no explicit per-agent distance counter exists. All-navigator and completed-only totals are separate.
- Median: conventional median, including averaging the two middle values for even-sized samples.
- Coverage: retains backward-compatible all-agent top-level keys and adds explicit all-agent and navigator-only populations.
- Severity-weighted exposure: sum of maximum node-damage severity in each agent's current-node/one-hop neighbourhood at each observed step. This is a proximity proxy, not injury, physical contact, or calibrated dose. Disabled-edge exposure remains separately reported.

## Validation performed here
- Python syntax compilation: passed for all modified Python files.
- Focused regression tests: 5 passed.
- The full project test suite was not run in this environment. Run it in the repository after installation.
