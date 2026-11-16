Coordination Replan-Attribution Fix
=================================

Replace these files in your repository's swarm-of-agents folder:

1. swarm_simulation_dynamic.py
2. collective_baselines.py

Also copy test_coordination_replan_attribution.py into the same folder
(or into your tests folder, keeping the import path configured).

What this fixes
---------------
- Exposes planner-replan source attribution in each parallel arm's JSON.
- Attributes a successful, hazard-linked planner route change to the agent's
  FIRST recorded awareness source: self_observation, teammate_message,
  shared_memory, or unattributed.
- Counts only invalidated planner events with both plan_found=True and
  route_changed=True for source attribution.
- Keeps planner attribution separate from the first event-response record,
  so an earlier default-policy response cannot hide a later genuine planner
  route change.
- Prints per-run and aggregate counts, plus per-agent attribution.

Validation performed
--------------------
- Python syntax compilation: passed for both production files and test file.
- Focused regression tests: 4 passed.

The full existing test suite was not validated in this environment because
its test_comms_roles fixture module was not present, and the available R-tree
stub cannot support one spatial-index integration test. Run your complete
suite in the project environment before using the results in a paper.

After replacing the files, run:

    python -m py_compile swarm_simulation_dynamic.py collective_baselines.py
    python -m pytest -q test_coordination_replan_attribution.py

Then run a small pilot first:

    python collective_baselines.py --type earthquake --workload collaboration --agents 10 --runs 2 --steps 200 --seed 42 --output collective_baselines_instrumentation_test.json

If the pilot runs correctly, run the 30 paired seeds with a NEW output filename.
Do not overwrite the earlier JSON. This patch changes instrumentation only;
it does not change navigation, communication, task generation, or policies.
