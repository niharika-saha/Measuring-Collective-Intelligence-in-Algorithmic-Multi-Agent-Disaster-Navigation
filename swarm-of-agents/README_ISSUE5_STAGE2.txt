ISSUE 5 — STAGE 2 PATCH
=======================

Updated files:
- comms.py
- swarm_simulation_dynamic.py
- test_comms_roles.py

Included unchanged current dependencies for a self-contained replacement bundle:
- agent_snn_v3.py
- test_dynamic_disaster.py

Changes:
1. Hazard awareness metrics now report mean delay across all awareness records,
   initially affected agent-event pairs, and agents outside the initially affected
   set. The legacy mean_awareness_delay_steps retains its prior affected-pair scope.
2. Added awareness source counts and record denominators to the metric output.
3. MessageBus.stats() exposes mean_scheduled_delay_steps in addition to measured
   mean_delay_steps.
4. Added tests for no-early-delivery at delays 0, 1, 2, and 3; inclusive radio
   radius boundary behavior; just-outside-radius exclusion; dead recipient
   exclusion; and awareness aggregation/source counts.

Validation performed in this environment:
- py_compile passed for all five included Python files.
- pytest -q test_comms_roles.py: 43 passed.
- The full test_dynamic_disaster.py suite was not run because this upload did not
  include hybrid_spatial_index.py (required during test collection). Run the full
  project tests from the repository where the project dependencies are present.

Recommended commands from your project experiment directory:
  python -m py_compile comms.py swarm_simulation_dynamic.py agent_snn_v3.py test_comms_roles.py test_dynamic_disaster.py
  pytest -q test_comms_roles.py test_dynamic_disaster.py

No communication routing algorithm or radius rule was changed. The radius tests
validate the existing inclusive rule: distance <= max(sender range, receiver range).
