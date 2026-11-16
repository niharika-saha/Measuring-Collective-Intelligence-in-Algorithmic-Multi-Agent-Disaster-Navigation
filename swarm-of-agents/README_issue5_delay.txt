Issue 5 communication-delay implementation
============================================

Files:
- comms.py: validated non-negative delay configuration; Message records its
  scheduled delay; bus stats expose mean scheduled delay and pending deliveries.
- test_dynamic_disaster.py: adds --mode delay, a paired 0/1/2/3-step sweep.

Existing files intentionally preserved:
- swarm_simulation_dynamic.py already passes CommConfig.delay_steps to local
  hazard/sync/advisory/assignment messages via MessageBus defaults, and passes
  board_delay_steps to shared-memory board posts. The comms phase delivers only
  messages whose due step has arrived.
- agent_snn_v3.py and test_comms_roles.py did not require changes.

The delay sweep applies the same selected delay to both local messages and
shared-memory board posts. It reuses the same generated tasks, event definition,
and seed across all delay arms in each run. Pheromones remain enabled and are
not delayed by this experiment.

Example:
python test_dynamic_disaster.py --mode delay --workload collaboration_local --type earthquake --agents 2 --steps 160 --runs 10 --seed 42 --delays 0 1 2 3 --output issue5_delay_sweep.json

The JSON includes per-run/per-delay raw results and aggregate metrics. Null
awareness/response delays mean no qualifying record occurred; they are not zero.

Validation performed here:
- Python compilation passed for the two modified files.
- Direct MessageBus checks passed for delays 0 and 2, no early delivery,
  pending queue accounting, and invalid delay rejection.
- Full pytest collection could not run in this isolated bundle because
  dynamic_disaster_engine.py, hybrid_spatial_index.py, and other project
  dependencies were not among the uploaded files. Run the full suite in the
  repository environment after copying these files into the experiment folder.
