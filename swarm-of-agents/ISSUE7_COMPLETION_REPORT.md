# Issue 7 — Collaboration workload and pheromone validation

## Status

**Core Issue 7 implementation and mechanism validation are complete in this
repository snapshot.** The original independent-task workload remains
available as a control; shared-task workloads are explicit alternatives.

This closes Issue 7 as an experimental-design/mechanism issue. It does **not**
establish that collaboration or pheromones improve overall evacuation
performance, and it does not close the separate Issue 2 adaptation-metrics
work.

## What changed in the simulation

- `test_dynamic_disaster.py` supports separate `independent`,
  `shared_shelter`, `collaboration`, and `collaboration_local` workloads.
- The collaboration workloads create feasible tasks with a common shelter and
  a shared route segment containing a controlled hazard event. The local
  variant constrains leader/follower separation to the configured radio range.
- Paired ablation runs generate the task/event manifest once and reuse it
  across communication configurations, so the arms differ by communication
  settings rather than task assignment.
- `swarm_simulation_dynamic.py` deposits pheromones online on newly occupied
  path nodes. It no longer waits for goal completion to replay the path.
  Deposits occur after all agents make their movement decisions for a step,
  avoiding sequential agent-order advantage.
- Pheromone publication provenance records the first contributor, step,
  path index and deposit strength.
- `comms.py` documents online pheromone deposition (corrected in this
  revision); `CommConfig` keeps shared memory, local messaging and pheromones
  independently switchable.

## Validation already recorded with this repository

The repository's `ISSUE7_REMAINING_NOTES.md` records:
- Python compilation passed.
- `pytest -q test_comms_roles.py test_dynamic_disaster.py`: 76 passed in the
  Windows environment with its native dependencies.
- Local-radio and shared-board-only paired tests demonstrated early teammate
  hazard warning/replanning before self-detection.
- The no-communication arm did not receive teammate information.
- Pheromone ablation removed trail publication/retrieval.
- A deterministic default-policy regression test demonstrated that a
  teammate pheromone can change the selected next hop.
- Online deposition was observed before goal completion.

The expanded pheromone study (`pheromone_guidance_experiment/
pheromone_expanded_30_scenarios.json`) contains 30 scenarios and 150 paired
trials. It demonstrates that pheromone information changes navigation
decisions, but it does not demonstrate a consistent route-efficiency or
success-rate improvement. Keep that negative/neutral performance result.

## Interpretation and limitations

1. **Mechanism validated:** agents can share hazard information through
   independent communication channels, and teammate pheromone can influence
   the default SNN-inspired policy's choice.
2. **Performance benefit not established:** the local and long-range
   collaboration experiments did not show a consistent evacuation-success or
   distance advantage. The expanded pheromone experiment also did not show a
   consistent route-distance benefit.
3. **Pheromone quality remains limited:** the current trail is a shared
   node-level field, not a goal-conditioned, directional route recommendation.
   A teammate's trail can therefore be irrelevant or lead to a longer route.
4. **Planner-assisted and default-policy evidence are distinct:** the
   collaboration hazard-propagation experiments use navigator/planner roles;
   the dedicated pheromone-guidance experiment isolates the default
   SNN-inspired policy with shared memory and local messaging disabled.
   Do not merge these into one performance claim.
5. **Issue 2 remains separate:** event-based hazard awareness, response and
   successful-replanning metrics still need their own implementation/audit.

## Reproduction commands (PowerShell)

Run from `swarm-of-agents`:

```powershell
python -m py_compile agent_snn_v3.py snn_core_v2.py comms.py roles.py swarm_simulation_dynamic.py test_dynamic_disaster.py
pytest -q test_comms_roles.py test_dynamic_disaster.py

# Long-range shared-board collaboration
python test_dynamic_disaster.py --mode paired --workload collaboration --type earthquake --agents 2 --steps 100 --runs 5 --seed 42 --output collaboration_paired_results.json

# Local-radio collaboration
python test_dynamic_disaster.py --mode paired --workload collaboration_local --type earthquake --agents 2 --steps 160 --runs 3 --seed 42 --output collaboration_local_paired_results.json

# Expanded default-policy pheromone experiment
python .\\pheromone_guidance_experiment\\test_pheromone_guidance.py --runs 5 --seed 42 --steps 120 --scenario-set expanded --generated-scenarios 25 --scenario-generation-seed 2026 --output .\\pheromone_guidance_experiment\\pheromone_expanded_30_scenarios.json
```

The repository snapshot's historical test record is 76 passing tests. Re-run
these commands on the target Windows machine before committing, because this
audit environment does not have the native `rtree` dependency installed.
