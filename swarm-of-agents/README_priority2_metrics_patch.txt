# Priority 2 metrics follow-up patch

This patch contains the two targeted changes discussed:

1. Correct completion makespan semantics in `swarm_simulation_dynamic.py`.
   - `makespan_steps` now means the maximum actual navigator goal-completion step.
   - It is `null` when no navigator completed.
   - Timeouts remain recorded in `finish_step_by_agent` / `finish_reason_by_agent`,
     but are right-censored and do not count as completed makespan.
   - Adds `completed_agent_makespan_steps`, `completed_finish_step_by_agent`,
     `completed_navigators`, `unfinished_navigators`,
     `unfinished_agent_indices`, and `observed_simulation_horizon_steps`.

2. Exports `execution_metrics` in each run result in `test_dynamic_disaster.py`.

3. Adds a dedicated `default_policy` workload:
   - All agents are navigators.
   - Planner assistance is explicitly disabled before the simulation starts.
   - The existing `SNNDecision` scorer and its no-pheromone counterfactual
     diagnostics can therefore be measured by `collective_metrics.py`.
   - This does not change the normal collaboration workload or force any
     outcome.

## Apply

Back up your current files, then replace them with the two files in this ZIP.

## Run the pheromone-policy diagnostic

Use the paired experiment so each ablation arm shares the same seed and task
manifest:

```powershell
python test_dynamic_disaster.py --mode paired --workload default_policy --agents 3 --steps 100 --runs 20 --seed 42 --output pheromone_policy_diagnostics.json
```

The existing paired communication configurations include a `no_pheromones`
arm. Compare the per-run `collective_metrics.pheromone` fields between `full`
and `no_pheromones`. The most informative fields are:

- `decisions_with_candidate_pheromone_diagnostics`
- `decisions_with_comparable_no_pheromone_counterfactual`
- `pheromone_changed_score_argmax`
- `mean_selected_candidate_probability_effect`
- `scored_choice_matches_executed_next_hop`
- `scored_choice_overridden_before_movement`

The counterfactual isolates the direct score/probability effect at a fixed
candidate set and temperature. It is not a replacement for the paired
pheromone-enabled versus pheromone-disabled performance comparison.

## Validation performed while preparing this patch

- Source syntax compilation: passed for the modified Python files.
- Synthetic pheromone-metric smoke test: passed.
- Full project pytest suite: not run in this environment. Run
  `python -m pytest -q` in your project after applying the patch.
