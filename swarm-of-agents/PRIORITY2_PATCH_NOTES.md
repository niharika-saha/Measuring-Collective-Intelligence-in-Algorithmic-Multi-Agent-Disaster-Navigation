# Priority 2 instrumentation patch

Replace these files in your repository's `swarm-of-agents` directory:
- `collective_metrics.py`
- `test_collective_metrics.py`
- `test_dynamic_disaster.py`

## Changes
1. Integrates `collective_metrics_from_sim(sim)` into every `run_single()` result, so all communication arms export the same Priority 2 schema.
2. Adds paired-experiment aggregates for collective coverage, information/early-warning, communication efficiency, pheromone diagnostics, and explicitly scoped hazard responses.
3. Fixes pheromone instrumentation to read the simulator's actual candidate fields (`pheromone_feature`, `pheromone_score`, and `pheromone_probability_effect`) rather than the nonexistent `pheromone` candidate field.
4. Uses the simulator's existing no-pheromone counterfactual (`pheromone_changed_score_argmax`) and probability deltas. These are policy-level counterfactual diagnostics; the paired pheromone-off arm remains the experiment-level comparison.
5. Treats Jaccard overlap of two empty coverage sets as undefined (`null`), not perfect overlap (`1.0`).
6. Preserves existing result fields and simulation behavior; this is post-run instrumentation only.

## Validation performed
- `python -m unittest -v test_collective_metrics.py`: 10 tests passed.
- `py_compile` passed for all three changed Python files.
- Full `test_dynamic_disaster.py` pytest collection could not run in this environment because the `rtree` dependency is not installed and network access is unavailable. The repository's `requirements.txt` lists `rtree`.

## Important interpretation
`hazard_exposure.agents_encountered_hazard` counts physical node encounters/occupancy. Event-linked awareness and safe navigation responses can also be recorded for agents who learn about an event through teammate messages or shared memory, including agents outside the initially affected set. Therefore, zero physical encounters does not by itself contradict nonzero event-linked response records. The paired aggregate now exports affected-pair and outside-initial-affected-set response counts separately.
