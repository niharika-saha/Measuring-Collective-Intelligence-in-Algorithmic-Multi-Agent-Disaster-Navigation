# Fixed-workload multi-agent scaling experiment

This patch adds a separate `scaling` mode to `test_dynamic_disaster.py` and
extends `DynamicSwarmSimulation` with independent per-agent task queues.

## What it compares

For each seed, the runner creates **one fixed manifest of 10 feasible tasks**.
The same task manifest and disaster seed are reused for team sizes 1, 2, 3, 5,
and 10 (or the values supplied with `--scaling-agents`).

- 1 agent receives all 10 tasks and completes them sequentially.
- Larger teams receive deterministic round-robin partitions of the same 10
  tasks; each agent completes its assigned queue sequentially.
- Every team size starts from the same deployment base (the first task's
  start node). Agents assigned elsewhere must travel to their first task's
  start. That initial deployment travel, as well as travel between later task
  endpoints and starts, is included in distance, elapsed time, and utilization.
- All agents are navigators. The configured default communication settings are
  retained consistently across team sizes.

The experiment does not fabricate task completions. If an agent dies or the
step budget expires, unfinished tasks are recorded as failed/not attempted.

## Run it

From the `swarm-of-agents` directory:

```powershell
python test_dynamic_disaster.py --mode scaling --type earthquake --runs 20 --seed 42 --scaling-steps 1500 --output scaling_earthquake_20runs.json
```

Other disaster types:

```powershell
python test_dynamic_disaster.py --mode scaling --type flood --runs 20 --seed 42 --scaling-steps 1500 --output scaling_flood_20runs.json
python test_dynamic_disaster.py --mode scaling --type fire --runs 20 --seed 42 --scaling-steps 1500 --output scaling_fire_20runs.json
python test_dynamic_disaster.py --mode scaling --type compound --runs 20 --seed 42 --scaling-steps 1500 --output scaling_compound_20runs.json
```

To change team sizes:

```powershell
python test_dynamic_disaster.py --mode scaling --scaling-agents 1 2 4 6 10 --runs 20 --seed 42 --scaling-steps 1500 --output scaling_custom.json
```

The serial baseline now receives an adaptive budget automatically. It first
runs with `--scaling-steps` (default 1500). If it does not complete all 10
tasks, the same seed and exact task queues are replayed from step 1 with a
larger budget, multiplied by `--scaling-step-growth-factor` (default 2.0),
until all tasks complete or `--scaling-max-steps` is reached (default 12000).
Parallel team sizes retain the original `--scaling-steps` budget. Each raw
serial record stores the actual budget and number of attempts. If the serial
baseline still cannot finish by the maximum, its makespan and speedups remain
`null`; it is not treated as a completed workload.

Example with a larger serial ceiling:

```powershell
python test_dynamic_disaster.py --mode scaling --type earthquake --runs 20 --seed 42 --scaling-steps 1500 --scaling-max-steps 16000 --scaling-step-growth-factor 2 --output scaling_earthquake_20runs.json
```

## Reported metrics

For each team size and seed, the JSON stores:

- tasks completed and completion rate;
- observed simulation steps, configured initial budget, actual budget used,
  serial retry count, and full-workload makespan (the latter is `null` unless
  all 10 tasks complete);
- throughput (completed tasks per observed simulation step);
- total distance travelled by all agents, including deployment and
  inter-task repositioning;
- agent utilization, calculated as recorded task plus reposition steps divided
  by `agent_count * observed_simulation_steps`;
- actual node/edge hazard encounters, encounters per kilometre and per 1,000
  agent-steps, hazard-sensing agent-steps, agents encountering hazards, and
  traversed route nodes affected by damage;
- parallel speedup relative to the single-agent full-workload makespan; and
- scaling efficiency (`speedup / number_of_agents`).

Speedup and scaling efficiency remain `null` when either the serial baseline or
the corresponding parallel run did not complete the entire fixed workload.
Summary statistics report mean, sample standard deviation, and valid `n`.
Raw per-seed task results and task assignments are retained for auditability.

## Validation performed for this patch

- `test_scaling_task_queues.py` covers independent queues, queue transitions,
  inter-task repositioning, and initial deployment repositioning.
- `test_scaling_budget_retry.py` covers deterministic serial retries, maximum-
  budget censoring, and the unchanged one-shot budget for parallel arms.
- Python syntax compilation passed for the modified runner and simulation file.
- The three adaptive-budget unit tests passed.
- Full simulation/queue integration tests were not run in this patch-validation
  environment because the optional `rtree` dependency is not installed. Run
  the project's full test suite in the environment where its dependencies are
  installed.

## Interpretation caution

This is a fixed-workload throughput/scaling experiment, not proof that swarm
coordination is causally beneficial. Because the same default communication
configuration is used at each team size, the results characterize scaling of
the configured multi-agent system. A separate communication-off ablation can
isolate the contribution of communication if needed.
