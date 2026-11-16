# Issue 5 — Stage 3: Communication Channel Isolation

## What this patch adds

This patch adds a dedicated `--mode channels` experiment to
`test_dynamic_disaster.py`.

It compares a controlled 2x2 ablation:

| Arm | Local messaging | Shared memory | Pheromones |
|---|---:|---:|---:|
| `both_channels` | ON | ON | ON |
| `local_only` | ON | OFF | ON |
| `shared_memory_only` | OFF | ON | ON |
| `neither_message_channel` | OFF | OFF | ON |

Pheromones are held ON in all four arms. This isolates the two message
channels without confounding the comparison with pheromone enablement.

For each seed, the paired runner generates one task manifest and reuses
the same seed, task manifest, disaster parameters, and collaboration scenario
across all four arms.

The existing `--mode paired` and its legacy configurations remain available.
This patch does not replace or rename them.

## Install

Replace the five corresponding files in your experiment folder with the
files in this ZIP. Four files are included unchanged from the latest supplied
source; `test_dynamic_disaster.py` contains the Stage 3 experiment changes.

## Run

Recommended first run:

```powershell
python test_dynamic_disaster.py --mode channels --workload collaboration_local --type earthquake --agents 2 --steps 160 --runs 10 --seed 42 --output issue5_channel_isolation.json
```

This runs 10 paired seeds × 4 channel conditions = 40 simulations.

For a quick smoke test:

```powershell
python test_dynamic_disaster.py --mode channels --workload collaboration_local --type earthquake --agents 2 --steps 80 --runs 1 --seed 42 --output issue5_channel_smoke.json
```

## Metrics

The paired aggregate now includes the existing outcome metrics plus:

- all-agent mean awareness delay
- all-agent mean response delay
- successful and failed hazard responses
- agents receiving teammate information
- shared-memory cross-agent retrievals
- route-relevant shared-memory retrievals
- cross-agent pheromone retrievals and route-relevant pheromone retrievals

Raw per-run records, task manifests, communication configs, and instrumentation
are retained in the JSON.

## Validation

Run:

```powershell
python -m py_compile comms.py swarm_simulation_dynamic.py agent_snn_v3.py test_comms_roles.py test_dynamic_disaster.py
pytest -q test_comms_roles.py test_dynamic_disaster.py
```

A configuration unit test verifies that all four arms vary only local messaging
and shared memory while keeping pheromones enabled.

## Interpretation cautions

- This is a mechanism-isolation experiment, not an assumption that full
  communication must perform best.
- If all four arms have identical goal success, report that honestly.
- Distinguish message delivery from actual hazard awareness and successful
  route response.
- The `neither_message_channel` arm still has pheromones enabled. It is not a
  total-communication-off baseline. It is specifically the no-local-message
  and no-shared-memory condition while pheromone behavior is held constant.
- The separate legacy `none` arm in `--mode paired` disables pheromones too;
  do not mix it into the clean 2x2 channel comparison.
