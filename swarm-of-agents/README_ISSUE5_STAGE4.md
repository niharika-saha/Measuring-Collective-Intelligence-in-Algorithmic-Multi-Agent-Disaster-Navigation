# Issue 5 — Stage 4: Full Communication-Channel Factorial Ablation

## What this stage adds

This patch adds a new experiment mode, `factorial`, that evaluates all eight
combinations of the three independent information channels:

- Shared memory: ON / OFF
- Local messaging: ON / OFF
- Pheromones: ON / OFF

This is a complete 2x2x2 factorial design. It is stronger than a four-arm
comparison because it allows the effects of each channel and their interactions
to be examined without leaving one channel permanently enabled.

## Files

Replace these files in your experiment directory:

- `comms.py`
- `swarm_simulation_dynamic.py`
- `agent_snn_v3.py`
- `test_comms_roles.py`
- `test_dynamic_disaster.py`

Only `test_dynamic_disaster.py` has functional changes in this stage. The other
four files are included unchanged so the ZIP is a consistent set based on the
latest uploaded versions.

## Collaboration metric clarification

The previous `retrieved_before_self_detection` field measures **explicit hazard
information** received through shared-memory hazard-board reads or local
teammate hazard messages. It does not count pheromone reads.

New fields make this distinction explicit:

- `explicit_hazard_info_before_self_detection`
- `pheromone_retrieved_before_self_detection`
- `first_pheromone_retrieval_step`
- `followers_received_explicit_hazard_info_before_self_detection`
- `followers_retrieved_pheromone_before_self_detection`

A pheromone trail is a navigation signal, not necessarily a hazard report.
Therefore, a pheromone read must not be counted as hazard awareness. This
prevents the collaboration summary from conflating two different mechanisms.

## Install

Extract the ZIP and overwrite the five corresponding files in your experiment
folder. Keep the files in the same directory structure used by your repository.

## Compile and regression tests

Run from the directory containing these files:

```powershell
python -m py_compile comms.py swarm_simulation_dynamic.py agent_snn_v3.py test_comms_roles.py test_dynamic_disaster.py
pytest -q test_comms_roles.py test_dynamic_disaster.py
```

## Smoke test

One paired seed, eight configurations:

```powershell
python test_dynamic_disaster.py --mode factorial --workload collaboration_local --type earthquake --agents 2 --steps 160 --runs 1 --seed 42 --output issue5_factorial_smoke.json
```

## Main experiment

Recommended initial run: 10 paired seeds, 80 simulations total:

```powershell
python test_dynamic_disaster.py --mode factorial --workload collaboration_local --type earthquake --agents 2 --steps 160 --runs 10 --seed 42 --output issue5_full_channel_factorial.json
```

The experiment reuses the same seed-specific task manifest and disaster
parameters across all eight configurations within each paired run.

## Eight configurations

| Configuration | Shared memory | Local messaging | Pheromones |
|---|---:|---:|---:|
| `all_on` | ON | ON | ON |
| `no_local_messaging` | ON | OFF | ON |
| `no_shared_memory` | OFF | ON | ON |
| `no_pheromones` | ON | ON | OFF |
| `shared_memory_only` | ON | OFF | OFF |
| `local_messaging_only` | OFF | ON | OFF |
| `pheromones_only` | OFF | OFF | ON |
| `all_off` | OFF | OFF | OFF |

## Interpretation guidance

- Compare `all_on` with `all_off` for the combined effect of all channels.
- Compare channel-on and channel-off conditions across matched settings of the
  other two channels to estimate each channel's contribution.
- Examine interactions: a channel may only help when another channel is active.
- Do not infer a performance benefit from message counts or retrieval counts
  alone. Examine route changes, response outcomes, distance/time and goal success.
- Do not claim a channel improves performance if paired results do not support it.
- Ten seeds are an initial evaluation, not definitive statistical evidence.
  Preserve raw runs and report paired differences and uncertainty.

## Validation performed while preparing this patch

- All five Python files passed `py_compile`.
- The eight configurations were checked to cover all eight Boolean combinations.
- A targeted metric-contract check confirmed that a pheromone-only read does
  not increment explicit hazard-information retrieval, while the separate
  pheromone-before-self-detection metric does increment.
- The full pytest suite could not be executed in this isolated workspace because
  the uploaded files did not include repository dependencies
  `dynamic_disaster_engine.py` and `hybrid_spatial_index.py`. Run the commands
  above in the full repository.

## Issue 5 status after this stage

This stage adds a complete channel-factorial experiment and clearer collaboration
instrumentation. It does not by itself finish Issue 5. The communication-dependent
hazard scenario, broader evaluation, and any desired reliability/drop tests
remain separate validation work.
