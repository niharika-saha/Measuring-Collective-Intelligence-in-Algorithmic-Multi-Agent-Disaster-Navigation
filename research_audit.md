# WMAC 2027 Research Results and Implementation Audit

Audit date: 2026-11-16. This audit is read-only with respect to the research implementation and existing experiment outputs. Only this report and derived result tables were created. No experiment was rerun for this audit.

## Executive verdict

The defensible canonical three-arm baseline files are [collective_baselines_earthquake_schema15_30runs.json](swarm-of-agents/collective_baselines_earthquake_schema15_30runs.json) and [collective_baselines_flood_schema15_30runs.json](swarm-of-agents/collective_baselines_flood_schema15_30runs.json). Each contains 30 paired seeds, 10 tasks, and the three expected arms. The embedded collective-intelligence records report schema `1.5`; the top-level files do not themselves expose a `schema_version` field.

The 25-run pheromone files are not incomplete fragments of the 150-run study. They are the legacy design: five hand-selected scenarios times five seeds. The complete expected design was executed and is in [pheromone_expanded_30_scenarios.json](swarm-of-agents/pheromone_guidance_experiment/pheromone_expanded_30_scenarios.json): 30 scenarios, five seeds each, 150 unique paired scenario-seed records, and zero rejected pairs.

The full factorial file is structurally complete: 10 seeds, all eight channel combinations per seed, no missing or duplicated combinations, and paired task/scenario generation. It contains descriptive per-configuration aggregates, but no formal factorial main effects, interaction estimates, confidence intervals, or p-values. Treat its conclusions as exploratory/descriptive.

## 1. Repository and canonical-file inventory

| Study | Canonical/current input | Runs/seeds | Schema/status | Caveat |
|---|---|---:|---|---|
| Three-arm earthquake baseline | `swarm-of-agents/collective_baselines_earthquake_schema15_30runs.json` | 30; seeds 42 through 126 by the runner's 42-step seed policy | Embedded CI schema 1.5; validated by `collective_baselines_audit_report.json` only for the older similarly named file and by direct structure checks here | Top-level schema field absent |
| Three-arm flood baseline | `swarm-of-agents/collective_baselines_flood_schema15_30runs.json` | 30 | Embedded CI schema 1.5 | Same top-level schema omission |
| Earthquake scaling | `swarm-of-agents/scaling_earthquake_20runs_v2.json` | 20 seeds, agent counts 1/2/3/5/10 | Scaling schema 1.1 | Separate fixed-workload study; not a coordination causal comparison |
| Full communication factorial | `swarm-of-agents/issue5_full_channel_factorial.json` | 10 seeds, 8 configurations each | No top-level schema field; runner emits current collective metrics inside results | Exploratory sample; no formal factorial effects stored |
| Communication delay sweep | `swarm-of-agents/issue5_delay_sweep.json` | 10 paired seeds, delays 0/1/2/3 | No top-level schema field | Local and board delays are varied together |
| Channel isolation | `swarm-of-agents/issue5_channel_isolation.json` | 10 paired seeds, 4 configurations | No top-level schema field | Pheromones are held on |
| Communication-dependent messaging | `swarm-of-agents/issue5_communication_dependent.json` | 20 paired seeds, 2 configurations | No top-level schema field | Two-agent local-messaging study |
| Pheromone legacy guidance | `swarm-of-agents/pheromone_guidance_experiment/pheromone_guidance_results.json` and diagnostics variants | 25; 5 scenarios x 5 seeds | Study-specific; no top-level schema field | Legacy controlled fork study |
| Pheromone expanded guidance | `swarm-of-agents/pheromone_guidance_experiment/pheromone_expanded_30_scenarios.json` | 150; 30 x 5 | Study-specific; no top-level schema field | Complete expected design |
| Pheromone weight sweep | `swarm-of-agents/pheromone_weight_0.json`, `_2`, `_4`, `_8`, `_16` | 20 per weight | Study-specific; no top-level schema field | Separate from guidance experiment |

Main runners and analysis sources are [collective_baselines.py](swarm-of-agents/collective_baselines.py), [audit_collective_baselines.py](swarm-of-agents/audit_collective_baselines.py), [test_dynamic_disaster.py](swarm-of-agents/test_dynamic_disaster.py), [collective_metrics.py](swarm-of-agents/collective_metrics.py), and [test_pheromone_guidance.py](swarm-of-agents/pheromone_guidance_experiment/test_pheromone_guidance.py). Notebooks are dataset construction/indexing notebooks under `dataset/`; no notebook was identified as the canonical statistical analysis for the listed paper results.

Older or preliminary files include `collective_baselines_30runs.json`, `collective_baselines_earthquake_30.json`, the flood pilot, smoke/diagnostic JSON files, and `scaling_flood_diagnostic*.json`. They should not be pooled with the canonical files.

## 2. Main baseline results

The raw run-level descriptive table is [baseline_descriptive_results.csv](baseline_descriptive_results.csv). Values are newly calculated from raw records; distance is `mean_distance_per_task_m`, effort is `agent_steps_total`, makespan is `makespan_steps`, repositioning uses the explicit reposition fields, and utilisation is `agent_step_budget_utilisation`. Standard deviations are sample SD across 30 runs. All three arms completed all 10 tasks in every run: 300 successful tasks and zero failed tasks per scenario.

### Earthquake

| Arm | Mean distance/task m | SD | Median | Mean total distance m | Mean agent steps | Mean makespan steps | Mean reposition m | Mean utilisation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Individual serial | 3004.223867 | 917.144332 | 3208.109850 | 61294.162822 | 871.566667 | 871.566667 | 27350.834333 | 0.435783 |
| Independent multi-agent | 3082.363060 | 926.803725 | 3246.533850 | 30823.630600 | 476.600000 | 52.266667 | 0 | 0.238300 |
| Coordinated swarm | 3016.142413 | 921.912685 | 3216.837050 | 30161.424133 | 458.766667 | 51.033333 | 0 | 0.229383 |

### Flood

| Arm | Mean distance/task m | SD | Median | Mean total distance m | Mean agent steps | Mean makespan steps | Mean reposition m | Mean utilisation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Individual serial | 3038.514403 | 1128.579594 | 3364.561800 | 64315.918595 | 923.433333 | 923.433333 | 27657.542500 | 0.461717 |
| Independent multi-agent | 3086.155753 | 1145.599864 | 3438.584400 | 30861.557533 | 492.733333 | 53.866667 | 0 | 0.246367 |
| Coordinated swarm | 3038.779817 | 1132.493094 | 3368.401450 | 30387.798167 | 483.100000 | 53.333333 | 0 | 0.241550 |

Success rate is not discriminative here because it is 1.0 in every arm and every run. Serial makespan equals serial agent-step effort in these records because one navigator performs the queue; this equality must not be generalized to multi-agent arms.

## 3. Paired coordinated-versus-independent results

The primary comparison is coordinated minus independent on paired run-level observations. The machine-readable table is [paired_coordinated_vs_independent.csv](paired_coordinated_vs_independent.csv). The confidence intervals are two-sided 95% Student-t intervals on paired differences. Wilcoxon values are newly calculated from raw differences; Holm correction is across the four listed metrics within each scenario. Effect-size values were not stored by the repository's existing analysis, so no unverified effect-size estimate is claimed here.

| Scenario | Metric | Mean difference | 95% CI | Wilcoxon p | Holm p | Positive/negative/zero |
|---|---|---:|---|---:|---:|---:|
| Earthquake | Distance/task m | -66.220647 | [-94.058404,-38.382890] | 0.0000182153 | 0.0000728612 | 0/24/6 |
| Earthquake | Agent steps | -17.833333 | [-29.905261,-5.761406] | 0.0018627850 | 0.0055913551 | 3/18/9 |
| Earthquake | Makespan | -1.233333 | [-2.849659,0.382993] | 0.0596565953 | 0.1193131906 | 5/16/9 |
| Flood | Distance/task m | -47.375937 | [-77.134504,-17.617369] | 0.0002598153 | 0.0007794459 | 1/21/8 |
| Flood | Agent steps | -9.633333 | [-20.363531,1.096864] | 0.0629116834 | 0.1258233668 | 8/13/9 |
| Flood | Makespan | -0.533333 | [-1.844326,0.777660] | 0.2712835138 | 0.2712835138 | 9/12/9 |

Success-rate Wilcoxon testing is undefined because all 30 paired differences are zero. The strongest supported baseline claim is a route-distance reduction for coordination relative to independent multi-agent execution in both scenarios, plus an earthquake effort reduction. Makespan and flood effort are not statistically supported after the stated paired tests.

## 4. Communication and shared-information results

The communication runner is `run_paired_communication_experiment()` in [test_dynamic_disaster.py](swarm-of-agents/test_dynamic_disaster.py). It generates one task/event scenario per seed and passes the same values to each configuration. `CommConfig` in [comms.py](swarm-of-agents/comms.py) independently controls shared memory, local messaging, pheromones, local delay, board delay, radio radius, and contact synchronisation.

Verified study designs:

- `issue5_full_channel_factorial.json`: 10 seeds, seeds 42, 84, ..., 420; 2 agents; 2 tasks; 160 steps; `collaboration_local`; all eight combinations of messaging, shared memory, and pheromones. Every seed has all eight configurations, no duplicates, and no missing boolean triple. Tasks and collaboration/event scenario are generated once per seed and reused. The file's `comparison_design.full_2x2x2_factorial` is true.
- `issue5_delay_sweep.json`: 10 seeds, delays 0/1/2/3, all three channels enabled; `delay_applied_to` is both `local_messaging` and `shared_memory_board`. It does not isolate a single channel's delay.
- `issue5_channel_isolation.json`: 10 seeds, four configurations, pheromones held on.
- `issue5_communication_dependent.json`: 20 seeds, two configurations, local messaging enabled/disabled with shared memory and pheromones disabled.

The full factorial output contains per-configuration descriptive aggregates only. It does not contain formal factorial main effects or interaction terms. Stored mean total distance is 6094.039 m for all-on, no-local-messaging, no-shared-memory, no-pheromones, shared-memory-only, and local-messaging-only; 6106.801 m for pheromones-only and all-off. All configurations have mean success rate 1.0. This supports only an exploratory statement that the controlled workload did not separate success and showed little aggregate distance variation, not a claim that channel interactions were established.

The implementation records transmissions, deliveries, dropped messages, fixed-model bytes, scheduled/delivery delay, hazard-awareness records, shared-memory reads, local-message reads, pheromone retrievals, planner events, and source-attribution records. The existing outputs do not provide a defensible causal numerator/denominator for generic “communication efficiency”; use the explicitly defined delivery/transmission and bytes/delivery measures only.

## 5. Pheromone guidance

### Discrepancy resolution

The runner is [test_pheromone_guidance.py](swarm-of-agents/pheromone_guidance_experiment/test_pheromone_guidance.py). Its loop is:

```text
scenarios = five legacy scenarios
for scenario in scenarios:
    for run_idx in range(runs):
        seed = base_seed + run_idx * 42
        run_arm(ON)
        run_arm(OFF)
```

With the default legacy invocation `--runs 5`, the result is 5 scenarios x 5 seeds = 25 paired records. Thus 25 is the number of completed scenario-seed pairs, not the number of seeds alone and not an aggregate-record count. Each record contains both `pheromones_on` and `pheromones_off`, so ON/OFF is paired for every completed record.

The runner's `LEGACY_SCENARIOS` list has exactly five scenarios. The files `pheromone_guidance_results.json`, `pheromone_guidance_diagnostics.json`, `pheromone_guidance_diagnostics_v2.json`, and `pheromone_legacy_deduplicated.json` are 25-record legacy outputs. Additional observations do exist: `pheromone_expanded_30_scenarios.json` has 30 observed scenarios, 5 seeds per scenario, 150 records, 0 rejected pairs, and 150 unique scenario-seed pairs. Therefore the expected 30 x 5 design was executed; the missing 125 observations do not need to be inferred from the 25-run files because they are present in the expanded file.

The expanded file reports 141/150 ON successes and 140/150 OFF successes, mean follower distance 2716.2 m ON versus 2699.983 m OFF, difference +16.217 m ON minus OFF, 60 changed paths, and mean trail probability effect 0.1472. This does not support a route-quality improvement claim; it does support a decision-influence claim, subject to the controlled-fork limitation.

Recommendation: retain the expanded file as the complete guidance study and cite the legacy files only as preliminary/legacy controlled subsets. A rerun is optional for provenance, not required to fill a missing design. If rerun for a clean artifact, use:

```powershell
Set-Location swarm-of-agents
python pheromone_guidance_experiment/test_pheromone_guidance.py --scenario-set expanded --generated-scenarios 25 --runs 5 --seed 42 --steps 120 --output pheromone_expanded_30_scenarios_rerun.json
```

The existing runner already supports the design; no implementation change is required. The experiment intentionally uses default SNN-inspired potential-field navigation with planner assistance, shared memory, and local messaging disabled.

## 6. Pheromone-weight sweep

The files `pheromone_weight_0.json`, `_2`, `_4`, `_8`, and `_16` are separate from the guidance study. Each declares 20 runs, 3 agents, 100 steps, `default_policy`. They should not be merged with the 150 guidance pairs. The current audit did not find a top-level schema/version field in these files. Exact probability-effect and argmax definitions are implemented in the decision diagnostics in [snn_core_v2.py](swarm-of-agents/snn_core_v2.py) and the aggregation paths in [collective_metrics.py](swarm-of-agents/collective_metrics.py); the files should be read as decision-level sensitivity evidence, not end-to-end route-quality evidence.

## 7. Team-size scaling

The canonical earthquake scaling file is [scaling_earthquake_20runs_v2.json](swarm-of-agents/scaling_earthquake_20runs_v2.json). It declares 20 seeds, team sizes 1, 2, 3, 5, and 10, fixed 10-task manifests, deterministic round-robin queues, shared communication configuration, and schema 1.1. The serial arm can replay the same seed and queues with geometrically increasing budgets up to the configured maximum; parallel arms retain the initial budget. Speedup is calculated per seed as serial workload makespan divided by parallel workload makespan, only when both complete all ten tasks. Efficiency is speedup divided by team size. These results demonstrate fixed-workload throughput scaling, not an isolated coordination benefit, because team size, assignment, budget, and parallelism all change.

## 8. Collective-intelligence metric availability

| Metric | Directly recorded | Derivable | Source/definition | Limitation |
|---|---|---|---|---|
| Message transmissions/deliveries/dropped/bytes | Yes | Yes | `communication_metrics`, `MessageBus`, `payload_bytes()` | Bytes are a fixed model, not wire capture |
| Communication latency | Yes for scheduled delivery delay | Yes | message `sent_step`, `retrieval_step`, delay fields | Availability and causal use differ |
| Hazard-awareness delay/early warning | Yes | Yes | `information_metrics()`; first-awareness vs self-discovery | Censored when no self-discovery |
| Shared-memory reads | Yes | Yes | `_shared_memory_reads`, deduplicated and raw forms | Reads show access, not causal use |
| Shared-memory writes | Publication attempts/deliveries recorded | Partly | board publication instrumentation | No generic byte-level write causal metric |
| Pheromone deposits/retrievals | Retrievals and provenance recorded | Partly | pheromone instrumentation | Exact contribution to outcome is not established |
| Pheromone candidate-score/probability effect | Yes where diagnostics exist | Yes | `pheromone_metrics`, decision diagnostics | Decision-level, not route improvement |
| Planner replans/route changes | Yes | Yes | `plan_events`, response metrics | Attribution is conditional on awareness records |
| Route changes by teammate/shared/self source | Yes | Yes | planner source attribution | Attribution is an instrumentation association, not proof of benefit |
| Unique coverage/redundant visits/overlap | Yes | Yes | `coverage_metrics()` | Population must be stated: all agents or navigators |
| Spatial-cell coverage | Yes | Yes | `cell_counts`, coverage metrics | Cell size is implementation-defined (`CELL_DEG=0.002`) |
| Agent utilisation/agent-step effort | Yes | Yes | `agent_step_budget_utilisation`, `agent_steps_total` | Utilisation denominator is budget-specific |
| Communication efficiency | No single causal metric | Partly | delivery/transmission and bytes/delivery are available | Do not claim a generic efficiency result without predefining numerator/denominator |

## 9. Section III implementation fact-check

### Tokyo graph

`dataset/tokyo_graph.json` is JSON with 77,575 nodes and 239,678 stored edges. Nodes contain `id`, latitude/longitude, type, and optional population. The graph is represented by adjacency lists and disaster state is mutated in place (`_node_damage`, `_disabled_edges`). Edge geometry uses stored lengths; fallback geographic distance is haversine. `HybridTokyoGraph` builds a KD-tree for road-node nearest-neighbour operations and an R-tree for edge-radius candidate queries. Facility connector edges are added by snapping facilities to nearby road edges and connecting to endpoints. Exact indexed counts are runtime/index-build properties, not all present as metadata in the JSON file.

### Disaster engine

[dynamic_disaster_engine.py](swarm-of-agents/dynamic_disaster_engine.py) supports earthquake, flood, fire, and compound. Earthquake aftershocks can occur during steps 1–5 with probability 0.55; severity is 0.30–0.65 times main severity and radius 0.40–0.75 times main radius. Rubble propagation starts at step 6 for 30 steps, adds 0.10 damage subject to 0.95 decay, and uses 0.6 as the blocked line. Flood grows 150 m per step, increases severity by 0.03 per step to 0.90, and is capped at twice the initial radius. Fire uses up to 3 neighbour jumps per step, base probability 0.45, station suppression radius 800 m, and radius cap 1.5 times the initial radius. Compound flood begins at step 15 and fire at step 30. The simulation calls `engine.tick(t)` before agents move. Agents do not receive ground truth as automatic knowledge; they sense, communicate, or read the board.

### Agents and policy

The main agent class is `SNNAgent` in [agent_snn_v3.py](swarm-of-agents/agent_snn_v3.py). The decision class is `SNNDecision` in [snn_core_v2.py](swarm-of-agents/snn_core_v2.py). It is not a conventional spiking neural network implementation; it is an SNN-inspired potential-field feature scorer. Features include goal distance, progress, hazard, visited penalty, backtrack, exploration, momentum, taboo/blacklist, two-hop goal, lookahead/dead-end risk, planner next-hop, and pheromone. Selection uses temperature-dependent softmax/greedy logic and exploration probability.

The agent also has taboo memory, permanent blacklist, cycle detection (`deque(maxlen=30)`, cycle length 10), oscillation limits, and progress/stall tracking. Navigator roles use known-hazard A* with `PLAN_MAX_EXPANSIONS=40000`; the current source also has failure backoff and stable-plan validation caching. Baseline homogeneous agents use `roles=None`, while the collective baseline harness explicitly uses navigator roles with planner enabled. Specialised scout, assessor, and allocator roles exist in [roles.py](swarm-of-agents/roles.py), but are not the three-arm homogeneous baseline.

### Coordination

`CommConfig` defaults to shared memory, local messaging, and pheromones enabled, 500 m base radius, zero local and board delay, and contact synchronisation enabled. Message delivery uses distance <= max(sender range, receiver range), where role multipliers can enlarge range. A broadcast counts as one transmission and one delivery per receiver; payload bytes use the fixed constants in [comms.py](swarm-of-agents/comms.py). Shared-board posts are delayed through the bus when configured. Pheromones are deposited by agents, evaporated by the simulation, and read through tracked views. The baseline and mechanism runners explicitly set channel configurations; do not infer channel status from filenames alone.

## 10. Discussion and limitations

### Supported claims

- Coordination reduced mean route distance relative to independent multi-agent execution in both canonical earthquake and flood baselines, with paired Wilcoxon evidence and Holm-adjusted p-values below 0.001.
- Earthquake coordination reduced total agent-step effort in the paired comparison; flood effort was directionally lower but not statistically supported at the stated sample and test.
- Success rate alone is uninformative for these workloads because all three arms completed all tasks in all 30 runs.
- The expanded pheromone study demonstrates measurable decision influence and path changes in controlled forks, not a general route-quality improvement.
- The communication factorial is a paired, complete eight-condition design, but its sample is exploratory.

### Claims not supported

Do not claim universal swarm superiority, universal channel benefit, pheromone route-quality improvement, significant makespan improvement, a conventional SNN, an LLM-based system, or a formally established three-way interaction from the 10-seed factorial.

### Main limitations

Serial execution includes inter-task repositioning and is not the primary causal comparison. Makespan, agent-step effort, and distance measure different properties and should not be collapsed. The controlled collaboration workload and fork scenarios are deliberately structured rather than representative random Tokyo samples. Communication-delay results vary local and board delay together. Scaling changes team size and parallelism by design and therefore cannot be interpreted as a pure coordination effect. Older schemas and smoke outputs remain in the repository and must be excluded from pooled analysis. Reproducibility is limited unless the paper records the code revision/hash, dependency versions, graph/index metadata, and exact command lines.

## 11. Recommended paper tables and figures

1. Table 1: three-arm earthquake/flood descriptive results. Use `baseline_descriptive_results.csv`; show success, distance/task, effort, makespan, repositioning, and utilisation with n=30.
2. Table 2: paired coordinated-minus-independent effects. Use `paired_coordinated_vs_independent.csv`; show difference, CI, Wilcoxon p, Holm p, and sign counts.
3. Figure 1: paired route-distance and effort differences by seed. Source raw baseline run records; x-axis scenario/seed, y-axis coordinated minus independent, separate panels for distance and agent steps.
4. Figure 2: team-size speedup from `scaling_earthquake_20runs_v2.json`; x-axis team size, y-axis per-run speedup with CI, and a separate efficiency line. State serial retry policy.
5. Figure 3: Tokyo graph with disaster overlays. Source `dataset/tokyo_graph.json`, disaster logs, and existing visualization scripts; label units and distinguish damaged nodes from disabled edges.
6. Table 3: mechanism study summary. Use factorial per-configuration aggregates, delay outputs, and expanded pheromone aggregate. Keep the 10-seed factorial and 150-pair pheromone study separate.

## 12. Unresolved issues and missing data

- Top-level schema/version fields are absent from most mechanism outputs; schema must be inferred from embedded fields or runner version and explicitly documented.
- Existing factorial output has no formal main-effect or interaction analysis. Any such values would be newly calculated, not retrieved.
- Pheromone-weight aggregate details require a separate raw-field extraction before publication; do not borrow them from the guidance study.
- The repository contains current source changes from prior work; this audit describes the current files, not a historical commit. Record a commit hash before final paper claims.
- Generic communication efficiency is not a defensible existing empirical metric without an explicit numerator and denominator.
- The canonical baseline audit report shown in the repository is for `collective_baselines_30runs.json`; rerun the audit script against the schema15 earthquake and flood files before publication if its assumptions are compatible.

## Reproduction commands, not executed in this audit

```powershell
# Complete pheromone guidance design
python pheromone_guidance_experiment/test_pheromone_guidance.py --scenario-set expanded --generated-scenarios 25 --runs 5 --seed 42 --steps 120 --output pheromone_expanded_30_scenarios_rerun.json

# Full 2^3 communication factorial
python test_dynamic_disaster.py --mode factorial --type earthquake --workload collaboration_local --agents 2 --steps 160 --runs 10 --seed 42 --output issue5_full_channel_factorial_rerun.json
```
