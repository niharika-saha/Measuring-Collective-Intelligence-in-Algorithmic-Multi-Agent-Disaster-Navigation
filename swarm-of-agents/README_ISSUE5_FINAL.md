# Issue 5 — Communication Layer: Validation Patch

This patch hardens the communication implementation, improves auditable
communication metrics, and adds an independent local-radio/shared-board delay
experiment. It preserves the existing Issue 7 and Issue 2 work.

## Code changes

- `comms.py`
  - Validates communication configuration and all send/delivery inputs,
    including invalid sends with an empty recipient list.
  - Snapshots message payloads at send time and de-duplicates recipients.
  - Tracks pending, dropped, late and delivered messages by recipient and
    message kind.
  - Reports transmission, delivery, byte and latency statistics separately
    for local messaging and shared-memory board traffic.
  - Keeps mean latency `null` when there are no successful deliveries.
  - Provides independent local-radio and board delay configurations.
- `swarm_simulation_dynamic.py`
  - Reports awareness count and delay statistics by awareness source, with
    explicit denominators.
  - Keeps all-agent, initially affected and outside-initially-affected delay
    populations separate.
  - Handles null communication latency safely in printed summaries.
- `test_dynamic_disaster.py`
  - Includes the independent local/board delay-matrix runner. It reuses each
    seed's task and hazard manifest across all delay combinations and records
    paired differences against the zero-delay baseline.
  - Includes communication-dependent manipulation checks.
- `test_comms_roles.py`
  - Covers configuration, send/delivery timing, recipient range and boundary,
    channel isolation, payload snapshots, message accounting, source-specific
    awareness metrics and communication-dependent behavior.

## Validation performed

From `swarm-of-agents`:

```powershell
python -m py_compile comms.py swarm_simulation_dynamic.py test_dynamic_disaster.py test_comms_roles.py
pytest -q test_comms_roles.py
```

The communication/role suite passes **53 tests** in the supplied environment.
The dynamic-disaster suite has **51 passing tests** when excluding five tests
that construct the native spatial index. Those five cannot be validated here
because the `rtree` dependency is absent. An import-only shim was used to load
the remaining tests; it does not emulate R-tree behavior and is not evidence of
Tokyo-graph spatial-index integration.

## Reproduction: independent channel delays

Smoke test:

```powershell
python test_dynamic_disaster.py --mode channel_delays --workload collaboration_local --type earthquake --agents 2 --steps 100 --runs 1 --seed 42 --local-delays 0 2 --board-delays 0 2 --output issue5_channel_delay_matrix_smoke.json
```

Paired evaluation:

```powershell
python test_dynamic_disaster.py --mode channel_delays --workload collaboration_local --type earthquake --agents 2 --steps 160 --runs 10 --seed 42 --local-delays 0 1 2 3 --board-delays 0 1 2 3 --output issue5_independent_channel_delay_matrix.json
```

This runs 16 local/board delay combinations per seed. Each seed's task and
hazard scenario are generated once and reused across conditions. Increase the
number of paired seeds (for example, 30 or 50) for the final paper analysis
once runtime and native dependencies are available.

## Interpretation rules

- Message delivery is not automatically an awareness event; the recipient
  must actually read/use the hazard information.
- A pheromone read is a navigation cue, not explicit hazard awareness.
- A safe greedy next-hop choice is not a planner replan.
- A failed manipulation check is an invalid test condition, not evidence for
  or against communication benefit.
- Equal success rates do not prove communication is useless; examine awareness,
  response, route changes, travel distance/time and communication overhead.
  Conversely, message counts alone do not prove a performance benefit.
- Probabilistic message loss remains an optional extension and is not
  implemented in this patch.

## Completion status

The deterministic communication layer, channel accounting, awareness-source
metrics and focused regression tests are implemented. Issue 5 should not be
marked fully validated until the full native-dependency test suite and fresh
paired Tokyo-graph experiments have been run. No general evacuation-success
advantage is claimed by this patch.
