# Issue 5 --- Communication Layer: Findings and Remaining Work

**Status:** Partially validated\
**Experiment:** Paired communication-delay sweep\
**Evidence file:** `issue5_delay_sweep.json`\
**Prepared:** 12 November 2026

------------------------------------------------------------------------

## 1. Executive summary

The communication layer supports configurable delivery delays, and the
delay sweep confirms that the configured delays are reflected in
measured message-delivery latency.

The experiment compared delays of **0, 1, 2, and 3 simulation steps**
across 10 paired seeds, using two agents, an earthquake scenario, and
the `collaboration_local` workload. The same task manifest was reused
across delay conditions within each seed.

The clearest observed effect is on the **source of hazard awareness**:

-   At 0-step delay, both agents became aware in all 10 runs: one
    through self-observation and one through a teammate message.
-   At 1-step delay, teammate-message awareness occurred in 8 of 10
    runs; the remaining 2 records were self-observations.
-   At 2- and 3-step delays, all 20 awareness records per condition came
    from self-observation.

This suggests that increasing communication latency reduced the
contribution of teammate information in this particular scenario. It
does **not** establish that longer delays reduce evacuation success: all
four conditions achieved 100% goal success.

**Overall conclusion:** the delay mechanism is functioning and the
experiment provides evidence that latency changes information
propagation. Issue 5 is not yet fully validated because communication
radius, channel-specific effects, and scenarios in which an agent
genuinely depends on teammate information still need testing.

## 2. Experiment setup

  Parameter              Value
  ---------------------- -----------------------------------------------
  Experiment mode        Paired communication-delay sweep
  Disaster               Earthquake
  Workload               `collaboration_local`
  Agents                 2
  Runs / paired seeds    10
  Base seed              42
  Simulation steps       160
  Delay conditions       0, 1, 2, 3 steps
  Delayed channels       Local messaging and shared-memory board posts
  Shared memory          Enabled
  Local messaging        Enabled
  Pheromones             Enabled
  Communication radius   500 m
  Total simulations      40

The experiment held the communication channels and workload constant
while changing the configured delay. The same seed-specific task
manifest was used for all four delay conditions within each paired run.

**Important design limitation:** because local messaging and
shared-memory board posts were both assigned the same delay in this
sweep, the results do not isolate the independent contribution of each
channel.

## 3. Aggregate results

The following values are taken from `aggregate_by_delay` in the
experiment JSON. Values are means over 10 runs.

  -------------------------------------------------------------------------
    Configured Actual mean        Goal    Reported    Reported         Mean
         delay     message     success        mean        mean   successful
                     delay               awareness    response    responses
                                             delay       delay 
  ------------ ----------- ----------- ----------- ----------- ------------
       0 steps         0.0        100%         0.0         1.0          1.0

        1 step         1.0        100%         0.0         0.9          1.0

       2 steps         2.0        100%         0.0         0.1          1.0

       3 steps         3.0        100%         0.0         0.1          1.0
  -------------------------------------------------------------------------

### Message accounting

  ---------------------------------------------------------------------------
      Configured            Mean           Mean   Mean dropped   Mean pending
           delay   transmissions     deliveries                        at end
  -------------- --------------- -------------- -------------- --------------
         0 steps             6.6            6.6              0              0

          1 step             6.4            6.4              0              0

         2 steps             6.4            6.4              0              0

         3 steps             6.4            6.4              0              0
  ---------------------------------------------------------------------------

The measured mean message delay exactly matches the configured delay in
all four conditions. All reported transmissions were delivered; there
were no reported drops or pending deliveries at simulation end.

This is evidence that the delay configuration is being applied. It is
not, by itself, evidence that agents' decisions or evacuation outcomes
depend on communication.

## 4. Raw awareness-source analysis

The aggregate `mean_awareness_delay_steps` field remains zero for all
four conditions. However, the per-agent, per-event `awareness_records`
contain a more informative picture.

Across the 10 runs, there were 20 awareness records per delay condition
(two agents per run).

  -------------------------------------------------------------------------
   Configured delay   Self-observation   Teammate-message Recalculated mean
                               records            records   awareness delay
                                                                 across all
                                                                    records
  ----------------- ------------------ ------------------ -----------------
            0 steps                 10                 10         0.0 steps

             1 step                 12                  8         0.5 steps

            2 steps                 20                  0         0.9 steps

            3 steps                 20                  0         0.9 steps
  -------------------------------------------------------------------------

The recalculated mean is computed from each raw record's
`awareness_delay_steps`, rather than the aggregate summary field.

### Interpretation

-   **0-step delay:** teammate communication contributed to awareness in
    every run.
-   **1-step delay:** teammate-message awareness was still recorded in 8
    of 10 runs, but the source distribution shifted toward
    self-observation.
-   **2- and 3-step delay:** the receiving agent was not recorded as
    becoming aware through a teammate message; awareness records were
    all attributed to self-observation.

This is the strongest evidence in the current experiment that latency
changes information propagation.

It should be described narrowly: it shows a change in recorded awareness
source in this workload and scenario. It does not yet prove that delayed
communication caused a worse route, slower evacuation, or a failed
evacuation.

## 5. Response and evacuation interpretation

All four conditions achieved **10/10 successful runs**. The mean number
of successful responses was 1.0 in every condition.

The reported mean response delay was 1.0, 0.9, 0.1, and 0.1 steps for
delays 0, 1, 2, and 3 respectively. This is not a monotonic
relationship. It should not be interpreted as evidence that larger
communication delays improve response.

A plausible explanation is that agents can independently observe hazards
and replan, so they may not need to wait for teammate information in
this particular scenario. That explanation is consistent with the source
counts, but it needs a controlled communication-dependent scenario to be
tested.

### What the current results do and do not establish

**Supported by the experiment:**

-   Configured message delays of 0--3 steps are reflected in measured
    delivery delays.
-   Messages were delivered without reported drops or end-of-run pending
    deliveries in this sweep.
-   The recorded source of hazard awareness changes as communication
    delay increases.
-   All tested conditions achieved 100% goal success in this workload.

**Not established by this experiment:**

-   That communication delay worsens evacuation success.
-   That communication delay consistently increases response time.
-   That local messaging and shared-memory board posts have independent
    effects.
-   That the communication radius is enforced correctly at its boundary.
-   That communication improves overall performance compared with no
    communication.
-   That results generalize beyond this two-agent, one-workload,
    one-disaster setup.

## 6. Reporting issue to fix

The JSON aggregate field `mean_awareness_delay_steps` is zero for all
four conditions, while the raw awareness records show mean delays of
0.0, 0.5, 0.9, and 0.9 steps across all agents.

The current aggregate is therefore not suitable as the only
awareness-latency measure for comparing all agents. The raw records are
available and support the recalculation above.

### Required reporting correction

The evaluator should report separate metrics:

1.  `mean_awareness_delay_all_agents_steps` --- average across all
    agent-event awareness records.
2.  `mean_awareness_delay_initially_affected_steps` --- average only for
    agents initially affected by the event.
3.  `mean_awareness_delay_outside_initial_affected_set_steps` ---
    average for agents informed or detecting the event outside the
    initial affected set.
4.  Awareness counts by source: `self_observation`, `teammate_message`,
    and `shared_memory`.
5.  Counts and denominators for each metric, so `null` can be
    distinguished from zero.

Do not overwrite or discard the existing raw event records. Keep the
metric definitions explicit.

## 7. Remaining work to finish Issue 5

Issue 5 concerns a real communication layer: bounded/local exchange,
information propagation, communication costs, and separable
communication mechanisms. The delay sweep is one part of that
validation, not the entire issue.

### 5.1 Fix and expand communication metrics

-   [ ] Correct the all-agent awareness-delay aggregate.
-   [ ] Report awareness counts and delay separately by information
    source.
-   [ ] Report local-message and shared-memory-board transmissions,
    deliveries, bytes, and delays separately.
-   [ ] Preserve per-agent and per-event records in the JSON output.
-   [ ] Include metric denominators and define how `null`, zero, and
    missing observations differ.

**Acceptance check:** the aggregate values can be independently
recomputed from raw records, and channel/source totals reconcile with
the raw message and awareness logs.

### 5.2 Validate communication radius

The current experiment uses a 500 m radius, but this sweep does not
establish that the radius is correctly enforced.

-   [ ] Test two agents clearly inside the configured radius.
-   [ ] Test two agents clearly outside the configured radius.
-   [ ] Test an exact or near-boundary case, with a documented
    tolerance.
-   [ ] Verify that moving agents are evaluated using the intended
    positions at message-send time or delivery time; document which
    semantics the implementation uses.
-   [ ] Verify that an out-of-range agent does not receive a local
    message merely because another channel is enabled.

**Acceptance check:** only eligible recipients receive local messages,
with deterministic boundary behavior.

### 5.3 Isolate communication channels

The current delay sweep applies the same delay to local messaging and
shared-memory board posts. It cannot tell us which channel caused the
observed awareness changes.

Run controlled comparisons with the same seed and task manifest:

-   [ ] Local messaging enabled; shared-memory board disabled.
-   [ ] Shared-memory board enabled; local messaging disabled.
-   [ ] Both enabled.
-   [ ] Both disabled.
-   [ ] If useful, vary delay independently by channel rather than
    applying one shared delay to both.

Keep pheromones fixed during the first channel-isolation experiment. If
pheromone communication or retrieval is included in the communication
architecture, test it separately rather than conflating it with message
delivery.

**Acceptance check:** each channel's effect can be attributed to that
channel, and disabled channels do not leak information through another
path.

### 5.4 Add a communication-dependent hazard scenario

The current `collaboration_local` scenario allows agents to become aware
through self-observation. That makes it difficult to measure the full
consequence of delayed teammate information.

Create a controlled scenario in which:

-   [ ] Agent A observes a hazard.
-   [ ] Agent B is not initially within its own sensing/observation
    range of that hazard.
-   [ ] The hazard affects Agent B's planned route.
-   [ ] Agent A can communicate the hazard to Agent B through the
    configured communication channel.
-   [ ] Agent B's awareness and route response are measured at each
    delay.
-   [ ] A no-communication condition is included as a control.

Do not simply remove all sensing from Agent B unless that is an
explicitly named experimental condition. The goal is to model a
realistic information gap, not to force a desired result.

**Acceptance check:** the raw records show whether and when Agent B
learns about the hazard, whether it replans, whether its route changes,
and whether it reaches its goal.

### 5.5 Test queue and delivery semantics

-   [ ] Confirm that a 0-step message is delivered at the documented
    point in the simulation tick.
-   [ ] Confirm that 1-, 2-, and 3-step messages cannot be read before
    their scheduled delivery step.
-   [ ] Test multiple messages scheduled for the same recipient and
    delivery step.
-   [ ] Test ordering/FIFO behavior if ordering is part of the intended
    design.
-   [ ] Test messages sent immediately before simulation termination.
-   [ ] Confirm pending messages are reported accurately at termination.
-   [ ] Test disabled channels and invalid delay values.
-   [ ] Confirm that delivery and awareness are separate events:
    delivery alone must not count as awareness unless the agent actually
    reads/uses the information.

**Acceptance check:** automated tests prove no early delivery, correct
scheduled delivery, and correct accounting for pending/dropped messages.

### 5.6 Test unreliable communication (recommended extension)

This is a useful extension after deterministic delay and radius behavior
are correct.

-   [ ] Add optional message-loss probability or deterministic drop
    injection.
-   [ ] Track attempted, sent, delivered, dropped, and pending messages
    separately.
-   [ ] Use paired seeds and repeatable loss schedules.
-   [ ] Verify that a dropped message never creates an awareness record.
-   [ ] Compare no-loss and loss conditions.

This is recommended for realism, but should be treated as a separate
extension if unreliable delivery is not part of the current Issue 5
scope.

### 5.7 Run a broader paired evaluation

After the metric and scenario fixes:

-   [ ] Keep the same seeds, tasks, hazard events, and initial
    conditions across communication conditions.
-   [ ] Include 0--3 step delays.
-   [ ] Include channel ablations and a no-communication baseline.
-   [ ] Use enough seeds to characterize variability; report the sample
    size.
-   [ ] Report awareness, response, route changes, travel distance/time,
    goal success, and communication costs.
-   [ ] Report uncertainty or paired comparisons, not only means.
-   [ ] Preserve all raw JSON outputs and the exact experiment
    command/configuration.

**Acceptance check:** conclusions are based on paired comparisons and
distinguish mechanism validation from performance improvement.

## 8. Suggested order of work

To avoid changing too many things at once, use this order:

1.  **Metric correction:** fix all-agent awareness and source/channel
    breakdowns.
2.  **Queue tests:** establish exact delivery semantics for 0--3 steps.
3.  **Radius tests:** validate inside, outside, and boundary behavior.
4.  **Channel isolation:** separate local messaging from shared-memory
    board behavior.
5.  **Communication-dependent scenario:** test whether delayed teammate
    information changes a recipient's awareness and route response.
6.  **Paired evaluation:** repeat across seeds and report uncertainty.
7.  **Optional reliability extension:** introduce controlled message
    loss only after the deterministic layer is validated.

## 9. Current Issue 5 status

  ---------------------------------------------------------------------------
  Component                 Current status          Evidence / remaining work
  ------------------------- ----------------------- -------------------------
  Configurable 0--3 step    Implemented and         Measured mean delays
  delay                     exercised               match configured values

  Delivery accounting       Partially validated     No drops or pending
                                                    messages in this sweep;
                                                    edge cases still need
                                                    tests

  Hazard-awareness linkage  Partially validated     Raw event records show
                                                    source changes; aggregate
                                                    needs correction

  Response-event linkage    Partially validated     Responses are recorded,
                                                    but response delay is not
                                                    monotonic

  Local communication       Not fully validated     Need
  radius                                            inside/outside/boundary
                                                    tests

  Shared-memory vs          Not fully validated     Current sweep delays both
  local-message separation                          together

  Communication-dependent   Not yet demonstrated    Need recipient unable to
  scenario                                          independently observe the
                                                    hazard

  Unreliable delivery       Optional / not yet      Add after core
                            tested                  deterministic behavior is
                                                    stable

  General performance       Not demonstrated        All conditions had 100%
  benefit                                           success in this scenario
  ---------------------------------------------------------------------------

**Recommended status label:** *Issue 5 --- partially implemented;
configurable latency mechanism validated in one paired workload, with
radius, channel isolation, and communication-dependence validation
remaining.*

## 10. Research-paper-safe wording

> We implemented configurable communication latency for local messages
> and shared-memory board updates and evaluated delays of 0--3
> simulation steps in a paired 10-seed earthquake collaboration
> experiment. The measured mean delivery latency matched each configured
> delay. Across the 20 agent-event awareness records per condition,
> teammate-message awareness decreased from 10 records at zero delay to
> 8 at one-step delay and zero at two- and three-step delays, while
> self-observation records increased correspondingly. All conditions
> achieved 100% goal success, so this experiment demonstrates an effect
> on information propagation but does not establish an effect on
> evacuation success. Further validation is required to isolate
> communication channels, verify radius enforcement, and test scenarios
> where agents depend on teammate information.

------------------------------------------------------------------------

## Appendix: Reproducibility record

**Input result file:** `issue5_delay_sweep.json`

**Experiment command used:**

``` powershell
python test_dynamic_disaster.py --mode delay --workload collaboration_local --type earthquake --agents 2 --steps 160 --runs 10 --seed 42 --delays 0 1 2 3 --output issue5_delay_sweep.json
```

**Note:** This document records the current experiment and planned
validation work. Update the status, tables, and conclusions after each
remaining Issue 5 experiment. Do not replace the current findings with
stronger claims unless new results support them.

---

## 10. Implementation hardening update (13 November 2026)

The communication implementation has received a further validation/hardening
patch. This is a code-level update; it does not replace the historical result
files above or claim new Tokyo-graph experiment outcomes.

### Changes in the updated implementation

- `CommConfig` now rejects non-finite/non-positive radio radii, invalid delay
  values, and non-boolean channel switches.
- `MessageBus.send()` now snapshots payloads at send time, removes duplicate
  recipients while preserving order, and validates message kind and send step.
- `MessageBus.deliver()` validates its simulation step and separately records
  late deliveries (delivered later than their scheduled due step).
- Message accounting now includes pending deliveries by message kind and
  recipient, dropped deliveries by recipient, recipient delivery counts, and
  per-kind pending counts.
- Mean delivery latency is `null` when no delivery was observed, rather than
  reporting a misleading zero.
- Added `build_delay_matrix_configs()` and a new `channel_delays` experiment
  mode. This creates a Cartesian paired sweep where local-radio delay and
  shared-board delay vary independently, while all three channels remain on.
- Collaboration summaries now emit explicit manipulation checks: whether the
  hazard lies on the reference route, whether the leader self-observed it,
  whether a follower received explicit hazard information before self-detection,
  and whether disabled channels remained unused.

### Validation performed for this patch

- `python -m py_compile comms.py swarm_simulation_dynamic.py
  test_dynamic_disaster.py test_comms_roles.py`: passed.
- `pytest -q test_comms_roles.py`: **51 passed**.
- Focused `test_dynamic_disaster.py` checks for channel configuration, factorial
  coverage, collaboration manipulation checks, and pheromone/hazard metric
  separation: **5 passed** using a temporary import-only R-tree shim.
- The complete `test_dynamic_disaster.py` suite was not run against a real R-tree
  backend in this environment because the `rtree` dependency is unavailable
  and package installation cannot reach the network. The shim was not used to
  claim spatial-index or Tokyo-graph integration correctness.

### New independent-delay experiment

Run from `swarm-of-agents`:

```powershell
python test_dynamic_disaster.py --mode channel_delays --workload collaboration_local --type earthquake --agents 2 --steps 160 --runs 10 --seed 42 --local-delays 0 1 2 3 --board-delays 0 1 2 3 --output issue5_independent_channel_delay_matrix.json
```

This runs 16 delay combinations per seed (160 simulations for 10 seeds),
reusing the same task/event manifest across conditions within each seed. The
JSON stores per-condition aggregates, per-run raw records, and paired metric
differences against `local_0_board_0`. A smaller smoke run can use one seed
and `--local-delays 0 2 --board-delays 0 2`.

### Remaining validation before calling Issue 5 fully complete

1. Run the complete regression suite in the project's normal environment with
   `rtree` and the required scientific dependencies installed.
2. Run the new independent-delay matrix on the Tokyo graph and inspect its
   manipulation checks and paired differences.
3. Rerun the communication-dependent experiment and require its manipulation
   checks to pass in the messaging-enabled arm; report failed manipulation
   seeds rather than silently pooling them as successful communication trials.
4. Decide whether message-loss simulation is required for the workshop paper.
   It remains an optional extension and is not implemented by this patch.

**Status:** communication-layer code hardening and focused unit validation
completed; full environment integration and final paired experiment validation
remain pending.

## 10. Follow-up hardening pass (13 November 2026)

The follow-up implementation adds the following code-level checks and
reporting improvements:

- `MessageBus.send()` validates the sender, step, kind, delay, payload and
  recipient IDs even when the recipient list is empty. An empty valid recipient
  list remains a zero-cost no-op.
- Message-bus statistics now include delivery delay, late-delivery count,
  pending/dropped counts, byte totals and denominators by message kind and by
  transport channel (`local_messaging` versus `shared_memory_board`).
- Hazard-awareness summaries now include `awareness_by_source`, with a count,
  delay sum, delay denominator and mean delay for each observed source. This
  preserves the raw records and makes zero observations distinguishable from
  zero delay.
- The human-readable simulation summary prints `n/a` rather than raising an
  exception when no messages have been delivered and mean latency is null.

Validation in this environment:

- `python -m py_compile comms.py swarm_simulation_dynamic.py test_dynamic_disaster.py test_comms_roles.py`: passed.
- `pytest -q test_comms_roles.py`: **53 passed**.
- `test_dynamic_disaster.py`: **51 tests passed** when the five tests requiring
  native R-tree index construction were excluded. The full set cannot be
  completed here because the `rtree` Python dependency is not installed. An
  import-only shim was used solely to allow the non-R-tree tests to load; it
  was not used to claim spatial-index or Tokyo-graph integration validation.

Still outstanding before Issue 5 can be called fully validated:

1. Run the complete test suite with the real `rtree` dependency installed.
2. Run the paired communication-dependent and independent local/board delay
   matrix experiments on the Tokyo graph in the standard project environment.
3. Reconcile the new channel-level aggregates against raw message records from
   those fresh runs.
4. Treat probabilistic message loss as an optional extension, not as a
   completed feature.

The code changes validate instrumentation and deterministic communication
semantics; they do not establish a general evacuation-performance advantage
for communication.
