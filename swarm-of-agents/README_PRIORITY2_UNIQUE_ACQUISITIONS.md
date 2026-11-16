# Priority 2 patch: unique information acquisitions

This patch updates the Priority 2 metrics implementation without changing agent policy or navigation behaviour.

## Files
- `swarm_simulation_dynamic.py`: aligns blocked-node novelty checks with the shared-memory setting.
- `collective_metrics.py`: schema 1.4. Keeps raw read counts, and adds deduplicated recipient-node and recipient-event acquisition metrics across local messages and shared-memory reads. Route relevance is taken from the first cross-agent acquisition.
- `test_unique_information_acquisitions.py`: regression tests for repeated messages, cross-channel duplicates, event overlap, self-observation novelty, and null denominators.

## Important interpretation
Raw reads are activity counts. Unique recipient-node acquisitions deduplicate repeated reads/messages across channels. Event-level counts are available only where the simulator has a node-to-event mapping. Route relevance is a snapshot at first arrival, not evidence of causal benefit. No new cross-agent-count scaling result is created by this patch; that still requires matched experiments at several agent counts.

## Apply
Back up the current `swarm-of-agents/swarm_simulation_dynamic.py` and `collective_metrics.py`, then replace them with the two files in this patch. Copy the test file into the same folder.

Run:
```powershell
python -m pytest test_unique_information_acquisitions.py -q
python -m pytest -q
```
