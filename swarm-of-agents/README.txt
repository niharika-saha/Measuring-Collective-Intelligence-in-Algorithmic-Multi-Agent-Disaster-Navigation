Priority 2 local-message instrumentation patch

Files:
- swarm_simulation_dynamic.py
- collective_metrics.py

Changes:
1. Records hazard-node information carried in delivered teammate messages (hazard/sync/advisory).
2. Records sender, receiver, message ID/kind, send and receive steps, route relevance, and whether the item was new to the recipient.
3. Avoids calling tracked agent lookup methods while checking novelty, so instrumentation does not create artificial shared-memory-read events.
4. Extends useful_information metrics to combine shared-memory hazard reads and local-message hazard information acquisitions.
5. Reports route relevance and novelty separately. These remain information-acquisition/relevance proxies, not proof of causal benefit.
6. Bumps collective metric schema to 1.3.

Apply by replacing the corresponding two files in swarm-of-agents. No changes to navigation decisions, communication policy, or agent behavior are intended.
