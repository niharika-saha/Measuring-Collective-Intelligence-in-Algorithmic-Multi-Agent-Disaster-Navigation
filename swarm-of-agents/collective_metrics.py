
"""Priority 2 collective-intelligence metrics.

Post-run instrumentation only: does not alter agent behaviour, random streams,
communication, task manifests, or experiment budgets.

Usage:
    from collective_metrics import collective_metrics_from_sim
    result = collective_metrics_from_sim(sim)
"""

from collections import Counter
from itertools import combinations
import math
import statistics


def _mean(values):
    values = [
        float(v) for v in values
        if v is not None and math.isfinite(float(v))
    ]
    return sum(values) / len(values) if values else None



def aggregate_numeric_metrics(records, precision=6):
    """Aggregate numeric scalar leaves across per-run metric dictionaries.

    Returns dotted metric paths with mean, population standard deviation,
    valid sample count, total run count, and missing sample count. Booleans,
    strings, arrays, and non-finite numbers are deliberately excluded.
    Missing/undefined values remain missing; they are never coerced to zero.
    """
    records = list(records or ())
    flattened = []

    def walk(value, prefix=""):
        found = {}
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                found.update(walk(child, path))
        elif isinstance(value, bool):
            return found
        elif isinstance(value, (int, float)):
            numeric = float(value)
            if math.isfinite(numeric):
                found[prefix] = numeric
        return found

    for record in records:
        flattened.append(walk(record or {}))

    all_paths = sorted(set().union(*(row.keys() for row in flattened))) if flattened else []
    result = {}
    total_runs = len(records)

    for path in all_paths:
        values = [row[path] for row in flattened if path in row]
        count = len(values)
        mean = sum(values) / count if count else None
        variance = (
            sum((value - mean) ** 2 for value in values) / count
            if count else None
        )
        result[path] = {
            "mean": round(mean, precision) if mean is not None else None,
            "std": round(math.sqrt(variance), precision) if variance is not None else None,
            "n": count,
            "total_runs": total_runs,
            "missing_n": total_runs - count,
        }

    return result

def _jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else None


def coverage_metrics(agents, indices=None):
    """Coverage from exact visited node-ID sets, with an explicit population."""
    if indices is None:
        indices = list(range(len(agents)))
    else:
        indices = list(indices)
    selected = [agents[i] for i in indices]

    node_sets = [set(getattr(a, "visited", set()) or ()) for a in selected]
    cell_sets = [set(getattr(a, "cell_counts", {}) or {}) for a in selected]

    all_nodes = set().union(*node_sets) if node_sets else set()
    all_cells = set().union(*cell_sets) if cell_sets else set()
    node_pairs, cell_pairs = [], []

    for local_i, local_j in combinations(range(len(selected)), 2):
        node_pairs.append({
            "agent_i": indices[local_i],
            "agent_j": indices[local_j],
            "shared_nodes": len(node_sets[local_i] & node_sets[local_j]),
            "jaccard": _jaccard(node_sets[local_i], node_sets[local_j]),
        })
        cell_pairs.append({
            "agent_i": indices[local_i],
            "agent_j": indices[local_j],
            "shared_cells": len(cell_sets[local_i] & cell_sets[local_j]),
            "jaccard": _jaccard(cell_sets[local_i], cell_sets[local_j]),
        })

    memberships = sum(map(len, node_sets))
    redundant = max(0, memberships - len(all_nodes))
    per_agent = []
    for local_i, nodes in enumerate(node_sets):
        other_nodes = set().union(*[
            node_sets[j] for j in range(len(node_sets)) if j != local_i
        ]) if len(node_sets) > 1 else set()
        other_cells = set().union(*[
            cell_sets[j] for j in range(len(cell_sets)) if j != local_i
        ]) if len(cell_sets) > 1 else set()
        exclusive_nodes = nodes - other_nodes
        exclusive_cells = cell_sets[local_i] - other_cells
        per_agent.append({
            "agent": indices[local_i],
            "visited_nodes": len(nodes),
            "unique_nodes": len(exclusive_nodes),
            "unique_node_fraction": (
                len(exclusive_nodes) / len(nodes) if nodes else None
            ),
            "unique_cells": len(exclusive_cells),
        })

    return {
        "population": "selected_agent_indices",
        "agent_indices": indices,
        "agent_count": len(selected),
        "unique_nodes_covered": len(all_nodes),
        "unique_cells_covered": len(all_cells),
        "total_agent_node_memberships": memberships,
        "redundant_agent_node_memberships": redundant,
        "redundant_membership_fraction": (
            redundant / memberships if memberships else None
        ),
        "mean_pairwise_node_jaccard": _mean(
            [r["jaccard"] for r in node_pairs]
        ),
        "mean_pairwise_cell_jaccard": _mean(
            [r["jaccard"] for r in cell_pairs]
        ),
        "pairwise_node_overlap": node_pairs,
        "pairwise_cell_overlap": cell_pairs,
        "per_agent_unique_coverage": per_agent,
    }


def _self_discovery_steps(sim):
    """Earliest self-observation for each (agent,event)."""
    result = {}

    for row in getattr(sim, "_hazard_discoveries", ()) or ():
        if row.get("source") != "self_observation":
            continue

        agent = row.get("agent")
        step = row.get("step")

        if step is None:
            continue

        for event_id in row.get("event_ids", ()) or ():
            key = (agent, event_id)
            result[key] = min(
                int(step), result.get(key, int(step))
            )

    return result


def information_metrics(sim):
    """First awareness, censored early-warning comparisons, and use proxies."""
    events = getattr(sim, "_hazard_event_records", {}) or {}
    self_steps = _self_discovery_steps(sim)

    source_counts = Counter()
    awareness_records = []
    lead_samples = []
    censored = 0

    for event_id, event in events.items():
        for row in (event.get("awareness", {}) or {}).values():
            agent = row.get("agent")
            source = row.get("awareness_source", "unknown")
            step = row.get("awareness_step")

            source_counts[source] += 1

            record = {
                "event_id": event_id,
                "agent": agent,
                "first_awareness_source": source,
                "first_awareness_step": step,
            }

            if source in {"teammate_message", "shared_memory"}:
                self_step = self_steps.get((agent, event_id))

                if self_step is None:
                    record["self_discovery_step"] = None
                    record["early_warning_lead_steps"] = None
                    record["comparison"] = "censored_no_self_discovery"
                    censored += 1
                else:
                    lead = int(self_step) - int(step)
                    record["self_discovery_step"] = self_step
                    record["early_warning_lead_steps"] = lead
                    record["comparison"] = (
                        "teammate_earlier" if lead > 0
                        else "same_step" if lead == 0
                        else "self_earlier"
                    )
                    lead_samples.append(lead)

            awareness_records.append(record)

    reads = list(getattr(sim, "_shared_memory_reads", ()) or ())

    hazard_reads = [
        r for r in reads
        if r.get("cross_agent")
        and r.get("channel") in {"blocked_nodes", "danger_nodes"}
    ]

    route_relevant_reads = [
        r for r in hazard_reads if r.get("route_relevant")
    ]

    pheromone_reads = [
        r for r in reads
        if r.get("cross_agent") and r.get("channel") == "pheromones"
    ]

    decisions = list(getattr(sim, "_navigation_decisions", ()) or ())

    informed_decisions = [
        d for d in decisions
        if set(d.get("preceding_sources", ()) or ())
        & {"teammate_message", "shared_memory"}
    ]

    changed_after_information = [
        d for d in informed_decisions
        if d.get("changed_from_previous_decision_at_same_node") is True
    ]

    response = getattr(sim, "_response", {}) or {}
    planner_records = list(
        response.get("planner_replan_source_records", ()) or ()
    )
    planner_by_source = Counter(
        r.get("awareness_source", "unattributed")
        for r in planner_records
    )

    return {
        "registered_hazard_events": len(events),
        "agent_event_first_awareness_pairs": len(awareness_records),
        "first_awareness_source_counts": dict(source_counts),
        "first_awareness_records": awareness_records,

        "early_warning": {
            "comparisons_with_self_discovery": len(lead_samples),
            "censored_no_self_discovery": censored,
            "mean_lead_steps_positive_means_teammate_was_earlier": (
                _mean(lead_samples)
            ),
            "lead_step_samples": lead_samples,
        },

        "information_retrieval": {
            "cross_agent_hazard_reads": len(hazard_reads),
            "route_relevant_cross_agent_hazard_reads": len(
                route_relevant_reads
            ),
            "cross_agent_pheromone_reads": len(pheromone_reads),
            "all_shared_memory_reads": len(reads),
        },

        "navigation_association_not_causation": {
            "decisions_with_preceding_teammate_or_board_information": len(
                informed_decisions
            ),
            "same_node_next_hop_changes_after_information": len(
                changed_after_information
            ),
            "planner_route_changes_by_first_awareness_source": dict(
                planner_by_source
            ),
            "planner_route_changes_after_teammate_or_board_awareness": (
                planner_by_source.get("teammate_message", 0)
                + planner_by_source.get("shared_memory", 0)
            ),
            "interpretation": (
                "Default-policy next-hop changes after information are "
                "temporal associations, not causal proof. Planner route "
                "changes retain the existing known-blocked-route trigger."
            ),
        },
    }


def communication_efficiency_metrics(sim):
    bus = getattr(sim, "bus", None)
    stats = bus.stats() if bus is not None else {}

    response = getattr(sim, "_response", {}) or {}
    source_counts = response.get(
        "planner_replan_source_attribution", {}
    ) or {}

    useful_changes = (
        int(source_counts.get("teammate_message", 0))
        + int(source_counts.get("shared_memory", 0))
    )

    transmissions = int(stats.get("transmissions", 0) or 0)
    deliveries = int(stats.get("deliveries", 0) or 0)
    bytes_tx = int(stats.get("bytes_tx", 0) or 0)

    return {
        "transmissions": transmissions,
        "deliveries": deliveries,
        "dropped_deliveries": int(stats.get("dropped", 0) or 0),
        "bytes_transmitted": bytes_tx,
        "mean_delivery_delay_steps": stats.get("mean_delay_steps"),
        "late_deliveries": stats.get("late_deliveries"),
        "pending_deliveries_at_end": stats.get("pending_deliveries"),

        "deliveries_per_transmission": (
            deliveries / transmissions if transmissions else None
        ),
        "bytes_per_delivery": (
            bytes_tx / deliveries if deliveries else None
        ),

        "planner_route_changes_attributed_to_teammate_or_board": (
            useful_changes
        ),
        "deliveries_per_attributed_planner_route_change": (
            deliveries / useful_changes if useful_changes else None
        ),
        "bytes_per_attributed_planner_route_change": (
            bytes_tx / useful_changes if useful_changes else None
        ),

        "by_channel": stats.get("by_channel", {}),
        "zero_denominator_policy": (
            "Undefined ratios are null, never zero or infinity."
        ),
    }


def pheromone_metrics(sim):
    """Measure pheromone exposure and its direct score/probability effect."""
    reads = [
        r for r in (getattr(sim, "_shared_memory_reads", ()) or ())
        if r.get("channel") == "pheromones"
    ]

    decisions = getattr(sim, "_navigation_decisions", ()) or ()
    decision_count = 0
    selected_positive = 0
    selected_max = 0
    argmax_changed = 0
    selected_probability_effects = []
    candidate_probability_effects = []
    selected_pheromone_scores = []
    maximum_pheromone_scores = []
    counterfactual_comparable = 0

    for decision in decisions:
        if decision.get("policy_type") != "default_snn":
            continue

        diagnostics = decision.get("decision_diagnostics") or {}
        candidates = diagnostics.get("candidates") or []
        parsed = []

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            try:
                feature = float(candidate.get(
                    "pheromone_feature",
                    candidate.get("pheromone", 0.0)
                ) or 0.0)
                score = float(candidate.get(
                    "pheromone_score", 0.0
                ) or 0.0)
            except (TypeError, ValueError):
                feature, score = 0.0, 0.0

            parsed.append({
                "node": candidate.get("node"),
                "feature": feature,
                "score": score,
                "probability_effect": candidate.get(
                    "pheromone_probability_effect"
                ),
            })

            effect = candidate.get("pheromone_probability_effect")
            if effect is not None:
                try:
                    candidate_probability_effects.append(float(effect))
                except (TypeError, ValueError):
                    pass

        if not parsed:
            continue

        decision_count += 1
        # The scorer's selected node is the correct candidate for measuring
        # the SNN policy's pheromone effect. The executed next hop can differ
        # when a later safety/stall override replaces the scorer's choice.
        scored_node = diagnostics.get("selected_node")
        selected_node = (
            scored_node if scored_node is not None else decision.get("next_node")
        )
        selected = next(
            (item for item in parsed if item["node"] == selected_node),
            None,
        )
        if selected is not None:
            selected_pheromone_scores.append(selected["score"])
            effect = selected["probability_effect"]
            if effect is not None:
                try:
                    selected_probability_effects.append(float(effect))
                except (TypeError, ValueError):
                    pass
            if selected["feature"] > 0:
                selected_positive += 1

            max_feature = max(item["feature"] for item in parsed)
            if max_feature > 0 and selected["feature"] == max_feature:
                selected_max += 1

        if diagnostics.get("pheromone_changed_score_argmax") is not None:
            counterfactual_comparable += 1
            if diagnostics.get("pheromone_changed_score_argmax") is True:
                argmax_changed += 1

        maximum_pheromone_scores.append(
            max(item["score"] for item in parsed)
        )

    return {
        "pheromone_reads": len(reads),
        "decisions_with_candidate_pheromone_diagnostics": decision_count,
        "decisions_with_comparable_no_pheromone_counterfactual": (
            counterfactual_comparable
        ),
        "selected_candidate_had_positive_pheromone": selected_positive,
        "selected_candidate_had_maximum_pheromone": selected_max,
        "scored_choice_matches_executed_next_hop": sum(
            1 for decision in decisions
            if decision.get("policy_type") == "default_snn"
            and isinstance(decision.get("decision_diagnostics"), dict)
            and decision["decision_diagnostics"].get("selected_node") is not None
            and decision["decision_diagnostics"].get("selected_node") == decision.get("next_node")
        ),
        "scored_choice_overridden_before_movement": sum(
            1 for decision in decisions
            if decision.get("policy_type") == "default_snn"
            and isinstance(decision.get("decision_diagnostics"), dict)
            and decision["decision_diagnostics"].get("selected_node") is not None
            and decision["decision_diagnostics"].get("selected_node") != decision.get("next_node")
        ),
        "selected_candidate_probability_effect_samples": (
            len(selected_probability_effects)
        ),
        "mean_selected_candidate_probability_effect": _mean(
            selected_probability_effects
        ),
        "mean_absolute_candidate_probability_effect": _mean(
            [abs(v) for v in candidate_probability_effects]
        ),
        "mean_selected_candidate_pheromone_score": _mean(
            selected_pheromone_scores
        ),
        "mean_max_candidate_pheromone_score": _mean(
            maximum_pheromone_scores
        ),
        "pheromone_changed_score_argmax": argmax_changed,
        "interpretation": (
            "The no-pheromone counterfactual holds the candidate set and "
            "selection temperature fixed and removes only the pheromone "
            "score term. Argmax changes show direct score-ranking influence; "
            "probability effects quantify the change in candidate selection "
            "probability. These are per-decision policy counterfactuals, not "
            "a substitute for the paired pheromone-off experiment."
        ),
    }



def _safe_int(value, default=None):
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_ratio(numerator, denominator):
    """Return None for an undefined ratio, never a misleading zero."""
    return numerator / denominator if denominator else None


def _navigator_indices(agents):
    """Return navigator indices without assuming a particular enum import."""
    result = []
    for i, agent in enumerate(agents):
        role = getattr(agent, "role", None)
        role_value = getattr(role, "value", role)
        if str(role_value).lower() == "navigator":
            result.append(i)
    # Some lightweight test doubles / older simulation versions do not expose
    # roles. In that case, report all agents rather than silently dropping them.
    return result if result else list(range(len(agents)))


def performance_metrics(sim, agents):
    """Per-run navigation performance with censoring and distance semantics explicit."""
    nav_indices = _navigator_indices(agents)
    total_navigators = len(nav_indices)
    current_step = _safe_int(getattr(sim, "current_step", None))
    if current_step is None:
        current_step = _safe_int(getattr(sim, "steps", None))

    goal_steps = getattr(sim, "_goal_step", {}) or {}
    finish_steps = getattr(sim, "_agent_finish_step", {}) or {}
    finish_reasons = getattr(sim, "_agent_finish_reason", {}) or {}
    agent_steps = getattr(sim, "_agent_steps_by_agent", {}) or {}

    completed = [
        i for i in nav_indices
        if bool(getattr(agents[i], "reached_goal", False))
    ]
    completion_steps, completed_decision_steps = [], []
    per_agent = []
    unfinished, failed = [], []

    for i in nav_indices:
        goal_step = _safe_int(goal_steps.get(i, goal_steps.get(str(i))))
        finish_step = _safe_int(finish_steps.get(i, finish_steps.get(str(i))))
        decision_steps = _safe_int(agent_steps.get(i, agent_steps.get(str(i))))
        is_completed = i in completed
        reason = finish_reasons.get(i, finish_reasons.get(str(i)))
        if is_completed:
            if goal_step is not None:
                completion_steps.append(goal_step)
            if decision_steps is not None:
                completed_decision_steps.append(decision_steps)
        elif reason == "timeout" or (
            reason is None and getattr(agents[i], "alive", True)
        ):
            unfinished.append(i)
        else:
            failed.append(i)

        per_agent.append({
            "agent": i,
            "completed": is_completed,
            "finish_reason": reason or (
                "goal" if is_completed else "unfinished_or_unclassified"
            ),
            "goal_completion_step": goal_step if is_completed else None,
            "finish_step": finish_step,
            "agent_decision_steps": decision_steps,
            "path_nodes_recorded": len(getattr(agents[i], "path", ()) or ()),
        })

    successful_makespan = max(completion_steps) if completion_steps else None
    all_completed = total_navigators > 0 and len(completed) == total_navigators
    team_completion_makespan = successful_makespan if all_completed else None

    distance_by_agent = {}
    distance_completed_by_agent = {}
    distance_total = 0.0
    completed_distance_total = 0.0
    path_distance = getattr(sim, "_path_distance", None)
    for i in nav_indices:
        agent = agents[i]
        distance = _safe_float(getattr(agent, "distance_travelled_m", None))
        if distance is None and callable(path_distance):
            try:
                distance = _safe_float(path_distance(list(getattr(agent, "path", ()) or ())))
            except (TypeError, ValueError, KeyError):
                distance = None
        if distance is not None:
            distance_by_agent[str(i)] = distance
            distance_total += distance
            if i in completed:
                distance_completed_by_agent[str(i)] = distance
                completed_distance_total += distance

    # Never label a partial sum as the total for a population. Keep the
    # per-agent observations, but expose a population total only when every
    # member of that population has a valid distance measurement.
    distance_agents_with_data = len(distance_by_agent)
    distance_agents_missing = [
        i for i in nav_indices if str(i) not in distance_by_agent
    ]
    all_navigator_distances_complete = (
        total_navigators > 0
        and distance_agents_with_data == total_navigators
    )
    completed_distance_complete = (
        len(completed) > 0
        and len(distance_completed_by_agent) == len(completed)
    )

    throughput_horizon = current_step if current_step is not None and current_step > 0 else None
    completed_count = len(completed)
    return {
        "navigator_count": total_navigators,
        "completed_navigators": completed_count,
        "unfinished_navigators": len(unfinished),
        "failed_navigators": len(failed),
        "unfinished_agent_indices": unfinished,
        "failed_agent_indices": failed,
        "completion_rate": _safe_ratio(completed_count, total_navigators),
        "travel_time": {
            "unit": "agent_decision_steps",
            "completed_agent_goal_times": completed_decision_steps,
            "mean_completed_agent_goal_time_steps": _mean(completed_decision_steps),
            "median_completed_agent_goal_time_steps": (
                statistics.median(completed_decision_steps)
                if completed_decision_steps else None
            ),
            "completion_timestamps_simulation_steps": completion_steps,
            "mean_completion_timestamp_simulation_step": _mean(completion_steps),
            "note": (
                "Elapsed per-agent decision steps are used for travel time. "
                "Global completion timestamps are reported separately and are "
                "not mislabeled as individual travel durations."
            ),
        },
        "makespan": {
            "successful_agent_makespan_steps": successful_makespan,
            "team_completion_makespan_steps": team_completion_makespan,
            "all_navigators_completed": all_completed,
            "observed_simulation_horizon_steps": current_step,
            "source": "_goal_step",
            "note": (
                "Team completion makespan is defined only when every navigator "
                "completed. Successful-agent makespan is reported separately "
                "for partial-completion runs; unfinished agents are right-censored."
            ),
        },
        "task_throughput": {
            "completed_agent_goals": completed_count,
            "simulation_step_horizon": throughput_horizon,
            "completed_agent_goals_per_step": _safe_ratio(
                completed_count, throughput_horizon
            ),
            "completed_agent_goals_per_100_steps": (
                100.0 * _safe_ratio(completed_count, throughput_horizon)
                if throughput_horizon else None
            ),
            "definition": (
                "Completed navigator goals divided by observed simulation "
                "steps. This is a run-level rate, not a per-agent travel speed."
            ),
        },
        "agent_work": {
            "total_agent_decision_steps": (
                sum(_safe_int(agent_steps.get(i, agent_steps.get(str(i))), 0)
                    for i in nav_indices) if agent_steps else None
            ),
            "mean_decision_steps_per_navigator": _safe_ratio(
                sum(_safe_int(agent_steps.get(i, agent_steps.get(str(i))), 0)
                    for i in nav_indices), total_navigators
            ) if agent_steps and total_navigators else None,
        },
        "distance": {
            "total_travelled_m_all_navigators": (
                distance_total if all_navigator_distances_complete else None
            ),
            "total_travelled_m_completed_navigators": (
                completed_distance_total if completed_distance_complete else None
            ),
            "distance_by_agent_m": distance_by_agent,
            "completed_distance_by_agent_m": distance_completed_by_agent,
            "agents_with_distance": distance_agents_with_data,
            "agents_missing_distance": distance_agents_missing,
            "all_navigator_distances_complete": all_navigator_distances_complete,
            "completed_distance_complete": completed_distance_complete,
            "population": (
                "All-navigator total is reported only when every navigator "
                "has a valid distance; completed-only total is reported only "
                "when every completed navigator has a valid distance."
            ),
            "note": (
                "Distance is computed from consecutive recorded path nodes "
                "using the simulator's geographic edge-distance function when "
                "an explicit distance counter is unavailable. Per-agent values "
                "may be present even when a population total is incomplete."
            ),
        },
        "per_navigator": per_agent,
    }


def hazard_exposure_metrics(sim, agents):
    """
    Summarise the simulator's existing ground-truth exposure instrumentation.

    Sensing-neighbourhood exposure, severity-weighted proximity dose, and
    actual traversal/occupancy encounters are reported separately. Severity
    dose is a proximity proxy based on maximum node-damage value in the
    current/one-hop neighbourhood, not physical contact or injury.
    """
    exposure = getattr(sim, "_hazard_exposure", {}) or {}
    local_by_step = exposure.get(
        "agents_with_hazard_in_sensing_neighborhood_by_step", {}
    ) or {}
    dynamic_local_by_step = exposure.get(
        "agents_with_dynamic_hazard_in_sensing_neighborhood_by_step", {}
    ) or {}

    def _aggregate_step_sets(history):
        per_agent_steps = Counter()
        total_agent_step_exposures = 0
        for _, indices in history.items():
            unique_indices = set(indices or ())
            total_agent_step_exposures += len(unique_indices)
            for idx in unique_indices:
                per_agent_steps[str(idx)] += 1
        return total_agent_step_exposures, dict(per_agent_steps)

    local_total, local_per_agent = _aggregate_step_sets(local_by_step)
    dynamic_total, dynamic_per_agent = _aggregate_step_sets(dynamic_local_by_step)

    actual_node_encounters = exposure.get("actual_node_encounters", set()) or set()
    actual_edge_encounters = exposure.get("actual_edge_encounters", set()) or set()
    actual_edge_attempts = exposure.get(
        "actual_edge_attempts_after_disablement", set()
    ) or set()
    damaged_occupants = exposure.get(
        "agents_occupying_dynamically_damaged_nodes", set()
    ) or set()
    agents_encountered = exposure.get("agents_encountered_hazard", set()) or set()

    severity_history = exposure.get(
        "severity_weighted_exposure_by_agent_by_step", {}
    ) or {}
    severity_by_agent = Counter()
    severity_total = 0.0
    observed_agent_step_samples = 0
    positive_severity_agent_step_samples = 0
    for step_values in severity_history.values():
        for agent_idx, dose in (step_values or {}).items():
            dose_value = _safe_float(dose, 0.0)
            severity_by_agent[str(agent_idx)] += dose_value
            severity_total += dose_value
            observed_agent_step_samples += 1
            if dose_value > 0.0:
                positive_severity_agent_step_samples += 1

    return {
        "sensing_neighborhood_exposure": {
            "agent_step_exposures": local_total,
            "unique_agents_exposed": len({
                idx for values in local_by_step.values() for idx in (values or ())
            }),
            "exposed_agent_steps_by_agent": local_per_agent,
            "observed_steps": len(local_by_step),
            "dynamic_agent_step_exposures": dynamic_total,
            "dynamic_exposed_agent_steps_by_agent": dynamic_per_agent,
            "dynamic_observed_steps": len(dynamic_local_by_step),
            "definition": (
                "An agent is counted once per step when a damaged node or "
                "disabled edge is in its current-node/one-hop sensing "
                "neighbourhood. This is proximity exposure, not physical contact."
            ),
        },
        "actual_encounters": {
            "unique_hazard_nodes_encountered": len(actual_node_encounters),
            "unique_hazard_edges_encountered": len(actual_edge_encounters),
            "attempts_to_traverse_disabled_edges": len(actual_edge_attempts),
            "agents_with_actual_hazard_encounters": len(agents_encountered),
            "agents_occupying_dynamically_damaged_nodes": len(damaged_occupants),
        },
        "severity_weighted_exposure": {
            "unit": "damage-severity-agent-steps",
            "total": severity_total if observed_agent_step_samples else None,
            "per_agent": dict(severity_by_agent),
            "observed_agent_step_samples": observed_agent_step_samples,
            # Backward-compatible alias retained for existing analysis scripts.
            "agent_step_samples": observed_agent_step_samples,
            "positive_severity_agent_step_samples": (
                positive_severity_agent_step_samples
            ),
            "mean_per_observed_agent_step": (
                _safe_ratio(severity_total, observed_agent_step_samples)
                if observed_agent_step_samples else None
            ),
            "mean_per_positive_severity_agent_step": (
                _safe_ratio(severity_total, positive_severity_agent_step_samples)
                if positive_severity_agent_step_samples else None
            ),
            "definition": (
                "For each observed agent-step, the maximum node-damage severity "
                "in the current node plus one-hop neighbourhood, summed over "
                "observed agent-steps. The observed-agent-step mean includes "
                "zero-severity observations; the positive-severity mean excludes "
                "them. This is a proximity-weighted dose proxy, not a "
                "physiological dose or physical-contact measure."
            ),
        },
        "severity_weighted_exposure_note": (
            "Uses simulator-recorded per-agent, per-step local maximum node "
            "damage. Compare only under identical sensing/neighbourhood "
            "definitions and simulation horizons."
        ),
    }


def useful_information_metrics(sim):
    """Report raw reads separately from deduplicated information acquisition.

    Raw reads describe access activity. Unique recipient-node acquisitions
    deduplicate across messages, repeated shared-memory lookups, and channels.
    Event-level acquisitions are additionally counted when node/event mapping
    is available. Route relevance is frozen at the first cross-agent arrival.
    Neither measure establishes causal navigation benefit.
    """
    reads = list(getattr(sim, "_shared_memory_reads", ()) or ())
    message_reads = list(getattr(sim, "_local_message_reads", ()) or ())
    cross_shared = [r for r in reads if r.get("cross_agent")]
    shared_hazard = [
        r for r in cross_shared
        if r.get("channel") in {"blocked_nodes", "danger_nodes"}
    ]
    cross_messages = [r for r in message_reads if r.get("cross_agent")]
    raw_hazard_reads = shared_hazard + cross_messages
    raw_route_relevant = [
        r for r in raw_hazard_reads if r.get("route_relevant") is True
    ]

    # Stable sort means simultaneous deliveries are resolved consistently;
    # messages are considered before board reads at an identical step.
    ordered = sorted(
        raw_hazard_reads,
        key=lambda r: (
            int(r.get("retrieval_step", 0)),
            0 if r.get("channel") == "teammate_message" else 1,
            str(r.get("node_id")),
        ),
    )
    first_by_recipient_node = {}
    for record in ordered:
        recipient = record.get("reader_agent")
        node = record.get("node_id")
        if recipient is None or node is None:
            continue
        first_by_recipient_node.setdefault((recipient, node), record)

    unique_records = list(first_by_recipient_node.values())
    unique_route_relevant = [
        r for r in unique_records if r.get("route_relevant") is True
    ]

    # Determine whether a recipient had already independently observed the
    # same node before its first cross-agent acquisition.
    discoveries = list(getattr(sim, "_hazard_discoveries", ()) or ())
    self_seen = {}
    for d in discoveries:
        if d.get("source") != "self_observation":
            continue
        key = (d.get("agent"), d.get("hazard_id"))
        step = int(d.get("step", 0))
        self_seen[key] = min(step, self_seen.get(key, step))
    unique_new = []
    for r in unique_records:
        key = (r.get("reader_agent"), r.get("node_id"))
        acquisition_step = int(r.get("retrieval_step", 0))
        independently_known = self_seen.get(key)
        if independently_known is None or independently_known > acquisition_step:
            unique_new.append(r)

    event_ids_by_node = getattr(sim, "_event_ids_by_node", {}) or {}
    first_by_recipient_event = {}
    for r in unique_records:
        for event_id in event_ids_by_node.get(r.get("node_id"), ()):
            first_by_recipient_event.setdefault(
                (r.get("reader_agent"), event_id), r
            )
    unique_event_records = list(first_by_recipient_event.values())
    unique_event_route_relevant = [
        r for r in unique_event_records if r.get("route_relevant") is True
    ]

    new_local_message_reads = [
        r for r in cross_messages if r.get("new_information") is True
    ]
    route_relevant_new_local = [
        r for r in new_local_message_reads if r.get("route_relevant") is True
    ]
    return {
        # Raw read-level measures retained for backwards compatibility.
        "cross_agent_shared_memory_reads": len(cross_shared),
        "cross_agent_shared_memory_hazard_reads": len(shared_hazard),
        "cross_agent_local_message_hazard_reads": len(cross_messages),
        "cross_agent_hazard_information_reads": len(raw_hazard_reads),
        "route_relevant_hazard_information_reads": len(raw_route_relevant),
        "route_relevant_hazard_read_ratio": _safe_ratio(
            len(raw_route_relevant), len(raw_hazard_reads)
        ),
        "raw_read_level_definition": (
            "Repeated successful hazard reads are counted; this is access "
            "activity, not a count of distinct information acquisitions."
        ),
        # Deduplicated recipient-node acquisitions across all channels.
        "unique_recipient_node_acquisitions": len(unique_records),
        "unique_route_relevant_recipient_node_acquisitions": len(unique_route_relevant),
        "unique_acquisition_route_relevance_ratio": _safe_ratio(
            len(unique_route_relevant), len(unique_records)
        ),
        "unique_new_recipient_node_acquisitions": len(unique_new),
        "unique_new_route_relevant_recipient_node_acquisitions": sum(
            r.get("route_relevant") is True for r in unique_new
        ),
        "unique_new_acquisition_route_relevance_ratio": _safe_ratio(
            sum(r.get("route_relevant") is True for r in unique_new),
            len(unique_new),
        ),
        "repeated_hazard_reads_after_first_recipient_node_acquisition": max(
            0, len(raw_hazard_reads) - len(unique_records)
        ),
        "unique_recipient_event_acquisitions": len(unique_event_records),
        "unique_route_relevant_recipient_event_acquisitions": len(
            unique_event_route_relevant
        ),
        "unique_event_acquisition_route_relevance_ratio": _safe_ratio(
            len(unique_event_route_relevant), len(unique_event_records)
        ),
        # Keep message-specific novelty diagnostics, but label their scope.
        "new_local_message_hazard_information_reads": len(new_local_message_reads),
        "route_relevant_new_local_message_hazard_information_reads": len(route_relevant_new_local),
        "route_relevant_new_local_message_information_ratio": _safe_ratio(
            len(route_relevant_new_local), len(new_local_message_reads)
        ),
        "message_read_records_if_instrumented": len(cross_messages),
        "usefulness_definition": (
            "Unique route-relevant recipient-node acquisitions divided by "
            "all unique cross-agent recipient-node acquisitions; deduplicated "
            "across shared-memory and local-message channels. Event-level "
            "counts are also reported when event mappings exist."
        ),
        "interpretation": (
            "Route relevance is a snapshot at first cross-agent arrival. "
            "Acquisition and novelty are instrumentation proxies, not proof "
            "that the agent causally used the information or improved an "
            "outcome. Raw repeated reads remain separately available."
        ),
    }


def scaling_efficiency_metrics(performance, agents):
    """
    Report within-run normalized productivity, while explicitly declining
    to infer cross-run scaling from a single agent-count observation.
    """
    count = performance.get("navigator_count", 0)
    completed = performance.get("completed_navigators", 0)
    throughput = performance.get("task_throughput", {}).get(
        "completed_agent_goals_per_step"
    )
    return {
        "agent_count": count,
        "completed_goals_per_agent": _safe_ratio(completed, count),
        "completed_goals_per_agent_per_step": (
            _safe_ratio(throughput, count) if throughput is not None else None
        ),
        "cross_agent_count_scaling_efficiency": None,
        "requires_matched_multi_agent_count_runs": True,
        "note": (
            "A single run cannot establish scaling efficiency. Compare "
            "matched workloads at multiple agent counts, then compute "
            "throughput(N)/[N * throughput(1)] or an explicitly chosen "
            "speedup/efficiency measure across those runs."
        ),
    }


def collective_metrics_from_sim(sim):
    """One consistent Priority-2 schema for any experimental arm."""
    agents = list(getattr(sim, "agents", ()) or ())
    performance = performance_metrics(sim, agents)
    navigator_indices = _navigator_indices(agents)

    return {
        "schema_version": "1.5",
        "performance": performance,
        "hazard_exposure": hazard_exposure_metrics(sim, agents),
        "coverage": {
            "all_agents": coverage_metrics(agents),
            "navigators_only": coverage_metrics(agents, navigator_indices),
            # Backward-compatible top-level keys retain the previous all-agent definition.
            **coverage_metrics(agents),
        },
        "information": information_metrics(sim),
        "useful_information": useful_information_metrics(sim),
        "communication_efficiency": communication_efficiency_metrics(sim),
        "pheromone": pheromone_metrics(sim),
        "scaling_efficiency": scaling_efficiency_metrics(performance, agents),
        "instrumentation_notes": {
            "coverage": "Exact visited node-ID sets and cell-count keys.",
            "awareness": "One first-awareness record per agent-event pair.",
            "early_warning": "Censored if the agent never independently discovers the event.",
            "travel_time": "Per-agent decision steps for completed navigators; global completion timestamps reported separately.",
            "makespan": "Team completion makespan is null unless all navigators complete; partial successful makespan and observed horizon are separate.",
            "throughput": "Completed navigator goals per observed simulation step.",
            "hazard_exposure": "Severity-weighted proximity is a maximum-neighbourhood node-damage proxy; actual traversal/occupancy encounters are separate.",
            "useful_information": "Reports raw read activity and deduplicated recipient-node/event acquisitions across shared-memory and local-message channels; route relevance is frozen at first acquisition and is not causal proof.",
            "coordination_overhead": "Communication volume is reported alongside attributed planner changes; attribution is not proof of benefit.",
            "scaling": "Cross-agent-count efficiency requires matched runs at multiple agent counts.",
            "causality": "Post-awareness navigation changes are not automatically causal.",
        },
    }
