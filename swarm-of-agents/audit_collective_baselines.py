
import argparse
import json
import math
import sys
from pathlib import Path

ARMS = (
    "individual_serial",
    "independent_multi_agent",
    "coordinated_swarm",
)
EVENT_TYPE = "dynamic_hazard_exposure"


class Audit:
    def __init__(self):
        self.rows = []

    def add(self, status, name, detail):
        self.rows.append((status, name, str(detail)))

    def show(self):
        print("\n" + "=" * 72)
        print("COLLECTIVE BASELINES — AUTOMATED AUDIT")
        print("=" * 72)

        for status, name, detail in self.rows:
            print(f"[{status:4}] {name}: {detail}")

        failures = sum(x[0] == "FAIL" for x in self.rows)
        warnings = sum(x[0] == "WARN" for x in self.rows)
        passes = sum(x[0] == "PASS" for x in self.rows)

        print("-" * 72)
        print(f"PASS: {passes}  WARN: {warnings}  FAIL: {failures}")

        result = (
            "FAIL" if failures
            else "PASS WITH WARNINGS" if warnings
            else "PASS"
        )
        print("AUDIT RESULT:", result)
        print("=" * 72)

        return failures, warnings


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    )


def walk(value):
    yield value

    if isinstance(value, dict):
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def get_events(arm):
    hazard = arm.get("hazard_response", {})
    records = hazard.get("event_records", [])

    if not isinstance(records, list):
        return []

    return [
        event for event in records
        if isinstance(event, dict)
        and event.get("event_type") == EVENT_TYPE
    ]


def event_signature(event):
    keys = (
        "event_id",
        "event_step",
        "centre",
        "center",
        "radius_m",
        "severity",
        "affected_nodes",
        "affected_edges",
    )

    result = {k: event[k] for k in keys if k in event}

    if "centre" not in result and "center" in result:
        result["centre"] = result.pop("center")

    result.pop("center", None)
    return result


def communication_counts(arm):
    communication = arm.get("communication", {})

    if not isinstance(communication, dict):
        return {}

    keys = ("transmissions", "deliveries", "dropped", "bytes_tx")

    return {
        key: communication[key]
        for key in keys
        if key in communication
    }


def awareness_sources(arm):
    hazard = arm.get("hazard_response", {})
    found = set()

    for item in walk(hazard):
        if not isinstance(item, dict):
            continue

        for key, value in item.items():
            if key in (
                "source",
                "awareness_source",
                "information_source",
                "replan_source",
            ) and isinstance(value, str):
                found.add(value)

            if key == "planner_replan_source_attribution":
                if isinstance(value, dict):
                    found.update(
                        str(k)
                        for k, v in value.items()
                        if isinstance(v, (int, float)) and v > 0
                    )

    return found


def audit_file(filename, report_filename):
    audit = Audit()

    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    experiment = data.get("experiment", {})
    runs = data.get("runs")

    if not isinstance(runs, list):
        audit.add("FAIL", "paired_runs", "Missing or invalid")
        audit.show()
        return 2

    declared_runs = experiment.get("n_runs")

    audit.add(
        "PASS" if declared_runs is None or declared_runs == len(runs)
        else "FAIL",
        "Run count",
        f"JSON={len(runs)}, declared={declared_runs}",
    )

    expected_budget = experiment.get(
        "total_agent_step_budget_per_arm"
    )

    design_arms = data.get("design", {}).get("arms", {})

    for index, run in enumerate(runs, 1):
        label = f"Run {run.get('run', index)}"
        arms = run.get("arms", {})

        missing = [
            name for name in ARMS
            if not isinstance(arms.get(name), dict)
        ]

        if missing:
            audit.add("FAIL", label + " arm presence", missing)
            continue

        audit.add("PASS", label + " arm presence", "All three arms exist")

        audit.add(
            "PASS" if run.get("seed") is not None else "FAIL",
            label + " seed",
            run.get("seed"),
        )

        audit.add(
            "PASS" if run.get("task_manifest") is not None else "FAIL",
            label + " task manifest",
            "Shared paired-run manifest present"
            if run.get("task_manifest") is not None
            else "Missing",
        )

        # -------------------------------------------------
        # 1. CONTROLLED EVENT PARITY
        # -------------------------------------------------

        signatures = {}
        expected_event = run.get("exposure_event")

        for arm_name in ARMS:
            arm = arms[arm_name]
            hazard = arm.get("hazard_response")

            if not isinstance(hazard, dict):
                audit.add(
                    "FAIL",
                    f"{label}/{arm_name} hazard response",
                    "Missing hazard_response",
                )
                continue

            events = get_events(arm)

            if len(events) != 1:
                audit.add(
                    "FAIL",
                    f"{label}/{arm_name} event count",
                    f"Expected 1 event; found {len(events)}",
                )
                continue

            event = events[0]
            signatures[arm_name] = event_signature(event)

            applied = hazard.get("exposure_event_applied")

            if applied is False:
                audit.add(
                    "FAIL",
                    f"{label}/{arm_name} event application",
                    "exposure_event_applied is false",
                )
            elif applied is True:
                audit.add(
                    "PASS",
                    f"{label}/{arm_name} event application",
                    "Event application flag is true",
                )

            # Compare the recorded event against the expected event.
                        # Compare the recorded event against the expected event.
            if isinstance(expected_event, dict):

                for key in (
                    "event_id",
                    "radius_m",
                    "severity",
                    "affected_nodes",
                    "affected_edges",
                ):
                    if key in expected_event:
                        if event.get(key) != expected_event[key]:
                            audit.add(
                                "FAIL",
                                f"{label}/{arm_name} event {key}",
                                f"Expected {expected_event[key]!r}; "
                                f"got {event.get(key)!r}",
                            )

                # Expected trigger_step corresponds to recorded event_step.
                if "trigger_step" in expected_event:
                    if event.get("event_step") != expected_event["trigger_step"]:
                        audit.add(
                            "FAIL",
                            f"{label}/{arm_name} event trigger step",
                            f"Expected {expected_event['trigger_step']}; "
                            f"got {event.get('event_step')}",
                        )

                    if key in expected_event:
                        if event.get(key) != expected_event[key]:
                            audit.add(
                                "FAIL",
                                f"{label}/{arm_name} event {key}",
                                f"Expected {expected_event[key]!r}; "
                                f"got {event.get(key)!r}",
                            )

                if (
                    "centre" in expected_event
                    or "center" in expected_event
                ):
                    expected_centre = expected_event.get(
                        "centre", expected_event.get("center")
                    )
                    actual_centre = event.get(
                        "centre", event.get("center", "<missing>")
                    )

                    if canonical(expected_centre) != canonical(actual_centre):
                        audit.add(
                            "FAIL",
                            f"{label}/{arm_name} event centre",
                            f"Expected {expected_centre}; got {actual_centre}",
                        )

        if len(signatures) == 3:
            matching = len({
                canonical(signatures[name]) for name in ARMS
            }) == 1

            audit.add(
                "PASS" if matching else "FAIL",
                label + " controlled-event parity",
                "All three arms have matching event signatures"
                if matching else signatures,
            )

        # -------------------------------------------------
        # 2. COMMUNICATION ISOLATION
        # -------------------------------------------------

        for arm_name in (
            "individual_serial",
            "independent_multi_agent",
        ):
            arm = arms[arm_name]
            counts = communication_counts(arm)

            if not counts:
                audit.add(
                    "WARN",
                    f"{label}/{arm_name} communication",
                    "Communication counters missing",
                )
            else:
                nonzero = {
                    key: value
                    for key, value in counts.items()
                    if isinstance(value, (int, float)) and value != 0
                }

                audit.add(
                    "FAIL" if nonzero else "PASS",
                    f"{label}/{arm_name} communication",
                    f"Non-zero counters: {nonzero}"
                    if nonzero else f"All zero: {counts}",
                )

            sources = awareness_sources(arm)

            forbidden = {
                source for source in sources
                if (
                    "teammate" in source.lower()
                    or "shared_memory" in source.lower()
                    or "shared memory" in source.lower()
                )
            }

            if forbidden:
                audit.add(
                    "FAIL",
                    f"{label}/{arm_name} awareness isolation",
                    sorted(forbidden),
                )
            elif sources:
                audit.add(
                    "PASS",
                    f"{label}/{arm_name} awareness isolation",
                    f"No teammate/shared-memory source: {sorted(sources)}",
                )
            else:
                audit.add(
                    "WARN",
                    f"{label}/{arm_name} awareness isolation",
                    "No parseable awareness source labels",
                )

        # Coordinated arm should show communication activity.
        coordinated_counts = communication_counts(
            arms["coordinated_swarm"]
        )

        transmissions = coordinated_counts.get("transmissions", 0)
        deliveries = coordinated_counts.get("deliveries", 0)

        active = (
            isinstance(transmissions, (int, float))
            and isinstance(deliveries, (int, float))
            and transmissions > 0
            and deliveries > 0
        )

        audit.add(
            "PASS" if active else "WARN",
            label + " coordinated communication",
            f"Transmissions={transmissions}, deliveries={deliveries}",
        )

        # Check declared experimental design.
        independent_config = design_arms.get(
            "independent_multi_agent", {}
        )
        coordinated_config = design_arms.get(
            "coordinated_swarm", {}
        )

        flags_correct = (
            independent_config.get("coordination") is False
            and coordinated_config.get("coordination") is True
        )

        audit.add(
            "PASS" if flags_correct else "FAIL",
            label + " design coordination flags",
            f"Independent={independent_config.get('coordination')!r}; "
            f"coordinated={coordinated_config.get('coordination')!r}",
        )

        # -------------------------------------------------
        # 3. AGENT-STEP BUDGET PARITY
        # -------------------------------------------------

        budgets = {}

        for arm_name in ARMS:
            arm = arms[arm_name]
            budget = arm.get("max_agent_step_budget")
            used = arm.get("agent_steps_total")

            if budget is None:
                audit.add(
                    "WARN",
                    f"{label}/{arm_name} declared budget",
                    "max_agent_step_budget missing",
                )
            else:
                budgets[arm_name] = budget

                correct = (
                    expected_budget is None
                    or budget == expected_budget
                )

                audit.add(
                    "PASS" if correct else "FAIL",
                    f"{label}/{arm_name} declared budget",
                    f"Budget={budget}; expected={expected_budget}",
                )

            if isinstance(budget, (int, float)) and isinstance(
                used, (int, float)
            ):
                audit.add(
                    "PASS" if used <= budget else "FAIL",
                    f"{label}/{arm_name} budget use",
                    f"{used}/{budget}",
                )
            else:
                audit.add(
                    "WARN",
                    f"{label}/{arm_name} budget use",
                    f"Used={used!r}; budget={budget!r}",
                )

        if len(budgets) == 3:
            matching = len(set(budgets.values())) == 1

            audit.add(
                "PASS" if matching else "FAIL",
                label + " cross-arm budget parity",
                budgets,
            )

        # -------------------------------------------------
        # 4. PAIRED DELTA CONSISTENCY
        # -------------------------------------------------

        independent = arms["independent_multi_agent"]
        coordinated = arms["coordinated_swarm"]

        metrics = (
            (
                "coordination_distance_delta_m_vs_independent",
                "mean_distance_per_task_m",
            ),
            (
                "coordination_makespan_delta_steps_vs_independent",
                "makespan_steps",
            ),
            (
                "coordination_agent_steps_delta_vs_independent",
                "agent_steps_total",
            ),
            (
                "coordination_success_rate_delta_vs_independent",
                "success_rate",
            ),
        )

        for delta_name, metric in metrics:
            left = independent.get(metric)
            right = coordinated.get(metric)
            recorded = run.get("paired_differences", {}).get(delta_name)

            if left is None or right is None:
                audit.add(
                    "WARN",
                    label + " " + delta_name,
                    f"Raw metric {metric} unavailable",
                )
                continue

            recomputed = float(right) - float(left)

            matches = (
                recorded is not None
                and math.isclose(
                    float(recorded),
                    recomputed,
                    rel_tol=1e-8,
                    abs_tol=1e-6,
                )
            )

            audit.add(
                "PASS" if matches else "FAIL",
                label + " " + delta_name,
                f"Recomputed={recomputed:.6f}; recorded={recorded!r}",
            )

    # -----------------------------------------------------
    # 5. SEED UNIQUENESS
    # -----------------------------------------------------

    seeds = [run.get("seed") for run in runs]
    unique = len(seeds) == len(set(seeds))

    audit.add(
        "PASS" if unique else "FAIL",
        "Unique run seeds",
        seeds,
    )

    failures, warnings = audit.show()

    report = {
        "input": str(filename),
        "runs": len(runs),
        "passes": sum(x[0] == "PASS" for x in audit.rows),
        "warnings": warnings,
        "failures": failures,
        "result": (
            "FAIL" if failures
            else "PASS_WITH_WARNINGS" if warnings
            else "PASS"
        ),
        "checks": [
            {"status": s, "name": n, "detail": d}
            for s, n, d in audit.rows
        ],
    }

    if report_filename:
        Path(report_filename).write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print("\nDetailed report saved:", report_filename)

    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        nargs="?",
        default="collective_baselines_test_fixed.json",
    )
    parser.add_argument(
        "--report",
        default="collective_baselines_audit_report.json",
    )

    args = parser.parse_args()

    if not Path(args.input).is_file():
        print("ERROR: File not found:", Path(args.input).resolve())
        return 2

    return audit_file(args.input, args.report)


if __name__ == "__main__":
    raise SystemExit(main())
