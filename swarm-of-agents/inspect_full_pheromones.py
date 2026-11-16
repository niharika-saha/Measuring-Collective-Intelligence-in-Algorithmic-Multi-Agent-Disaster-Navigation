
import json

with open("priority2_diagnostics_check.json", encoding="utf-8") as f:
    data = json.load(f)

for run_index, run in enumerate(data["runs"]):
    print("\n" + "=" * 65)
    print("RUN", run_index + 1)
    print("=" * 65)

    configurations = run["configurations"]

    for config_name in ["full", "no_pheromones"]:
        config = configurations[config_name]
        metrics = config["collective_metrics"]["pheromone"]

        print(f"\n--- {config_name.upper()} ---")

        for key, value in metrics.items():
            print(f"{key}: {value}")

        decisions = config["response_metrics"].get(
            "planner_assisted_navigation_decisions", []
        )

        valid = [
            d for d in decisions
            if d.get("decision_diagnostics") is not None
        ]

        print("\nTotal recorded decisions:", len(decisions))
        print("Decisions with diagnostics:", len(valid))

        if valid:
            print("\nExample diagnostic:")
            print(json.dumps(
                valid[0]["decision_diagnostics"],
                indent=2
            )[:4000])
        else:
            print("NO NON-NULL DECISION DIAGNOSTICS")
