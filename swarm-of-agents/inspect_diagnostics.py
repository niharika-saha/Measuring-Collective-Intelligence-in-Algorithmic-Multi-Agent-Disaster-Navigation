import json

with open("priority2_diagnostics_check.json", encoding="utf-8") as f:
    data = json.load(f)

def inspect(obj, path="root"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if any(term in key.lower() for term in (
                "pheromone",
                "decision_diagnostics",
                "candidate_diagnostic",
                "counterfactual",
            )):
                print(f"\n{path}.{key}")
                print(json.dumps(value, indent=2)[:3000])

            inspect(value, f"{path}.{key}")

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            inspect(value, f"{path}[{i}]")

inspect(data)