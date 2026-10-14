import json
from pathlib import Path

DATASET_PATH = Path("synthetic_clinical_trajectories.jsonl")


def main():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        trajectory = json.loads(next(file))

    steps = trajectory["steps"]

    print("=" * 60)
    print("PATIENT:", trajectory["patient_id"])
    print("CONDITION:", trajectory["condition_key"])
    print("NUMBER OF STEPS:", len(steps))

    for index, step in enumerate(steps[:2], start=1):
        print("\n" + "=" * 60)
        print(f"STEP {index}")

        print("\nAction:")
        print(json.dumps(
            step.get("action"),
            indent=2,
            ensure_ascii=False,
        ))

        print("\nReward:", step.get("reward"))
        print("Done:", step.get("done"))

        print("\nState:")
        print(json.dumps(
            step.get("state"),
            indent=2,
            ensure_ascii=False,
        ))

        print("\nNext state:")
        print(json.dumps(
            step.get("next_state"),
            indent=2,
            ensure_ascii=False,
        ))


if __name__ == "__main__":
    main()