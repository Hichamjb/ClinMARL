import json
from pathlib import Path
from collections import Counter

DATASET_PATH = Path("synthetic_clinical_trajectories.jsonl")


def main():
    if not DATASET_PATH.exists():
        print(f"❌ File not found: {DATASET_PATH}")
        print("Place this script in the same directory as the JSONL dataset.")
        return

    trajectories = []

    # Read JSONL: each line should contain one trajectory
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                trajectories.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"❌ Invalid JSON on line {line_number}: {exc}")
                return

    print("\n========== DATASET REPORT ==========")
    print(f"Number of trajectories: {len(trajectories)}")

    if not trajectories:
        print("❌ Dataset is empty.")
        return

    # Inspect the first trajectory
    first = trajectories[0]
    print("\nFirst trajectory keys:")
    print(list(first.keys()))

    print("\nFirst trajectory preview:")
    print(json.dumps(first, indent=2, ensure_ascii=False)[:2500])

    # Identify the list containing the sequential steps
    possible_step_keys = [
        key for key, value in first.items()
        if isinstance(value, list)
        and value
        and isinstance(value[0], dict)
    ]

    if not possible_step_keys:
        print("\n⚠️ Could not identify the steps list automatically.")
        print("Check the keys printed above.")
        return

    # Prefer common names for the steps field
    preferred_names = ["steps", "trajectory", "transitions", "events"]
    step_key = next(
        (key for key in preferred_names if key in possible_step_keys),
        possible_step_keys[0]
    )

    print(f"\nSteps field detected: {step_key}")

    step_counts = []
    action_counts = Counter()
    errors = []

    patient_ids = []

    for index, trajectory in enumerate(trajectories, start=1):
        patient_id = trajectory.get("patient_id", f"row-{index}")
        patient_ids.append(patient_id)

        steps = trajectory.get(step_key)

        if not isinstance(steps, list) or not steps:
            errors.append(f"{patient_id}: missing or empty steps")
            continue

        step_counts.append(len(steps))

        # Check the actions and terminal flag
        for step_index, step in enumerate(steps):
            action = step.get("action")

            if action is not None:
                # Handle actions represented as dictionaries
                if isinstance(action, dict):
                    action = action.get("name", str(action))
                action_counts[str(action)] += 1

            if step_index == len(steps) - 1:
                if step.get("done") is not True:
                    errors.append(
                        f"{patient_id}: final step is not marked done=True"
                    )
            elif step.get("done") is True:
                errors.append(
                    f"{patient_id}: step {step_index + 1} is terminal "
                    "before the final step"
                )

        # Check that next_state matches the next step's state, if present
        for step_index in range(len(steps) - 1):
            next_state = steps[step_index].get("next_state")
            following_state = steps[step_index + 1].get("state")

            if (
                next_state is not None
                and following_state is not None
                and next_state != following_state
            ):
                errors.append(
                    f"{patient_id}: state-chain mismatch "
                    f"between steps {step_index + 1} and {step_index + 2}"
                )

    print("\n========== STATISTICS ==========")
    print(f"Unique patient IDs: {len(set(patient_ids))}")
    print(f"Total steps: {sum(step_counts)}")

    if step_counts:
        print(f"Minimum steps per trajectory: {min(step_counts)}")
        print(f"Maximum steps per trajectory: {max(step_counts)}")
        print(f"Average steps per trajectory: {sum(step_counts) / len(step_counts):.2f}")

    print("\nAction distribution:")
    for action, count in action_counts.most_common():
        print(f"  {action}: {count}")

    print("\n========== VALIDATION ==========")

    if len(trajectories) != 200:
        errors.append(
            f"Expected 200 trajectories, found {len(trajectories)}"
        )

    if len(set(patient_ids)) != len(patient_ids):
        errors.append("Duplicate patient IDs detected")

    if errors:
        print(f"⚠️ Found {len(errors)} validation issue(s):")
        for error in errors[:20]:
            print(f"  - {error}")

        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    else:
        print("✅ Basic dataset checks passed.")

    print(
        "\nNote: these checks verify the file structure and consistency, "
        "not clinical validity."
    )


if __name__ == "__main__":
    main()