from rl.dataset_loader import (
    load_trajectories,
    flatten_trajectories,
    get_dataset_summary,
)


def main():
    trajectories = load_trajectories()
    transitions = flatten_trajectories(trajectories)

    summary = get_dataset_summary(
        trajectories,
        transitions,
    )

    print("\n========== LOADER TEST ==========")
    print(f"Trajectories: {summary['num_trajectories']}")
    print(f"Transitions: {summary['num_transitions']}")
    print(
        "Terminal transitions:",
        summary["num_terminal_transitions"],
    )
    print(f"Actions: {summary['num_actions']}")

    print("\nAction distribution:")
    for action, count in summary["action_distribution"].items():
        print(f"  {action}: {count}")

    print("\nFirst transition:")
    first = transitions[0]

    print("Patient:", first["patient_id"])
    print("Condition:", first["condition_key"])
    print("Action ID:", first["action_id"])
    print("Action name:", first["action_name"])
    print("Reward:", first["reward"])
    print("Done:", first["done"])

    print("\nState type:", type(first["state"]).__name__)
    print("Next state type:", type(first["next_state"]).__name__)

    print("\n✅ Loader test completed.")


if __name__ == "__main__":
    main()