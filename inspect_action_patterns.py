from collections import defaultdict

import numpy as np

from rl.dataset_loader import (
    load_trajectories,
    flatten_trajectories,
)


def main():
    trajectories = load_trajectories()
    transitions = flatten_trajectories(trajectories)

    actions = defaultdict(list)
    severity_by_action = defaultdict(list)
    condition_by_action = defaultdict(list)

    for transition in transitions:
        action_name = transition["action_name"]
        state = transition["state"]

        severity = state.get("severity_index")

        if severity is not None:
            severity_by_action[action_name].append(
                float(severity)
            )

        actions[action_name].append(state)

        condition_by_action[action_name].append(
            transition.get("condition_key", "unknown")
        )

    print("\n========== ACTION PATTERN AUDIT ==========")

    for action_name in sorted(actions):
        states = actions[action_name]
        severities = severity_by_action[action_name]
        conditions = condition_by_action[action_name]

        print(f"\n--- {action_name} ---")
        print(f"Occurrences: {len(states)}")

        if severities:
            print(
                f"Mean severity: {np.mean(severities):.3f}"
            )
            print(
                f"Severity std: {np.std(severities):.3f}"
            )
            print(
                f"Min severity: {np.min(severities):.3f}"
            )
            print(
                f"Max severity: {np.max(severities):.3f}"
            )

        condition_counts = {}

        for condition in conditions:
            condition_counts[condition] = (
                condition_counts.get(condition, 0) + 1
            )

        print("Most common conditions:")

        for condition, count in sorted(
            condition_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]:
            print(f"  {condition}: {count}")

    print("\n========== AUDIT COMPLETE ==========")
    print(
        "Synthetic data only: these associations do not "
        "establish clinical appropriateness."
    )


if __name__ == "__main__":
    main()