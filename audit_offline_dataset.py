import random
import numpy as np

from collections import Counter, defaultdict

from rl.dataset_loader import (
    load_trajectories,
    flatten_trajectories,
)

from rl.state_encoder import (
    encode_state,
    encode_action_mask,
)


SEED = 42


def normalize_outcome(trajectory) -> str:
    """Return a hashable string label for a trajectory's final outcome.

    `final_outcome` may be a string, None, or a dict with one of several
    possible label keys. This normalizes all of those into a plain string
    so it can safely be used as a Counter key.
    """
    outcome = trajectory.get("final_outcome", "unknown")

    if isinstance(outcome, dict):
        outcome = (
            outcome.get("label")
            or outcome.get("outcome")
            or outcome.get("status")
            or outcome.get("category")
            or "dictionary_without_label"
        )

    if outcome is None:
        outcome = "unknown"

    return str(outcome)


def main():
    trajectories = load_trajectories()
    transitions = flatten_trajectories(trajectories)

    print("\n========== OFFLINE DATASET AUDIT ==========")

    # 1. Reward statistics
    rewards = np.asarray(
        [t["reward"] for t in transitions],
        dtype=np.float32,
    )

    print("\n--- Rewards ---")
    print(f"Count: {len(rewards)}")
    print(f"Mean: {rewards.mean():.4f}")
    print(f"Std: {rewards.std():.4f}")
    print(f"Min: {rewards.min():.4f}")
    print(f"Max: {rewards.max():.4f}")

    # 2. Final outcome distribution
    # (Count final outcomes safely, even when they are dictionaries.)
    outcomes = Counter(
        normalize_outcome(trajectory)
        for trajectory in trajectories
    )

    print("\n--- Synthetic final outcomes ---")
    for outcome, count in sorted(outcomes.items()):
        print(f"{outcome}: {count}")

    # 3. Check whether recorded actions are permitted
    # by the action mask of the corresponding state.
    violations = []

    for t in transitions:
        mask = encode_action_mask(t["state"])
        action_id = t["action_id"]

        if not mask[action_id]:
            violations.append(
                (
                    t["patient_id"],
                    t["step_index"],
                    t["action_name"],
                    action_id,
                )
            )

    print("\n--- Action-mask consistency ---")
    print(f"Transitions checked: {len(transitions)}")
    print(f"Actions disallowed by mask: {len(violations)}")

    for violation in violations[:10]:
        print("Violation:", violation)

    # 4. Check state vector dimensions
    feature_lengths = {
        len(encode_state(t["state"]))
        for t in transitions
    }

    print("\n--- State encoding ---")
    print("Feature dimensions found:", feature_lengths)

    # 5. Split by patient, not by individual transitions.
    # This prevents a patient's trajectory from appearing
    # in more than one split.
    shuffled = list(trajectories)
    random.Random(SEED).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * 0.70)
    n_validation = int(n * 0.15)

    train_patients = shuffled[:n_train]
    validation_patients = shuffled[
        n_train:n_train + n_validation
    ]
    test_patients = shuffled[n_train + n_validation:]

    split_patients = {
        "train": train_patients,
        "validation": validation_patients,
        "test": test_patients,
    }

    print("\n--- Patient-level split ---")

    patient_sets = {}

    for split_name, split in split_patients.items():
        patient_ids = {p["patient_id"] for p in split}
        patient_sets[split_name] = patient_ids

        num_transitions = sum(
            len(p["steps"]) for p in split
        )

        print(
            f"{split_name}: "
            f"{len(split)} patients, "
            f"{num_transitions} transitions"
        )

    # Ensure no patient is shared across splits.
    assert patient_sets["train"].isdisjoint(
        patient_sets["validation"]
    )
    assert patient_sets["train"].isdisjoint(
        patient_sets["test"]
    )
    assert patient_sets["validation"].isdisjoint(
        patient_sets["test"]
    )

    print("\nPatient overlap: 0")

    # 6. Final summary
    print("\n========== AUDIT SUMMARY ==========")

    if feature_lengths != {49}:
        print("WARNING: State vectors do not all have 49 features.")

    if violations:
        print(
            "WARNING: Some recorded actions are disallowed "
            "by their state masks. Investigate before training."
        )
    else:
        print("All recorded actions pass the action-mask check.")

    print(
        "Reminder: this is synthetic data with manually "
        "designed rewards, not clinical evidence."
    )


if __name__ == "__main__":
    main()