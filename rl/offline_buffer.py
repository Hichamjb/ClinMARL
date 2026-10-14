# rl/offline_buffer.py

from typing import Any

import numpy as np

from rl.dataset_loader import load_trajectories
from rl.state_encoder import encode_state, encode_action_mask


NUM_ACTIONS = 7


def to_transitions(
    trajectories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert synthetic trajectories into offline RL transitions."""

    transitions = []

    for trajectory in trajectories:
        patient_id = trajectory["patient_id"]

        for step_index, step in enumerate(trajectory["steps"]):
            state = step["state"]
            next_state = step["next_state"]
            action = step["action"]

            if not isinstance(action, dict):
                raise ValueError(
                    f"Invalid action in {patient_id}, step {step_index}"
                )

            action_id = int(action["action_id"])

            if not 0 <= action_id < NUM_ACTIONS:
                raise ValueError(
                    f"Invalid action ID: {action_id}"
                )

            # The encoder returns 49 features:
            # 42 state features + 7 action-mask features.
            # Keep the mask separate from the state.
            state_vector = encode_state(state)[:-NUM_ACTIONS]
            next_state_vector = encode_state(next_state)[:-NUM_ACTIONS]

            action_mask = encode_action_mask(state)
            next_action_mask = encode_action_mask(next_state)

            if not action_mask[action_id]:
                raise ValueError(
                    f"Action {action_id} is not allowed for "
                    f"{patient_id}, step {step_index}"
                )

            reward = step["reward"]

            if isinstance(reward, dict):
                reward = reward.get(
                    "total_reward",
                    reward.get("total"),
                )

            if not isinstance(reward, (int, float)):
                raise ValueError(
                    f"Invalid reward for {patient_id}, "
                    f"step {step_index}"
                )

            transitions.append(
                {
                    "patient_id": patient_id,
                    "step_index": step_index,
                    "state": state_vector,
                    "action": action_id,
                    "reward": float(reward),
                    "next_state": next_state_vector,
                    "done": bool(step.get("done", False)),
                    "action_mask": action_mask,
                    "next_action_mask": next_action_mask,
                }
            )

    return transitions


def main():
    trajectories = load_trajectories()
    buffer = to_transitions(trajectories)

    print("========== OFFLINE BUFFER ==========")
    print(f"Patients: {len(trajectories)}")
    print(f"Transitions: {len(buffer)}")

    if buffer:
        print("State dimensions:", buffer[0]["state"].shape)
        print("Next-state dimensions:", buffer[0]["next_state"].shape)
        print("First action ID:", buffer[0]["action"])
        print("First reward:", buffer[0]["reward"])
        print("First transition done:", buffer[0]["done"])

    assert len(buffer) == sum(
        len(trajectory["steps"])
        for trajectory in trajectories
    )

    assert all(
        item["state"].shape == (42,)
        and item["next_state"].shape == (42,)
        for item in buffer
    )

    assert all(
        item["action_mask"][item["action"]]
        for item in buffer
    )

    print("All basic transition checks passed.")


if __name__ == "__main__":
    main()