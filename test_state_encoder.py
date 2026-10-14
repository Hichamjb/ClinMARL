import numpy as np

from rl.dataset_loader import (
    load_trajectories,
    flatten_trajectories,
)

from rl.state_encoder import (
    encode_state,
    encode_action_mask,
)


def main():
    trajectories = load_trajectories()
    transitions = flatten_trajectories(trajectories)

    encoded_states = []
    encoded_next_states = []

    for transition in transitions:
        state_vector = encode_state(
            transition["state"]
        )

        next_state_vector = encode_state(
            transition["next_state"]
        )

        assert state_vector.ndim == 1
        assert next_state_vector.ndim == 1

        assert state_vector.shape == next_state_vector.shape

        assert np.isfinite(state_vector).all()
        assert np.isfinite(next_state_vector).all()

        assert 0 <= transition["action_id"] < 7

        mask = encode_action_mask(
            transition["state"]
        )

        assert mask.shape == (7,)

        encoded_states.append(state_vector)
        encoded_next_states.append(next_state_vector)

    X = np.stack(encoded_states)
    X_next = np.stack(encoded_next_states)

    print("\n========== STATE ENCODER TEST ==========")
    print(f"Trajectories: {len(trajectories)}")
    print(f"Transitions: {len(transitions)}")
    print(f"State matrix shape: {X.shape}")
    print(f"Next-state matrix shape: {X_next.shape}")
    print(f"Feature count: {X.shape[1]}")
    print(f"All values finite: {np.isfinite(X).all()}")
    print(f"Action IDs valid: True")
    print("Action-mask dimensions: 7")
    print("\nFirst encoded state:")
    print(X[0])

    print("\n✅ State encoder tests passed.")


if __name__ == "__main__":
    main()