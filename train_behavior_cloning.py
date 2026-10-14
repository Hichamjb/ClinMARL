from collections import Counter
import random

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from rl.dataset_loader import (
    load_trajectories,
    flatten_trajectories,
)
from rl.state_encoder import (
    encode_state,
    encode_action_mask,
)


SEED = 42
NUM_ACTIONS = 7


def split_by_patient(trajectories):
    """Split complete patient trajectories to avoid patient leakage."""

    patients = list(trajectories)
    random.Random(SEED).shuffle(patients)

    n = len(patients)
    n_train = int(n * 0.70)
    n_validation = int(n * 0.15)

    train = patients[:n_train]
    validation = patients[n_train:n_train + n_validation]
    test = patients[n_train + n_validation:]

    return train, validation, test


def prepare_transitions(trajectories):
    """Flatten trajectories and encode state features."""

    transitions = flatten_trajectories(trajectories)

    X = []
    y = []
    masks = []

    for transition in transitions:
        # The current encoder returns 49 features:
        # 42 state features + 7 action-mask features.
        encoded = encode_state(transition["state"])

        # Keep the state features separate from the action mask.
        state_features = encoded[:-NUM_ACTIONS]

        action_mask = encode_action_mask(transition["state"])
        action_id = transition["action_id"]

        if not 0 <= action_id < NUM_ACTIONS:
            raise ValueError(
                f"Invalid action ID: {action_id}"
            )

        if not action_mask[action_id]:
            raise ValueError(
                f"Recorded action is disallowed: "
                f"{transition['patient_id']} "
                f"step {transition['step_index']}"
            )

        X.append(state_features)
        y.append(action_id)
        masks.append(action_mask)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(masks, dtype=bool),
    )


def evaluate_model(name, model, X, y, masks):
    """Evaluate predictions, including action-mask compliance."""

    probabilities = model.predict_proba(X)

    # Map model probability columns to their corresponding action IDs.
    class_to_column = {
        int(class_id): column
        for column, class_id in enumerate(model.classes_)
    }

    masked_probabilities = np.zeros(
        (len(X), NUM_ACTIONS),
        dtype=np.float64,
    )

    for action_id in range(NUM_ACTIONS):
        column = class_to_column.get(action_id)

        if column is not None:
            masked_probabilities[:, action_id] = (
                probabilities[:, column]
            )

    # Prevent choosing actions disallowed by the state mask.
    masked_probabilities[~masks] = -1.0

    predictions = np.argmax(masked_probabilities, axis=1)

    print(f"\n========== {name} ==========")
    print(f"Samples: {len(y)}")
    print(f"Accuracy: {accuracy_score(y, predictions):.4f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y, predictions):.4f}"
    )
    print(
        "Action-mask violations: "
        f"{sum(not masks[i, pred] for i, pred in enumerate(predictions))}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            labels=list(range(NUM_ACTIONS)),
            zero_division=0,
        )
    )

    return predictions


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading synthetic clinical trajectories...")
    trajectories = load_trajectories()

    train_patients, validation_patients, test_patients = (
        split_by_patient(trajectories)
    )

    train_ids = {p["patient_id"] for p in train_patients}
    validation_ids = {
        p["patient_id"] for p in validation_patients
    }
    test_ids = {p["patient_id"] for p in test_patients}

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    X_train, y_train, masks_train = prepare_transitions(
        train_patients
    )
    X_val, y_val, masks_val = prepare_transitions(
        validation_patients
    )
    X_test, y_test, masks_test = prepare_transitions(
        test_patients
    )

    print("\n========== DATA SPLIT ==========")
    print(f"Training patients: {len(train_patients)}")
    print(f"Validation patients: {len(validation_patients)}")
    print(f"Test patients: {len(test_patients)}")
    print(f"Training transitions: {len(y_train)}")
    print(f"Validation transitions: {len(y_val)}")
    print(f"Test transitions: {len(y_test)}")
    print(f"State feature dimensions: {X_train.shape[1]}")

    print("\nTraining action distribution:")
    for action_id, count in sorted(
        Counter(y_train.tolist()).items()
    ):
        print(f"Action {action_id}: {count}")

    # Fit preprocessing only on training data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=False,
        random_state=SEED,
    )

    print("\nTraining Behavior Cloning model...")
    model.fit(X_train_scaled, y_train)

    evaluate_model(
        "VALIDATION RESULTS",
        model,
        X_val_scaled,
        y_val,
        masks_val,
    )

    evaluate_model(
        "FINAL TEST RESULTS",
        model,
        X_test_scaled,
        y_test,
        masks_test,
    )

    print("\nTraining completed.")
    print(
        "Reminder: the model imitates synthetic recorded actions. "
        "It is not a clinically validated decision system."
    )


if __name__ == "__main__":
    main()