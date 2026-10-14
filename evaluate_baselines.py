import random
from collections import Counter

import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from rl.dataset_loader import load_trajectories
from rl.state_encoder import encode_state, encode_action_mask
from train_behavior_cloning import split_by_patient, prepare_transitions


SEED = 42
NUM_ACTIONS = 7


def masked_predict(model, X, masks):
    """Predict only actions permitted by each state's action mask."""

    probabilities = model.predict_proba(X)

    class_to_column = {
        int(class_id): column
        for column, class_id in enumerate(model.classes_)
    }

    scores = np.zeros((len(X), NUM_ACTIONS), dtype=float)

    for action_id in range(NUM_ACTIONS):
        column = class_to_column.get(action_id)

        if column is not None:
            scores[:, action_id] = probabilities[:, column]

    scores[~masks] = -1.0

    return scores.argmax(axis=1)


def report(name, y_true, y_pred):
    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_true, y_pred):.4f}"
    )


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    trajectories = load_trajectories()
    train_patients, validation_patients, test_patients = (
        split_by_patient(trajectories)
    )

    X_train, y_train, _ = prepare_transitions(train_patients)
    X_val, y_val, masks_val = prepare_transitions(validation_patients)
    X_test, y_test, masks_test = prepare_transitions(test_patients)

    print("Training action distribution:")
    counts = Counter(y_train.tolist())

    for action_id in range(NUM_ACTIONS):
        print(f"Action {action_id}: {counts.get(action_id, 0)}")

    # Baseline 1: predict the most frequent training action.
    majority_action = counts.most_common(1)[0][0]

    majority_val = np.full(len(y_val), majority_action)
    majority_test = np.full(len(y_test), majority_action)

    # Enforce masks for baseline predictions too.
    for i, allowed in enumerate(masks_val):
        if not allowed[majority_val[i]]:
            majority_val[i] = np.flatnonzero(allowed)[0]

    for i, allowed in enumerate(masks_test):
        if not allowed[majority_test[i]]:
            majority_test[i] = np.flatnonzero(allowed)[0]

    report("Majority baseline - validation", y_val, majority_val)
    report("Majority baseline - test", y_test, majority_test)

    # Baseline 2: most-frequent DummyClassifier.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    dummy_val = masked_predict(dummy, X_val, masks_val)
    dummy_test = masked_predict(dummy, X_test, masks_test)

    report("Dummy baseline - validation", y_val, dummy_val)
    report("Dummy baseline - test", y_test, dummy_test)

    # Behavior Cloning reference model.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    bc = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=False,
        random_state=SEED,
    )

    bc.fit(X_train_scaled, y_train)

    bc_val = masked_predict(bc, X_val_scaled, masks_val)
    bc_test = masked_predict(bc, X_test_scaled, masks_test)

    report("Behavior Cloning - validation", y_val, bc_val)
    report("Behavior Cloning - test", y_test, bc_test)

    print("\nEvaluation complete.")
    print(
        "All results measure imitation of synthetic actions, "
        "not clinical correctness."
    )


if __name__ == "__main__":
    main()