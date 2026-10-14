# rl/dataset_loader.py

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "synthetic_clinical_trajectories.jsonl"
)


def load_trajectories(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> list[dict[str, Any]]:
    """
    Load synthetic clinical trajectories from a JSONL file.

    Each line represents one patient trajectory.
    """

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    trajectories = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                trajectory = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from exc

            if "patient_id" not in trajectory:
                raise ValueError(
                    f"Missing patient_id at line {line_number}"
                )

            if not isinstance(trajectory.get("steps"), list):
                raise ValueError(
                    f"Missing or invalid steps at line {line_number}"
                )

            trajectories.append(trajectory)

    return trajectories


def flatten_trajectories(
    trajectories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert trajectories into individual RL transitions.

    Output fields:
        patient_id
        condition_key
        state
        action_id
        action_name
        reward
        next_state
        done
    """

    transitions = []

    for trajectory in trajectories:
        patient_id = trajectory["patient_id"]
        condition_key = trajectory.get("condition_key")

        for step_index, step in enumerate(trajectory["steps"]):
            action = step.get("action")

            if isinstance(action, dict):
                action_id = action.get("action_id")
                action_name = action.get("action_name")
            elif isinstance(action, int):
                action_id = action
                action_name = None
            else:
                raise ValueError(
                    f"Invalid action in {patient_id}, "
                    f"step {step_index + 1}: {action!r}"
                )

            if action_id is None:
                raise ValueError(
                    f"Missing action_id in {patient_id}, "
                    f"step {step_index + 1}"
                )

            reward = step.get("reward")

            # Support a reward represented as a dictionary too.
            if isinstance(reward, dict):
                reward = reward.get(
                    "total_reward",
                    reward.get("total"),
                )

            if not isinstance(reward, (int, float)):
                raise ValueError(
                    f"Invalid reward in {patient_id}, "
                    f"step {step_index + 1}: {reward!r}"
                )

            state = step.get("state")
            next_state = step.get("next_state")

            if state is None or next_state is None:
                raise ValueError(
                    f"Missing state or next_state in {patient_id}, "
                    f"step {step_index + 1}"
                )

            transitions.append(
                {
                    "patient_id": patient_id,
                    "condition_key": condition_key,
                    "step_index": step_index,
                    "state": state,
                    "action_id": int(action_id),
                    "action_name": action_name,
                    "reward": float(reward),
                    "next_state": next_state,
                    "done": bool(step.get("done", False)),
                }
            )

    return transitions


def get_dataset_summary(
    trajectories: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return basic dataset statistics."""

    action_counts: dict[str, int] = {}

    for transition in transitions:
        name = (
            transition["action_name"]
            or str(transition["action_id"])
        )
        action_counts[name] = action_counts.get(name, 0) + 1

    return {
        "num_trajectories": len(trajectories),
        "num_transitions": len(transitions),
        "num_terminal_transitions": sum(
            transition["done"] for transition in transitions
        ),
        "num_actions": len(action_counts),
        "action_distribution": action_counts,
    }