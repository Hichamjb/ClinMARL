from dataclasses import dataclass

from rl.actions import ClinicalAction
from rl.state_encoder import RLState


@dataclass
class StepResult:
    state: RLState
    reward: float
    done: bool
    info: dict


class ClinicalEnvironment:

    def __init__(self):

        self.state = None
        self.steps = 0
        self.max_steps = 5

    def reset(self, initial_state: RLState) -> RLState:
        """
        Start a new episode.
        """

        self.state = initial_state
        self.steps = 0

        return self.state

    def step(
        self,
        action: ClinicalAction
    ) -> StepResult:

        if self.state is None:
            raise RuntimeError(
                "Environment must be reset before step()."
            )

        self.steps += 1

        reward = self._calculate_reward(
            action
        )

        next_state = self._transition(
            action
        )

        done = (
            self.steps >= self.max_steps
        )

        self.state = next_state

        return StepResult(
            state=next_state,
            reward=reward,
            done=done,
            info={
                "action": action.name,
                "step": self.steps
            }
        )

    def _calculate_reward(
        self,
        action: ClinicalAction
    ) -> float:

        """
        Educational/synthetic reward function.

        This is NOT a clinical recommendation.
        """

        state = self.state

        # --------------------------------
        # High-information finding
        # --------------------------------

        if state.breast_lump == 1:

            if action == (
                ClinicalAction.REQUEST_DIAGNOSTIC_IMAGING
            ):
                return 5.0

            if action == (
                ClinicalAction.SPECIALIST_REFERRAL
            ):
                return 4.0

        # --------------------------------
        # Missing information
        # --------------------------------

        if action == (
            ClinicalAction.GATHER_MORE_INFORMATION
        ):
            return 2.0

        # --------------------------------
        # Laboratory follow-up
        # --------------------------------

        if state.low_hemoglobin == 1:

            if action == (
                ClinicalAction.REQUEST_LAB_TEST
            ):
                return 2.0

        # --------------------------------
        # Clinical review
        # --------------------------------

        if action == ClinicalAction.CLINICAL_REVIEW:

            return 1.0

        # --------------------------------
        # Reassessment
        # --------------------------------

        if action == ClinicalAction.REASSESS:

            return 0.5

        # --------------------------------
        # Default
        # --------------------------------

        return -1.0

    def _transition(
        self,
        action: ClinicalAction
    ) -> RLState:

        """
        Synthetic transition function.

        In a real research environment,
        transitions should come from a
        validated simulator or offline data.
        """

        current = self.state

        # Copy current state
        next_state = current.model_copy()

        # --------------------------------
        # Gathering information
        # --------------------------------

        if action == (
            ClinicalAction.GATHER_MORE_INFORMATION
        ):

            return next_state

        # --------------------------------
        # Diagnostic imaging
        # --------------------------------

        if action == (
            ClinicalAction.REQUEST_DIAGNOSTIC_IMAGING
        ):

            return next_state

        # --------------------------------
        # Laboratory test
        # --------------------------------

        if action == (
            ClinicalAction.REQUEST_LAB_TEST
        ):

            return next_state

        # --------------------------------
        # Clinical review
        # --------------------------------

        if action == (
            ClinicalAction.CLINICAL_REVIEW
        ):

            return next_state

        # --------------------------------
        # Specialist referral
        # --------------------------------

        if action == (
            ClinicalAction.SPECIALIST_REFERRAL
        ):

            return next_state

        # --------------------------------
        # Reassessment
        # --------------------------------

        if action == ClinicalAction.REASSESS:

            return next_state

        return next_state