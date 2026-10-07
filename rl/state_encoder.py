from pydantic import BaseModel

from models.clinical_state import ClinicalState


class RLState(BaseModel):
    age_over_50: int

    fatigue: int
    weight_loss: int
    breast_lump: int

    family_history_breast_cancer: int

    low_hemoglobin: int

    hypertension: int

    def to_tuple(self) -> tuple:
        """
        Convert the RL state into a tuple.

        Q-Learning will use this tuple
        as the state identifier.
        """

        return (
            self.age_over_50,
            self.fatigue,
            self.weight_loss,
            self.breast_lump,
            self.family_history_breast_cancer,
            self.low_hemoglobin,
            self.hypertension,
        )


class StateEncoder:

    def encode(
        self,
        clinical_state: ClinicalState
    ) -> RLState:

        # --------------------------------
        # Age
        # --------------------------------

        age_over_50 = int(
            clinical_state.age >= 50
        )

        # --------------------------------
        # Symptoms
        # --------------------------------

        symptom_names = {
            symptom.name.lower()
            for symptom in clinical_state.symptoms.symptoms
        }

        fatigue = int(
            "fatigue" in symptom_names
        )

        weight_loss = int(
            "weight_loss" in symptom_names
        )

        breast_lump = int(
            "breast_lump" in symptom_names
        )

        # --------------------------------
        # Family history
        # --------------------------------

        family_history = {
            condition.lower()
            for condition
            in clinical_state.history.family_history
        }

        family_history_breast_cancer = int(
            "breast_cancer" in family_history
        )

        # --------------------------------
        # Laboratory
        # --------------------------------

        low_hemoglobin = 0

        for lab in clinical_state.laboratory.laboratory:

            if lab.name.lower() == "hemoglobin":

                if lab.value < 12:

                    low_hemoglobin = 1

        # --------------------------------
        # Medical conditions
        # --------------------------------

        conditions = {
            condition.lower()
            for condition
            in clinical_state.history.conditions
        }

        hypertension = int(
            "hypertension" in conditions
        )

        # --------------------------------
        # Build RL state
        # --------------------------------

        return RLState(

            age_over_50=age_over_50,

            fatigue=fatigue,

            weight_loss=weight_loss,

            breast_lump=breast_lump,

            family_history_breast_cancer=(
                family_history_breast_cancer
            ),

            low_hemoglobin=low_hemoglobin,

            hypertension=hypertension,
        )