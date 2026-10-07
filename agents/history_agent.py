from pydantic import BaseModel, Field
from typing import List

from models.patient import PatientData


class HistoryState(BaseModel):
    patient_id: str

    conditions: List[str] = Field(default_factory=list)

    previous_treatments: List[str] = Field(default_factory=list)

    family_history: List[str] = Field(default_factory=list)

    previous_procedures: List[str] = Field(default_factory=list)


class HistoryAgent:

    def run(self, patient: PatientData) -> HistoryState:

        history = patient.medical_history

        return HistoryState(
            patient_id=patient.patient_id,
            conditions=history.conditions,
            previous_treatments=history.previous_treatments,
            family_history=history.family_history,
            previous_procedures=history.previous_procedures
        )