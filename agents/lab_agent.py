from pydantic import BaseModel, Field
from typing import List, Optional

from models.patient import PatientData


class LaboratoryState(BaseModel):
    name: str
    value: float
    unit: str
    reference_range: Optional[str] = None


class LabState(BaseModel):
    patient_id: str
    laboratory: List[LaboratoryState] = Field(default_factory=list)


class LabAgent:

    def run(self, patient: PatientData) -> LabState:

        laboratory_state = []

        for lab in patient.laboratory:

            state = LaboratoryState(
                name=lab.name,
                value=lab.value,
                unit=lab.unit,
                reference_range=lab.reference_range
            )

            laboratory_state.append(state)

        return LabState(
            patient_id=patient.patient_id,
            laboratory=laboratory_state
        )