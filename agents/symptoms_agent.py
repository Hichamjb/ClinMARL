from pydantic import BaseModel, Field
from typing import List, Optional

from models.patient import PatientData


class SymptomState(BaseModel):
    name: str
    present: bool
    duration_days: Optional[int] = None
    severity: Optional[str] = None
    location: Optional[str] = None


class SymptomsState(BaseModel):
    patient_id: str
    symptoms: List[SymptomState] = Field(default_factory=list)


class SymptomsAgent:

    def run(self, patient: PatientData) -> SymptomsState:

        symptoms_state = []

        for symptom in patient.symptoms:

            state = SymptomState(
                name=symptom.name,
                present=True,
                duration_days=symptom.duration_days,
                severity=symptom.severity,
                location=symptom.location
            )

            symptoms_state.append(state)

        return SymptomsState(
            patient_id=patient.patient_id,
            symptoms=symptoms_state
        )