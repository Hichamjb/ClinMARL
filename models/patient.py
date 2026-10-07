from pydantic import BaseModel, Field
from typing import List, Optional


class Symptom(BaseModel):
    name: str
    duration_days: Optional[int] = None
    severity: Optional[str] = None
    location: Optional[str] = None


class LabResult(BaseModel):
    name: str
    value: float
    unit: str
    reference_range: Optional[str] = None


class MedicalHistory(BaseModel):
    conditions: List[str] = Field(default_factory=list)
    previous_treatments: List[str] = Field(default_factory=list)
    family_history: List[str] = Field(default_factory=list)
    previous_procedures: List[str] = Field(default_factory=list)


class PatientData(BaseModel):
    patient_id: str
    age: int
    sex: str

    symptoms: List[Symptom] = Field(default_factory=list)

    laboratory: List[LabResult] = Field(default_factory=list)

    medical_history: MedicalHistory





    