from pydantic import BaseModel

from agents.symptoms_agent import SymptomsState
from agents.lab_agent import LabState
from agents.history_agent import HistoryState


class ClinicalState(BaseModel):

    patient_id: str

    age: int
    sex: str

    symptoms: SymptomsState

    laboratory: LabState

    history: HistoryState