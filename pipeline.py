from models.patient import PatientData

from agents.symptoms_agent import SymptomsAgent
from agents.lab_agent import LabAgent
from agents.history_agent import HistoryAgent

from models.clinical_state import ClinicalState


class ClinicalPipeline:

    def __init__(self):

        self.symptoms_agent = SymptomsAgent()
        self.lab_agent = LabAgent()
        self.history_agent = HistoryAgent()

    def run(self, patient: PatientData) -> ClinicalState:

        # 1. Analyze symptoms
        symptoms_state = self.symptoms_agent.run(patient)

        # 2. Analyze laboratory results
        lab_state = self.lab_agent.run(patient)

        # 3. Analyze medical history
        history_state = self.history_agent.run(patient)

        # 4. Build unified clinical state
        clinical_state = ClinicalState(
            patient_id=patient.patient_id,
            age=patient.age,
            sex=patient.sex,
            symptoms=symptoms_state,
            laboratory=lab_state,
            history=history_state
        )

        return clinical_state