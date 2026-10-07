from pydantic import BaseModel

from models.clinical_state import ClinicalState
from rag.vector_store import MedicalVectorStore


class Evidence(BaseModel):

    document_id: str
    title: str
    content: str
    distance: float


class RAGResult(BaseModel):

    patient_id: str
    query: str
    evidence: list[Evidence]


class MedicalRetriever:

    def __init__(self):

        self.vector_store = MedicalVectorStore()

    def _build_query(
        self,
        clinical_state: ClinicalState
    ) -> str:

        parts = []

        # Patient information
        parts.append(
            f"patient age {clinical_state.age}"
        )

        parts.append(
            f"sex {clinical_state.sex}"
        )

        # Symptoms
        for symptom in clinical_state.symptoms.symptoms:

            parts.append(symptom.name)

            if symptom.location:
                parts.append(
                    symptom.location
                )

        # Medical history
        for condition in clinical_state.history.conditions:

            parts.append(condition)

        # Family history
        for family_condition in (
            clinical_state.history.family_history
        ):

            parts.append(
                f"family history {family_condition}"
            )

        # Laboratory
        for lab in clinical_state.laboratory.laboratory:

            parts.append(
                f"{lab.name} {lab.value} {lab.unit}"
            )

        return " ".join(parts)

    def retrieve(
        self,
        clinical_state: ClinicalState,
        top_k: int = 3
    ) -> RAGResult:

        query = self._build_query(
            clinical_state
        )

        results = self.vector_store.search(
            query=query,
            top_k=top_k
        )

        evidence = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        for i in range(len(ids)):

            evidence.append(
                Evidence(
                    document_id=ids[i],
                    title=metadatas[i]["title"],
                    content=documents[i],
                    distance=round(
                        distances[i],
                        4
                    )
                )
            )

        return RAGResult(
            patient_id=clinical_state.patient_id,
            query=query,
            evidence=evidence
        )