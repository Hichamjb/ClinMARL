from pydantic import BaseModel


class MedicalDocument(BaseModel):
    document_id: str
    title: str
    content: str
    disease: str
    keywords: list[str]


MEDICAL_KNOWLEDGE_BASE = [

    MedicalDocument(
        document_id="DOC001",
        title="Breast Mass Evaluation",
        disease="breast_cancer",
        keywords=[
            "breast_lump",
            "breast_mass",
            "breast",
            "mass"
        ],
        content=(
            "A new breast mass requires clinical assessment. "
            "Further evaluation may include clinical examination "
            "and appropriate diagnostic imaging."
        )
    ),

    MedicalDocument(
        document_id="DOC002",
        title="Family History of Breast Cancer",
        disease="breast_cancer",
        keywords=[
            "family_history",
            "breast_cancer",
            "hereditary"
        ],
        content=(
            "A family history of breast cancer can be relevant "
            "when assessing an individual's clinical risk profile."
        )
    ),

    MedicalDocument(
        document_id="DOC003",
        title="Fatigue and Weight Loss",
        disease="general",
        keywords=[
            "fatigue",
            "weight_loss",
            "unexplained_weight_loss"
        ],
        content=(
            "Persistent fatigue and unexplained weight loss "
            "are nonspecific clinical findings that may require "
            "further medical evaluation."
        )
    ),

    MedicalDocument(
        document_id="DOC004",
        title="Hemoglobin",
        disease="general",
        keywords=[
            "hemoglobin",
            "anemia"
        ],
        content=(
            "Hemoglobin values should be interpreted according "
            "to the laboratory reference range and clinical context."
        )
    )
]