from models.patient import (
    PatientData,
    Symptom,
    LabResult,
    MedicalHistory
)

from pipeline import ClinicalPipeline


def main():

    # =====================================
    # PATIENT DATA
    # =====================================

    patient = PatientData(

        patient_id="P001",

        age=55,

        sex="female",

        symptoms=[
            Symptom(
                name="fatigue",
                duration_days=30,
                severity="moderate"
            ),

            Symptom(
                name="weight_loss",
                duration_days=60,
                severity="moderate"
            ),

            Symptom(
                name="breast_lump",
                duration_days=20,
                severity="moderate",
                location="breast"
            )
        ],

        laboratory=[
            LabResult(
                name="hemoglobin",
                value=11.2,
                unit="g/dL",
                reference_range="12-16"
            ),

            LabResult(
                name="WBC",
                value=7200,
                unit="cells/uL",
                reference_range="4000-11000"
            )
        ],

        medical_history=MedicalHistory(

            conditions=[
                "hypertension"
            ],

            previous_treatments=[],

            family_history=[
                "breast_cancer"
            ],

            previous_procedures=[]
        )
    )

    # =====================================
    # RUN PIPELINE
    # =====================================

    pipeline = ClinicalPipeline()

    clinical_state = pipeline.run(patient)
    from rag.retriever import MedicalRetriever
    retriever = MedicalRetriever()

    rag_result = retriever.retrieve(
        clinical_state,
        top_k=3
    )

    print("\n==============================")
    print("RAG QUERY")
    print("==============================")

    print(rag_result.query)

    print("\n==============================")
    print("RETRIEVED MEDICAL EVIDENCE")
    print("==============================")

    print(
        rag_result.model_dump_json(
            indent=2
        )
    )

    # # =====================================
    # # DISPLAY
    # # =====================================

    # print("\n==============================")
    # print("SYMPTOMS STATE")
    # print("==============================")

    # print(
    #     clinical_state.symptoms.model_dump_json(
    #         indent=2
    #     )
    # )

    # print("\n==============================")
    # print("LABORATORY STATE")
    # print("==============================")

    # print(
    #     clinical_state.laboratory.model_dump_json(
    #         indent=2
    #     )
    # )

    # print("\n==============================")
    # print("HISTORY STATE")
    # print("==============================")

    # print(
    #     clinical_state.history.model_dump_json(
    #         indent=2
    #     )
    # )

    # print("\n==============================")
    # print("CLINICAL STATE")
    # print("==============================")

    # print(
    #     clinical_state.model_dump_json(
    #         indent=2
    #     )
    # )


if __name__ == "__main__":
    main()