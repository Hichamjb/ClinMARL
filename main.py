from models.patient import (
    PatientData,
    Symptom,
    LabResult,
    MedicalHistory,
)

from pipeline import ClinicalPipeline

from rag.retriever import MedicalRetriever

from rl.state_encoder import StateEncoder

from rl.environment import ClinicalEnvironment
from rl.actions import ClinicalAction
from rl.q_learning import QLearningAgent


def print_section(title: str):
    print("\n" + "=" * 30)
    print(title)
    print("=" * 30)


def main():

    # ==========================================================
    # 1. PATIENT DATA
    # ==========================================================

    patient = PatientData(

        patient_id="P001",

        age=55,

        sex="female",

        symptoms=[
            Symptom(
                name="fatigue",
                duration_days=30,
                severity="moderate",
            ),

            Symptom(
                name="weight_loss",
                duration_days=60,
                severity="moderate",
            ),

            Symptom(
                name="breast_lump",
                duration_days=20,
                severity="moderate",
                location="breast",
            ),
        ],

        laboratory=[
            LabResult(
                name="hemoglobin",
                value=11.2,
                unit="g/dL",
                reference_range="12-16",
            ),

            LabResult(
                name="WBC",
                value=7200,
                unit="cells/uL",
                reference_range="4000-11000",
            ),
        ],

        medical_history=MedicalHistory(

            conditions=[
                "hypertension",
            ],

            previous_treatments=[],

            family_history=[
                "breast_cancer",
            ],

            previous_procedures=[],
        ),
    )

    # ==========================================================
    # 2. AGENTS
    # ==========================================================

    pipeline = ClinicalPipeline()

    clinical_state = pipeline.run(patient)

    print_section("SYMPTOMS STATE")

    print(
        clinical_state.symptoms.model_dump_json(
            indent=2
        )
    )

    print_section("LABORATORY STATE")

    print(
        clinical_state.laboratory.model_dump_json(
            indent=2
        )
    )

    print_section("HISTORY STATE")

    print(
        clinical_state.history.model_dump_json(
            indent=2
        )
    )

    # ==========================================================
    # 3. CLINICAL STATE
    # ==========================================================

    print_section("CLINICAL STATE")

    print(
        clinical_state.model_dump_json(
            indent=2
        )
    )

    # ==========================================================
    # 4. RAG
    # ==========================================================

    retriever = MedicalRetriever()

    rag_result = retriever.retrieve(
        clinical_state,
        top_k=3,
    )

    print_section("RAG QUERY")

    print(rag_result.query)

    print_section("RETRIEVED MEDICAL EVIDENCE")

    print(
        rag_result.model_dump_json(
            indent=2
        )
    )

    # ==========================================================
    # 5. STATE ENCODER
    # ==========================================================

    encoder = StateEncoder()

    rl_state = encoder.encode(
        clinical_state
    )

    print_section("RL STATE")

    print(
        rl_state.model_dump_json(
            indent=2
        )
    )

    print("\nRL STATE TUPLE:")

    print(
        rl_state.to_tuple()
    )

    # ==========================================================
    # 6. ACTION SPACE
    # ==========================================================

    print_section("ACTION SPACE")

    for action in ClinicalAction:

        print(
            f"{action.value} -> {action.name}"
        )

    # ==========================================================
    # 7. CLINICAL ENVIRONMENT
    # ==========================================================

    environment = ClinicalEnvironment()

    initial_state = environment.reset(
        rl_state
    )

    print_section("ENVIRONMENT")

    print("Initial state:")

    print(
        initial_state.to_tuple()
    )

    # ==========================================================
    # 8. TEST ONE ACTION
    # ==========================================================

    action = (
        ClinicalAction.REQUEST_DIAGNOSTIC_IMAGING
    )

    result = environment.step(
        action
    )

    print("\nSelected action:")

    print(
        action.name
    )

    print("\nReward:")

    print(
        result.reward
    )

    print("\nNext state:")

    print(
        result.state.to_tuple()
    )

    print("\nEpisode done:")

    print(
        result.done
    )

    # ==========================================================
    # 9. TEST ALL ACTIONS
    # ==========================================================

    print_section("ACTION / REWARD TEST")

    for action in ClinicalAction:

        environment.reset(
            rl_state
        )

        result = environment.step(
            action
        )

        print(
            f"{action.name:35s}"
            f" -> reward = {result.reward}"
        )

            # ==========================================================
    # 10. Q-LEARNING
    # ==========================================================

    print_section("Q-LEARNING TRAINING")

    q_agent = QLearningAgent(
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    )

    training_rewards = q_agent.train(
        environment=environment,
        initial_state=rl_state,
        episodes=1000,
    )

    print("Training completed.")

    print(
        f"Number of episodes: "
        f"{len(training_rewards)}"
    )

    print(
        f"Final epsilon: "
        f"{q_agent.epsilon:.4f}"
    )

    # ==========================================================
    # 11. Q-VALUES
    # ==========================================================

    print_section("LEARNED Q-VALUES")

    q_values = q_agent.get_q_values(
        rl_state
    )

    for action, value in q_values.items():

        print(
            f"{action.name:35s}"
            f" -> Q = {value:.4f}"
        )

    # ==========================================================
    # 12. BEST ACTION
    # ==========================================================

    best_action = q_agent.best_action(
        rl_state
    )

    print_section("BEST ACTION")

    print(
        f"Recommended action according to "
        f"the learned Q-table:"
    )

    print(
        best_action.name
    )

    print(
        f"Q-value: "
        f"{q_values[best_action]:.4f}"
    )

    # ==========================================================
    # 13. TRAINING REWARD
    # ==========================================================

    print_section("TRAINING REWARD")

    print(
        f"First episode reward: "
        f"{training_rewards[0]:.2f}"
    )

    print(
        f"Last episode reward: "
        f"{training_rewards[-1]:.2f}"
    )

    # Average reward over last 100 episodes

    last_100 = training_rewards[-100:]

    average_reward = (
        sum(last_100)
        / len(last_100)
    )

    print(
        f"Average reward "
        f"(last 100 episodes): "
        f"{average_reward:.2f}"
    )


if __name__ == "__main__":
    main()