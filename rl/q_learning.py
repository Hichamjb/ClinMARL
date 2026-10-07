from collections import defaultdict
import random

from rl.actions import ClinicalAction
from rl.environment import ClinicalEnvironment
from rl.state_encoder import RLState


class QLearningAgent:

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
    ):
        """
        Q-Learning agent.

        Parameters
        ----------
        alpha : float
            Learning rate.

        gamma : float
            Discount factor.

        epsilon : float
            Initial exploration probability.

        epsilon_decay : float
            Decay applied to epsilon after each episode.

        epsilon_min : float
            Minimum exploration probability.
        """

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table:
        # state -> action -> Q-value
        self.q_table = defaultdict(
            lambda: {
                action: 0.0
                for action in ClinicalAction
            }
        )

    # ==========================================================
    # GET Q VALUES
    # ==========================================================

    def get_q_values(self, state: RLState):
        """
        Return Q-values for all actions in a state.
        """

        state_key = state.to_tuple()

        return self.q_table[state_key]

    # ==========================================================
    # BEST ACTION
    # ==========================================================

    def best_action(self, state: RLState) -> ClinicalAction:
        """
        Return the action with the highest Q-value.
        """

        state_key = state.to_tuple()

        q_values = self.q_table[state_key]

        return max(
            q_values,
            key=q_values.get
        )

    # ==========================================================
    # EPSILON-GREEDY ACTION
    # ==========================================================

    def choose_action(self, state: RLState) -> ClinicalAction:
        """
        Epsilon-greedy action selection.

        With probability epsilon:
            explore

        Otherwise:
            exploit the best known action.
        """

        if random.random() < self.epsilon:

            return random.choice(
                list(ClinicalAction)
            )

        return self.best_action(state)

    # ==========================================================
    # UPDATE Q VALUE
    # ==========================================================

    def update(
        self,
        state: RLState,
        action: ClinicalAction,
        reward: float,
        next_state: RLState,
        done: bool,
    ):
        """
        Q-Learning update rule:

        Q(s,a) <- Q(s,a) +
                  alpha * [
                      reward
                      + gamma * max Q(s',a')
                      - Q(s,a)
                  ]
        """

        state_key = state.to_tuple()
        next_state_key = next_state.to_tuple()

        current_q = self.q_table[
            state_key
        ][action]

        if done:

            target = reward

        else:

            max_next_q = max(
                self.q_table[
                    next_state_key
                ].values()
            )

            target = (
                reward
                + self.gamma * max_next_q
            )

        new_q = (
            current_q
            + self.alpha
            * (target - current_q)
        )

        self.q_table[
            state_key
        ][action] = new_q

    # ==========================================================
    # EPSILON DECAY
    # ==========================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    # ==========================================================
    # TRAIN
    # ==========================================================

    def train(
        self,
        environment: ClinicalEnvironment,
        initial_state: RLState,
        episodes: int = 1000,
    ):
        """
        Train the Q-Learning agent.

        Returns
        -------
        list
            Total reward obtained in each episode.
        """

        episode_rewards = []

        for episode in range(episodes):

            state = environment.reset(
                initial_state
            )

            total_reward = 0.0

            done = False

            while not done:

                action = self.choose_action(
                    state
                )

                result = environment.step(
                    action
                )

                self.update(
                    state=state,
                    action=action,
                    reward=result.reward,
                    next_state=result.state,
                    done=result.done,
                )

                state = result.state

                total_reward += result.reward

                done = result.done

            self.decay_epsilon()

            episode_rewards.append(
                total_reward
            )

        return episode_rewards