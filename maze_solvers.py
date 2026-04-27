import numpy as np
from tqdm.auto import trange

class ValueIteration:
    def __init__(self, env, discount=0.9):
        self.env = env
        self.discount = discount
        self.state_values = np.zeros(env.snum)

    def bellman_equation(self, state, action_space=[0, 1, 2, 3], discount_factor=0.9):
        """
        V(s) = max_a [ R(s, a) + γ * Σ P(s'|s,a) * V(s') ]
        """
        action_values = np.zeros(len(action_space))

        for action in action_space:
            expected_reward, expected_next_value = self.get_expected_reward_and_next_state(state, action)
            action_values[action] = expected_reward + discount_factor * expected_next_value

        return np.max(action_values)

    def get_expected_reward_and_next_state(self, state, action):
        """Returns (expected_reward, expected_next_state_value) under the slip model."""
        slip_chance = self.env.slip

        # Intended transition
        self.env.slip = 0.0
        reward, next_state, _ = self.env.step(state, action)
        self.env.slip = slip_chance

        # Slip transition
        self.env.slip = 1.0
        reward_slip, next_state_slip, _ = self.env.step(state, action)
        self.env.slip = slip_chance

        expected_reward = reward * (1 - slip_chance) + reward_slip * slip_chance
        expected_next_value = self.state_values[next_state] * (1 - slip_chance) + self.state_values[next_state_slip] * slip_chance

        return expected_reward, expected_next_value

    def learn(self, max_iterations = 1000, tolerance = 1e-4):
        steps = trange(max_iterations)
        for i in steps:
            og_state_values = self.state_values.copy()

            for s in range(self.env.snum):
                self.state_values[s] = self.bellman_equation(s)

            # Check for convergence
            if np.max(np.abs(self.state_values - og_state_values)) < tolerance:
                print(f"Converged at iteration {i}")
                steps.colour = 'green' # Making it pretty
                break


class QLearning:
    def __init__(self, env, discount=0.9, learning_rate=0.1, exploration_rate=0.1):
        self.env = env
        self.discount = discount
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.q_lookup = np.zeros((env.snum, env.anum))

    def learn(self, num_episodes=5000):
        for _ in trange(num_episodes):
            state = np.random.randint(self.env.snum)  # randomize start to cover all flag-state combos
            done = False

            while not done:
                # ε-greedy action selection
                if np.random.rand() < self.exploration_rate:
                    action = np.random.randint(self.env.anum)
                else:
                    action = np.argmax(self.q_lookup[state])

                reward, next_state, done = self.env.step(state, action)

                # Q-learning update
                self.q_lookup[state, action] += self.learning_rate * (
                    reward + self.discount * np.max(self.q_lookup[next_state]) - self.q_lookup[state, action]
                )

                state = next_state