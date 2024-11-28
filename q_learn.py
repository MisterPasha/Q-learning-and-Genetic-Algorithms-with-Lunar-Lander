import numpy as np
import collections
import random


class QLearn:
    """
    Simple Q-Learning agent
    """
    def __init__(self, env):
        self.env = env
        self.rewards = []
        # Initialize the Q-table
        self.n_actions = self.env.action_space.n
        self.q_table = collections.defaultdict(lambda: np.zeros(self.n_actions))
        # Discretisation parameters. First two values are the range of values, third is number of bins
        self.bins = [
            np.linspace(-1.5, 1.5, 15),  # x position
            np.linspace(-1.5, 1.5, 15),  # y position
            np.linspace(-5.0, 5.0, 50),  # x velocity
            np.linspace(-5.0, 5.0, 50),  # y velocity
            np.linspace(-np.pi, np.pi, 30),  # angle
            np.linspace(-5.0, 5.0, 50),  # angular velocity
            [0, 1],  # left leg contact
            [0, 1]  # right leg contact
        ]

    # Discrete function
    def discretize_state(self, state, bins):
        """
        Convert continuous values into discrete values using bins
        :param state: current State
        :param bins: Bins
        :return: Tuple of bin indices
        """
        indices = []
        for i, b in enumerate(bins):
            indices.append(np.digitize(state[i], b) - 1)  # Convert state to bin index
        return tuple(indices)

    def epsilon_greedy(self, state, q_table, epsilon):
        """
        Epsilon greedy action selection
        :param state: Current State
        :param q_table: Current Q-Table
        :param epsilon: Current epsilon
        :return: action (integer in range 0-3)
        """
        if random.random() < epsilon:
            return self.env.action_space.sample()  # random Action
        else:
            return np.argmax(q_table[state])

    def train(self, n_episodes):
        """
        Training Loop
        :param n_episodes: Number of episodes
        :return: Rewards gained throughout training
        """
        # Hyperparameters
        max_steps = 1000
        learning_rate = 0.01
        discount_factor = 0.99  # determines the importance of future rewards rather than immediate rewards
        epsilon = 1.0           # Probability of choosing random action
        epsilon_min = 0.05      # Minimum epsilon probability
        epsilon_decay = 0.9999  # Factor on which epsilon decreases
        rewards = []
        top_reward = -1000      # Initial top reward

        # Training loop
        for episode in range(n_episodes):
            state, _ = self.env.reset()   # reset Environment
            state = self.discretize_state(state, self.bins)  # Convert states into discrete values
            total_reward = 0

            for step in range(max_steps):
                action = self.epsilon_greedy(state, self.q_table, epsilon)  # Choose Action
                next_state, reward, terminated, truncated, _ = self.env.step(action)  # Execute Action
                next_state = self.discretize_state(next_state, self.bins)  # Convert next states into discrete values

                best_next_action = np.argmax(self.q_table[next_state])  # Choose action for next step

                # Calculate temporal difference and update Q-Table
                td_target = reward + discount_factor * self.q_table[next_state][best_next_action] * (
                            not terminated and not truncated)
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += learning_rate * td_error

                state = next_state
                total_reward += reward

                # End the episode if Lunar Lander Crashes or beyond allowed window
                if terminated or truncated:
                    break

            if total_reward > top_reward:
                top_reward = total_reward
            rewards.append(top_reward)

            # Decrease Epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if episode % 500 == 0:
                print(f"Episode {episode}/{n_episodes}, Top Reward: {top_reward}")
                self.rewards.append(total_reward)

        # Close the environment
        self.env.close()
        return rewards
