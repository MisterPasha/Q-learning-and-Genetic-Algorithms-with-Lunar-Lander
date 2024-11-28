import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


class DQN(nn.Module):
    """
    Neural network model for approximating Q-values
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialise NN with 3 layers
        :param input_dim: Input dimension (state)
        :param output_dim: output dimension (action space)
        """
        super(DQN, self).__init__()
        # First layer
        self.fc1 = nn.Linear(input_dim, 128)
        # Second Layer
        self.fc2 = nn.Linear(128, 128)
        # Third Layer
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        Define forward pass of the network
        :param x: input state
        :return:
        """
        # Apply Relu activation function on each layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """
    replay buffer to store experience tuples for training reinforcement learning agents.
    """

    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        :param capacity (int): The maximum number of experiences the buffer can hold.
        """
        self.buffer = deque(maxlen=capacity)  # Initialize a deque (double-ended queue)

    def push(self, state, action, reward, next_state, done):
        """
        Add an experience tuple to the buffer.

        :param state: The current state
        :param action: The current action taken
        :param reward: The current reward
        :param next_state: The next state resulting from the action
        :param done: A boolean indicating whether the episode has ended.
        """
        # Add experience tuple to the buffer.
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        :param batch_size: The number of experiences to sample.
        :return : A tuple of tensors
        """
        # Randomly sample a batch of experiences from the buffer.
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))

        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        """
        Return the current size of the buffer.
        :return: The number of experiences currently stored in the buffer.
        """
        return len(self.buffer)


class DQNAgent:
    def __init__(self, env, buffer_capacity=10000, batch_size=64, learning_rate=0.001,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, target_update=10):
        self.env = env  # Environment
        self.buffer = ReplayBuffer(buffer_capacity)  # Create Replay Buffer with fixed buffer capacity
        self.batch_size = batch_size  # number of experience tuples sampled from the replay buffer in each training step
        self.gamma = gamma  # discount factor that determines the importance of future rewards rather than immediate rewards
        self.epsilon = epsilon  # Probability of choosing random action
        self.epsilon_min = epsilon_min  # Minimum Epsilon Probability
        self.epsilon_decay = epsilon_decay  # Factor on which Epsilon will decrease each episode
        self.target_update = target_update  # Number of episodes
        self.n_actions = env.action_space.n  # number of actions in action space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        # Policy network, which is the main network used for making decisions
        self.policy_net = DQN(env.observation_space.shape[0], self.n_actions).to(self.device)
        # Target network, which is used for stabilising training
        self.target_net = DQN(env.observation_space.shape[0], self.n_actions).to(self.device)
        # Copy the weights from the policy network to the target network to ensure they start with the same parameters
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Set the target network to evaluation mode
        # This ensures it is used only for inference and not for training updates
        self.target_net.eval()

        # Optimizer using Adam
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def select_action(self, state, evaluate=False):
        """
        Select action either randomly or using Q-Table
        :param state: Current State
        :param evaluate: Boolean that enables the use of random actions. True during training, False during Testing
        :return: Action (integer in range 0-3)
        """

        if not evaluate and random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].item()

    def update_policy(self):
        """
        Updates the policy network by learning from past experiences
        """
        # Check if there are enough samples in the buffer to form a batch
        if len(self.buffer) < self.batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        # Move the sampled experiences to the appropriate device (CPU or GPU)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # Q-values for current states
        q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # Q-values for next states using target network
        next_q_values = self.target_net(next_state).max(1)[0]
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values

        # Compute loss and optimize
        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes):
        """
        Training Loop
        :param num_episodes: Number of episodes to train
        :return: Gained top rewards during training
        """
        rewards = []
        top_rewards = []
        top_reward = - 1000  # Initial top Reward
        for episode in range(num_episodes):
            state, _ = self.env.reset()  # Reset Environment
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)  # Select action based on the current state
                next_state, reward, terminated, truncated, _ = self.env.step(action)  # Execute action
                done = terminated or truncated

                self.buffer.push(state, action, reward, next_state, done)  # Store experience in the buffer
                state = next_state
                total_reward += reward

                # Update policy network
                self.update_policy()

            # Calculate top reward based on average within last 3 rewards
            if len(rewards) >= 3:
                mean_reward = sum([rewards[len(rewards) - 1], rewards[len(rewards) - 2], rewards[len(rewards) - 3]]) / 3
                if mean_reward > top_reward:
                    top_reward = mean_reward
            else:
                if total_reward > top_reward:
                    top_reward = total_reward
            rewards.append(total_reward)
            top_rewards.append(top_reward)

            # Reduce epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if episode % 100 == 0:
                print(f"Episode {episode}/{num_episodes}, Top Reward: {top_reward}, Epsilon: {self.epsilon:.2f}")

        return top_rewards
