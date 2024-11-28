import gymnasium as gym
import numpy as np
from q_learn import QLearn
from qd_learn import DQNAgent
from ga import GA, GaWith2Layers, GaWith1Layer

# Create Environment for training
env = gym.make("LunarLander-v2")

# Q Learn
#q_learn = QLearn(env)
#rewards_q = q_learn.train(20000)  # Returns gained rewards during training

# Microbial GA Agents
# Agent with 1 hidden layer
#ga_agent1 = GaWith1Layer(env, generations=2500, mutation_rate=0.4, crossover_rate=0.5, population_size=30)
#geno1, rewards_ga1 = ga_agent1.run()  # Returns best performing genotype and top rewards during training

# Agent with 2 hidden layers
ga_agent2 = GaWith2Layers(env, generations=2000, mutation_rate=0.4, crossover_rate=0.5, population_size=30)
geno2, rewards_ga2 = ga_agent2.run()  # Returns best performing genotype and top rewards during training

# Simple Perceptron agent
#ga_agent = GA(env, generations=3000, mutation_rate=0.4, crossover_rate=0.5, population_size=50)
#geno, rewards_ga = ga_agent.run()  # Returns best performing genotype and top rewards during training

# DQL Agent
#dqn_agent = DQNAgent(env, buffer_capacity=10000, batch_size=64, learning_rate=0.001,
#                     gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, target_update=10)
#rewards_dq = dqn_agent.train(1000)  # Returns top rewards during training


# Evaluate the trained agent
# Create environment for testing
# param: render_mode="human" allows visualisation of trained Lunar Lander
env = gym.make("LunarLander-v2", render_mode="human")

# Maximum number of steps per episode
max_steps = 1000

# testing GA agents with no hidden layers
for episode in range(15):
    state, _ = env.reset()  # Each episode reset environment
    total_reward = 0
    # Choose action for each step and record reward to total_reward
    for _ in range(max_steps):
        action = ga_agent2.predict_action(geno2, state)  # Predict action based on best chosen genotype and current state
        next_state, reward, terminated, truncated, _ = env.step(action)  # Execute an action and return next state of the environment and reward
        total_reward += reward  # Add step reward to total reward
        state = next_state
        # if Lunar Lander is crashed or beyond allowed window - it ends current episode
        if terminated or truncated:
            break
    # Print Total Reward of each episode
    print(f"{episode} Reward: {total_reward}")
# Close environment
env.close()

## Testing DQN agent
#for episode in range(15):
#    state, _ = env.reset()  # Each episode reset environment
#    total_reward = 0
#    # Choose action for each step and record reward to total_reward
#    for _ in range(max_steps):
#        action = dqn_agent.select_action(state, evaluate=True)  # Predict action based on current state
#        next_state, reward, terminated, truncated, _ = env.step(action)  # Execute an action and return next state of the environment and reward
#        total_reward += reward  # Add step reward to total reward
#        state = next_state
#        # if Lunar Lander is crashed or beyond allowed window - it ends current episode
#        if terminated or truncated:
#            break
#    print(f"{episode} Reward: {total_reward}")
#
#env.close()
#
## Testing Q-Learning agent
#for episode in range(15):
#    state, _ = env.reset()   # Each episode reset environment
#    state = q_learn.discretize_state(state, q_learn.bins)
#    total_reward = 0
#    # Choose action for each step and record reward to total_reward
#    for _ in range(max_steps):
#        action = np.argmax(q_learn.q_table[state])  # Predict action based on current state
#        next_state, reward, terminated, truncated, _ = env.step(action)  # Execute an action and return next state of the environment and reward
#        next_state = q_learn.discretize_state(next_state, q_learn.bins)  # Convert continuous values into discrete values
#        total_reward += reward  # Add step reward to total reward
#        state = next_state
#        # if Lunar Lander is crashed or beyond allowed window - it ends current episode
#        if terminated or truncated:
#            break
#    print(f"{episode} Reward: {total_reward}")
#
#env.close()
#