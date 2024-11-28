import random
import numpy as np
import torch


class GA:
    """
    GA Class that uses Simple Perceptron network
    """
    def __init__(self, env, generations=1000, mutation_rate=0.05, crossover_rate=0.5, population_size=30, episodes_per_individual=5):
        """
        Initialisation of the class
        :param env: Environment
        :param generations: Number of generations to train
        :param mutation_rate: Probability of mutation
        :param crossover_rate: Probability of Crossover
        :param population_size: Size of the Population
        :param episodes_per_individual: Number of episodes Fitness function will run to evaluate genotype
        """
        self.env = env
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.episodes_per_individual = episodes_per_individual
        self.population = self.generate_population()

    def generate_population(self):
        """
        Generates population of random genotypes
        :return: population of genotypes. Size of each Genotype = 36
        """
        # Initialize the population with random weights and biases
        population = []
        for _ in range(self.population_size):
            genotype = np.concatenate([
                np.random.randn(8, 4).flatten(),  # Weights (8 inputs, 4 outputs)
                np.zeros(4)  # Biases (4 outputs) as zeros
            ])
            population.append(genotype)
        return np.array(population)

    def predict_action(self, genotype, state):
        """

        :param genotype: Current Genotype
        :param state: Current State
        :return: action (integer in range 0-3)
        """
        # Extract weights and biases from the genotype
        weights = genotype[:32].reshape(8, 4)  # First 32 values are weights
        biases = genotype[32:]  # Remaining 4 values are biases
        state = np.array(state).reshape(1, -1)

        # Calculate scores for each action using weights and biases
        scores = np.tanh(np.dot(state, weights) + biases)
        # Find index with highest value
        action = np.argmax(scores)

        return action

    def fitness(self, genotype):
        """
        Evaluate current genotype with average reward throughout multiple episodes
        :param genotype: Genotype to evaluate
        :return: average reward
        """
        # Holds rewards from multiple runs
        global_reward = 0
        # Calculate reward for each episode
        for _ in range(self.episodes_per_individual):
            state = self.env.reset()[0]  # Reset environment and return current state
            episode_reward = 0
            done = False
            # While not Lunar Lander is crashed and within window - predict action and calculate reward
            while not done:
                action = self.predict_action(genotype, state)  # Predict action based on current genotype and state
                state, reward, terminated, truncated, _ = self.env.step(action)  # Execute action and return next state and reward
                episode_reward += reward
                if terminated or truncated:
                    done = True

            global_reward += episode_reward

        return global_reward / self.episodes_per_individual

    def crossover(self, parent1, parent2):
        """
        Perform genotypes crossover between winner and loser genotypes
        :param parent1: Winner Genotype
        :param parent2: Loser Genotype
        :return: Modified Loser Genotype
        """
        child = parent1.copy()
        for i in range(len(parent1)):
            # If random value is within crossover rate - copy gene of the winner into loser gene
            if random.random() < self.crossover_rate:
                child[i] = parent2[i]
        return child

    def mutate(self, genotype, mutation_rate):
        """
        Perform random mutation of Loser Genotype
        :param genotype: Loser Genotype
        :param mutation_rate: Current Mutation rate
        :return: Mutated Genotype
        """
        for gene_i in range(len(genotype)):   # for each gene in the genotype
            # If random value is within mutation rate - Generate random value from Gaussian Distribution
            # and add it to the current gene
            if random.random() < mutation_rate:
                mutation = np.random.normal(0, 0.1)  # 0 is the mean that centres distribution around 0, 0.1 is std
                genotype[gene_i] += mutation
        return genotype

    def run(self):
        """
        Training loop
        :return: Best performing Genotype and list of top rewards to evaluate training process
        """
        top_fit = -1000  # Initial top fitness value
        top_geno = None
        fitnesses = []  # List that holds top fitness's

        for generation in range(self.generations):
            # Find two distinct indices within population
            index1 = np.random.randint(0, self.population_size)
            index2 = np.random.randint(0, self.population_size)
            while index2 == index1:
                index2 = np.random.randint(0, self.population_size)

            # Apply fitness function on each genotype with chosen index
            fit1 = self.fitness(self.population[index1])
            fit2 = self.fitness(self.population[index2])

            # Evaluate both fitnesses and assign genotype with lower fitness as a loser and second genotype as winner
            if fit1 > fit2:
                winner, loser = index1, index2
                if fit1 > top_fit:
                    top_fit = fit1
                    top_geno = self.population[winner].copy()
            else:
                winner, loser = index2, index1
                if fit2 > top_fit:
                    top_fit = fit2
                    top_geno = self.population[winner].copy()

            # Perform crossover and mutation on Loser Genotype
            self.population[loser] = self.crossover(self.population[winner], self.population[loser])
            self.population[loser] = self.mutate(self.population[loser], self.mutation_rate)

            # Print training performance so far
            if generation % 100 == 0:
                print(f"Generation {generation}/{self.generations}, Top Reward {top_fit}, Mutation Rate {self.mutation_rate}")

            # Reduce mutation rate after each generation
            if self.mutation_rate > 0.2:
                self.mutation_rate = self.mutation_rate * 0.9999
            fitnesses.append(top_fit)

        return top_geno, fitnesses


class GaWithKnn:
    """
    Same as GA, but uses K-Nearest Neighbour technique during training
    """
    def __init__(self, env, generations=1000, mutation_rate=0.05, crossover_rate=0.5, population_size=30, episodes_per_individual=5, k=3):
        self.env = env
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.episodes_per_individual = episodes_per_individual
        self.population = self.generate_population()
        self.k = k

    def generate_population(self):
        # Initialize the population with random weights and biases
        population = []
        for _ in range(self.population_size):
            genotype = np.concatenate([
                np.random.randn(8, 4).flatten(),  # Weights (8 inputs, 4 outputs)
                np.random.randn(4)  # Biases (4 output nodes)
            ])
            population.append(genotype)
        return np.array(population)

    def predict_action(self, genotype, state):
        # Extract weights and biases from the genotype
        weights = genotype[:32].reshape(8, 4)  # First 32 values are weights
        biases = genotype[32:]  # Remaining 4 values are biases
        state = np.array(state).reshape(1, -1)

        # Calculate scores for each action using weights and biases
        scores = np.tanh(np.dot(state, weights) + biases)
        action = np.argmax(scores)

        return action

    def fitness(self, genotype):
        # Evaluate fitness over multiple episodes
        global_reward = 0

        for _ in range(self.episodes_per_individual):
            state = self.env.reset()[0]
            episode_reward = 0
            done = False
            while not done:
                action = self.predict_action(genotype, state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    done = True

            global_reward += episode_reward

        return global_reward / self.episodes_per_individual

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        for i in range(len(parent1)):
            if random.random() < self.crossover_rate:
                child[i] = parent2[i]
        return child

    def mutate(self, genotype, mutation_rate):
        for gene_i in range(len(genotype)):
            if random.random() < mutation_rate:
                genotype[gene_i] += np.random.normal(0, 0.1)
        return genotype

    def run(self):
        """
        Training loop with KNN implementation
        :return:
        """
        top_fit = -1000
        top_geno = None
        fitnesses = []

        for generation in range(self.generations):
            # Randomly select index 1
            index1 = np.random.randint(0, self.population_size)

            # Define the range for index2 within k distance from index1
            min_bound = max(0, index1 - self.k)
            max_bound = min(self.population_size - 1, index1 + self.k)

            # Select index2 within the range [min_bound, max_bound] but not equal to index1
            index2 = np.random.randint(min_bound, max_bound + 1)
            while index2 == index1:
                index2 = np.random.randint(min_bound, max_bound + 1)

            fit1 = self.fitness(self.population[index1])
            fit2 = self.fitness(self.population[index2])

            if fit1 > fit2:
                winner, loser = index1, index2
                if fit1 > top_fit:
                    top_fit = fit1
                    top_geno = self.population[winner].copy()
            else:
                winner, loser = index2, index1
                if fit2 > top_fit:
                    top_fit = fit2
                    top_geno = self.population[winner].copy()

            self.population[loser] = self.crossover(self.population[winner], self.population[loser])
            self.population[loser] = self.mutate(self.population[loser], self.mutation_rate)

            if generation % 100 == 0:
                print(f"Generation {generation}/{self.generations}, Top Reward {top_fit}, Mutation Rate {self.mutation_rate}")

            if self.mutation_rate > 0.2:
                self.mutation_rate = self.mutation_rate * 0.9998
            fitnesses.append(top_fit)

        return top_geno, fitnesses


class GaWith2Layers:
    """
    GA implementation with deeper NN using 2 hidden layers
    """
    def __init__(self, env, generations=1000, mutation_rate=0.05, crossover_rate=0.5, population_size=30, episodes_per_individual=5):
        self.env = env
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.episodes_per_individual = episodes_per_individual
        self.population = self.generate_population()

    def generate_population(self):
        """
        Generate Population of DNN genotypes
        :return: population of DNN genotypes. Size of each Genotype = 226
        """
        population = []
        for _ in range(self.population_size):
            # Input Layer
            first_layer_weights = np.random.randn(8, 8).flatten()  # Weights (8 inputs, 8 outputs)
            first_layer_biases = np.zeros(8)  # Biases (8 outputs)

            # First Hidden Layer
            second_layer_weights = np.random.randn(8, 8).flatten()  # Weights (8 inputs, 8 outputs)
            second_layer_biases = np.zeros(8)  # Biases (8 outputs)

            # Second Hidden Layer
            third_layer_weights = np.random.randn(8, 6).flatten()  # Weights (8 inputs, 6 outputs)
            third_layer_biases = np.zeros(6)  # Biases (6 outputs)

            # Output Layer
            output_layer_weights = np.random.randn(6, 4).flatten()  # Weights (6 inputs, 4 outputs)
            output_layer_biases = np.zeros(4)  # Biases (4 outputs)

            # Concatenate all parameters into one genotype
            genotype = np.concatenate([
                first_layer_weights, first_layer_biases,
                second_layer_weights, second_layer_biases,
                third_layer_weights, third_layer_biases,
                output_layer_weights, output_layer_biases
            ])
            population.append(genotype)
        return np.array(population)

    def predict_action(self, genotype, state):
        # Set a common dtype for all tensors
        dtype = torch.float32

        # Extract weights and biases from the genotype
        weights1 = torch.tensor(genotype[:64].reshape(8, 8), dtype=dtype)
        biases1 = torch.tensor(genotype[64:72], dtype=dtype)

        weights2 = torch.tensor(genotype[72:136].reshape(8, 8), dtype=dtype)
        biases2 = torch.tensor(genotype[136:144], dtype=dtype)

        weights3 = torch.tensor(genotype[144:192].reshape(8, 6), dtype=dtype)
        biases3 = torch.tensor(genotype[192:198], dtype=dtype)

        weights4 = torch.tensor(genotype[198:222].reshape(6, 4), dtype=dtype)
        biases4 = torch.tensor(genotype[222:226], dtype=dtype)

        # Convert state to tensor and ensure correct dtype
        state = torch.tensor(state, dtype=dtype).reshape(1, -1)

        # Calculate scores for each action using weights and biases, and tanh as activation function
        scores = torch.tanh(torch.mm(state, weights1) + biases1)
        scores = torch.tanh(torch.mm(scores, weights2) + biases2)
        scores = torch.tanh(torch.mm(scores, weights3) + biases3)
        scores = torch.tanh(torch.mm(scores, weights4) + biases4)

        # Convert scores to numpy array and get the action with the highest score
        action = np.argmax(scores.numpy())

        return action

    def fitness(self, genotype):
        # Evaluate fitness over multiple episodes
        global_reward = 0

        for _ in range(self.episodes_per_individual):
            state = self.env.reset()[0]
            episode_reward = 0
            done = False
            while not done:
                action = self.predict_action(genotype, state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    done = True

            global_reward += episode_reward

        return global_reward / self.episodes_per_individual

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        for i in range(len(parent1)):
            if random.random() < self.crossover_rate:
                child[i] = parent2[i]
        return child

    def mutate(self, genotype, mutation_rate):
        for gene_i in range(len(genotype)):
            if random.random() < mutation_rate:
                genotype[gene_i] += np.random.normal(0, 0.1)
        return genotype

    def run(self):
        top_fit = -1000
        top_geno = None
        fitnesses = []

        for generation in range(self.generations):
            index1 = np.random.randint(0, self.population_size)
            index2 = np.random.randint(0, self.population_size)
            while index2 == index1:
                index2 = np.random.randint(0, self.population_size)

            fit1 = self.fitness(self.population[index1])
            fit2 = self.fitness(self.population[index2])

            if fit1 > fit2:
                winner, loser = index1, index2
                if fit1 > top_fit:
                    top_fit = fit1
                    top_geno = self.population[winner].copy()
            else:
                winner, loser = index2, index1
                if fit2 > top_fit:
                    top_fit = fit2
                    top_geno = self.population[winner].copy()

            if top_fit >= 200:
                self.population[loser] = self.crossover(top_geno, self.population[loser]).copy()
                self.population[loser] = self.mutate(self.population[loser], self.mutation_rate).copy()
            else:
                self.population[loser] = self.crossover(self.population[winner], self.population[loser]).copy()
                self.population[loser] = self.mutate(self.population[loser], self.mutation_rate).copy()

            if generation % 100 == 0:
                print(f"Generation {generation}/{self.generations}, Top Reward {top_fit}, mutation rate {self.mutation_rate}")

            self.mutation_rate = self.mutation_rate * 0.999

            fitnesses.append(top_fit)
        return top_geno, fitnesses


class GaWith1Layer:
    """
    GA implementation with deeper NN using 1 hidden layer
    """
    def __init__(self, env, generations=1000, mutation_rate=0.05, crossover_rate=0.5, population_size=30, episodes_per_individual=5):
        self.env = env
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.episodes_per_individual = episodes_per_individual
        self.population = self.generate_population()

    def generate_population(self):
        """
        Generate Population of DNN genotypes
        :return: population of DNN genotypes. Size of each Genotype = 154
        """
        population = []
        for _ in range(self.population_size):
            # Input Layer
            first_layer_weights = np.random.randn(8, 8).flatten()  # Weights (8 inputs, 8 outputs)
            first_layer_biases = np.zeros(8)  # Biases (8 outputs)

            # Hidden Layer
            second_layer_weights = np.random.randn(8, 6).flatten()  # Weights (8 inputs, 6 outputs)
            second_layer_biases = np.zeros(6)  # Biases (6 outputs)

            # Output Layer
            output_layer_weights = np.random.randn(6, 4).flatten()  # Weights (6 inputs, 4 outputs)
            output_layer_biases = np.zeros(4)  # Biases (4 outputs)

            # Concatenate all parameters into one genotype
            genotype = np.concatenate([
                first_layer_weights, first_layer_biases,
                second_layer_weights, second_layer_biases,
                output_layer_weights, output_layer_biases
            ])
            population.append(genotype)
        return np.array(population)

    def predict_action(self, genotype, state):
        # Set a common dtype for all tensors
        dtype = torch.float32

        # Extract weights and biases from the genotype
        weights1 = torch.tensor(genotype[:64].reshape(8, 8), dtype=dtype)
        biases1 = torch.tensor(genotype[64:72], dtype=dtype)

        weights2 = torch.tensor(genotype[72:120].reshape(8, 6), dtype=dtype)
        biases2 = torch.tensor(genotype[120:126], dtype=dtype)

        weights3 = torch.tensor(genotype[126:150].reshape(6, 4), dtype=dtype)
        biases3 = torch.tensor(genotype[150:154], dtype=dtype)

        # Convert state to tensor and ensure correct dtype
        state = torch.tensor(state, dtype=dtype).reshape(1, -1)

        # Calculate scores for each action using weights and biases, and tanh as activation function
        scores = torch.tanh(torch.mm(state, weights1) + biases1)
        scores = torch.tanh(torch.mm(scores, weights2) + biases2)
        scores = torch.tanh(torch.mm(scores, weights3) + biases3)

        # Convert scores to numpy array and get the action with the highest score
        action = np.argmax(scores.numpy())

        return action

    def fitness(self, genotype):
        # Evaluate fitness over multiple episodes
        global_reward = 0

        for _ in range(self.episodes_per_individual):
            state = self.env.reset()[0]
            episode_reward = 0
            done = False
            while not done:
                action = self.predict_action(genotype, state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    done = True

            global_reward += episode_reward

        return global_reward / self.episodes_per_individual

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        for i in range(len(parent1)):
            if random.random() < self.crossover_rate:
                child[i] = parent2[i]
        return child

    def mutate(self, genotype, mutation_rate):
        for gene_i in range(len(genotype)):
            if random.random() < mutation_rate:
                genotype[gene_i] += np.random.normal(0, 0.1)
        return genotype

    def run(self):
        top_fit = -1000
        top_geno = None
        fitnesses = []

        for generation in range(self.generations):
            index1 = np.random.randint(0, self.population_size)
            index2 = np.random.randint(0, self.population_size)
            while index2 == index1:
                index2 = np.random.randint(0, self.population_size)

            fit1 = self.fitness(self.population[index1])
            fit2 = self.fitness(self.population[index2])

            if fit1 > fit2:
                winner, loser = index1, index2
                if fit1 > top_fit:
                    top_fit = fit1
                    top_geno = self.population[winner].copy()
            else:
                winner, loser = index2, index1
                if fit2 > top_fit:
                    top_fit = fit2
                    top_geno = self.population[winner].copy()

            if top_fit >= 200:
                self.population[loser] = self.crossover(top_geno, self.population[loser]).copy()
                self.population[loser] = self.mutate(self.population[loser], self.mutation_rate).copy()
            else:
                self.population[loser] = self.crossover(self.population[winner], self.population[loser]).copy()
                self.population[loser] = self.mutate(self.population[loser], self.mutation_rate).copy()

            if generation % 100 == 0:
                print(f"Generation {generation}/{self.generations}, Top Reward {top_fit}, mutation rate {self.mutation_rate}")

            self.mutation_rate = self.mutation_rate * 0.9995

            fitnesses.append(top_fit)
        return top_geno, fitnesses
