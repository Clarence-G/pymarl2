import torch
import torch.nn.init as init
import random


class EAPopulation:
    agent = None
    score = 0
    reward = 0
    states = 0

    def __init__(self, agent):
        self.agent = agent

    def flatten_parameters(self):
        return torch.cat([p.view(-1) for p in self.agent.parameters()])

    def unflatten_parameters(self, flattened):
        start_idx = 0
        for p in self.agent.parameters():
            param_size = p.numel()
            p.data.copy_(flattened[start_idx:start_idx + param_size].view(p.size()))
            start_idx += param_size

    def initialize_parameters(self):
        for param in self.agent.parameters():
            if param.dim() > 1:
                init.xavier_uniform_(param)
            else:
                init.constant_(param, 0)

    def save(self, filename):
        """
        Save the population to a file using PyTorch's state_dict format
        """
        torch.save(self.agent.state_dict(), filename)

    def load(self, filename):
        """
        Load the population from a file using PyTorch's state_dict format
        """
        self.agent.load_state_dict(torch.load(filename))

    def random_mutation(self, mutation_prob=0.1, mutation_scale=0.1):
        flattened_params = self.flatten_parameters()
        for i in range(len(flattened_params)):
            if random.uniform(0, 1) < mutation_prob:
                flattened_params[i] += mutation_scale * torch.randn(1)
        child = EAPopulation(self.agent)
        child.unflatten_parameters(flattened_params)

        return child

    def crossover(self, other):
        my_flattened_params = self.flatten_parameters()
        other_flattened_params = other.flatten_parameters()
        crossover_point = random.randint(0, len(my_flattened_params))

        new_flattened_params = torch.cat(
            [my_flattened_params[:crossover_point], other_flattened_params[crossover_point:]])

        child = EAPopulation(self.agent)
        child.unflatten_parameters(new_flattened_params)

        return child


class EA:
    def __init__(self, agent, pop_size, mutation_rate, crossover_rate):
        self.populations = []
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        for i in range(self.pop_size):
            population = EAPopulation(agent)
            population.initialize_parameters()
            self.populations.append(population)

    def add_population(self, population):
        self.populations.append(population)

    def remove_population(self, index):
        del self.populations[index]

    def mutate(self):
        for population in self.populations:
            if random.random() < self.mutation_rate:
                population.random_mutation()

    def crossover(self):
        for i in range(len(self.populations)):
            if random.random() < self.crossover_rate:
                j = random.randint(0, len(self.populations) - 1)
                self.populations[i], self.populations[j] = \
                    self.populations[i].crossover(self.populations[j])

    def evaluate_fitness(self, evaluation_function):
        for population in self.populations:
            population.score, population.states = evaluation_function(population.agent)

    def rank(self):
        self.populations.sort(key=lambda x: x.score, reverse=True)

    def select(self):
        total_fitness = sum([p.score for p in self.populations])
        if total_fitness == 0:
            return random.choice(self.populations)

        r = random.uniform(0, total_fitness)
        running_total = 0

        for population in self.populations:
            running_total += population.score
            if running_total > r:
                return population

    def evolve(self, evaluation_function, num_generations):
        for i in range(num_generations):
            # Evaluate each population
            for population in self.populations:
                population.reward, population.states = evaluation_function(population.agent)
            # Select the top performers as survivors
            # Crossover the survivors
            self.crossover()
            # Mutate the populations
            self.mutate()
            # Print the best reward so far
            print("Generation:", i, "Best Reward:",
                  sorted(self.populations, key=lambda x: x.reward, reverse=True)[0].reward)

