import copy
import json
import pickle
import random

import torch

from my_run import init_learner, agent_evaluate_sequential, agent_train_sequential, conv_args
from params import AgentParam
import utils.logging

_log = utils.logging.get_logger()
logger = utils.logging.Logger(_log)


def flatten_parameters(agent):
    return torch.cat([pu.view(-1) for pu in agent.parameters()])


class EAPopulation:
    def with_param(self, agent_param: AgentParam):
        self.agent_flat = flatten_parameters(agent_param.agent)
        self.agent_param = agent_param
        return self

    def __init__(self, args):
        self.score = -1
        self.reward = -1
        self.states = -1
        self.train_cnt = 0
        self.mutation_cnt = 0
        self.args = args
        learner, _, _ = init_learner(args, logger)
        self.agent_param = AgentParam()
        self.agent_param.save_params(learner)
        self.agent_flat = flatten_parameters(self.agent_param.agent)

    def train(self, epoch=0):
        self.unflatten_parameters()
        args_tmp = copy.deepcopy(self.args)
        if epoch > 0:
            args_tmp.t_max = epoch
        post_param = agent_train_sequential(args_tmp, logger, self.agent_param)
        self.agent_param = post_param
        self.train_cnt += 1
        self.agent_flat = flatten_parameters(self.agent_param.agent)

    def evaluate(self, epoch=0):
        self.unflatten_parameters()
        args_tmp = copy.deepcopy(self.args)
        if epoch > 0:
            args_tmp.test_nepisode = epoch
        self.states, self.reward = agent_evaluate_sequential(args_tmp, logger, self.agent_param)
        self.score = self.reward * 50 + self.states

    def unflatten_parameters(self):
        flattened = self.agent_flat
        start_idx = 0
        for p in self.agent_param.agent.parameters():
            param_size = p.numel()
            p.data.copy_(flattened[start_idx:start_idx + param_size].view(p.size()))
            start_idx += param_size

    def random_mutation(self, mutation_rate=0.1, perturbation_scale=0.06):
        # Create a copy of the agent's flattened parameters
        mutated_agent_flat = self.agent_flat.clone()

        # Iterate over the flattened parameters and apply mutation
        for i in range(len(mutated_agent_flat)):
            if random.random() < mutation_rate:
                # Apply mutation by adding a small perturbation
                perturbation = torch.randn_like(mutated_agent_flat[i]) * perturbation_scale
                mutated_agent_flat[i] += perturbation

        # Return a new EAPopulation instance with the mutated agent
        mutated_population = self.clone()
        mutated_population.agent_flat = mutated_agent_flat
        mutated_population.unflatten_parameters()
        mutated_population.mutation_cnt += 1
        return mutated_population

    def crossover(self, other_population):
        # Perform crossover by mixing the flattened parameters

        # Determine the crossover point (random index in the flattened parameters)
        crossover_point = random.randint(0, len(self.agent_flat) - 1)

        # Create new agent_flat by combining the parameters of self and other_population
        new_agent_flat = torch.cat((self.agent_flat[:crossover_point], other_population.agent_flat[crossover_point:]))

        # Create a new agent_param with the crossover parameters
        crossover_agent_param = self.agent_param.clone()
        crossover_agent_param.agent.load_state_dict(new_agent_flat)

        # Return a new EAPopulation instance with the crossover agent
        crossover_population = EAPopulation(crossover_agent_param)
        return crossover_population

    def clone(self):
        # Create a new EAPopulation instance with the same agent parameters
        clone_population = copy.deepcopy(self)
        return clone_population

    def save_to_file(self, path):
        # save obj to file
        with open(path, 'wb+') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __str__(self):
        return f"score: {self.score}, reward: {self.reward}, states: {self.states}, train_cnt: {self.train_cnt}, " \
               f"mutation_cnt: {self.mutation_cnt}"


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


if __name__ == '__main__':

    config = json.load(open("running_args.json", "r"))
    args = conv_args(config, _log)
    p = EAPopulation(args)
    filename = "ea_3.pkl"
    # for i in range(3):
    #     print("------- train rounds: ", i)
    #     p.train(epoch=30000-i*5000)
    #     p.evaluate(epoch=64)
    #     print(p)
    #     p.save_to_file(filename)
    p = EAPopulation.load_from_file(filename)
    print(p)
    p1 = p.random_mutation()
    p1.evaluate()
    print(p1)
    p1.train()
    p1.evaluate()
    print(p1)

