from pathlib import Path
import sys

for i in range(3):
    path_root = Path(__file__).parents[i]
    sys.path.append(str(path_root))

import copy
import json
import os
import pickle
import random

import torch

from my_run import init_learner, agent_evaluate_sequential, agent_train_sequential, conv_args
from params import AgentParam
from utils import logging
from pareto import cal_all_pareto_frontier


_log = logging.get_logger()
logger = logging.Logger(_log)


def flatten_parameters(agent):
    return torch.cat([pu.view(-1) for pu in agent.parameters()])


def cd_select(population, n):
    return random.sample(population, n)


def pop_select(population, n, func):
    fronts = cal_all_pareto_frontier(population, func)
    new_p = []
    for front in fronts:
        if len(new_p) + len(front) <= n:
            new_p.extend(front)
        else:
            new_p.extend(cd_select(front, n - len(new_p)))

    return new_p


class EAPopulation:
    def __init__(self, args):
        self.score = -1
        self.reward = -1
        self.states = -1
        self.seq_num = -1
        self.train_cnt = 0
        self.mutation_cnt = 0
        self.args = args
        learner, _, _ = init_learner(args, logger)
        self.agent_param = AgentParam()
        self.agent_param.save_params(learner)
        self.agent_flat = flatten_parameters(self.agent_param.agent)

    def with_param(self, agent_param: AgentParam):
        self.agent_flat = flatten_parameters(agent_param.agent)
        self.agent_param = agent_param
        return self

    def __gt__(self, other):
        return self.reward > other.reward and self.states > other.states

    def __ge__(self, other):
        return self.reward >= other.reward and self.states >= other.states

    def train(self, epoch=0):
        self.unflatten_parameters()
        args_tmp = copy.deepcopy(self.args)
        if epoch > 0:
            args_tmp.t_max = epoch
        post_param = agent_train_sequential(args_tmp, logger, self.agent_param)
        self.agent_param = post_param
        self.train_cnt += 1
        self.agent_flat = flatten_parameters(self.agent_param.agent)

    def evaluate(self, epoch=0, file=None):
        self.unflatten_parameters()
        args_tmp = copy.deepcopy(self.args)
        if epoch > 0:
            args_tmp.test_nepisode = epoch

        self.states, self.reward, self.seq_num = agent_evaluate_sequential(args_tmp, logger, self.agent_param)
        self.score = self.reward * 50 + self.states
        # save len(state_set) and avg_reward   to file
        if file is not None:
            with open(file, "a+") as f:
                f.write(str(self) + "\n")

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

        # Return a new EAPopulation instance with the crossover agent
        crossover_population = self.clone()
        crossover_population.agent_flat = new_agent_flat
        crossover_population.unflatten_parameters()

        return crossover_population

    def clone(self):
        # Create a new EAPopulation instance with the same parameters
        clone_population = EAPopulation(self.args)
        clone_population.agent_flat = self.agent_flat.clone()
        clone_population.agent_param = self.agent_param.clone()
        clone_population.score = self.score
        clone_population.reward = self.reward
        clone_population.states = self.states
        clone_population.train_cnt = self.train_cnt
        clone_population.mutation_cnt = self.mutation_cnt
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
               f"mutation_cnt: {self.mutation_cnt}, seq_num: {self.seq_num}"


class EA:
    def __init__(self, args, pop_size, mutation_rate, crossover_rate, file=None):
        self.populations = []
        self.offspring_tmp = []
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        for i in range(self.pop_size):
            population = EAPopulation(args)
            population.evaluate(file=file)
            self.populations.append(population)
        print("############ init finished ############")

    def mutation_train(self, num_epochs):
        self.offspring_tmp = []
        for i in range(num_epochs):
            print("############ mutation train epoch: ", i, " ############")
            for p in self.populations:
                p.train()
                p.evaluate(file="mutation_train.txt")
                if random.random() < self.mutation_rate:
                    p = p.random_mutation()
                    p.evaluate(file="mutation_train.txt")
                self.offspring_tmp.append(p)
            self.populations = self.offspring_tmp
            self.offspring_tmp = []

    def random_choice(self):
        return random.choice(self.populations)

    def random_pareto_selection(self):
        x1, x2 = self.random_choice(), self.random_choice()
        x = random.choice([x1, x2])
        if x1 >= x2:
            x = x1
        elif x2 >= x1:
            x = x2

        return x

    def cross_mutate(self):
        b = []
        for i in range(self.pop_size):
            x1, x2 = self.random_pareto_selection(), self.random_pareto_selection()
            x3 = x1.crossover(x2)
            x4 = x3.random_mutation()
            b.append(x4)

        return b

    # save populations, params to file
    def save_to_file(self, path):
        # save obj to file
        with open(path, 'wb+') as f:
            pickle.dump(self, f)

    # load populations, params from file
    @classmethod
    def load_from_file(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def evolve(self, num_generations, path):
        if len(self.offspring_tmp) > 0:
            # load offspring_tmp if it is not empty
            self.populations = pop_select(self.offspring_tmp, self.pop_size, lambda x: (x.reward, x.states))
            self.offspring_tmp = []
        print(self)
        for i in range(num_generations):
            print("############ evolution rounds {} ############: ".format(i))

            offspring_p = self.cross_mutate()
            for pp in offspring_p:
                pp.evaluate()
            print(self)

            self.offspring_tmp = offspring_p
            for i in range(self.pop_size):
                q = random.choice(offspring_p+self.populations).clone()
                q.train()
                q.evaluate()
                self.offspring_tmp.append(q)
                self.save_to_file(path)

            offspring = offspring_p + self.populations
            self.populations = pop_select(offspring, self.pop_size, lambda x: (x.reward, x.states))

            self.save_to_file(path)
            self.offspring_tmp = []

    def __str__(self):
        pop_str = ""
        for p in self.populations:
            pop_str += str(p) + "\n"

        return f"populations: {pop_str}, pop_size: {self.pop_size}, mutation_rate: {self.mutation_rate}, " \
               f"crossover_rate: {self.crossover_rate}"


if __name__ == '__main__':
    config = json.load(open("running_args.json", "r"))
    args = conv_args(config, _log)
    pu = EAPopulation(args)
    filename = "ea_alo_2333.pkl"
    # if filename exists, load from file else create new
    if os.path.exists(filename):
        ea = EA.load_from_file(filename)
    else:
        ea = EA(args, pop_size=10, mutation_rate=0.1, crossover_rate=0.1)
    ea.evolve(10, filename)
    print(ea)
