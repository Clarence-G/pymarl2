import json
import os

from run.ea import EAPopulation, EA
from run.my_run import conv_args
from utils import logging

_log = logging.get_logger()

def ea():
    config = json.load(open("run/running_args.json", "r"))
    args = conv_args(config, _log)
    pu = EAPopulation(args)
    filename = "run/ea_alo_2.pkl"
    # if filename exists, load from file else create new
    if os.path.exists(filename):
        ea = EA.load_from_file(filename)
    else:
        ea = EA(args, pop_size=10, mutation_rate=0.1, crossover_rate=0.1)
    ea.evolve(10, filename)
    print(ea)

def mutation_train():
    config = json.load(open("run/running_args.json", "r"))
    args = conv_args(config, _log)
    filename = "run/mutation_train.pkl"
    # if filename exists, load from file else create new
    if os.path.exists(filename):
        ea = EA.load_from_file(filename)
    else:
        ea = EA(args, pop_size=10, mutation_rate=0.1, crossover_rate=0.1, file="mutation_train.txt")
    ea.mutation_train(10)
    print(ea)


if __name__ == '__main__':
    mutation_train()