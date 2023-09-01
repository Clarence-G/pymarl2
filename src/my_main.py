import json
import os

from run.ea import EAPopulation, EA
from run.my_run import conv_args
from utils import logging

_log = logging.get_logger()

def ea_ours():
    config = json.load(open("run/running_args.json", "r"))
    args = conv_args(config, _log)
    filename = "run/ea_seq_5.pkl"
    log_path = "ea_log_5.txt"
    q = EAPopulation(args)
    # if filename exists, load from file else create new
    if os.path.exists(filename):
        ea_full = EA.load_from_file(filename)
    else:
        ea_full = EA(args, pop_size=1, mutation_rate=0.1, crossover_rate=0.1, file=log_path)

    time = 60 * 60 * 8
    ea_full.evolve(200000, filename, time)


def drl():
    config = json.load(open("run/running_args.json", "r"))
    args = conv_args(config, _log)
    filename = "run/drl_4.pkl"
    log_path = "drl_log_4.txt"
    # if filename exists, load from file else create new
    if os.path.exists(filename):
        ea_full = EA.load_from_file(filename)
    else:
        ea_full = EA(args, pop_size=1, mutation_rate=0.1, crossover_rate=0.1, file=log_path)

    d = 60 * 60 * 8
    ea_full.drl(filename, d)


def random_action():
    config = json.load(open("run/running_args.json", "r"))
    args = conv_args(config, _log)
    filename = "random_2.pkl"
    log_path = "random_log_2.txt"
    # if filename exists, load from file else create new
    if os.path.exists(filename):
        ea_random = EA.load_from_file(filename)
    else:
        ea_random = EA(args, pop_size=10, mutation_rate=0.1, crossover_rate=0.1, file=log_path)
    ea_random.random_action()

def ea_test_s():
    config = json.load(open("run/running_args.json", "r"))
    args = conv_args(config, _log)
    filename = "ea_s.pkl"
    log_path = "ea_s_log.txt"
    # if filename exists, load from file else create new
    if os.path.exists(filename):
        ea_s = EA.load_from_file(filename)
    else:
        ea_s = EA(args, pop_size=10, mutation_rate=0.1, crossover_rate=0.1, file=log_path)
    ea_s.ea_s(20, filename)

def ea_test_s_drl():
    config = json.load(open("run/running_args.json", "r"))
    args = conv_args(config, _log)
    filename = "ea_s_drl_1.pkl"
    log_path = "ea_s_drl_log_1.txt"
    # if filename exists, load from file else create new
    if os.path.exists(filename):
        ea_s = EA.load_from_file(filename)
    else:
        ea_s = EA(args, pop_size=10, mutation_rate=0.1, crossover_rate=0.1, file=log_path)
    ea_s.ea_s(20, filename)


def ea_test_m():
    config = json.load(open("run/running_args.json", "r"))
    args = conv_args(config, _log)
    filename = "ea_m_2.pkl"
    log_path = "ea_m_log_2.txt"
    # if filename exists, load from file else create new
    if os.path.exists(filename):
        ea_m = EA.load_from_file(filename)
    else:
        ea_m = EA(args, pop_size=10, mutation_rate=0.1, crossover_rate=0.1, file=log_path)
    ea_m.ea_m(20, filename)


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
    drl()