import argparse
import copy
import datetime
import json
import os
import pickle
import pprint
import sys
import time
import threading
import torch as th
from types import SimpleNamespace as SN

import utils.logging
from ea import EA, EAPopulation
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from smac.env import StarCraft2Env
from params import PreParam


def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return 4 + sc_env.shield_bits_ally + sc_env.unit_type_bits


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # sys.exit(0)


def modify_params(runner):
    agent = runner.mac.agent
    # 对 fc1 和 fc2 的参数增加一个随机扰动
    fc1_weights = agent.fc1.weight
    fc1_bias = agent.fc1.bias
    fc2_weights = agent.fc2.weight
    fc2_bias = agent.fc2.bias

    # fc1_weights = torch.nn.parameter.Parameter(fc1_weights + torch.randn_like(fc1_weights) * 0.01,
    #                                            requires_grad=True)
    # fc1_bias = torch.nn.parameter.Parameter(fc1_bias + torch.randn_like(fc1_bias) * 0.01, requires_grad=True)
    fc2_weights = th.nn.parameter.Parameter(fc2_weights + th.randn_like(fc2_weights) * 0.04,
                                            requires_grad=True)
    fc2_bias = th.nn.parameter.Parameter(fc2_bias + th.randn_like(fc2_bias) * 0.04, requires_grad=True)

    # 将修改后的参数设置回模型
    agent.fc1.weight = fc1_weights
    agent.fc1.bias = fc1_bias
    agent.fc2.weight = fc2_weights
    agent.fc2.bias = fc2_bias


def evaluate_sequential(args, runner):
    state_set = set()
    total_reward = 0
    for _ in range(args.test_nepisode):
        batch, para = runner.run(test_mode=True)
        for s in para.state_list:
            state_set.add(tuple(s))
        total_reward += para.reward

    avg_reward = total_reward / args.test_nepisode
    print("state 数量:", len(state_set))
    print("reward:", avg_reward)

    if args.save_replay:
        runner.save_replay()

    # write len(state_set) and avg_reward to file for analysis
    with open("state_reward.txt", "a+") as f:
        f.write(str(len(state_set)) + " " + str(avg_reward) + "\n")

    runner.close_env()
    return len(state_set), avg_reward


def run_sequential(args, logger, preparam: PreParam = None):
    # Init runner so we can get env info

    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    # pre-params for loading
    if preparam:
        preparam.load_params(learner)

    if args.evaluate:
        evaluate_sequential(args, runner)
        return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time

        with th.no_grad():
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")

    postParam = PreParam()
    postParam.save_params(learner)
    return postParam


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config


def my_train_sequential(args, logger, preparam: PreParam = None):
    # copy args
    args = copy.deepcopy(args)
    args.runner = "parallel"
    args.batch_size_run = 8
    learner, runner, buffer = init_learner(args, logger, preparam)

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        with th.no_grad():
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")

    postParam = PreParam()
    postParam.save_params(learner)
    return postParam


def init_learner(args, logger, preparam):
    # Init runner so we can get env info

    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    # pre-params for loading
    if preparam:
        preparam.load_params(learner)

    return learner, runner, buffer


if __name__ == '__main__':
    # load from running_args.json
    args_json = json.load(open("running_args.json", "r"))

    args = argparse.Namespace(**args_json)
    _log = utils.logging.get_logger()
    logger = utils.logging.Logger(_log)
    args.device = "cuda" if args.use_cuda else "cpu"

    logger.console_logger.info("Experiment Parameters:", args_json)

    # configure tensorboard bn
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # configure results logger
    # para = my_train_sequential(args, logger)
    run_id = "11"
    para = PreParam.load_params_from_file(os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "param_{}.pickle".format(run_id)))
    learner, runner, buffer = init_learner(args, logger, para)
    evaluate_sequential(args, runner)
    # my_train_sequential(args, logger, para)

    sys.exit(0)


    eval = False

    # load para from /result/param_id.pickle if exists
    if os.path.exists(
            os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "param_{}.pickle".format(run_id))):
        print("load para from /result/param_{}.pickle".format(run_id))
        with open(
                os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "param_{}.pickle".format(run_id)),
                "rb") as f:
            para = pickle.load(f)
    else:
        print("first time to train, run_id=", run_id)
        para = my_train_sequential(args, logger)
        with open(
                os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "param_{}.pickle".format(run_id)),
                "wb") as f:
            pickle.dump(para, f)

    # eval

    if eval:
        learner, runner, buffer = init_learner(args, logger, para)
        evaluate_sequential(args, runner)

    else:
        for i in range(100):
            para = my_train_sequential(args, logger, para)
            logger.console_logger.info("第{}次训练 : {}".format(i, para.agent.fc1.bias[0:5]))

            # dump para to /result/param_id.pickle
            with open(os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results",
                                   "param_{}.pickle".format(run_id)), "wb") as f:
                pickle.dump(para, f)

            learner, runner, buffer = init_learner(args, logger, para)
            evaluate_sequential(args, runner)

    # ea = EA()
