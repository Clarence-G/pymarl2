import numpy as np
import os
import collections
from os.path import dirname, abspath, join
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th

import runners
from utils.logging import get_logger
import yaml

from run import REGISTRY as run_REGISTRY
def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = runners.EpisodeRunner(args, logger)

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

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
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

if __name__ == '__main__':
    args_json = {'action_selector': 'epsilon_greedy',
                 'agent': 'n_rnn',
                 'agent_output_type': 'q',
                 'batch_size': 128,
                 'batch_size_run': 1,
                 'buffer_cpu_only': True,
                 'buffer_size': 5000,
                 'checkpoint_path': 'C:/Users/tryso/py/pymarl2/results/models/qmix_env=8_adam_td_lambda__2023-02-17_15-40-46',
                 'critic_lr': 0.0005,
                 'env': 'sc2',
                 'env_args': {'continuing_episode': True,
                              'debug': False,
                              'difficulty': '7',
                              'game_version': None,
                              'heuristic_ai': False,
                              'heuristic_rest': False,
                              'map_name': '2m_vs_1z',
                              'move_amount': 2,
                              'obs_all_health': True,
                              'obs_instead_of_state': False,
                              'obs_last_action': False,
                              'obs_own_health': True,
                              'obs_pathing_grid': False,
                              'obs_terrain_height': False,
                              'obs_timestep_number': False,
                              'replay_dir': '',
                              'replay_prefix': '',
                              'reward_death_value': 10,
                              'reward_defeat': 0,
                              'reward_negative_scale': 0.5,
                              'reward_only_positive': True,
                              'reward_scale': True,
                              'reward_scale_rate': 20,
                              'reward_sparse': False,
                              'reward_win': 200,
                              'seed': 315055481,
                              'state_last_action': True,
                              'state_timestep_number': False,
                              'step_mul': 8},
                 'epsilon_anneal_time': 100000,
                 'epsilon_finish': 0.05,
                 'epsilon_start': 1.0,
                 'evaluate': False,
                 'gain': 0.01,
                 'gamma': 0.99,
                 'grad_norm_clip': 10,
                 'hypernet_embed': 64,
                 'label': 'default_label',
                 'learner': 'nq_learner',
                 'learner_log_interval': 10000,
                 'load_step': 0,
                 'local_results_path': 'results',
                 'log_interval': 10000,
                 'lr': 0.001,
                 'mac': 'n_mac',
                 'mixer': 'qmix',
                 'mixing_embed_dim': 32,
                 'name': 'qmix_env=8_adam_td_lambda',
                 'obs_agent_id': True,
                 'obs_last_action': True,
                 'optim_alpha': 0.99,
                 'optim_eps': 1e-05,
                 'optimizer': 'adam',
                 'per_alpha': 0.6,
                 'per_beta': 0.4,
                 'q_lambda': False,
                 'repeat_id': 1,
                 'return_priority': False,
                 'rnn_hidden_dim': 64,
                 'run': 'default',
                 'runner': 'episode',
                 'runner_log_interval': 10000,
                 'save_model': True,
                 'save_model_interval': 5000,
                 'save_replay': True,
                 'seed': 315055481,
                 't_max': 10050000,
                 'target_update_interval': 200,
                 'td_lambda': 0.6,
                 'test_greedy': True,
                 'test_interval': 10000,
                 'test_nepisode': 15,
                 'use_cuda': False,
                 'use_layer_norm': False,
                 'use_orthogonal': False,
                 'use_per': False,
                 'use_tensorboard': False}

    args = argparse.Namespace(**args_json)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = utils.logging.get_logger()

    logger.info("Experiment Parameters:")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
    runner = runners.EpisodeRunner(args, logger)



