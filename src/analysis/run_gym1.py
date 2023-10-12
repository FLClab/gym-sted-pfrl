
import gym
import json
import os
import gym_sted
import pfrl
import torch
import sys
import random
import pickle
import logging
import functools
import numpy
import h5py

from tqdm.auto import trange, tqdm
from matplotlib import pyplot
from collections import defaultdict
from skimage import io

while "../.." in sys.path:
    sys.path.remove("../..")
sys.path.insert(0, "../..")
from src import models, WrapPyTorch, GymnasiumWrapper

from gym_sted.defaults import action_spaces, FLUO
from gym_sted.envs.sted_env import scales_dict, bounds_dict
from gym_sted.utils import BleachSampler

# Defines constants
PATH = "../../data"
ROUTINES = {
    "high-signal_low-bleach" : {
        "bleach" : {
            "p_ex" : 10e-6,
            "p_sted" : 150e-3,
            "pdt" : 30.0e-6,
            "target" : 0.2
        },
        "signal" : {
            "p_ex" : 10.0e-6,
            "p_sted" : 0.,
            "pdt" : 10.0e-6,
            "target" : 200.
        },
    },
    "high-signal_high-bleach" : {
        "bleach" : {
            "p_ex" : 2e-6,
            "p_sted" : 150e-3,
            "pdt" : 25.0e-6,
            "target" : 0.7
        },
        "signal" : {
            "p_ex" : 10.0e-6,
            "p_sted" : 0.,
            "pdt" : 10.0e-6,
            "target" : 200.
        },
    },
    "low-signal_low-bleach" : {
        "bleach" : {
            "p_ex" : 10e-6,
            "p_sted" : 150e-3,
            "pdt" : 30.0e-6,
            "target" : 0.2
        },
        "signal" : {
            "p_ex" : 10.0e-6,
            "p_sted" : 0.,
            "pdt" : 10.0e-6,
            "target" : 30.
        },
    },
    "low-signal_high-bleach" : {
        "bleach" : {
            "p_ex" : 2e-6,
            "p_sted" : 150e-3,
            "pdt" : 25.0e-6,
            "target" : 0.7
        },
        "signal" : {
            "p_ex" : 10.0e-6,
            "p_sted" : 0.,
            "pdt" : 10.0e-6,
            "target" : 30.
        },
    }
}

def aggregate(items):
    """
    Aggregates a list of dict into a single dict

    :param items: A `list` of dict

    :returns : A `dict` with aggregated items
    """
    out = defaultdict(list)
    for item in items:
        for key, value in item.items():
            out[key].append(value)
    return out

def _batch_run_episodes_record(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple episodes and return returns in a batch manner."""
    assert (n_steps is None) != (n_episodes is None)

    logger = logger or logging.getLogger(__name__)
    num_envs = env.num_envs
    episode_returns = dict()
    episode_lengths = dict()
    episode_infos = dict()
    episode_indices = numpy.zeros(num_envs, dtype="i")
    episode_idx = 0
    for i in range(num_envs):
        episode_indices[i] = episode_idx
        episode_idx += 1
    episode_r = numpy.zeros(num_envs, dtype=numpy.float64)
    episode_len = numpy.zeros(num_envs, dtype="i")
    episode_info = [[] for _ in range(num_envs)]

    obss = env.reset()
    rs = numpy.zeros(num_envs, dtype="f")

    termination_conditions = False
    timestep = 0
    while True:

        # a_t
        actions = agent.batch_act(obss)
        timestep += 1
        # o_{t+1}, r_{t+1}
        obss, rs, dones, infos = env.step(actions)
        episode_r += rs
        episode_len += 1
        for i, info in enumerate(infos):
            episode_info[i].append(info)

        # Compute mask for done and reset
        if max_episode_len is None:
            resets = numpy.zeros(num_envs, dtype=bool)
        else:
            resets = episode_len == max_episode_len
        resets = numpy.logical_or(
            resets, [info.get("needs_reset", False) for info in infos]
        )

        # Make mask. 0 if done/reset, 1 if pass
        end = numpy.logical_or(resets, dones)
        not_end = numpy.logical_not(end)

        for index in range(len(end)):
            if end[index]:
                episode_returns[episode_indices[index]] = episode_r[index]
                episode_lengths[episode_indices[index]] = episode_len[index]
                episode_infos[episode_indices[index]] = episode_info[index]
                # Give the new episode an a new episode index
                episode_indices[index] = episode_idx
                episode_idx += 1

        # Resets done episode
        episode_r[end] = 0
        episode_len[end] = 0
        for index in range(len(end)):
            if end[index]:
                episode_info[index] = []

        # find first unfinished episode
        first_unfinished_episode = 0
        while first_unfinished_episode in episode_returns:
            first_unfinished_episode += 1

        # Check for termination conditions
        eval_episode_returns = []
        eval_episode_lens = []
        eval_episode_infos = []
        if n_steps is not None:
            total_time = 0
            for index in range(first_unfinished_episode):
                total_time += episode_lengths[index]
                # If you will run over allocated steps, quit
                if total_time > n_steps:
                    break
                else:
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])
                    eval_episode_infos.append(episode_infos[index])
            termination_conditions = total_time >= n_steps
            if not termination_conditions:
                unfinished_index = numpy.where(
                    episode_indices == first_unfinished_episode
                )[0]
                if total_time + episode_len[unfinished_index] >= n_steps:
                    termination_conditions = True
                    if first_unfinished_episode == 0:
                        eval_episode_returns.append(episode_r[unfinished_index])
                        eval_episode_lens.append(episode_len[unfinished_index])
                        eval_episode_infos.append(episode_infos[index])
        else:
            termination_conditions = first_unfinished_episode >= n_episodes
            if termination_conditions:
                # Get the first n completed episodes
                for index in range(n_episodes):
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])
                    eval_episode_infos.append(episode_infos[index])

        if termination_conditions:
            # If this is the last step, make sure the agent observes reset=True
            resets.fill(True)

        # Agent observes the consequences.
        agent.batch_observe(obss, rs, dones, resets)

        if termination_conditions:
            break
        else:
            obss = env.reset(not_end)

    for i, (epi_len, epi_ret) in enumerate(
        zip(eval_episode_lens, eval_episode_returns)
    ):
        logger.info("evaluation episode %s length: %s R: %s", i, epi_len, epi_ret)
    scores = [float(r) for r in eval_episode_returns]
    lengths = [float(ln) for ln in eval_episode_lens]
    infos = [info for info in eval_episode_infos]
    return scores, lengths, infos

def _batch_run_episodes_with_delayed_reward_record(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple episodes and return returns in a batch manner."""
    assert (n_steps is None) != (n_episodes is None)

    logger = logger or logging.getLogger(__name__)
    num_envs = env.num_envs
    episode_returns = dict()
    episode_lengths = dict()
    episode_infos = dict()
    episode_indices = numpy.zeros(num_envs, dtype="i")
    episode_idx = 0
    for i in range(num_envs):
        episode_indices[i] = episode_idx
        episode_idx += 1
    episode_r = numpy.zeros(num_envs, dtype=numpy.float64)
    episode_len = numpy.zeros(num_envs, dtype="i")
    episode_info = [[] for _ in range(num_envs)]

    obss = env.reset()
    rs = numpy.zeros(num_envs, dtype="f")

    termination_conditions = False
    timestep = 0
    while True:

        # a_t
        actions = agent.batch_act(obss)
        timestep += 1
        # o_{t+1}, r_{t+1}
        obss, rs, dones, infos = env.step(actions)
        episode_len += 1
        for i, info in enumerate(infos):
            episode_info[i].append(info)

        # Compute mask for done and reset
        if max_episode_len is None:
            resets = numpy.zeros(num_envs, dtype=bool)
        else:
            resets = episode_len == max_episode_len
        resets = numpy.logical_or(
            resets, [info.get("needs_reset", False) for info in infos]
        )

        # Make mask. 0 if done/reset, 1 if pass
        end = numpy.logical_or(resets, dones)
        not_end = numpy.logical_not(end)

        # If episode end we update the episode reward
        for idx in numpy.argwhere(end).ravel():
            episode_r[idx] = numpy.sum(rs[idx])

        for index in range(len(end)):
            if end[index]:
                episode_returns[episode_indices[index]] = episode_r[index]
                episode_lengths[episode_indices[index]] = episode_len[index]
                episode_infos[episode_indices[index]] = episode_info[index]
                # Give the new episode an a new episode index
                episode_indices[index] = episode_idx
                episode_idx += 1

        # Resets done episode
        episode_r[end] = 0
        episode_len[end] = 0
        for index in range(len(end)):
            if end[index]:
                episode_info[index] = []

        # find first unfinished episode
        first_unfinished_episode = 0
        while first_unfinished_episode in episode_returns:
            first_unfinished_episode += 1

        # Check for termination conditions
        eval_episode_returns = []
        eval_episode_lens = []
        eval_episode_infos = []
        if n_steps is not None:
            total_time = 0
            for index in range(first_unfinished_episode):
                total_time += episode_lengths[index]
                # If you will run over allocated steps, quit
                if total_time > n_steps:
                    break
                else:
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])
                    eval_episode_infos.append(episode_infos[index])
            termination_conditions = total_time >= n_steps
            if not termination_conditions:
                unfinished_index = numpy.where(
                    episode_indices == first_unfinished_episode
                )[0]
                if total_time + episode_len[unfinished_index] >= n_steps:
                    termination_conditions = True
                    if first_unfinished_episode == 0:
                        eval_episode_returns.append(episode_r[unfinished_index])
                        eval_episode_lens.append(episode_len[unfinished_index])
                        eval_episode_infos.append(episode_infos[index])
        else:
            termination_conditions = first_unfinished_episode >= n_episodes
            if termination_conditions:
                # Get the first n completed episodes
                for index in range(n_episodes):
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])
                    eval_episode_infos.append(episode_infos[index])

        if termination_conditions:
            # If this is the last step, make sure the agent observes reset=True
            resets.fill(True)

        # Agent observes the consequences.
        agent.batch_observe(obss, rs, dones, resets)

        if termination_conditions:
            break
        else:
            obss = env.reset(not_end)

    for i, (epi_len, epi_ret) in enumerate(
        zip(eval_episode_lens, eval_episode_returns)
    ):
        logger.info("evaluation episode %s length: %s R: %s", i, epi_len, epi_ret)
    scores = [float(r) for r in eval_episode_returns]
    lengths = [float(ln) for ln in eval_episode_lens]
    infos = [info for info in eval_episode_infos]
    return scores, lengths, infos

def _batch_run_episodes_recurrent_record(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple episodes and return returns in a batch manner."""
    assert (n_steps is None) != (n_episodes is None)

    logger = logger or logging.getLogger(__name__)
    num_envs = env.num_envs
    episode_returns = dict()
    episode_lengths = dict()
    episode_infos = dict()
    episode_indices = numpy.zeros(num_envs, dtype="i")
    episode_idx = 0
    for i in range(num_envs):
        episode_indices[i] = episode_idx
        episode_idx += 1
    episode_r = numpy.zeros(num_envs, dtype=numpy.float64)
    episode_len = numpy.zeros(num_envs, dtype="i")
    episode_info = [[] for _ in range(num_envs)]

    obss = env.reset()
    rs = numpy.zeros(num_envs, dtype="f")

    termination_conditions = False
    timestep = 0
    while True:

        # a_t
        actions = agent.batch_act(obss)
        timestep += 1
        # o_{t+1}, r_{t+1}
        obss, rs, dones, infos = env.step(actions)
        episode_r += rs
        episode_len += 1
        for i, info in enumerate(infos):

            # Extract recurrent states from agent
            for key, recurrent_states in zip(["policy_recurrent_states", "value_recurrent_states"], agent.test_recurrent_states):
                states = []
                for recurrent_state in recurrent_states:
                    # for each recurrent layer in recurrent_states
                    # h_n : final hidden state
                    # c_n : final cell state
                    h_n, c_n = recurrent_state
                    states.append([h_n[:, i, :].cpu().data.numpy(), c_n[:, i, :].cpu().data.numpy()])
                info[key] = states

            episode_info[i].append(info)

        # Compute mask for done and reset
        if max_episode_len is None:
            resets = numpy.zeros(num_envs, dtype=bool)
        else:
            resets = episode_len == max_episode_len
        resets = numpy.logical_or(
            resets, [info.get("needs_reset", False) for info in infos]
        )

        # Make mask. 0 if done/reset, 1 if pass
        end = numpy.logical_or(resets, dones)
        not_end = numpy.logical_not(end)

        for index in range(len(end)):
            if end[index]:
                episode_returns[episode_indices[index]] = episode_r[index]
                episode_lengths[episode_indices[index]] = episode_len[index]
                episode_infos[episode_indices[index]] = episode_info[index]
                # Give the new episode an a new episode index
                episode_indices[index] = episode_idx
                episode_idx += 1

        # Resets done episode
        episode_r[end] = 0
        episode_len[end] = 0
        for index in range(len(end)):
            if end[index]:
                episode_info[index] = []

        # find first unfinished episode
        first_unfinished_episode = 0
        while first_unfinished_episode in episode_returns:
            first_unfinished_episode += 1

        # Check for termination conditions
        eval_episode_returns = []
        eval_episode_lens = []
        eval_episode_infos = []
        if n_steps is not None:
            total_time = 0
            for index in range(first_unfinished_episode):
                total_time += episode_lengths[index]
                # If you will run over allocated steps, quit
                if total_time > n_steps:
                    break
                else:
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])
                    eval_episode_infos.append(episode_infos[index])
            termination_conditions = total_time >= n_steps
            if not termination_conditions:
                unfinished_index = numpy.where(
                    episode_indices == first_unfinished_episode
                )[0]
                if total_time + episode_len[unfinished_index] >= n_steps:
                    termination_conditions = True
                    if first_unfinished_episode == 0:
                        eval_episode_returns.append(episode_r[unfinished_index])
                        eval_episode_lens.append(episode_len[unfinished_index])
                        eval_episode_infos.append(episode_infos[index])
        else:
            termination_conditions = first_unfinished_episode >= n_episodes
            if termination_conditions:
                # Get the first n completed episodes
                for index in range(n_episodes):
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])
                    eval_episode_infos.append(episode_infos[index])

        if termination_conditions:
            # If this is the last step, make sure the agent observes reset=True
            resets.fill(True)

        # Agent observes the consequences.
        agent.batch_observe(obss, rs, dones, resets)

        if termination_conditions:
            break
        else:
            obss = env.reset(not_end)

    for i, (epi_len, epi_ret) in enumerate(
        zip(eval_episode_lens, eval_episode_returns)
    ):
        logger.info("evaluation episode %s length: %s R: %s", i, epi_len, epi_ret)
    scores = [float(r) for r in eval_episode_returns]
    lengths = [float(ln) for ln in eval_episode_lens]
    infos = [info for info in eval_episode_infos]
    return scores, lengths, infos

def batch_run_evaluation_episodes_record_actions(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
    recurrent=False,
    with_delayed_reward=False
):
    """Run multiple evaluation episodes and return returns in a batch manner.

    Args:
        env (VectorEnv): Environment used for evaluation.
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of total timesteps to evaluate the agent.
        n_episodes (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes
            longer than this value will be truncated.
        logger (Logger or None): If specified, the given Logger
            object will be used for logging results. If not
            specified, the default logger of this module will
            be used.

    Returns:
        List of returns of evaluation runs.
    """
    with agent.eval_mode():
        if recurrent:
            return _batch_run_episodes_recurrent_record(
                env=env,
                agent=agent,
                n_steps=n_steps,
                n_episodes=n_episodes,
                max_episode_len=max_episode_len,
                logger=logger,
            )
        elif with_delayed_reward:
            return _batch_run_episodes_with_delayed_reward_record(
                env=env,
                agent=agent,
                n_steps=n_steps,
                n_episodes=n_episodes,
                max_episode_len=max_episode_len,
                logger=logger,
            )
        else:
            return _batch_run_episodes_record(
                env=env,
                agent=agent,
                n_steps=n_steps,
                n_episodes=n_episodes,
                max_episode_len=max_episode_len,
                logger=logger,
            )

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True,
                        help="The name of the model")
    parser.add_argument("--savedir", type=str, default=PATH,
                        help="The directory containing all models")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="The number of simultaneous environment to use")
    parser.add_argument("--eval-n-runs", type=int, default=1,
                        help="The number of episodes to run")
    parser.add_argument("--env", type=str, default=None,
                        help="If given it overwrites the env that the model was trained with")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Wheter gpu should be used")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Wheter gpu should be used")
    parser.add_argument("--overwrite", action="store_true",
                        help="(optional) will overwrite a previous checkpoint file if already existing")    
    args = parser.parse_args()

    assert os.path.isdir(os.path.join(args.savedir, args.model_name)), f"This is not a valid model name : {args.model_name}"

    os.makedirs(os.path.join(args.savedir, args.model_name, "panels"), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, args.model_name, "eval"), exist_ok=True)

    loaded_args = json.load(open(os.path.join(args.savedir, args.model_name, "args.txt"), "r"))
    if args.env:
        loaded_args["env"] = args.env

    process_seeds = numpy.arange(args.num_envs) + 42

    def make_env(idx, test, **kwargs):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env = gym.make(loaded_args["env"], disable_env_checker=True)
        # Normalize the action space
        env = pfrl.wrappers.NormalizeActionSpace(env)
        # Use different random seeds for train and test envs
        env.reset(seed=env_seed)
        # Converts the openAI Gym to PyTorch tensor shape
        env = WrapPyTorch(env)
        # Converts the new gymnasium implementation to old gym implementation
        env = GymnasiumWrapper(env)

        if "fluo" in kwargs:
            fluo = kwargs.get("fluo")
            if "sigma_abs" in fluo:
                env.update_(bleach_sampler=BleachSampler("constant", kwargs.get("fluo")))
            else:
                env.update_(bleach_sampler=BleachSampler("uniform", criterions=kwargs.get("fluo")))

        return env

    def make_batch_env(test, **kwargs):
        vec_env = pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test, **kwargs)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )
        # vec_env = pfrl.wrappers.VectorFrameStack(vec_env, 4)
        return vec_env

    env = make_env(0, True)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    action_space = env.action_space

    # Creates the agent
    policy = models.Policy(obs_space=obs_space, action_size=action_space.shape[0])
    vf = models.ValueFunction(obs_space=obs_space)            
    model = pfrl.nn.Branched(policy, vf)

    opt = torch.optim.Adam(model.parameters(), lr=loaded_args["lr"])

    agent = pfrl.agents.PPO(
        model,
        opt,
        gpu=args.gpu,
        minibatch_size=loaded_args["batchsize"],
        max_grad_norm=1.0,
        update_interval=loaded_args["update_interval"],
        recurrent=loaded_args["recurrent"],
        act_deterministically=True
    )
    agent.load(os.path.join(args.savedir, args.model_name, "best"))
    if isinstance(args.checkpoint, int):
        checkpoint_path = os.path.join(args.savedir, args.model_name, f"{args.checkpoint}_checkpoint")
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint : {checkpoint_path} does not exists")
        agent.load(checkpoint_path)
        
    savename = f"stats_{args.checkpoint}_checkpoint.hdf5" if args.checkpoint else "stats_best.hdf5"
    savepath = os.path.join(args.savedir, args.model_name, "eval", savename)
    if not args.overwrite and os.path.isfile(savepath):
        raise FileExistsError(f"Checkpoint : {savepath} already exists. Use option `--overwrite` to overwrite this file")

    # Runs the agent
    all_records = {}
    for key, fluo in ROUTINES.items():
        print(key)
        # Creates the batch envs
        env = make_batch_env(test=True, fluo=fluo)
        scores, lengths, records = batch_run_evaluation_episodes_record_actions(
            env, agent, n_steps=None, n_episodes=args.eval_n_runs,
            recurrent=loaded_args["recurrent"], with_delayed_reward="WithDelayedReward" in loaded_args["env"]
        )
        # for i in range(len(records[0])):
        #     print(records[0][i]["action"], records[0][i]["mo_objs"], records[0][i]["reward"])
        #     print(records[0][i]["conf1"].max(), records[0][i]["sted_image"].max())
        all_records[key] = records

    # Avoids pending with multiprocessing
    if not env.closed:
        env.close()

    # # Saves all runs
    with h5py.File(savepath, "w") as file:
        for routine_name, routine in all_records.items():
            routine_group = file.create_group(routine_name)
            for eval_run, record in enumerate(routine):
                eval_group = routine_group.create_group(str(eval_run))
                for key, values in aggregate(record).items():
                    if key == "nanodomains-coords":
                        step_group = eval_group.create_group(key)
                        for step, value in enumerate(values):
                            step_group.create_dataset(str(step), data=value)
                    elif isinstance(values, list) and isinstance(values[0], dict):
                        dict_group = eval_group.create_group(key)
                        data = aggregate(values)
                        for k, v in data.items():
                            dict_group.create_dataset(k, data=v)
                    else:
                        if isinstance(values[0], str):
                            data = numpy.array(values, dtype="S16") # Assumes 16char is enough to encode string
                        else:
                            data = numpy.array(values)
                        eval_group.create_dataset(
                            key, data=data, compression="gzip", compression_opts=5
                        )
