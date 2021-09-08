
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

from tqdm.auto import trange, tqdm
from matplotlib import pyplot
from collections import defaultdict
from skimage import io

while "../.." in sys.path:
    sys.path.remove("../..")
sys.path.insert(0, "../..")
from src import models, WrapPyTorch

from gym_sted.envs.sted_env import action_spaces, scales_dict, bounds_dict
from gym_sted.utils import BleachSampler

# Defines constants
PATH = "../../data"
PHY_REACTS = {
    "low-bleach" : gym_sted.defaults.FLUO["phy_react"],
    "mid-bleach" : {488: 0.5e-7 + 3 * 0.25e-7, 575: 50.0e-11 + 3 * 25.0e-11},
    "high-bleach" : {488: 0.5e-7 + 10 * 0.25e-7, 575: 50.0e-11 + 10 * 25.0e-11},
}

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


def batch_run_evaluation_episodes_record_actions(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
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
        env = gym.make(loaded_args["env"])
        # Use different random seeds for train and test envs
        env.seed(env_seed)
        # Converts the openAI Gym to PyTorch tensor shape
        env = WrapPyTorch(env)
        # Normalize the action space
        env = pfrl.wrappers.NormalizeActionSpace(env)

        if "phy_react" in kwargs:
            env.update_(bleach_sampler=BleachSampler("constant", kwargs.get("phy_react")))

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
    policy = models.Policy2(action_size=action_space.shape[0], obs_space=obs_space)
    vf = models.ValueFunction2(obs_space=obs_space)
    model = pfrl.nn.Branched(policy, vf)
    opt = torch.optim.Adam(model.parameters(), lr=loaded_args["lr"])
    agent = pfrl.agents.PPO(
        model,
        opt,
        gpu=0,
        minibatch_size=loaded_args["batchsize"],
        max_grad_norm=1.0,
        update_interval=512
    )
    agent.load(os.path.join(args.savedir, args.model_name, "best"))

    # Runs the agent
    all_records = {}
    for key, phy_react in tqdm(PHY_REACTS.items()):
        # Creates the batch envs
        env = make_batch_env(test=True, phy_react=phy_react)
        scores, lengths, records = batch_run_evaluation_episodes_record_actions(env, agent, n_steps=None, n_episodes=args.eval_n_runs)
        all_records[key] = records

        # Avoids pending with multiprocessing
        if not env.closed:
            env.close()

    # Saves all runs
    # pickle.dump(all_records, open(os.path.join(args.savedir, args.model_name, "eval", "stats.pkl"), "wb"))
