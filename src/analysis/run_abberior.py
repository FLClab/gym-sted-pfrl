
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
import matplotlib
matplotlib.use("TkAgg")

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

        return env

    env = make_env(0, True)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    action_space = env.action_space

    # Creates the agent
    if loaded_args["recurrent"]:
        policy = models.RecurrentPolicy(obs_space=obs_space, action_size=action_space.shape[0])
        vf = models.RecurrentValueFunction(obs_space=obs_space)
        model = pfrl.nn.RecurrentBranched(policy, vf)
    else:
        if loaded_args["model"] == "default":
            policy = models.Policy2(obs_space=obs_space, action_size=action_space.shape[0])
            vf = models.ValueFunction2(obs_space=obs_space)            
        elif loaded_args["model"] == "pooled":
            policy = models.PooledPolicy(obs_space=obs_space, action_size=action_space.shape[0])
            vf = models.PooledValueFunction(obs_space=obs_space)
        else:
            print(f"Model type `{loaded_args['model']}` not found... Exiting")
            exit()
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
        
    # Runs the agent
    episode_memory = defaultdict(list)
        
    observation = env.reset(seed=None)
    episode_r, t, episode_len = 0, 0, 0
    max_episode_len = env.spec.max_episode_steps

    episode_memory["action"].append([0.] * len(env.actions))
    episode_memory["observation"].append(observation)
    episode_memory["info"].append({})

    while True:
        
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)

        t += 1
        episode_r += reward
        episode_len += 1

        reset = episode_len == max_episode_len or info.get("needs_reset", False)
        
        agent.observe(observation, reward, done, reset)

        episode_memory["action"].append(action)
        episode_memory["observation"].append(observation)
        episode_memory["info"].append(info)

        if done or reset:
            break
        
    env.close()

    # # Saves all runs
    # with h5py.File(savepath, "w") as file:
    #     for routine_name, routine in all_records.items():
    #         routine_group = file.create_group(routine_name)
    #         for eval_run, record in enumerate(routine):
    #             eval_group = routine_group.create_group(str(eval_run))
    #             for key, values in aggregate(record).items():
    #                 if key == "nanodomains-coords":
    #                     step_group = eval_group.create_group(key)
    #                     for step, value in enumerate(values):
    #                         step_group.create_dataset(str(step), data=value)
    #                 else:
    #                     data = numpy.array(values)
    #                     eval_group.create_dataset(
    #                         key, data=numpy.array(values), compression="gzip", compression_opts=5
    #                     )
