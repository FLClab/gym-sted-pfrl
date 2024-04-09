
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
import datetime
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

import abberior

# Defines constants
# PATH = os.path.join(
#     "C:", os.sep, "Users", "abberior", "Desktop", "DATA", "abilodeau",
#     "20230424_STED-RL"
# )
PATH = os.path.join(os.getcwd(), "data", "20230409_STED-RL")

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
    if loaded_args["model"] == "default":
        policy = models.Policy(obs_space=obs_space, action_size=action_space.shape[0])
        vf = models.ValueFunction(obs_space=obs_space)
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
    # agent.load(os.path.join(args.savedir, args.model_name, "best"))
    if isinstance(args.checkpoint, int):
        checkpoint_path = os.path.join(args.savedir, args.model_name, f"{args.checkpoint}_checkpoint")
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint : {checkpoint_path} does not exists")
        agent.load(checkpoint_path)

    dtime = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
    savename = f"{dtime}_stats_{args.checkpoint}_checkpoint.hdf5" if args.checkpoint else f"{dtime}_stats_best.hdf5"
    savepath = os.path.join(PATH, dtime)
    os.makedirs(os.path.join(savepath), exist_ok=True)

    # Save template
    abberior.microscope.measurement.save_as(os.path.join(savepath, "template.msr"))
    env_state = env.get_state()
    env_state["loaded-args"] = loaded_args
    env_state["model-name"] = args.model_name
    env_state["model-checkpoint"] = args.checkpoint if args.checkpoint else "best"
    json.dump(env_state, open(os.path.join(savepath, "env-state.json"), "w"), sort_keys=True, indent=4)

    # Runs the agent
    episode_memory = defaultdict(list)

    observation = env.reset(seed=None)
    episode_r, t, episode_len = 0, 0, 0
    max_episode_len = env.spec.max_episode_steps

    episode_memory["action"].append([0.] * len(env.actions))
    episode_memory["observation"].append(observation)
    episode_memory["info"].append({})

    try:
        while True:

            print(f"[----] Timestep: {t}")

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

            pickle.dump(episode_memory, open(os.path.join(savepath, "checkpoint.pkl"), "wb"))

            if done or reset:
                break
    except KeyboardInterrupt:
        pass
    
    env.close()

    # Saves all runs
    with h5py.File(os.path.join(savepath, savename), "w") as file:
        for key, values in episode_memory.items():
            group = file.create_group(key)
            for step, items in enumerate(values):
                if key == "observation":
                    step_group = group.create_group(str(step))
                    obs, history = items
                    step_group.create_dataset(
                        "state", data=numpy.array(obs),
                        compression="gzip", compression_opts=5
                    )
                    step_group.create_dataset(
                        "history", data=numpy.array(history)
                    )
                else:
                    if isinstance(items, dict):
                        step_group = group.create_group(str(step))
                        for item_key, item_value in items.items():
                            data = numpy.array(item_value)
                            step_group.create_dataset(
                                str(item_key), data=data
                            )
                    else:
                        data = numpy.array(items)
                        group.create_dataset(
                            str(step), data=data
                        )
