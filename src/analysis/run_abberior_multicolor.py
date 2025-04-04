
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
import yaml
# matplotlib.use("TkAgg")

from tqdm.auto import trange, tqdm
from matplotlib import pyplot
from collections import defaultdict
from skimage import io

while "../.." in sys.path:
    sys.path.remove("../..")
sys.path.insert(0, "../..")
from src import models, WrapPyTorch, GymnasiumWrapper

from gym_sted import defaults
from gym_sted.defaults import action_spaces, FLUO
from gym_sted.envs.sted_env import scales_dict, bounds_dict
from gym_sted.utils import BleachSampler
from gym_sted.microscopes.abberior import AbberiorMicroscope
from gym_sted.envs.abberior_env import MetaEnv

import abberior

# Defines constants
# PATH = os.path.join(
#     "C:", os.sep, "Users", "abberior", "Desktop", "DATA", "abilodeau",
#     "20230424_STED-RL"
# )
PATH = os.path.join(os.getcwd(), "data", "20230409_STED-RL")
PATH = "/home-local2/projects/pysted"

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
    print("Loaded args : ", loaded_args)

    process_seeds = numpy.arange(args.num_envs) + 42

    CONFIGURATIONS = {
        "488" : {
            "abberior_action_spaces" : {
                    "p_sted" : {"low" : 0., "high" : 80.},
                    "p_ex" : {"low" : 0., "high" : 18.},
                    "pdt" : {"low" : 1.0e-6, "high" : 100.0e-6},
            },
            "conf_params" : {
                "p_sted" : 0.,
                "p_ex" : 9.0,
                "pdt" : 10.0e-6,
            }
        },
        "561" : {
            "abberior_action_spaces" : {
                    "p_sted" : {"low" : 0., "high" : 80.},
                    "p_ex" : {"low" : 0., "high" : 18.},
                    "pdt" : {"low" : 1.0e-6, "high" : 100.0e-6},
            },
            "conf_params" : {
                "p_sted" : 0.,
                "p_ex" : 9.0,
                "pdt" : 10.0e-6,
            }
        },
        "640" : {
            "abberior_action_spaces" : {
                    "p_sted" : {"low" : 0., "high" : 80.},
                    "p_ex" : {"low" : 0., "high" : 18.},
                    "pdt" : {"low" : 1.0e-6, "high" : 100.0e-6},
            },
            "conf_params" : {
                "p_sted" : 0.,
                "p_ex" : 9.0,
                "pdt" : 10.0e-6,
            }
        }
    }

    def make_env(idx, test, **kwargs):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env = gym.make(loaded_args["env"], disable_env_checker=True)
        # Normalize the action space
        env = pfrl.wrappers.NormalizeActionSpace(env)
        # Use different random seeds for train and test envs
        # env.reset(seed=env_seed)
        # Converts the openAI Gym to PyTorch tensor shape
        env = WrapPyTorch(env)
        # Converts the new gymnasium implementation to old gym implementation
        env = GymnasiumWrapper(env)

        return env

    envs = []
    agents = []
    for env_name, configuration in CONFIGURATIONS.items():

        # This is hacky but should work
        defaults.abberior_action_spaces = configuration["abberior_action_spaces"]

        print("Creating env : ", env_name)
        
        # Creates the environment
        env = make_env(0, True)
        
        obs_space = env.observation_space
        action_space = env.action_space

        # We need to update the `AbberiorMicroscope` so that the measurements are using the correct configuration
        env.microscope = None
        config = yaml.load(open(os.path.join(
            os.path.dirname(__file__), "configs", f"abberior-config-{env_name}.yml"), "r"), Loader=yaml.FullLoader)
        env.microscope = AbberiorMicroscope(
            env.measurements, config=config
        )

        # We update the conf_params of the microscope
        env.conf_params = configuration["conf_params"]

        print("Done creating env")

        envs.append(env)

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
        
        agents.append(agent)

    # Creates the MetaEnv
    print("Creating the MetaEnv")
    env = MetaEnv(envs)
    
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

    observations = env.reset(seed=None)
    episode_r, t, episode_len = [0] * len(env.envs), 0, 0
    max_episode_len = env.max_episode_steps

    episode_memory["action"].append([[0.] * len(env.actions) for env in env.envs])
    episode_memory["observation"].append(observations)
    episode_memory["info"].append({})

    try:
        while True:

            print(f"[----] Timestep: {t}")

            actions = []
            for agent, observation, history in zip(agents, *observations):
                obs = numpy.array(observation).astype(numpy.float32)
                history = numpy.array(history).astype(numpy.float32)

                obs = numpy.transpose(obs, (2, 0, 1))
                                
                # Convert to torch
                obs = torch.tensor(obs)
                history = torch.tensor(history)
                obs = obs.to(agent.device)
                history = history.to(agent.device)

                observation = (obs, history)

                action = agent.act(observation)
                actions.append(action)

            observations, rewards, done, _, info = env.step(actions)

            t += 1
            for i, reward in enumerate(rewards):
                episode_r[i] += reward
            episode_len += 1

            reset = episode_len == max_episode_len or info.get("needs_reset", False)
            
            for agent, observation, history, reward in zip(agents, *observations, rewards):
                observation = (observation, history)
                agent.observe(observation, reward, done, reset)

            episode_memory["action"].append(actions)
            episode_memory["observation"].append(observations)
            episode_memory["info"].append(info)

            pickle.dump(episode_memory, open(os.path.join(savepath, "checkpoint.pkl"), "wb"))

            if done or reset:
                break

    except KeyboardInterrupt:
        pass
    
    env.close()

    print("Saving the results...")

    # Saves all runs
    with h5py.File(os.path.join(savepath, savename), "w") as file:
        for key, values in episode_memory.items():
            group = file.create_group(key)
            for step, items in enumerate(values):
                if key == "observation":
                    for env_name, env_observation, env_history in zip(CONFIGURATIONS.keys(), *items):

                        step_group = group.create_group(f"{step}_{env_name}")
                        step_group.create_dataset(
                            "state", data=numpy.array(env_observation),
                            compression="gzip", compression_opts=5
                        )
                        step_group.create_dataset(
                            "history", data=numpy.array(env_history)
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
