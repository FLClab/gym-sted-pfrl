"""
In this version the agent will select the params by itself like a grand garçon :)
"""

import argparse
import numpy
import datetime
import functools

import gym
import gym.spaces
import torch
from torch import nn

import pfrl
from pfrl import experiments, utils
from pfrl.policies import GaussianHeadWithFixedCovariance, SoftmaxCategoricalHead

from src import models, WrapPyTorch

TIMEFMT = "%Y%m%d-%H%M%S"

def main():
    # que es e plano capitao
    # the plan here is to load a trained a agent an have it act on an env and see the actions taken / images resulting

    import logging
    from matplotlib import pyplot as plt
    import metrics
    from skimage.feature import peak_local_max

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="gym_sted:STEDtimed-v3")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--outdir",
        type=str,
        default="./data",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--exp_id", type=str, default=datetime.datetime.now().strftime(TIMEFMT),
        help=(
            "Identification of the experiment"
        ),
    )
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10 ** 5)
    parser.add_argument("--eval-interval", type=int, default=1e+3)
    parser.add_argument("--eval-n-runs", type=int, default=100)
    parser.add_argument("--reward-scale-factor", type=float, default=1.)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--n-runs", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)
    process_seeds = numpy.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    save_path = args.outdir
    args.outdir = experiments.prepare_output_dir(args, args.outdir, exp_id=args.exp_id)

    def make_env(idx, test):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed

        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        env.seed(env_seed)
        # Converts the openAI Gym to PyTorch tensor shape
        env = WrapPyTorch(env)
        # Normalize the action space
        env = pfrl.wrappers.NormalizeActionSpace(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(env, args.outdir)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = pfrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if args.render and not test:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        vec_env = pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )
        # vec_env = pfrl.wrappers.VectorFrameStack(vec_env, 4)
        return vec_env

    sample_env = make_env(0, test=False)
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space

    policy = models.Policy2(obs_space=obs_space, action_size=action_space.shape[0])
    vf = models.ValueFunction2(obs_space=obs_space)
    model = pfrl.nn.Branched(policy, vf)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    agent = pfrl.agents.PPO(
        model,
        opt,
        gpu=args.gpu,
        minibatch_size=args.batchsize,
        max_grad_norm=1.0,
        update_interval=100
    )
    if args.load:
        agent.load(args.load)



    # train_env = make_env(0, test=False)
    # eval_env = make_env(0, test=True)

    action_spaces = {
        # changed p_sted low to 0 as I want to 0. as I want to take confocals if the flash is not yet happening
        "p_sted": {"low": 0., "high": 5.0e-3},
        "p_ex": {"low": 0., "high": 5.0e-6},  # jveux tu lui laisser prendre un p_ex = 0 ? ferait la wait action...
        "pdt": {"low": 10.0e-6, "high": 150.0e-6},
    }
    valid_actions = ["pdt", "p_ex", "p_sted"]

    action_space = gym.spaces.Box(
        low=numpy.array([action_spaces[name]["low"] for name in valid_actions]),
        high=numpy.array([action_spaces[name]["high"] for name in valid_actions]),
        shape=(3,),
        dtype=numpy.float32
    )

    laser_received_per_episode = []
    n_nanodomains_per_episode = []
    n_nanodomains_identified_per_episode = []
    # save_path = args.outdir
    for i in range(args.n_runs):
        laser_received_this_episode = 0
        train_env = make_env(0, test=False)
        eval_env = make_env(0, test=True)

        obs = eval_env.reset()
        nanodomain_coords = numpy.array(eval_env.temporal_datamap.synapse.nanodomains_coords)
        n_nanodomains_per_episode.append(nanodomain_coords.shape[0])
        nd_assigned_truth_list = []
        for i in range(nanodomain_coords.shape[0]):
            nd_assigned_truth_list.append(0)
        done = False
        while not done:
            a = agent.act(obs)

            action_clipped = numpy.clip(a, action_space.low, action_space.high)

            laser_received_this_episode += action_clipped[0] * (action_clipped[1] + action_clipped[2])

            obs, r, done, info = eval_env.step(a)

            guess_coords = peak_local_max(obs[0][-1], min_distance=2, threshold_rel=0.5)
            guess_coords_list.append(guess_coords)
            detector = metrics.CentroidDetectionError(nanodomain_coords, guess_coords, 2, algorithm="hungarian")
            for nd in detector.truth_couple:
                nd_assigned_truth_list[nd] = 1

        laser_received_per_episode.append(laser_received_this_episode)
        n_nanodomains_identified_per_episode.append(numpy.sum(nd_assigned_truth_list))

    laser_received_per_episode = numpy.asarray(laser_received_per_episode)
    n_nanodomains_per_episode = numpy.asarray(n_nanodomains_per_episode)
    n_nanodomains_identified_per_episode = numpy.asarray(n_nanodomains_identified_per_episode)
    numpy.save("gym-sted-pfrl/analysis/laser_received/laser_dose_agent_params.npy", laser_received_per_episode)
    numpy.save("gym-sted-pfrl/analysis/laser_received/gt_nb_nanodomains_agent_params.npy", n_nanodomains_per_episode)
    numpy.save("gym-sted-pfrl/analysis/laser_received/nb_nanodomains_id_agent_params.npy", n_nanodomains_identified_per_episode)


if __name__ == "__main__":

    # Run the following line of code
    # python main.py --env gym_sted:STED-v0 --batchsize=16 --gpu=None --reward-scale-factor=1.0 --eval-interval=100 --eval-n-runs=5
    main()