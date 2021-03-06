import argparse
import numpy
import datetime
import functools
import uuid
import os

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
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="gym_sted:ContextualMOSTED-easy-v0")
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
        "--exp-id", type=str, default=datetime.datetime.now().strftime(TIMEFMT),
        help=(
            "Identification of the experiment"
        ),
    )
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10 ** 5)
    parser.add_argument("--eval-interval", type=int, default=1e+3)
    parser.add_argument("--eval-n-runs", type=int, default=100)
    parser.add_argument("--update-interval", type=int, default=512)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--reward-scale-factor", type=float, default=1.)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--load-ckpt", type=int, default=0)
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--bleach-sampling", type=str, default="constant")
    parser.add_argument("--recurrent", action="store_true", default=False)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)
    process_seeds = numpy.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    if args.load:
        args.outdir = experiments.prepare_output_dir(args, args.outdir, exp_id=args.load)
    else:
        exp_id = "{}_{}".format(args.exp_id, str(uuid.uuid4())[:8]) if args.exp_id != "debug" else args.exp_id
        args.outdir = experiments.prepare_output_dir(args, args.outdir, exp_id=exp_id)

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
        return vec_env

    sample_env = make_env(0, test=False)
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space

    policy = models.Policy(obs_space=obs_space, action_size=action_space.shape[0])
    vf = models.ValueFunction(obs_space=obs_space)
    model = pfrl.nn.Branched(policy, vf)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    agent = pfrl.agents.PPO(
        model,
        opt,
        gpu=args.gpu,
        minibatch_size=args.batchsize,
        max_grad_norm=1.0,
        update_interval=args.update_interval,
        gamma=args.gamma
    )
    if args.load:
        model_path = os.path.join(args.outdir, f"{args.load_ckpt}_checkpoint")
        agent.load(model_path)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_env(0, True),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        if args.num_envs > 1:
            experiments.train_agent_batch_with_evaluation(
                agent=agent,
                env=make_batch_env(test=False),
                eval_env=make_batch_env(test=True),
                outdir=args.outdir,
                steps=args.steps,
                step_offset=args.load_ckpt,
                eval_n_steps=None,
                eval_n_episodes=args.eval_n_runs,
                eval_interval=args.eval_interval,
                checkpoint_freq=args.checkpoint_freq
            )
        else:
            experiments.train_agent_with_evaluation(
                agent=agent,
                env=make_env(0, test=False),
                eval_env=make_env(0, test=True),
                outdir=args.outdir,
                steps=args.steps,
                step_offset=args.load_ckpt,
                eval_n_steps=None,
                eval_n_episodes=args.eval_n_runs,
                eval_interval=args.eval_interval,
                checkpoint_freq=args.checkpoint_freq
            )


if __name__ == "__main__":

    main()
