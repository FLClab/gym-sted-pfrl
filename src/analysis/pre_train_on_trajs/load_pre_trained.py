import argparse
import numpy
import datetime
import functools
import uuid

import gym
import gym.spaces
import torch
from torch import nn

import pfrl
from pfrl import experiments, utils
from pfrl.policies import GaussianHeadWithFixedCovariance, SoftmaxCategoricalHead

from src import models, WrapPyTorch
from gym_sted import defaults

TIMEFMT = "%Y%m%d-%H%M%S"

def action_rescaler(action_val, action_type, action_spaces=defaults.action_spaces, way="range_to_space"):
    """
    rescales the action from range [-1, 1] to the action space space of the corresponding action
    Args:
        action_val: the input value (float)
        action_type: pdt, p_ex or p_sted (str)

    Returns:

    """
    way_vals = ["range_to_space", "space_to_range"]
    if way not in way_vals:
        raise ValueError(f"invalid way, valid values are {way_vals}")

    if way == "range_to_space":
        if action_val <= -1. :
            action_val = -1.
        elif action_val >= 1. :
            action_val = 1.

        action_range_min = action_spaces[action_type]['low']
        action_range_max = action_spaces[action_type]['high']
        input_range_min, input_range_max = -1., 1.

        m = (action_val - input_range_min) / (input_range_max - input_range_min) * (action_range_max - action_range_min) + action_range_min
    elif way == "space_to_range":
        if action_val <= action_spaces[action_type]['low']:
            action_val = action_spaces[action_type]['low']
        elif action_val >= action_spaces[action_type]['high']:
            action_val = action_spaces[action_type]['high']

        action_range_min = action_spaces[action_type]['low']
        action_range_max = action_spaces[action_type]['high']
        input_range_min, input_range_max = -1., 1.

        m = (action_val - action_range_min) / (action_range_max - action_range_min) * (input_range_max - input_range_min) + input_range_min

    return m

def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="gym_sted:STEDtimed-exp-easy-v5")
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
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--bleach-sampling", type=str, default="constant")
    parser.add_argument("--recurrent", action="store_true", default=False)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)
    process_seeds = numpy.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir, exp_id="{}_{}".format(args.exp_id, str(uuid.uuid4())[:8]))

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

    if args.recurrent:
        policy = models.RecurrentPolicy(obs_space=obs_space, action_size=action_space.shape[0])
        vf = models.RecurrentValueFunction(obs_space=obs_space)
        model = pfrl.nn.RecurrentBranched(policy, vf)
    else:
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
        update_interval=args.update_interval,
        recurrent=args.recurrent,
        max_recurrent_sequence_len=10
    )
    if args.load:
        agent.load(args.load)

    eval_env = make_env(None, test=True)
    obs = eval_env.reset()
    done = False
    episode_len = 0
    max_episode_len = 50
    while not done:
        action = agent.act(obs)
        pdt_val = action_rescaler(action[0], 'pdt')
        p_ex_val = action_rescaler(action[1], 'p_ex')
        p_sted_val = action_rescaler(action[2], 'p_sted')
        print(f"stepping! pdt = {pdt_val}, p_ex = {p_ex_val}, p_sted = {p_sted_val}")
        obs, r, done, info = eval_env.step(action)

        episode_len += 1
        reset = episode_len == max_episode_len or info.get("needs_reset", False)

        agent.observe(obs, r, done, reset)


if __name__ == "__main__":

    # Run the following line of code
    # python main.py --env gym_sted:STED-v0 --batchsize=16 --gpu=None --reward-scale-factor=1.0 --eval-interval=100 --eval-n-runs=5
    main()