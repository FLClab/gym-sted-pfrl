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

import os

TIMEFMT = "%Y%m%d-%H%M%S"

def make_path_sane(p):
    """Function to uniformly return a real, absolute filesystem path."""
    # ~/directory -> /home/user/directory
    p = os.path.expanduser(p)
    # A/.//B -> A/B
    p = os.path.normpath(p)
    # Resolve symbolic links
    p = os.path.realpath(p)
    # Ensure path is absolute
    p = os.path.abspath(p)
    return p

def main():
    import logging
    import metrics
    from skimage.feature import peak_local_max
    from matplotlib import pyplot as plt

    parser = argparse.ArgumentParser()

    parser.add_argument("--main_dir", type=str, default="")
    parser.add_argument("--analysis_n_runs", type=int, default=1)
    parser.add_argument("--checkpoints_interval", type=int, default=5000)
    parser.add_argument("--env", type=str, default="gym_sted:STEDtimed-v4")
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
    args = parser.parse_args()
    # main_path = make_path_sane(args.main_dir)

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)
    process_seeds = numpy.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    save_path = args.main_dir

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

    # if not os.path.exists(args.main_dir + "/analysis_results"):
    #     os.makedirs(args.main_dir + "/analysis_results")

    checkpoints = numpy.arange(1000, 100000 + args.checkpoints_interval, args.checkpoints_interval)
    if checkpoints[-1] > 100000:
        checkpoints[-1] = 100000
    checkpoints = checkpoints.tolist()
    checkpoints.append("best")
    checkpoints_n_successes = numpy.zeros(len(checkpoints))
    for idx_checkpoint, checkpoint in enumerate(checkpoints):
        current_dir = args.main_dir + f"/{checkpoint}_checkpoint"
        if not os.path.exists(current_dir + "/analysis_results"):
            os.makedirs(current_dir + "/analysis_results")
        os.makedirs(current_dir + "/analysis_results/NDs")
        os.makedirs(current_dir + "/analysis_results/actions")

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

        agent.load(current_dir)

        fixed_n_nanodomains_threshold = 0.8

        for i in range(args.analysis_n_runs):
            file_nd_analysis = open(current_dir + f"/analysis_results/NDs/{i}.txt", "w")
            eval_env = make_env(0, test=True)

            action_spaces = {
                # changed p_sted low to 0 as I want to 0. as I want to take confocals if the flash is not yet happening
                "p_sted": {"low": 0., "high": 5.0e-3},
                "p_ex": {"low": 0., "high": 5.0e-6},
                # jveux tu lui laisser prendre un p_ex = 0 ? ferait la wait action...
                "pdt": {"low": 10.0e-6, "high": 150.0e-6},
            }
            valid_actions = ["pdt", "p_ex", "p_sted"]

            action_space = gym.spaces.Box(
                low=numpy.array([action_spaces[name]["low"] for name in valid_actions]),
                high=numpy.array([action_spaces[name]["high"] for name in valid_actions]),
                shape=(3,),
                dtype=numpy.float32
            )

            # save_path = args.outdir

            acquisitions, actions_selected_vals, actions_corresponding_vals = [], [], []
            actions_before, actions_during, actions_after = [], [], []
            obs = eval_env.reset()
            # là faut jtrouve les flash step des pour les 3 différentes sections : avant, pendant, après le flash
            summed_flashes = numpy.sum(eval_env.temporal_datamap.flash_tstack, axis=(1, 2))
            during_threshold = 65 * len(eval_env.temporal_datamap.synapse.nanodomains_coords)
            during_idx = numpy.min(numpy.squeeze(numpy.argwhere(summed_flashes >= during_threshold)))

            flash_end_threshold = 15 * len(eval_env.temporal_datamap.synapse.nanodomains_coords)
            during_start_idx = numpy.argmax(summed_flashes)
            after_start_idx = numpy.argwhere(summed_flashes <= flash_end_threshold)

            after_start_idx = numpy.squeeze(after_start_idx).tolist()
            after_start_idx.reverse()

            after_idx = None
            for idx, elt in enumerate(after_start_idx):
                if elt - 1 != after_start_idx[idx + 1]:
                    after_idx = elt
                    break


            # je veux save les positions gt des nanodomaines
            nanodomain_coords = numpy.array(eval_env.temporal_datamap.synapse.nanodomains_coords)
            file_nd_analysis.write(f"{nanodomain_coords.shape[0]}\n")
            nd_assigned_truth_list = []
            new_threshold = numpy.floor(fixed_n_nanodomains_threshold *
                                        nanodomain_coords.shape[0]) / nanodomain_coords.shape[0]
            file_nd_analysis.write(f"{new_threshold}\n")
            for j in range(nanodomain_coords.shape[0]):
                nd_assigned_truth_list.append(0)
            done = False
            action_nb = 0
            guess_coords_list = []
            while not done:
                a = agent.act(obs)
                # print(f"action number {action_nb} = {a}")
                actions_selected_vals.append(a)
                actions_corresponding_vals.append(numpy.clip(a, action_space.low, action_space.high))
                if eval_env.temporal_experiment.flash_tstep < during_idx:
                    actions_before.append(numpy.clip(a, action_space.low, action_space.high))
                elif (eval_env.temporal_experiment.flash_tstep >= during_idx) and \
                        (eval_env.temporal_experiment.flash_tstep < after_idx):
                    actions_during.append(numpy.clip(a, action_space.low, action_space.high))
                else:
                    actions_after.append(numpy.clip(a, action_space.low, action_space.high))

                obs, r, done, info = eval_env.step(a)

                acquisitions.append(obs[0][-1])
                action_nb += 1

                # try thresholding on the acquired img obs[0][-1]
                guess_coords = peak_local_max(obs[0][-1], min_distance=2, threshold_rel=0.5)
                guess_coords_list.append(guess_coords)
                detector = metrics.CentroidDetectionError(nanodomain_coords, guess_coords, 2, algorithm="hungarian")
                for nd in detector.truth_couple:
                    nd_assigned_truth_list[nd] = 1

            acquisitions = numpy.array(acquisitions)
            actions_selected_vals = numpy.array(actions_selected_vals)
            actions_corresponding_vals = numpy.array(actions_corresponding_vals)

            if numpy.sum(nd_assigned_truth_list) / nanodomain_coords.shape[0] >= new_threshold:
                checkpoints_n_successes[idx_checkpoint] += 1
            file_nd_analysis.write(f"{numpy.sum(nd_assigned_truth_list)}\n")
            file_nd_analysis.close()

            actions_before = numpy.array(actions_before)
            actions_during = numpy.array(actions_during)
            actions_after = numpy.array(actions_after)
            numpy.save(current_dir + f"/analysis_results/actions/actions_before_{i}.npy", actions_before)
            numpy.save(current_dir + f"/analysis_results/actions/actions_during_{i}.npy", actions_during)
            numpy.save(current_dir + f"/analysis_results/actions/actions_after_{i}.npy", actions_after)


        # exit()   # juste tester que la méchanique marche pour 1 image pour le premier checkpoint

    # checkpoints_n_successes_percentage = checkpoints_n_successes / args.analysis_n_runs
    #
    # numpy.save(save_path + "/checkpoints_n_successes", checkpoints_n_successes)
    # numpy.save(save_path + "/checkpoints_n_successes_percentage", checkpoints_n_successes_percentage)


if __name__ == "__main__":
    main()