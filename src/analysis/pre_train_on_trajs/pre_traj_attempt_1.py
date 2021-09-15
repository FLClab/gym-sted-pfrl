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

import itertools
from pfrl.utils.batch_states import batch_states
from pfrl.experiments.evaluator import save_agent

TIMEFMT = "%Y%m%d-%H%M%S"


def _add_log_prob_and_value_to_episodes(
    episodes,
    model,
    phi,
    batch_states,
    obs_normalizer,
    device,
):

    dataset = list(itertools.chain.from_iterable(episodes))

    # Compute v_pred and next_v_pred
    states = batch_states([b["state"] for b in dataset], device, phi)
    next_states = batch_states([b["next_state"] for b in dataset], device, phi)

    if obs_normalizer:
        states = obs_normalizer(states, update=False)
        next_states = obs_normalizer(next_states, update=False)

    with torch.no_grad(), pfrl.utils.evaluating(model):
        distribs, vs_pred = model(states)
        _, next_vs_pred = model(next_states)

        actions = torch.tensor([b["action"] for b in dataset], device=device)
        log_probs = distribs.log_prob(actions).cpu().numpy()
        vs_pred = vs_pred.cpu().numpy().ravel()
        next_vs_pred = next_vs_pred.cpu().numpy().ravel()

    for transition, log_prob, v_pred, next_v_pred in zip(
        dataset, log_probs, vs_pred, next_vs_pred
    ):
        transition["log_prob"] = log_prob
        transition["v_pred"] = v_pred
        transition["next_v_pred"] = next_v_pred


def _add_advantage_and_value_target_to_episode(episode, gamma, lambd):
    """Add advantage and value target values to an episode."""
    adv = 0.0
    for transition in reversed(episode):
        td_err = (
            transition["reward"]
            + (gamma * transition["nonterminal"] * transition["next_v_pred"])
            - transition["v_pred"]
        )
        adv = td_err + gamma * lambd * adv
        transition["adv"] = adv
        transition["v_teacher"] = adv + transition["v_pred"]


def _add_advantage_and_value_target_to_episodes(episodes, gamma, lambd):
    """Add advantage and value target values to a list of episodes."""
    for episode in episodes:
        _add_advantage_and_value_target_to_episode(episode, gamma=gamma, lambd=lambd)


def _make_dataset(
    episodes, model, phi, batch_states, obs_normalizer, gamma, lambd, device
):
    """Make a list of transitions with necessary information."""

    _add_log_prob_and_value_to_episodes(
        episodes=episodes,
        model=model,
        phi=phi,
        batch_states=batch_states,
        obs_normalizer=obs_normalizer,
        device=device,
    )

    _add_advantage_and_value_target_to_episodes(episodes, gamma=gamma, lambd=lambd)

    return list(itertools.chain.from_iterable(episodes))


def make_agent_act(agent, action, obs):
    """
    ??????????????
    Not sure how to do this ok :)
    ??????????????
    """
    assert agent.training
    b_state = agent.batch_states(obs, agent.device, agent.phi)

    if agent.obs_normalizer:
        b_state = agent.obs_normalizer(b_state, update=False)

    num_envs = len(obs)
    if agent.batch_last_episode is None:
        agent._initialize_batch_variables(num_envs)
    assert len(agent.batch_last_episode) == num_envs
    assert len(agent.batch_last_state) == num_envs
    assert len(agent.batch_last_action) == num_envs

    # action_distrib will be recomputed when computing gradients
    with torch.no_grad(), pfrl.utils.evaluating(agent.model):
        if agent.recurrent:
            assert agent.train_prev_recurrent_states is None
            agent.train_prev_recurrent_states = agent.train_recurrent_states
            (
                (action_distrib, batch_value),
                agent.train_recurrent_states,
            ) = one_step_forward(
                agent.model, b_state, agent.train_prev_recurrent_states
            )
        else:
            # là jpense que j'overwrite mon action, comment je fais pour pas faire ça? :)
            # genre ça va tu me calculer des trucs avec les mauvaises valeurs d'état et tout
            # vu que ça va essayer de sélectionner sa propre action ?
            action_distrib, batch_value = agent.model(b_state)
        # je pense que dans mon cas je fais juste faire que batch_action = action en input,
        # agent.entropy_record.extend(np.array([0]))
        # pour la value je fais quoi tho?
        # I guess jpeux add la valeur pareil direct vu que ça dépend pas de l'action jouée,
        # ça prend juste l'état en entrée sooo I should be good ?

        # batch_action = action_distrib.sample().cpu().numpy()
        # agent.entropy_record.extend(action_distrib.entropy().cpu().numpy())

        batch_action = numpy.array([action])
        agent.entropy_record.extend(numpy.array([0]))
        agent.value_record.extend(batch_value.cpu().numpy())

    agent.batch_last_state = list(obs)
    agent.batch_last_action = list(batch_action)

    return batch_action


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="gym_sted:STEDsum-v0")
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

    # PPO.initialize_batch_variables(num_envs)
    agent.batch_last_episode = [[] for _ in range(args.num_envs)]
    agent.batch_last_state = [None] * args.num_envs
    agent.batch_last_action = [None] * args.num_envs

    obs = sample_env.reset()

    done = False
    episode_len = 0
    max_episode_len = 50
    while not done:
        print("stepping!")
        # make this into a function or something
        # jpense qu'il faut que j'émulate ce qui se passe dans PPO._batch_act_train() ?
        action = numpy.array([10., 10., 10.])
        # the make_agent_act function makes it so everything has the right format as if the agent had
        # selected the action itself or something like that
        action = make_agent_act(agent, action, [obs])
        obs, r, done, info = sample_env.step(action[0])

        episode_len += 1
        reset = episode_len == max_episode_len or info.get("needs_reset", False)

        # agent.batch_last_action = list(action)
        # agent.batch_last_state = list(obs)

        agent.observe(obs, r, done, reset)
    print("done stepping")

    # build the dataset from the agent's memory
    dataset = _make_dataset(
        episodes=agent.memory,
        model=agent.model,
        phi=agent.phi,
        batch_states=agent.batch_states,
        obs_normalizer=agent.obs_normalizer,
        gamma=agent.gamma,
        lambd=agent.lambd,
        device=agent.device,
    )
    
    agent._update(dataset)

    logger = None or logging.getLogger(__name__)

    save_agent(agent, 0, "./data/pre_traj_tests", logger, suffix="_plswork")


if __name__ == "__main__":

    # Run the following line of code
    # python main.py --env gym_sted:STED-v0 --batchsize=16 --gpu=None --reward-scale-factor=1.0 --eval-interval=100 --eval-n-runs=5
    main()