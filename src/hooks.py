
from abc import ABCMeta, abstractmethod

from pfrl.experiments import EvaluationHook

class EvaluationActionHook(EvaluationHook):

    support_train_agent = True
    support_train_agent_batch = True
    support_train_agent_async = False

    def __init__(self):
        super().__init__()

    def __call__(self, env, agent, evaluator, step, eval_stats, agent_stats, env_stats):
        """Call the hook.

        Args:
            env: Environment.
            agent: Agent.
            evaluator: Evaluator.
            step: Current timestep. (Not the number of evaluations so far)
            eval_stats (dict): Last evaluation stats from
                pfrl.experiments.evaluator.eval_performance().
            agent_stats (List of pairs): Last agent stats from
                agent.get_statistics().
            env_stats: Last environment stats from
                env.get_statistics().
        """
        pass
        # print(agent_stats)
        # print(env_stats)
        # print("CALLLED")
