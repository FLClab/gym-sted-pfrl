
import logging
import time

from abc import ABCMeta, abstractmethod

from pfrl.experiments import EvaluationHook
from pfrl.experiments import StepHook

class ProgressStepHook(StepHook):
    """
    Hook function that will be called in training to log current progress
    """
    def __init__(self, log_interval=256, logger=None):
        super().__init__()
        self.log_interval = log_interval
        self.logger = logger or logging.getLogger(__name__)

        self.start_time = time.time()

    def __call__(self, env, agent, step):
        """Call the hook.

        Args:
            env: Environment.
            agent: Agent.
            step: Current timestep.
        """
        if step % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            self.logger.info(
                "elapsed:{:0.2f}min step:{} time-per-step:{:0.2f}s".format(  # NOQA
                    elapsed / 60,
                    step,
                    elapsed / step,
                )
            )

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
