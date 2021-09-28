
import numpy

from pfrl import agent
from matplotlib import pyplot

def scale(x, space, key):
    return x * (space[key]["high"] - space[key]["low"]) + space[key]["low"]

def rescale(x, space, key):
    return (x - space[key]["low"]) / (space[key]["high"] - space[key]["low"])

class HumanAgent(agent.BatchAgent):
    """
    Implements a manual selection of the desired action at each time steps in the
    environment.
    """
    def __init__(self, action_space):
        """
        Instantiates the `HumanAgent`

        :param action_space: A `dict` of the action space
        """
        super(HumanAgent, self).__init__()

        self.action_space = action_space

    def batch_act(self, batch_obs):
        """Select a batch of actions.

        Args:
            batch_obs (Sequence of ~object): Observations.

        Returns:
            Sequence of ~object: Actions.
        """
        image, context = batch_obs[0]

        reshaped = context.reshape(10, -1)
        print("p_sted\t --\tp_ex\t--\tpdt\t--\tResol\t--\tBleach\t--\tSNR")
        for r in reshaped:
            print("{:0.2f}\t--\t{:0.2f}\t--\t{:0.2f}\t--\t{:0.2f}\t--\t{:0.2f}\t--\t{:0.2f}".format(
                rescale(r[0], self.action_space, "p_sted"),
                rescale(r[1], self.action_space, "p_ex"),
                max(0, rescale(r[2], self.action_space, "pdt")),
                r[3],
                r[4],
                r[5]
            ))

        # fig, ax = pyplot.subplots()
        # ax.imshow(image[0], vmin=0, cmap="hot")
        # pyplot.show(block=True)

        p_sted = float(input("p_sted [%] : ")) / 100
        p_ex = float(input("p_ex [%] : ")) / 100
        pdt = float(input("pdt [%] : ")) / 100

        # p_sted, p_ex, pdt = numpy.random.rand(3)

        actions = numpy.array([[
            scale(p_sted, self.action_space, "p_sted"),
            scale(p_ex, self.action_space, "p_ex"),
            scale(pdt, self.action_space, "pdt"),
        ]], dtype=numpy.float32)
        return actions

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        """Observe a batch of action consequences.

        Args:
            batch_obs (Sequence of ~object): Observations.
            batch_reward (Sequence of float): Rewards.
            batch_done (Sequence of boolean): Boolean values where True
                indicates the current state is terminal.
            batch_reset (Sequence of boolean): Boolean values where True
                indicates the current episode will be reset, even if the
                current state is not terminal.

        Returns:
            None
        """
        pass

    def load(self):
        pass

    def save(self):
        pass

    def get_statistics(self):
        pass
