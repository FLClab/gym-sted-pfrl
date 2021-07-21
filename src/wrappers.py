
import numpy
import gym
import gym.spaces

class WrapPyTorch(gym.ObservationWrapper):
    """
    Wraps the observation of an OpenAI gym into a PyTorch gym

    :param env: A `gym.env`
    """
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        width, height, features = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.transpose(2, 0, 1),
            env.observation_space.high.transpose(2, 0, 1),
            [features, width, height], dtype=env.observation_space.dtype)

    def observation(self, observation):
        """
        Converts the observation. We rescale the observation values within a semi
        0-1 range by dividing by 2**10 (1024).

        :param observation: A `numpy.ndarray` of the current observation

        :returns : A converted `numpy.ndarray` of the current observation
        """
        # We rescale the observation into a semi 0-1 range
        observation = observation / 2**10
        return observation.transpose((2, 0, 1))
