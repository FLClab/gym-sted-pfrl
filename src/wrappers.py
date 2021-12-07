
import numpy
import gym
import gym.spaces

class WrapPyTorch(gym.ObservationWrapper):
    """
    Wraps the observation of an OpenAI gym into a PyTorch gym

    :param env: A `gym.env`
    """
    def __init__(self, env=None, dtype=numpy.float32):
        super(WrapPyTorch, self).__init__(env)
        self.dtype = dtype
        if isinstance(env.observation_space, gym.spaces.Tuple):
            width, height, features = env.observation_space[0].shape
            self.observation_space = gym.spaces.Tuple((
                gym.spaces.Box(
                    env.observation_space[0].low.transpose(2, 0, 1),
                    env.observation_space[0].high.transpose(2, 0, 1),
                    [features, width, height], dtype=env.observation_space[0].dtype
                ),
                gym.spaces.Box(
                    env.observation_space[1].low,
                    env.observation_space[1].high,
                    env.observation_space[1].shape, dtype=env.observation_space[1].dtype,
                )
            ))
        else:
            if len(env.observation_space.shape) == 3:
                width, height, features = env.observation_space.shape
                self.observation_space = gym.spaces.Box(
                    env.observation_space.low.transpose(2, 0, 1),
                    env.observation_space.high.transpose(2, 0, 1),
                    [features, width, height], dtype=env.observation_space.dtype)
            else:
                self.observation_space = env.observation_space

    def observation(self, observation):
        """
        Converts the observation. We rescale the observation values within a semi
        0-1 range by dividing by 2**10 (1024).

        :param observation: A `numpy.ndarray` of the current observation

        :returns : A converted `numpy.ndarray` of the current observation
        """
        # Case where the observation also contains the preference of the model
        if isinstance(observation, (tuple, list)):
            observation, articulation = observation
            # We rescale the observation into a semi 0-1 range
            observation = observation / 2**10
            return tuple((observation.transpose((2, 0, 1)).astype(self.dtype), articulation.astype(self.dtype)))
        # Case where the observation contains only the current image
        elif len(self.observation_space.shape) == 3:
            # We rescale the observation into a semi 0-1 range
            observation = observation / 2**10
            return observation.transpose((2, 0, 1)).astype(self.dtype)
        else:
            return observation.astype(self.dtype)
