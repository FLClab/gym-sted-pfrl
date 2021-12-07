
import numpy
import gym
import pfrl
import torch
import functools

from torch import nn

def calc_shape(shape, layers):
    """
    Calculates the shape of the tensor after the layers

    :param shape: A `tuple` of the input shape
    :param layers: A `list`-like of layers

    :returns : A `tuple` of the output shape
    """
    _shape = numpy.array(shape[-2:])
    for layer in layers:
        if not isinstance(layer, (nn.Conv2d)):
            continue
        _shape = (_shape + 2 * numpy.array(layer.padding) - numpy.array(layer.dilation) * (numpy.array(layer.kernel_size) - 1) - 1) / numpy.array(layer.stride) + 1
        _shape = _shape.astype(int)
    return _shape

class Policy(nn.Module):
    """
    Implements the `Policy` model of the `PPO` agent.

    The model encodes the image information using a Large Atari network.
    The model encodes the history information using a fully connected layer.
    Both encoding are concatenated and fed to the policy layer of the model.

    :param in_channels: A `int` of the number of input channels
    :param action_size: A `int` of the number of possible actions
    :param obs_space: A `gym` observation space object
    :param encoded_signal_shape: An `int` of the encoded signal shape
    :param activation: An activation function for the model
    """
    def __init__(
        self, in_channels=1, action_size=1, obs_space=None, encoded_signal_shape=4,
        activation=nn.functional.leaky_relu
    ):
        self.in_channels = in_channels
        self.action_size = action_size
        self.obs_space = obs_space
        self.activation = activation
        super(Policy, self).__init__()

        self.encoded_signal_shape = encoded_signal_shape
        self.img_shape = (self.in_channels, 64, 64)

        # RecordingQueue encoder (3 images to 3)
        self.recording_queue_encoder_layers = nn.ModuleList([
            nn.Conv2d(self.obs_space[0].shape[0], self.in_channels, 3, stride=1, padding=1)
        ])

        # Image encoder (1 image to a vector) (This is the LargeAtari architecture)
        self.image_encoder_layers = nn.ModuleList([
            nn.Conv2d(self.in_channels, 32, 8, stride=4),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.Conv2d(64, 64, 3, stride=1),
        ])

        # Signal encoder ([SNR, Resolution, Bleach] to a vector of length 4
        self.signal_encoder_layers = nn.ModuleList([
            nn.Linear(self.obs_space[1].shape[0], 16),
            nn.Linear(16, self.encoded_signal_shape)
        ])

        out_shape = calc_shape(self.img_shape, self.image_encoder_layers)
        in_features = 64 * numpy.prod(out_shape) + self.encoded_signal_shape

        # Action selection
        self.policy_to_actions_layer = nn.Linear(in_features, action_size)
        self.pfrl_head = pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        )

    def forward(self, x):
        # Split image and articulation
        if isinstance(self.obs_space, gym.spaces.Tuple):
            x, articulation = x

        # Encode to 3-channel image
        for layer in self.recording_queue_encoder_layers:
            x = self.activation(layer(x))

        # Encode image in AtariNetwork
        for layer in self.image_encoder_layers:
            x = self.activation(layer(x))
        x = x.view(x.size(0), -1)

        # Encode articulation
        for layer in self.signal_encoder_layers:
            articulation = self.activation(layer(articulation))

        # Combines encoded image and articulation
        if isinstance(self.obs_space, gym.spaces.Tuple):
            x = torch.cat([x, articulation], dim=1)

        # Runs the encoded into the policy
        x = self.policy_to_actions_layer(x)
        x = self.pfrl_head(x)
        return x

class ValueFunction(nn.Module):
    """
    Implements the `ValueFunction` model of the `PPO` agent.

    The model encodes the image information using a Large Atari network.
    The model encodes the history information using a fully connected layer.
    Both encoding are concatenated and fed to a fully connectd layer.

    :param in_channels: A `int` of the number of input channels
    :param action_size: A `int` of the number of possible actions
    :param obs_space: A `gym` observation space object
    :param encoded_signal_shape: An `int` of the encoded signal shape
    :param activation: An activation function for the model
    """
    def __init__(
        self, in_channels=1, action_size=1, obs_space=None, encoded_signal_shape=4,
        activation=torch.tanh
    ):
        self.in_channels = in_channels
        self.action_size = action_size
        self.obs_space = obs_space
        self.activation = activation
        super(ValueFunction, self).__init__()

        self.encoded_signal_shape = encoded_signal_shape
        self.img_shape = (self.in_channels, 64, 64)

        # RecordingQueue encoder (4 images to 1)
        self.recording_queue_encoder_layers = nn.ModuleList([
            nn.Conv2d(self.obs_space[0].shape[0], self.in_channels, 3, stride=1, padding=1)
        ])

        # Image encoder (1 image to a vector) (This is the LargeAtari architecture)
        self.image_encoder_layers = nn.ModuleList([
            nn.Conv2d(self.in_channels, 32, 8, stride=4),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.Conv2d(64, 64, 3, stride=1),
        ])

        # Signal encoder ([SNR, Resolution, Bleach] to a vector of length 4
        self.signal_encoder_layers = nn.ModuleList([
            nn.Linear(self.obs_space[1].shape[0], 16),
            nn.Linear(16, self.encoded_signal_shape)
        ])

        out_shape = calc_shape(self.img_shape, self.image_encoder_layers)
        in_features = 64 * numpy.prod(out_shape) + self.encoded_signal_shape

        self.linear1 = nn.Linear(in_features, 64)
        self.linear2 = nn.Linear(64, action_size)

    def forward(self, x):
        if isinstance(self.obs_space, gym.spaces.Tuple):
            x, articulation = x

        for layer in self.recording_queue_encoder_layers:
            x = self.activation(layer(x))

        for layer in self.image_encoder_layers:
            x = self.activation(layer(x))
        x = x.view(x.size(0), -1)

        for layer in self.signal_encoder_layers:
            articulation = self.activation(layer(articulation))

        if isinstance(self.obs_space, gym.spaces.Tuple):
            x = torch.cat([x, articulation], dim=1)

        x = self.activation(self.linear1(x))
        x = self.linear2(x)

        return x
