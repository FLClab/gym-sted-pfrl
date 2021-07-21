
import numpy
import gym
import pfrl
import torch

from torch import nn

def calc_shape(shape, layers):
    """
    Calculates the shape of the tensor after the layers

    :param shape: A `tuple` of the input shape
    :param layers: A `list`-like of layers

    :returns : A `tuple` of the output shape
    """
    _shape = numpy.array(shape[1:])
    for layer in layers:
        _shape = (_shape + 2 * numpy.array(layer.padding) - numpy.array(layer.dilation) * (numpy.array(layer.kernel_size) - 1) - 1) / numpy.array(layer.stride) + 1
        _shape = _shape.astype(int)
    return (shape[0], *_shape)

class Policy(nn.Module):
    def __init__(
        self, in_channels=1, action_size=1, obs_size=(1, 64, 64),
        activation=nn.functional.leaky_relu
    ):
        self.in_channels = in_channels
        self.action_size = action_size
        self.obs_size = obs_size
        self.activation = activation
        super(Policy, self).__init__()

        # Creates the layers of the model
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, 16, 8, stride=4),
            nn.Conv2d(16, 32, 4, stride=2),
        ])
        out_shape = calc_shape(obs_size, self.layers)
        self.policy =  nn.Sequential(
            nn.Linear(32 * numpy.prod(out_shape), action_size),
            pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=action_size,
                var_type="diagonal",
                var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            )
        )

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = x.view(x.size(0), -1)
        x = self.policy(x)
        return x

class ValueFunction(nn.Module):
    def __init__(
        self, in_channels=1, action_size=1, obs_size=(1, 64, 64),
        activation=torch.tanh
    ):
        self.in_channels = in_channels
        self.action_size = action_size
        self.obs_size = obs_size
        self.activation = activation
        super(ValueFunction, self).__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, 16, 8, stride=4),
            nn.Conv2d(16, 32, 4, stride=2),
        ])
        out_shape = calc_shape(obs_size, self.layers)
        self.linears = nn.ModuleList([
            nn.Linear(32 * numpy.prod(out_shape), 64),
            nn.Linear(64, action_size)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = x.view(x.size(0), -1)
        for layer in self.linears:
            x = self.activation(layer(x))
        return x
