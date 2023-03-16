
import numpy
import gym
import pfrl
import torch
import functools

from torch import nn
from gym_sted.utils import RecordingQueue

def calc_shape(shape, layers):
    """
    Calculates the shape of the tensor after the layers

    :param shape: A `tuple` of the input shape
    :param layers: A `list`-like of layers

    :returns : A `tuple` of the output shape
    """
    # _shape = numpy.array(shape[1:])
    _shape = numpy.array(shape[-2:])
    for layer in layers:
        if not isinstance(layer, (nn.Conv2d)):
            continue
        _shape = (_shape + 2 * numpy.array(layer.padding) - numpy.array(layer.dilation) * (numpy.array(layer.kernel_size) - 1) - 1) / numpy.array(layer.stride) + 1
        _shape = _shape.astype(int)
    # return (shape[0], *_shape)
    return _shape


class Policy(nn.Module):
    def __init__(
        self, in_channels=1, action_size=1, obs_space=None,
        activation=nn.functional.leaky_relu
    ):
        if isinstance(obs_space, gym.spaces.Tuple):
            self.in_channels = obs_space[0].shape[0]
        elif isinstance(obs_space, gym.spaces.Box):
            self.in_channels = obs_space.shape[0]
        else:
            self.in_channels = in_channels
        # self.in_channels = in_channels
        self.action_size = action_size
        self.obs_space = obs_space
        self.activation = activation
        super(Policy, self).__init__()

        # Creates the layers of the model
        self.layers = nn.ModuleList([
            nn.Conv2d(self.in_channels, 16, 8, stride=4),
            nn.Conv2d(16, 32, 4, stride=2),
        ])
        if isinstance(self.obs_space, gym.spaces.Tuple):
            out_shape = calc_shape(self.obs_space[0].shape, self.layers)
            in_features = 32 * numpy.prod(out_shape) + self.obs_space[1].shape[0]
        else:
            out_shape = calc_shape(self.obs_space.shape, self.layers)
            in_features = 32 * numpy.prod(out_shape)
        self.policy =  nn.Sequential(
            nn.Linear(in_features, action_size),
            pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=action_size,
                var_type="diagonal",
                var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            )
        )

    def forward(self, x):
        if isinstance(self.obs_space, gym.spaces.Tuple):
            x, articulation = x
        for layer in self.layers:
            x = self.activation(layer(x))
        x = x.view(x.size(0), -1)
        if isinstance(self.obs_space, gym.spaces.Tuple):
            x = torch.cat([x, articulation], dim=1)
        x = self.policy(x)
        return x

class Policy2(nn.Module):
    """
    if Policy was so good, why isn't there a Policy 2?
    """
    def __init__(
        self, in_channels=1, action_size=1, obs_space=None, encoded_signal_shape=4,
        activation=nn.functional.leaky_relu
    ):
        self.in_channels = in_channels
        self.action_size = action_size
        self.obs_space = obs_space
        self.activation = activation
        super(Policy2, self).__init__()

        self.encoded_signal_shape = encoded_signal_shape   # param d'entrée ?
        self.img_shape = self.obs_space[0].shape

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

        # Signal encoder ([SNR, Resolution, Bleach] to a vector of length 4 (?)
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

        # Encode to single channel image
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

class PolicyWithSideAction(nn.Module):
    """
    if Policy was so good, why isn't there a Policy 2?
    """
    def __init__(
        self, in_channels=1, action_size=1, obs_space=None, encoded_signal_shape=4,
        activation=nn.functional.leaky_relu
    ):
        self.in_channels = in_channels
        self.action_size = action_size
        self.obs_space = obs_space
        self.activation = activation
        super(PolicyWithSideAction, self).__init__()

        self.encoded_signal_shape = encoded_signal_shape   # param d'entrée ?
        self.img_shape = (1, 64, 64)

        # mon obs_space est un tuple, maintenant il faut déterminer la shape du premier elt du tuple:
        # c'est soit (1, 64, 64) (gym1) ou (4, 64, 64)
        # il faut stoquer cette info qqpart pour pouvoir faire le forward comme il faut
        if self.obs_space[0].shape[0] == 1:
            self.is_timed_env = False
        else:
            self.is_timed_env = True

        # RecordingQueue encoder (4 images to 1)
        self.recording_queue_encoder_layers = nn.ModuleList([
            nn.Conv2d(self.obs_space[0].shape[0], 1, 3, stride=1, padding=1)
        ])

        # Image encoder (1 image to a vector) (This is the LargeAtari architecture)
        self.image_encoder_layers = nn.ModuleList([
            nn.Conv2d(self.in_channels, 32, 8, stride=4),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.Conv2d(64, 64, 3, stride=1),
        ])

        # Signal encoder ([SNR, Resolution, Bleach] to a vector of length 4 (?)
        self.signal_encoder_layers = nn.ModuleList([
            nn.Linear(self.obs_space[1].shape[0], 16),
            nn.Linear(16, self.encoded_signal_shape)
        ])

        out_shape = calc_shape(self.img_shape, self.image_encoder_layers)
        in_features = 64 * numpy.prod(out_shape) + self.encoded_signal_shape

        # Action selection
        self.policy_to_ranking_layer = nn.Linear(in_features, 1)
        if self.is_timed_env:
            self.policy_to_actions_layer = nn.Linear(in_features, action_size)
            self.pfrl_head = pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=action_size,
                var_type="diagonal",
                var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            )
        else:
            self.policy_to_actions_layer = nn.Linear(in_features, action_size - 1)
            self.pfrl_head = pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=action_size,   # + 1,
                var_type="diagonal",
                var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            )

    def forward(self, x):
        # je crois que oui peu importe que ce soit gym 1 ou gym 2
        # si c'est gym 1, x = (img, [SNR, Resolution, Bleach, 1hot])
        # si c'est gym 2, x = ([4 imgs], [SNR, Resolution, Bleach, 1hot])
        if isinstance(self.obs_space, gym.spaces.Tuple):
            x, articulation = x

        # passer x dans une couche qui send de 4 imgs vers 1
        if self.is_timed_env:
            for layer in self.recording_queue_encoder_layers:
                x = self.activation(layer(x))

        # passer l'image dans le nn standard
        for layer in self.image_encoder_layers:
            x = self.activation(layer(x))
        x = x.view(x.size(0), -1)

        # passer le signal dans une couche linéaire X --> 4
        for layer in self.signal_encoder_layers:
            articulation = self.activation(layer(articulation))

        if isinstance(self.obs_space, gym.spaces.Tuple):
            x = torch.cat([x, articulation], dim=1)

        if self.is_timed_env:
            x = self.policy_to_actions_layer(x)
        else:
            x_1 = self.policy_to_actions_layer(x)
            x_2 = self.policy_to_ranking_layer(x)
            x = torch.cat([x_1, x_2], dim=1)

        x = self.pfrl_head(x)
        return x

class RecurrentPolicy(nn.Module, pfrl.nn.Recurrent):
    """
    Implements a `ReccurentPolicy` using the provided utilitaries in the `pfrl`
    library
    """
    def __init__(
        self, in_channels=1, action_size=1, obs_space=None, encoded_signal_shape=4,
        activation=nn.LeakyReLU
    ):
        self.in_channels = in_channels
        self.action_size = action_size
        self.obs_space = obs_space
        self.activation = activation
        super(RecurrentPolicy, self).__init__()

        self.encoded_signal_shape = encoded_signal_shape   # param d'entrée ?
        self.img_shape = (self.in_channels, 64, 64)

        # Image encoder (1 image to a vector) (This is the LargeAtari architecture)
        self.image_encoder_layers = pfrl.nn.RecurrentSequential(
            nn.Conv2d(self.obs_space[0].shape[0], self.in_channels, 3, stride=1, padding=1),
            self.activation(),
            nn.Conv2d(self.in_channels, 32, 8, stride=4),
            self.activation(),
            nn.Conv2d(32, 64, 4, stride=2),
            self.activation(),
            nn.Conv2d(64, 64, 3, stride=1),
            self.activation(),
            nn.Flatten()
        )

        # Signal encoder ([SNR, Resolution, Bleach] to a vector of length 4 (?)
        self.signal_encoder_layers = pfrl.nn.RecurrentSequential(
            nn.Linear(self.obs_space[1].shape[0], 16),
            self.activation(),
            nn.Linear(16, self.encoded_signal_shape),
            self.activation()
        )

        out_shape = calc_shape(self.img_shape, self.image_encoder_layers)
        in_features = 64 * numpy.prod(out_shape) + self.encoded_signal_shape

        # Action selection using LSTM network
        self.policy_to_actions_layer = pfrl.nn.RecurrentSequential(
            nn.LSTM(
                input_size=in_features, hidden_size=16, num_layers=1, batch_first=False
            ),
            nn.Linear(
                in_features=16, out_features=self.action_size
            )
        )

        # Policy
        self.pfrl_head = pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        )

    def forward(self, x, r):

        # Separates the image and the articulation
        x, articulation = x

        # Encodes the image signal
        # There are no recurrent layer in this layer
        # We do not modify the recurrent state
        x, _ = self.image_encoder_layers(x, None)

        # Encodes the articulation signal
        # There are no recurrent layer in this layer
        # We do not modify the recurrent state
        articulation, _ = self.signal_encoder_layers(articulation, None)

        # Combines the encoded image and articulation layers
        batch_sizes, sorted_indices, unsorted_indices = x.batch_sizes, x.sorted_indices, x.unsorted_indices
        x = torch.cat([x.data, articulation.data], dim=1)
        x = nn.utils.rnn.PackedSequence(
            data=x, batch_sizes=batch_sizes,
            sorted_indices=sorted_indices, unsorted_indices=unsorted_indices
        )

        # Applies the lstm models on both inputs
        x, r = self.policy_to_actions_layer(x, r)
        x = pfrl.utils.recurrent.unwrap_packed_sequences_recursive(x)

        # Creates the sampling distribution
        x = self.pfrl_head(x)
        return x, r

class RecurrentPolicyWithoutVision(nn.Module, pfrl.nn.Recurrent):
    """
    Implements a `ReccurentPolicy` using the provided utilitaries in the `pfrl`
    library
    """
    def __init__(
        self, in_channels=1, action_size=1, obs_space=None, encoded_signal_shape=4,
        activation=nn.LeakyReLU
    ):
        self.in_channels = in_channels
        self.action_size = action_size
        self.obs_space = obs_space
        self.activation = activation
        super(RecurrentPolicyWithoutVision, self).__init__()

        self.encoded_signal_shape = encoded_signal_shape   # param d'entrée ?
        self.img_shape = (1, 64, 64)

        # Signal encoder ([SNR, Resolution, Bleach] to a vector of length 4 (?)
        self.signal_encoder_layers = pfrl.nn.RecurrentSequential(
            nn.Linear(self.obs_space.shape[0], 16),
            self.activation(),
            nn.Linear(16, self.encoded_signal_shape),
            self.activation()
        )

        # Action selection using LSTM network
        in_features = self.encoded_signal_shape
        self.policy_to_actions_layer = pfrl.nn.RecurrentSequential(
            nn.LSTM(
                input_size=in_features, hidden_size=16, num_layers=1, batch_first=False
            ),
            nn.Linear(
                in_features=16, out_features=self.action_size
            )
        )

        # Policy
        self.pfrl_head = pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        )

    def forward(self, x, r):

        # Encodes the articulation signal
        # There are no recurrent layer in this layer
        # We do not modify the recurrent state
        # print("in x:", x)
        # print("in r:", r)
        x, _ = self.signal_encoder_layers(x, None)

        # Applies the lstm models on both inputs
        x, r = self.policy_to_actions_layer(x, r)
        batch_sizes, sorted_indices = pfrl.utils.recurrent.get_packed_sequence_info(x)
        x = pfrl.utils.recurrent.unwrap_packed_sequences_recursive(x)
        # Creates the sampling distribution
        x = self.pfrl_head(x)

        # Wraps back the sequence
        x = pfrl.utils.recurrent.wrap_packed_sequences_recursive(x, batch_sizes, sorted_indices)
        return x, r

class RecurrentPolicyWithSideAction(nn.Module, pfrl.nn.Recurrent):
    """
    Implements a `ReccurentPolicy` using the provided utilitaries in the `pfrl`
    library
    """
    def __init__(
        self, in_channels=1, action_size=1, obs_space=None, encoded_signal_shape=4,
        activation=nn.LeakyReLU
    ):
        self.in_channels = in_channels
        self.action_size = action_size
        self.obs_space = obs_space
        self.activation = activation
        super(RecurrentPolicyWithSideAction, self).__init__()

        self.encoded_signal_shape = encoded_signal_shape   # param d'entrée ?
        self.img_shape = (1, 64, 64)

        # Image encoder (1 image to a vector) (This is the LargeAtari architecture)
        self.image_encoder_layers = pfrl.nn.RecurrentSequential(
            nn.Conv2d(self.in_channels, 32, 8, stride=4),
            self.activation(),
            nn.Conv2d(32, 64, 4, stride=2),
            self.activation(),
            nn.Conv2d(64, 64, 3, stride=1),
            self.activation(),
            nn.Flatten()
        )

        # Signal encoder ([SNR, Resolution, Bleach] to a vector of length 4 (?)
        self.signal_encoder_layers = pfrl.nn.RecurrentSequential(
            nn.Linear(self.obs_space[1].shape[0], 16),
            self.activation(),
            nn.Linear(16, self.encoded_signal_shape),
            self.activation()
        )

        out_shape = calc_shape(self.img_shape, self.image_encoder_layers)
        in_features = 64 * numpy.prod(out_shape) + self.encoded_signal_shape

        # Action selection using LSTM network
        self.policy_to_actions_layer = pfrl.nn.RecurrentSequential(
            nn.LSTM(
                input_size=in_features, hidden_size=512, num_layers=1, batch_first=False
            ),
            nn.Linear(
                in_features=512, out_features=self.action_size - 1
            )
        )
        self.policy_to_ranking_layer = pfrl.nn.RecurrentSequential(
            nn.LSTM(
                input_size=in_features, hidden_size=512, num_layers=1, batch_first=False
            ),
            nn.Linear(
                in_features=512, out_features=1
            )
        )

        # Policy
        self.pfrl_head = pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,   # + 1,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        )

    def forward(self, x, r):

        # Separates the image and the articulation
        x, articulation = x

        # Encodes the image signal
        # There are no recurrent layer in this layer
        # We do not modify the recurrent state
        x, _ = self.image_encoder_layers(x, None)

        # Encodes the articulation signal
        # There are no recurrent layer in this layer
        # We do not modify the recurrent state
        articulation, _ = self.signal_encoder_layers(articulation, None)

        # Combines the encoded image and articulation layers
        batch_sizes, sorted_indices, unsorted_indices = x.batch_sizes, x.sorted_indices, x.unsorted_indices
        x = torch.cat([x.data, articulation.data], dim=1)
        x = nn.utils.rnn.PackedSequence(
            data=x, batch_sizes=batch_sizes,
            sorted_indices=sorted_indices, unsorted_indices=unsorted_indices
        )

        # Applies the lstm models on both inputs
        x_1, x_1_r = self.policy_to_actions_layer(x, r)
        x_1 = pfrl.utils.recurrent.unwrap_packed_sequences_recursive(x_1)
        x_2, x_2_r = self.policy_to_ranking_layer(x, r)
        x_2 = pfrl.utils.recurrent.unwrap_packed_sequences_recursive(x_2)

        # Concatenate state
        x = torch.cat([x_1, x_2], dim=1)

        # Combines recurrent state by an element-wise sum
        r = tuple([tuple([functools.reduce(lambda x1, x2: x1 + x2, rs) for rs in zip(x_1_r[0], x_2_r[0])])])

        # Creates the sampling distribution
        x = self.pfrl_head(x)
        return x, r

class ValueFunction(nn.Module):
    def __init__(
        self, in_channels=1, action_size=1, obs_space=None,
        activation=torch.tanh
    ):
        if isinstance(obs_space, gym.spaces.Tuple):
            self.in_channels = obs_space[0].shape[0]
        elif isinstance(obs_space, gym.spaces.Box):
            self.in_channels = obs_space.shape[0]
        else:
            self.in_channels = in_channels
        # self.in_channels = in_channels
        self.action_size = action_size
        self.obs_space = obs_space
        self.activation = activation
        super(ValueFunction, self).__init__()

        # Creates the layers of the model
        self.layers = nn.ModuleList([
            nn.Conv2d(self.in_channels, 16, 8, stride=4),
            nn.Conv2d(16, 32, 4, stride=2),
        ])
        if isinstance(self.obs_space, gym.spaces.Tuple):
            out_shape = calc_shape(self.obs_space[0].shape, self.layers)
            in_features = 32 * numpy.prod(out_shape) + self.obs_space[1].shape[0]
        else:
            out_shape = calc_shape(self.obs_space.shape, self.layers)
            in_features = 32 * numpy.prod(out_shape)
        self.linears = nn.ModuleList([
            nn.Linear(in_features, 64),
            nn.Linear(64, action_size)
        ])

    def forward(self, x):
        if isinstance(self.obs_space, gym.spaces.Tuple):
            x, articulation = x
        for layer in self.layers:
            x = self.activation(layer(x))
        x = x.view(x.size(0), -1)

        if isinstance(self.obs_space, gym.spaces.Tuple):
            x = torch.cat([x, articulation], dim=1)

        for layer in self.linears:
            x = self.activation(layer(x))
        return x

class ValueFunction2(nn.Module):
    """
    if ValueFunction was so good, why isn't there a ValueFunction 2?
    """
    def __init__(
        self, in_channels=1, action_size=1, obs_space=None, encoded_signal_shape=4,
        activation=torch.tanh
    ):
        self.in_channels = in_channels
        self.action_size = action_size
        self.obs_space = obs_space
        self.activation = activation
        super(ValueFunction2, self).__init__()

        self.encoded_signal_shape = encoded_signal_shape
        self.img_shape = self.obs_space[0].shape

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

        # Signal encoder ([SNR, Resolution, Bleach] to a vector of length 4 (?)
        self.signal_encoder_layers = nn.ModuleList([
            nn.Linear(self.obs_space[1].shape[0], 16),
            nn.Linear(16, self.encoded_signal_shape)
        ])

        out_shape = calc_shape(self.img_shape, self.image_encoder_layers)
        in_features = 64 * numpy.prod(out_shape) + self.encoded_signal_shape

        self.linear1 = nn.Linear(in_features, 64)
        self.linear2 = nn.Linear(64, action_size)

    def forward(self, x):
        # je crois que oui peu importe que ce soit gym 1 ou gym 2
        # si c'est gym 1, x = (img, [SNR, Resolution, Bleach, 1hot])
        # si c'est gym 2, x = ([4 imgs], [SNR, Resolution, Bleach, 1hot])
        if isinstance(self.obs_space, gym.spaces.Tuple):
            x, articulation = x

        for layer in self.recording_queue_encoder_layers:
            x = self.activation(layer(x))

        # passer l'image dans le nn standard
        for layer in self.image_encoder_layers:
            x = self.activation(layer(x))
        x = x.view(x.size(0), -1)

        # passer le signal dans une couche linéaire X --> 4
        for layer in self.signal_encoder_layers:
            articulation = self.activation(layer(articulation))

        if isinstance(self.obs_space, gym.spaces.Tuple):
            x = torch.cat([x, articulation], dim=1)

        x = self.activation(self.linear1(x))
        x = self.linear2(x)

        return x

class RecurrentValueFunction(nn.Module, pfrl.nn.Recurrent):
    """
    Implements a `ReccurentValueFunction` using the provided utilitaries in the `pfrl`
    library
    """
    def __init__(
        self, in_channels=1, action_size=1, obs_space=None, encoded_signal_shape=4,
        activation=nn.Tanh
    ):
        self.in_channels = in_channels
        self.action_size = action_size
        self.obs_space = obs_space
        self.activation = activation
        super(RecurrentValueFunction, self).__init__()

        self.encoded_signal_shape = encoded_signal_shape
        self.img_shape = (self.in_channels, 64, 64)

        # Image encoder (1 image to a vector) (This is the LargeAtari architecture)
        self.image_encoder_layers = pfrl.nn.RecurrentSequential(
            nn.Conv2d(self.obs_space[0].shape[0], self.in_channels, 3, stride=1, padding=1),
            self.activation(),
            nn.Conv2d(self.in_channels, 32, 8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        # Signal encoder ([SNR, Resolution, Bleach] to a vector of length 4 (?)
        self.signal_encoder_layers = pfrl.nn.RecurrentSequential(
            nn.Linear(self.obs_space[1].shape[0], 16),
            nn.LeakyReLU(),
            nn.Linear(16, self.encoded_signal_shape),
            nn.LeakyReLU()
        )

        out_shape = calc_shape(self.img_shape, self.image_encoder_layers)
        in_features = 64 * numpy.prod(out_shape) + self.encoded_signal_shape

        # Action selection using LSTM network
        self.lstm = pfrl.nn.RecurrentSequential(
            nn.LSTM(
                input_size=in_features, hidden_size=16, num_layers=1, batch_first=False
            ),
            nn.Linear(
                in_features=16, out_features=self.action_size
            ),
            self.activation()
        )


    def forward(self, x, r):

        # Separates the image and the articulation
        x, articulation = x

        # Encodes the image signal
        # There are no recurrent layer in this layer
        # We do not modify the recurrent state
        x, _ = self.image_encoder_layers(x, None)

        # Encodes the articulation signal
        # There are no recurrent layer in this layer
        # We do not modify the recurrent state
        articulation, _ = self.signal_encoder_layers(articulation, None)

        # Combines the encoded image and articulation layers
        batch_sizes, sorted_indices, unsorted_indices = x.batch_sizes, x.sorted_indices, x.unsorted_indices
        x = torch.cat([x.data, articulation.data], dim=1)
        x = nn.utils.rnn.PackedSequence(
            data=x, batch_sizes=batch_sizes,
            sorted_indices=sorted_indices, unsorted_indices=unsorted_indices
        )

        # Applies the lstm models on both inputs
        x, r = self.lstm(x, r)

        return x, r

class RecurrentValueFunctionWithoutVision(nn.Module, pfrl.nn.Recurrent):
    """
    Implements a `ReccurentValueFunction` using the provided utilitaries in the `pfrl`
    library
    """
    def __init__(
        self, in_channels=1, action_size=1, obs_space=None, encoded_signal_shape=4,
        activation=nn.Tanh
    ):
        self.in_channels = in_channels
        self.action_size = action_size
        self.obs_space = obs_space
        self.activation = activation
        super(RecurrentValueFunctionWithoutVision, self).__init__()

        self.encoded_signal_shape = encoded_signal_shape
        # Signal encoder ([SNR, Resolution, Bleach] to a vector of length 4 (?)
        self.signal_encoder_layers = pfrl.nn.RecurrentSequential(
            nn.Linear(self.obs_space.shape[0], 16),
            nn.LeakyReLU(),
            nn.Linear(16, self.encoded_signal_shape),
            nn.LeakyReLU()
        )

        in_features = self.encoded_signal_shape
        # Action selection using LSTM network
        self.lstm = pfrl.nn.RecurrentSequential(
            nn.LSTM(
                input_size=in_features, hidden_size=16, num_layers=1, batch_first=False
            ),
            nn.Linear(
                in_features=16, out_features=self.action_size
            ),
            self.activation()
        )


    def forward(self, x, r):

        x, _ = self.signal_encoder_layers(x, None)

        # Applies the lstm models on both inputs
        x, r = self.lstm(x, r)

        return x, r
