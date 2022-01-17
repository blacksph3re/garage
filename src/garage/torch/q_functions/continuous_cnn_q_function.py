"""This modules creates a continuous Q-function network."""

import torch
from torch import nn
import akro

from garage import InOutSpec
from garage.torch.modules import CNNModule, MLPModule


class ContinuousCNNQFunction(nn.Module):
    """Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(self, 
                 env_spec,
                 image_format='NCHW',
                 kernel_sizes=(5,5),
                 hidden_channels=(32,32),
                 strides=1,
                 paddings=0,
                 padding_mode='zeros',
                 max_pool=False,
                 pool_shape=None,
                 hidden_nonlinearity=nn.ReLU,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 pool_stride=1,
                 hidden_sizes_mlp=(32, 32),
                 hidden_nonlinearity_mlp=nn.Tanh,
                 hidden_w_init_mlp=nn.init.xavier_uniform_,
                 hidden_b_init_mlp=nn.init.zeros_,
                 layer_normalization_mlp=False,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                ):
        """Initialize class with multiple attributes.

        Args:
            env_spec (EnvSpec): Environment specification.
            **kwargs: Keyword arguments.

        """
        super().__init__()
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        self._cnn_module = CNNModule(InOutSpec(
                                      self._env_spec.observation_space, None),
                                     image_format=image_format,
                                     kernel_sizes=kernel_sizes,
                                     strides=strides,
                                     hidden_channels=hidden_channels,
                                     hidden_w_init=hidden_w_init,
                                     hidden_b_init=hidden_b_init,
                                     hidden_nonlinearity=hidden_nonlinearity,
                                     paddings=paddings,
                                     padding_mode=padding_mode,
                                     max_pool=max_pool,
                                     pool_shape=pool_shape,
                                     pool_stride=pool_stride,
                                     layer_normalization=layer_normalization)
        self._mlp_module = MLPModule(self._cnn_module.spec.output_space.flat_dim + self._env_spec.action_space.flat_dim, 
                                     1,
                                     hidden_sizes=hidden_sizes_mlp,
                                     hidden_nonlinearity=hidden_nonlinearity_mlp,
                                     hidden_w_init=hidden_w_init_mlp,
                                     hidden_b_init=hidden_b_init_mlp,
                                     output_nonlinearity=output_nonlinearity,
                                     output_w_init=output_w_init,
                                     output_b_init=output_b_init,
                                     layer_normalization=layer_normalization_mlp)
        
        

    # pylint: disable=arguments-differ
    def forward(self, observations, actions):
        """Return Q-value(s).

        Args:
            observations (np.ndarray): observations.
            actions (np.ndarray): actions.

        Returns:
            torch.Tensor: Output value
        """
        # We're given flattened observations.
        observations = observations.reshape(
            -1, *self._env_spec.observation_space.shape)
        
        actions = actions.reshape(-1, *env_spec.action_space.shape)
        cnn_output = self._cnn_module(observations)
        mlp_output = self._mlp_module(torch.cat([cnn_output, actions], 1))

        return mlp_output
