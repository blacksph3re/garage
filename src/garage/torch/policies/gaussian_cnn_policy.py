"""CategoricalCNNPolicy."""
import akro
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent

from garage.torch.distributions import TanhNormal
from garage import InOutSpec
from garage.torch.modules import CNNModule, GaussianMLPTwoHeadedModule
from garage.torch.policies.stochastic_policy import StochasticPolicy


class GaussianCNNPolicy(StochasticPolicy):
    """GaussianCNNPolicy.

    A policy that contains a CNN and a MLP to make prediction based on
    a Gaussian distribution.

    It only works with akro.Box action space.

    Args:
        env_spec (garage.EnvSpec): Environment specification.
        image_format (str): Either 'NCHW' or 'NHWC'. Should match env_spec. Gym
            uses NHWC by default, but PyTorch uses NCHW by default.
        kernel_sizes (tuple[int]): Dimension of the conv filters.
            For example, (3, 5) means there are two convolutional layers.
            The filter for first layer is of dimension (3 x 3)
            and the second one is of dimension (5 x 5).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        hidden_channels (tuple[int]): Number of output channels for CNN.
            For example, (3, 32) means there are two convolutional layers.
            The filter for the first conv layer outputs 3 channels
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        paddings (tuple[int]):  Zero-padding added to both sides of the input
        padding_mode (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        max_pool (bool): Bool for using max-pooling or not.
        pool_shape (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            shape (2, 2).
        pool_stride (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.

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
                 name='GaussianCNNPolicy',
                 hidden_sizes_mlp=(32, 32),
                 hidden_nonlinearity_mlp=nn.Tanh,
                 hidden_w_init_mlp=nn.init.xavier_uniform_,
                 hidden_b_init_mlp=nn.init.zeros_,
                 layer_normalization_mlp=False,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 init_std=1.0,
                 min_std=np.exp(-20.),
                 max_std=np.exp(2.),
                 std_parameterization='exp',
                 normal_distribution_cls=TanhNormal):

        if not isinstance(env_spec.action_space, akro.Box):
            raise ValueError('GaussianMLPPolicy only works '
                             'with akro.Box action space.')
        if isinstance(env_spec.observation_space, akro.Dict):
            raise ValueError('CNN policies do not support '
                             'akro.Dict observation spaces.')

        super().__init__(env_spec, name)

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
        self._mlp_module = GaussianMLPTwoHeadedModule(
            input_dim=self._cnn_module.spec.output_space.flat_dim,
            output_dim=self._env_spec.action_space.flat_dim,
            hidden_sizes=hidden_sizes_mlp,
            hidden_nonlinearity=hidden_nonlinearity_mlp,
            hidden_w_init=hidden_w_init_mlp,
            hidden_b_init=hidden_b_init_mlp,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization_mlp,
            normal_distribution_cls=normal_distribution_cls)

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Observations to act on.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors.
                Do not need to be detached, and can be on any device.
        """
        # We're given flattened observations.
        observations = observations.reshape(
            -1, *self._env_spec.observation_space.shape)
        cnn_output = self._cnn_module(observations)
        dist = self._mlp_module(cnn_output)
        
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)
        
        ret_mean = dist.mean.cpu()
        ret_log_std = (dist.variance.sqrt()).log().cpu()
        return dist, dict(mean=ret_mean, log_std=ret_log_std)
