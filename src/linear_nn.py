import sys

import dataclasses
import math
from typing import Optional
import warnings

import haiku as hk

import jax
import jax.numpy as jnp


@dataclasses.dataclass
class LinearNN(hk.Module):
    """A multi layer perceptron.

    This module is fully connected neural network.
    """

    def __init__(
            self,
            output_dim: int = 0,
            use_bias_p: bool = False,
            init_scale: float = 0.002,
            name: Optional[str] = None
    ):
        """Initialises the module.

        Args:
          w_init: Initializer for weights in the linear maps.
          use_bias_p: Use bias parameters in linear layers.
          output_dim: Output dimension of the MLP.
          name: Optional name for this module.
        """
        super().__init__(name=name)

        self.output_dim = output_dim
        self.use_bias_p = use_bias_p
        self.init_scale = init_scale
        self.w_init = hk.initializers.VarianceScaling(self.init_scale)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # MLP forward pass
        x = hk.Linear(self.output_dim, with_bias=self.use_bias_p, w_init=self.w_init)(x)
        x = jnp.reshape(x, (-1, self.output_dim))
        x = jnp.concatenate([x, jnp.zeros((x.shape[0], 1))], axis=-1)

        out = jax.nn.softmax(x, axis=-1)

        return out
