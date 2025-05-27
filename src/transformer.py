"""Fleixble Transformer model.

The multi-head-attention class is to some degree copied from
https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/attention.py.

This code makes heavy use of Haiku but enables ablation studies on architecture
choices such as including linear projections, skip connections, normalization
layer. The aim is to interpolate between stripped down linear and the classic 
transformer architecture.
"""

import dataclasses
import sys
from typing import Optional, Tuple, Union, Any, List

import haiku as hk
import jax
import jax.numpy as jnp
from jax import Array

from attn import MultiHeadAttention, MLP


@dataclasses.dataclass
class Transformer(hk.Module):
    """A flexible Transformer implementation."""

    def __init__(
            self,
            num_heads: int = 1,
            num_layers: int = 3,
            key_size: int = 5,
            input_size: int = 2,
            output_size: int = 1,
            emb_size: int = 5,
            widening_factor: int = 4,
            mlp_output_dim: int = 0,
            second_layer: bool = False,
            input_mapping: bool = True,
            use_bias_p: bool = True,
            deq: bool = True,
            init_scale: float = 0.02,
            sum_norm: bool = False,
            ana_copy: bool = False,
            return_logits: bool = False,
            include_query: bool = False,
            kernel: str = 'linear',
            gamma: float = None,
            sigma: float = None,
            num_queries: int = 1,
            linear_approx: bool = False,
            use_mlp: bool = False,
            name: Optional[str] = None,
    ):

        """
        Initialises the module.

        Args:
            num_heads: Number of heads in the inner self-attention module.
            num_layers: Number of transformer layers, usually one due DEQ behaviour.
            key_size: Key and query size.
            output_size: Output size.
            use_bias_p: Use bias parameter in the linear operations in the network.
            deq: Use recurrent transformer.
            sum_norm: Use sum normalization from Schlag et. al 2012
            ana_copy: Return full prediction stack instead of last entry.
            include_query: Include query vector in computation.
            kernel: Kernel to use in outer self-attention module.
            gamma: Parameter for exp kernel in outer self-attention module.
            sigma: Parameter for rbf kernel in outer self-attention module.
            num_queries: Number of queries in each contextual dataset.
            name : Optional name for this module.
        """

        super().__init__(name=name)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.key_size = key_size
        self.input_mapping = input_mapping
        self.use_bias_p = use_bias_p
        self.input_size = input_size
        self.output_size = output_size
        self.emb_size = emb_size
        self.widening_factor = widening_factor
        self.mlp_output_dim = mlp_output_dim
        self.second_layer = second_layer
        self.init_scale = init_scale
        self.deq = deq
        self.sum_norm = sum_norm
        self.ana_copy = ana_copy
        self.return_logits = return_logits
        self.include_query = include_query
        self.kernel = kernel
        self.gamma = gamma
        self.sigma = sigma
        self.num_queries = num_queries
        self.linear_approx = linear_approx
        self.use_mlp = use_mlp

        # matrix to extract w_e's
        one_hots = jax.nn.one_hot(jnp.arange(0, self.output_size, step=1), num_classes=self.output_size)
        self.W_e_proj = jnp.concatenate([jnp.zeros((self.output_size, self.input_size)), one_hots,
                                         jnp.zeros((self.output_size, self.output_size + self.emb_size))], axis=1)

    def trans_block(self, h, w_e, nl):
        if self.deq:
            # print("Input to inner", h[0, 0, :])
            # self attention to update f
            if not self.include_query:
                key = h[:, :-self.num_queries, :]
                value = h[:, :-self.num_queries, :]
            else:
                key = h
                value = h

            # print("h input shape:", h.shape)
            # print("key input shape: ", key.shape)
            # print("value input shape: ", value.shape)

            h_attn_f, att_map_f = self.attn_block_f(h, key, value)

            h = h + h_attn_f

            # print("h output shape: ", h.shape)

            # print("Output of inner", h[0, 0, :])

            # self attention to update e
            if self.kernel == 'rbf' or self.kernel == 'laplacian':
                h_attn_e, att_map_e = self.attn_block_e(h, w_e[None, ...], w_e)
            else:
                h_attn_e, att_map_e = self.attn_block_e(h, w_e, w_e)

            h = h + h_attn_e

            if self.use_mlp:
                ff_input = h[:, :, -self.emb_size:]

                ff_output = h[:, :, -self.emb_size:] + self.mlp_block(ff_input)

                h = h.at[:, :, -self.emb_size:].set(ff_output)

            # print("Final output", h[0, 0, :])
        else:
            # self attention to update f
            attn_block_f = MultiHeadAttention(num_heads=2,
                                              key_size=self.key_size,
                                              model_size=self.model_size,
                                              w_init=self.w_init,
                                              use_bias_p=self.use_bias_p,
                                              sum_normalization=self.sum_norm,
                                              kernel=self.kernel,
                                              gamma=self.gamma,
                                              sigma=self.sigma,
                                              name="layer_{}_inner".format(nl))

            attn_block_e = MultiHeadAttention(num_heads=1,
                                              key_size=self.key_size,
                                              model_size=self.model_size,
                                              w_init=self.w_init,
                                              use_bias_p=self.use_bias_p,
                                              sum_normalization=self.sum_norm,
                                              kernel=self.kernel,
                                              gamma=self.gamma,
                                              sigma=self.sigma,
                                              name="layer_{}_outer".format(nl))

            if not self.include_query:
                key = h[:, :-self.num_queries, :]
                value = h[:, :-self.num_queries, :]
            else:
                key = h
                value = h

            h_attn_f, att_map_f = attn_block_f(h, key, value)

            h = h + h_attn_f

            # self attention to update e
            if self.kernel == 'rbf' or self.kernel == 'laplacian':
                h_attn_e, att_map_e = attn_block_e(h, w_e[None, ...], w_e)
            else:
                h_attn_e, att_map_e = attn_block_e(h, w_e, w_e)

            h = h + h_attn_e

            if self.use_mlp:
                mlp_block = MLP(w_init=self.w_init,
                                widening_factor=self.widening_factor,
                                second_layer=self.second_layer,
                                use_bias_p=self.use_bias_p,
                                output_dim=self.emb_size,
                                name="layer_{}_mlp".format(nl))

                ff_input = h[:, :, -self.emb_size:]

                ff_output = h[:, :, -self.emb_size:] + mlp_block(ff_input)

                h = h.at[:, :, -self.emb_size:].set(ff_output)

            # h = h + ff_output
            # att_map_e = None

        return h, att_map_f, att_map_e

    def __call__(
            self,
            x: jnp.ndarray,
            is_training: bool,
            predict_test: bool
    ) -> tuple[Array, Union[list[Any], list[Array]], list[Any], list[Any]]:
        """Computes the transformer forward pass.

        Args:
          x: Inputs.
          is_training: Whether we're training or not.
          predict_test: Test or train prediction.
        Returns:
          Array of shape [B, T, H].
        """
        self.w_init = hk.initializers.VarianceScaling(self.init_scale)

        # print("emb size: ", self.emb_size)

        # apply embeddings
        self.embedding_layer = hk.Linear(self.input_size + 3 * self.emb_size, with_bias=False, w_init=None,
                                         name="emb")
        embeddings = self.embedding_layer(x)

        # print(embeddings.shape)

        # extract embeddings
        W_e = self.embedding_layer(self.W_e_proj)[:, self.input_size:(self.input_size + self.emb_size)]

        # print("W_e shape: ", W_e.shape)

        h = embeddings

        if len(h.shape) == 2:
            _, model_size = h.shape
        elif len(h.shape) == 3:
            _, _, model_size = h.shape
        self.model_size = model_size

        # recurrent transformer
        if self.deq:
            self.attn_block_f = MultiHeadAttention(num_heads=2,
                                                   key_size=self.key_size,
                                                   model_size=self.model_size,
                                                   w_init=self.w_init,
                                                   use_bias_p=self.use_bias_p,
                                                   sum_normalization=self.sum_norm,
                                                   kernel=self.kernel,
                                                   gamma=self.gamma,
                                                   sigma=self.sigma,
                                                   name="update_inner")

            self.attn_block_e = MultiHeadAttention(num_heads=1,
                                                   key_size=self.key_size,
                                                   model_size=self.model_size,
                                                   w_init=self.w_init,
                                                   use_bias_p=self.use_bias_p,
                                                   sum_normalization=self.sum_norm,
                                                   kernel=self.kernel,
                                                   gamma=self.gamma,
                                                   sigma=self.sigma,
                                                   name="update_outer")

            if self.use_mlp:
                self.mlp_block = MLP(w_init=self.w_init,
                                     widening_factor=self.widening_factor,
                                     second_layer=self.second_layer,
                                     use_bias_p=self.use_bias_p,
                                     output_dim=self.emb_size)

        stack_h = []
        stack_att_f = []
        stack_att_e = []

        for nl in range(self.num_layers):
            h, att_map_f, att_map_e = self.trans_block(h, W_e, nl)

            # intermediate readout of test prediction
            f = jnp.squeeze(h[:, -self.num_queries:, -self.emb_size:])
            logits = f @ W_e.T
            st = jnp.squeeze(jax.nn.softmax(logits, axis=-1))

            stack_h.append(st)
            stack_att_f.append(att_map_f)
            stack_att_e.append(att_map_e)

        # apply softmax
        f = jnp.squeeze(h[:, -self.num_queries:, -self.emb_size:])
        logits = f @ W_e.T

        out = jax.nn.softmax(logits, axis=-1)

        return out, stack_h, stack_att_f
