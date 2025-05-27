"""
Multi Head Attention

This class is, to a high degree, copied from
https://github.com/google-research/self-organising-systems/blob/master/transformers_learn_icl_by_gd/src/attn.py
"""
import sys

import dataclasses
import math
from typing import Optional
import warnings

import haiku as hk

import jax
import jax.numpy as jnp


@dataclasses.dataclass
class MultiHeadAttention(hk.Module):
    """Multi-headed attention (MHA) module.

    This module is intended for attending over sequences of vectors.
    Rough sketch:
    - Compute keys (K), queries (Q), and values (V) as projections of inputs.
    - Attention weights are computed as W = softmax(QK^T / sqrt(key_size)).
    - Output is another projection of WV^T.
    For more detail, see the original Transformer paper:
      "Attention is all you need" https://arxiv.org/abs/1706.03762.
    Glossary of shapes:
    - T: Sequence length.
    - D: Vector (embedding) size.
    - H: Number of attention heads.
    """

    def __init__(
            self,
            num_heads: int,
            key_size: int,
            w_init_scale: Optional[float] = None,
            *,
            w_init: Optional[hk.initializers.Initializer] = None,
            value_size: Optional[int] = None,
            model_size: Optional[int] = None,
            use_bias_p: Optional[bool] = False,
            sum_normalization: Optional[bool] = False,
            kernel: Optional[str] = 'linear',
            gamma: Optional[float] = None,
            sigma: Optional[float] = None,
            name: Optional[str] = None,
    ):
        """Initializes the module.

        Args:
            num_heads: Number of independent attention heads (H).
            key_size: The size of keys (K) and queries used for attention.
            w_init_scale: DEPRECATED. Please use w_init instead.
            w_init: Initializer for weights in the linear map.
            value_size: Optional size of the value projection (V). If None, defaults
              to the key size (K).
            model_size: Optional size of the output embedding (D'). If None, defaults
              to the key size multiplied by the number of heads (K * H).
            use_bias_p: Use bias parameters in the linear operations of the network.
            sum_normalization: Use sum normalization for the linear Transformer.
            kernel: Use specified kernel within the Transformer.
            gamma: Optional parameter for exp kernel.
            sigma: Optional parameter for rbf kernel.
            name: Optional name for this module.
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.use_bias_p = use_bias_p
        self.sum_normalization = sum_normalization
        self.kernel = kernel
        self.gamma = gamma
        self.sigma = sigma

        # Backwards-compatibility for w_init_scale.
        if w_init_scale is not None:
            warnings.warn(
                "w_init_scale is deprecated; please pass an explicit weight "
                "initializer instead.", DeprecationWarning)
        if w_init and w_init_scale:
            raise ValueError("Please provide only `w_init`, not `w_init_scale`.")
        if w_init is None and w_init_scale is None:
            raise ValueError("Please provide a weight initializer: `w_init`.")
        if w_init is None:
            w_init = hk.initializers.VarianceScaling(w_init_scale)
        self.w_init = w_init

    def __call__(
            self,
            query: jnp.ndarray,
            key: jnp.ndarray,
            value: jnp.ndarray,
            mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more 'batch-like' leading dimensions.
        Args:
            query: Embeddings sequence used to compute queries; shape [..., T', D_q].
            key: Embeddings sequence used to compute keys; shape [..., T, D_k].
            value: Embeddings sequence used to compute values; shape [..., T, D_v].
            mask: Optional mask applied to attention weights; shape [..., H=1, T', T].
        Returns:
            A new sequence of embeddings, consisting of a projection of the
              attention-weighted value projections; shape [..., T', D'].
        """

        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        *leading_dims, sequence_length, _ = query.shape
        projection = self._linear_projection

        # print("key shape: ", key.shape)

        # Compute key/query/values
        query_heads = projection(query, self.key_size, self.use_bias_p, "query")
        key_heads = projection(key, self.key_size, self.use_bias_p, "key")
        if self.sum_normalization:
            query_heads = query_heads / (jnp.sum(query_heads, axis=-1, keepdims=True) + 1e-6)
            key_heads = key_heads / (jnp.sum(key_heads, axis=-1)[..., None] + 1e-6)
        value_heads = projection(value, self.value_size, self.use_bias_p, "value")

        # Apply kernel
        if self.kernel == 'linear':
            # Compute linear attention logits.
            attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        if self.kernel == 'exp':
            # Compute exp attention logits.
            attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
            attn_logits = jnp.exp(attn_logits / self.gamma)
        elif self.kernel == 'rbf':
            # Compute RBF attention logits.
            # print("query heads shape: ", query_heads.shape)
            # print("key heads shape: ", key_heads.shape)
            l2 = jnp.sum((query_heads[:, :, None, :] - key_heads[:, None, :, :])**2, axis=-1)
            # print("l2 sum shape: ", l2.shape)
            l2 = jnp.moveaxis(l2, -1, 1)
            # print("l2 final shape: ", l2.shape)

            attn_logits = jnp.exp(-1 * l2 / (jnp.square(self.sigma)))

            # print("l2 shape: ", l2.shape)
        elif self.kernel == 'softmax':
            # Compute softmax attention logits.
            attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
            attn_logits = jax.nn.softmax(attn_logits)
        elif self.kernel == 'laplacian':
            # Compute laplacian attention logits.
            l1 = jnp.sum(jnp.abs(query_heads[:, :, None, :] - key_heads[:, None, :, :]), axis=-1)
            l1 = jnp.moveaxis(l1, -1, 1)

            attn_logits = jnp.exp(-1 * l1 / (jnp.square(self.sigma)))
        elif self.kernel != 'linear':
            raise Exception("Kernel must be linear, exp, rbf, laplacian, or softmax")

        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim}."
                )
            attn_logits = jnp.where(mask, attn_logits, -1e30)

        # [H, T', T]
        attn_weights = attn_logits

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings
        final_projection = hk.Linear(self.model_size, w_init=self.w_init,
                                     with_bias=self.use_bias_p, name="linear")
        attn = final_projection(attn)

        return attn, attn_weights  # [T', D']

    @hk.transparent
    def _linear_projection(
            self,
            x: jnp.ndarray,
            head_size: int,
            with_bias: Optional[bool] = False,
            name: Optional[str] = None,
    ) -> jnp.ndarray:
        y = hk.Linear(self.num_heads * head_size, with_bias=with_bias,
                      w_init=self.w_init, name=name)(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_heads, head_size))


@dataclasses.dataclass
class MLP(hk.Module):
    """A multi layer perceptron.

    This module is fully connected neural network, intended to process the
    result of the self-attention module. A couple of classic design choices
    have been already made such as using the gelu non-linearity,
    https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.gelu.html, as well
    as fixing the depth to 2. Since the depth of the MLP is not part of our
    analyses (for now) we do not allow for this flexibility.
    """

    def __init__(
            self,
            w_init: hk.initializers.Initializer,
            widening_factor: int = 4,
            second_layer: bool = False,
            use_bias_p: bool = False,
            output_dim: int = 0,
            name: Optional[str] = None
    ):
        """Initialises the module.

        Args:
          w_init: Initialiser for weights in the linear maps.
          widening_factor: Blow up in the hidden layer compared to input dimension.
          use_bias_p: Use pias parameters in linear layers.
          name: Optional name for this module.
        """
        super().__init__(name=name)
        self.w_init = w_init
        self.widening_factor = widening_factor
        self.second_layer = second_layer
        self.use_bias_p = use_bias_p
        self.output_dim = output_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        x = hk.Linear(self.widening_factor * hiddens, with_bias=self.use_bias_p,
                      w_init=self.w_init)(x)
        x = jax.nn.gelu(x)

        if self.second_layer:
            x = hk.Linear(self.widening_factor * hiddens, with_bias=self.use_bias_p,
                          w_init=self.w_init)(x)
            x = jax.nn.gelu(x)

        if self.output_dim == 0:
            return hk.Linear(hiddens, with_bias=self.use_bias_p,
                             w_init=self.w_init)(x)
        else:
            return hk.Linear(self.output_dim, with_bias=self.use_bias_p,
                             w_init=self.w_init)(x)
