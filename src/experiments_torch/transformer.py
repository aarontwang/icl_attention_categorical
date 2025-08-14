import dataclasses
import sys
from typing import Optional, Tuple, Union, Any, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerGDInterleaved(nn.Module):
    """A flexible Transformer implementation."""

    def __init__(
            self,
            num_layers: int = 3,
            input_size: int = 2,
            num_categories: int = 1,
            emb_size: int = 5,
            init_scale: float = 0.02,
            include_query: bool = False,
            kernel: str = 'linear',
            num_queries: int = 1,
            device: str = None,
            init_seed: int = 0
    ):

        """
        Initialises the module.

        Args:
            num_layers: Number of transformer layers, usually one due DEQ behaviour.
            num_categories: Number of categories.
            emb_size: Embedding dimension of transformer.
            init_scale: Weight initialization scale.
            include_query: Include query vector in computation.
            kernel: Kernel to use in first self-attention module.
            num_queries: Number of queries in each contextual dataset.
        """

        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.num_categories = num_categories  # number of categories
        self.emb_size = emb_size
        self.init_scale = init_scale
        self.include_query = include_query
        self.kernel = kernel
        self.num_queries = num_queries
        self.device = device

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        gen = torch.Generator().manual_seed(init_seed)

        # Initialize learnable parameters
        self.alpha = nn.Parameter(torch.randn((self.num_layers, self.emb_size), generator=gen).to(self.device))
        self.kernel_param = nn.Parameter(torch.ones(1).to(self.device))

        # Initialize initial embedding matrix
        self.embedding_matrix = nn.Parameter(
            torch.randn((self.emb_size, self.num_categories), generator=gen).to(self.device))

        # Initialize query and key matrices for self-attention
        self.W_Q_1 = torch.zeros((2, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size)).to(
            self.device)
        self.W_K_1 = torch.zeros((2, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size)).to(
            self.device)

        self.W_Q_1[0, :self.input_size, :self.input_size] = torch.eye(self.input_size).to(self.device)
        self.W_K_1[0, :self.input_size, :self.input_size] = torch.eye(self.input_size).to(self.device)

        self.lam = 1e5
        self.W_Q_1[1, :self.input_size, :self.input_size] = self.lam * torch.eye(self.input_size).to(self.device)
        self.W_K_1[1, :self.input_size, :self.input_size] = self.lam * torch.eye(self.input_size).to(self.device)

        # Initialize query and key matrices for cross-attention
        self.W_Q_2 = torch.zeros((1, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size)).to(
            self.device)
        self.W_K_2 = torch.zeros((1, self.input_size + 3 * self.emb_size, self.emb_size)).to(
            self.device)

        self.W_Q_2[0, -self.emb_size:, -self.emb_size:] = torch.eye(self.emb_size).to(self.device)

        self.W_K_2[0, -self.emb_size:] = torch.eye(self.emb_size).to(self.device)

        # Initialize value matrix for self-attention
        self.W_V_1 = torch.zeros((2, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size)).to(
            self.device)
        self.W_V_1[1, self.input_size + self.emb_size:-self.emb_size,
        self.input_size + self.emb_size:-self.emb_size] = torch.eye(self.emb_size).to(self.device)

        # Initialize value matrix for cross-attention
        self.W_V_2 = torch.zeros((1, self.input_size + 3 * self.emb_size, self.emb_size)).to(
            self.device)

        self.W_V_2[0, self.input_size + self.emb_size:-self.emb_size] = torch.eye(self.emb_size).to(self.device)

        # Initialize projection matrix for self-attention
        self.P_1 = torch.zeros((2, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size)).to(
            self.device)
        self.P_1[0] = torch.eye(self.input_size + 3 * self.emb_size).to(self.device)
        self.P_1[1] = -1 * torch.eye(self.input_size + 3 * self.emb_size).to(self.device)

        # Initialize projection matrix for cross-attention
        self.P_2 = torch.zeros((1, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size)).to(
            self.device)
        self.P_2[0] = torch.eye(self.input_size + 3 * self.emb_size).to(self.device)

        self.W_e_proj = torch.zeros(
            (self.input_size + 3 * self.emb_size, self.input_size + 2 * self.num_categories + self.emb_size)).to(
            self.device)
        self.W_e_proj[:self.input_size, :self.input_size] = torch.eye(self.input_size).to(self.device)
        self.W_e_proj[-self.emb_size:, -self.emb_size:] = torch.eye(self.emb_size).to(self.device)

    def self_attn(self, key, query, value, layer):
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more 'batch-like' leading dimensions.
        Args:
            query: Embeddings sequence used to compute queries; shape [..., D_q, N].
            key: Embeddings sequence used to compute keys; shape [..., D_k, M].
            value: Embeddings sequence used to compute values; shape [..., D_v, M].
            layer: Layer number used for value of alpha.
        Returns:
            A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [..., T', D'].
        """
        n_context = key.size(-1)

        # h = number of heads
        # i = input_size + 3 * emb_size
        # j = input_size + 3 * emb_size
        # d = D_q = D_k = D_v
        # m = number of queries
        # n = number of keys/values (context length)

        # get keys and queries
        Q = torch.einsum("hid,bdm -> bhim", self.W_Q_1, query)  # [batch_size, 2, i_size + 3*e_size, query_length]
        K = torch.einsum("hid,bdn -> bhin", self.W_K_1, key)  # [batch_size, 2, i_size + 3*e_size, key_length]

        # shape = [b, h, i, n]
        W_V = self.W_V_1.clone()
        W_V[0, -self.emb_size:, self.input_size:self.input_size + self.emb_size] = torch.diag_embed(
            self.alpha[layer]).to(
            self.device)
        W_V[0, -self.emb_size:, self.input_size + self.emb_size:-self.emb_size] = -1 * torch.diag_embed(
            self.alpha[layer]).to(self.device)

        V = torch.einsum("hid,bdn -> bhin", W_V, value)

        # get attention scores
        if self.kernel == 'linear':
            scores_1 = self.kernel_param * torch.einsum('bim,bin -> bmn', Q[:, 0], K[:, 0])
        elif self.kernel == 'rbf':
            diff = Q[:, 0].unsqueeze(-1) - K[:, 0].unsqueeze(-2)
            sqdist = torch.sum(diff ** 2, dim=1)
            scores_1 = torch.exp(-sqdist / (2 * self.kernel_param ** 2))
        elif self.kernel == 'softmax':  # default to scaled dot-product
            scores_1 = self.kernel_param * torch.einsum('bim,bin -> bmn', Q[:, 0], K[:, 0])
            scores_1 = scores_1 / math.sqrt(Q.size(-2))
            scores_1 = F.softmax(scores_1, dim=-1)

        # for erasure component, always use softmax
        scores_2 = torch.einsum('bim,bin -> bmn', Q[:, 1], K[:, 1])
        scores_2 = scores_2 / math.sqrt(Q.size(-2))
        scores_2 = F.softmax(scores_2, dim=-1)

        # shape = [b, h, n, m]
        attention_scores = torch.stack([scores_1, scores_2], dim=1)

        # multiply by values
        # shape = [b, h, i, m]
        attn = torch.einsum("bhin,bhmn -> bhim", V, attention_scores)
        if self.kernel != 'softmax':
            attn = attn * 1 / n_context

        output = torch.einsum("hji,bhim -> bjm", self.P_1, attn)

        return output

    def cross_attn(self, key, query, value, layer):
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more 'batch-like' leading dimensions.
        Args:
            query: Embeddings sequence used to compute queries; shape [..., D_q, N].
            key: Embeddings sequence used to compute keys; shape [..., D_k, M].
            value: Embeddings sequence used to compute values; shape [..., D_v, M].
            layer: Layer number used for value of alpha.
        Returns:
            A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [..., T', D'].
        """

        # h = number of heads
        # i = input_size + 3 * emb_size
        # j = input_size + 3 * emb_size
        # d = emb_size
        # m = number of queries
        # n = number of keys/values (categories of embedding vectors)

        # get keys and queries
        Q = torch.einsum("hid,bdm -> bhim", self.W_Q_2,
                         query)  # [batch_size, num_heads, i_size + 3*e_size, query_length]
        K = torch.einsum("hid,...dn -> ...hin", self.W_K_2,
                         key)  # [batch_size, num_heads, i_size + 3*e_size, key_length]

        # shape = [b, h, i, n]
        V = torch.einsum("hid,...dn -> ...hin", self.W_V_2, value)

        scores = torch.einsum('...him,...hin -> ...hmn', Q, K)
        scores = F.softmax(scores, dim=-1)

        # shape = [b, h, n, m]
        attention_scores = scores

        # multiply by values
        # shape = [b, h, i, m]
        attn = torch.einsum("...hin,...hmn -> ...him", V, attention_scores)

        # multiply by projection matrix
        output = torch.einsum("hji,...him -> ...jm", self.P_2, attn)
        # print("output shape: ", output.shape)
        return output

    def trans_block(self, h, nl):
        # first attention block computes updated f_i and erases E(w_e)
        if not self.include_query:
            key = h[:, :, :-self.num_queries]
            value = h[:, :, :-self.num_queries]
        else:
            key = h
            value = h

        query = h

        h = h + self.self_attn(key, query, value, nl)

        # second attention block uses w_e to compute updated E(w_e)
        h = h + self.cross_attn(self.embedding_matrix, h, self.embedding_matrix, nl)

        return h

    def forward(self, x):
        """Computes the transformer forward pass.

        Args:
          x: Inputs.
          is_training: Whether we're training or not.
          predict_test: Test or train prediction.
        Returns:
          Array of shape [B, T, H].
        """
        # x shape = [b, input_size + 2 * num_categories + emb_size, num_samples]

        # transform input
        # shape = [b, input_size + 3 * emb_size, num_samples]
        W = self.W_e_proj.clone()
        W[self.input_size:self.input_size + self.emb_size,
        self.input_size:self.input_size + self.num_categories] = self.embedding_matrix
        W[self.input_size + self.emb_size:-self.emb_size,
        self.input_size + self.num_categories:-self.emb_size] = self.embedding_matrix

        h = torch.matmul(W, x)

        for nl in range(self.num_layers):
            # shape = [b, input_size + 3 * emb_size, num_samples]
            h = self.trans_block(h, nl)

        # shape = [b, emb_size, 1]
        f = torch.squeeze(h[:, -self.emb_size:, :])

        # embedding_matrix shape = [emb_size, num_categories]
        logits = torch.einsum("bdn,dc -> bnc", f, self.embedding_matrix)

        # shape = [b, num_samples, num_categories
        predictions = F.softmax(logits, dim=-1)

        return logits, predictions

    def train_step(self, features, labels):
        # get query labels
        # shape = [batch_size, num_queries]
        query_labels = labels[:, -self.num_queries:]

        # get transformer output
        # shape = [batch_size, num_samples, num_categories]
        logits, predictions = self.forward(features)

        # shape = [batch_size, num_queries, num_categories]
        query_logits = logits[:, -self.num_queries:, :]
        query_preds = predictions[:, -self.num_queries:, :]

        query_logits = torch.reshape(query_logits, (-1, self.num_categories))
        query_labels = torch.reshape(query_labels, (-1,))

        # compute loss
        loss_fn = nn.CrossEntropyLoss()

        query_loss = loss_fn(query_logits, query_labels)

        # compute accuracy
        # shape = [batch_size, num_queries]
        pred_cats = query_logits.argmax(dim=-1)
        query_accuracy = (pred_cats == query_labels).float().mean()

        return query_loss, query_accuracy


class TransformerInterleaved(nn.Module):
    """A flexible Transformer implementation."""

    def __init__(
            self,
            num_layers: int = 3,
            input_size: int = 2,
            num_categories: int = 1,
            emb_size: int = 5,
            init_scale: float = 0.02,
            include_query: bool = False,
            kernel: str = 'linear',
            num_queries: int = 1,
            device: str = None,
            init_seed: int = 0
    ):

        """
        Initialises the module.

        Args:
            num_layers: Number of transformer layers, usually one due DEQ behaviour.
            num_categories: Number of categories.
            emb_size: Embedding dimension of transformer.
            init_scale: Weight initialization scale.
            include_query: Include query vector in computation.
            kernel: Kernel to use in first self-attention module.
            num_queries: Number of queries in each contextual dataset.
        """

        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.num_categories = num_categories  # number of categories
        self.emb_size = emb_size
        self.init_scale = init_scale
        self.include_query = include_query
        self.kernel = kernel
        self.num_queries = num_queries
        self.device = device

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        gen = torch.Generator().manual_seed(init_seed)

        # Initialize query and key matrices for self-attention
        self.W_Q_1 = nn.Parameter(
            torch.randn((self.num_layers, 2, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size),
                        generator=gen).to(
                self.device) * self.init_scale)
        self.W_K_1 = nn.Parameter(
            torch.randn((self.num_layers, 2, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size),
                        generator=gen).to(
                self.device) * self.init_scale)

        # Initialize query and key matrices for cross-attention
        self.W_Q_2 = nn.Parameter(
            torch.randn((self.num_layers, 1, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size),
                        generator=gen).to(
                self.device) * self.init_scale)
        self.W_K_2 = nn.Parameter(
            torch.randn((self.num_layers, 1, self.input_size + 3 * self.emb_size, self.emb_size), generator=gen).to(
                self.device) * self.init_scale)

        # Initialize value matrix for self-attention
        self.W_V_1 = nn.Parameter(
            torch.randn((self.num_layers, 2, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size),
                        generator=gen).to(
                self.device) * self.init_scale)

        # Initialize value matrix for cross-attention
        self.W_V_2 = nn.Parameter(
            torch.randn((self.num_layers, 1, self.input_size + 3 * self.emb_size, self.emb_size), generator=gen).to(
                self.device) * self.init_scale)

        # Initialize projection matrix for self-attention
        self.P_1 = nn.Parameter(
            torch.randn((self.num_layers, 2, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size),
                        generator=gen).to(
                self.device) * self.init_scale)

        # Initialize projection matrix for cross-attention
        self.P_2 = nn.Parameter(
            torch.randn((self.num_layers, 1, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size),
                        generator=gen).to(
                self.device) * self.init_scale)

        # Initialize initial embedding matrix
        self.embedding_matrix = nn.Parameter(
            torch.randn((self.emb_size, self.num_categories), generator=gen).to(self.device))

        # Projection matrix
        self.W_e_proj = torch.zeros(
            (self.input_size + 3 * self.emb_size, self.input_size + 2 * self.num_categories + self.emb_size)).to(
            self.device)
        self.W_e_proj[:self.input_size, :self.input_size] = torch.eye(self.input_size).to(self.device)
        self.W_e_proj[-self.emb_size:, -self.emb_size:] = torch.eye(self.emb_size).to(self.device)

    def self_attn(self, key, query, value, layer):
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more 'batch-like' leading dimensions.
        Args:
            query: Embeddings sequence used to compute queries; shape [..., D_q, N].
            key: Embeddings sequence used to compute keys; shape [..., D_k, M].
            value: Embeddings sequence used to compute values; shape [..., D_v, M].
            layer: Layer number used for value of alpha.
        Returns:
            A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [..., T', D'].
        """
        n_context = key.size(-1)

        # h = number of heads
        # i = input_size + 3 * emb_size
        # j = input_size + 3 * emb_size
        # d = D_q = D_k = D_v
        # m = number of queries
        # n = number of keys/values (context length)

        # get keys and queries
        Q = torch.einsum("hid,bdm -> b him", self.W_Q_1[layer],
                         query)  # [batch_size, 2, i_size + 3*e_size, query_length]
        K = torch.einsum("hid,bdn -> bhin", self.W_K_1[layer], key)  # [batch_size, 2, i_size + 3*e_size, key_length]

        # shape = [b, h, i, n]
        V = torch.einsum("hid,bdn -> bhin", self.W_V_1[layer], value)

        if self.kernel == 'linear':
            scores_1 = torch.einsum('bim,bin -> bmn', Q[:, 0], K[:, 0])
        elif self.kernel == 'rbf':
            diff = Q[:, 0].unsqueeze(-1) - K[:, 0].unsqueeze(-2)
            sqdist = torch.sum(diff ** 2, dim=1)
            scores_1 = torch.exp(-sqdist / (2 * 1))
        elif self.kernel == 'softmax':  # default to scaled dot-product
            scores_1 = torch.einsum('bim,bin -> bmn', Q[:, 0], K[:, 0])
            scores_1 = scores_1 / math.sqrt(Q.size(-2))
            scores_1 = F.softmax(scores_1, dim=-1)

        # for erasure component, always use softmax
        scores_2 = torch.einsum('bim,bin -> bmn', Q[:, 1], K[:, 1])
        scores_2 = scores_2 / math.sqrt(Q.size(-2))
        scores_2 = F.softmax(scores_2, dim=-1)

        # shape = [b, h, n, m]
        attention_scores = torch.stack([scores_1, scores_2], dim=1)

        # multiply by values
        # shape = [b, h, i, m]
        attn = torch.einsum("bhin,bhmn -> bhim", V, attention_scores)
        if self.kernel != 'softmax':
            attn = attn * 1 / n_context

        output = torch.einsum("hji,bhim -> bjm", self.P_1[layer], attn)

        return output

    def cross_attn(self, key, query, value, layer):
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more 'batch-like' leading dimensions.
        Args:
            query: Embeddings sequence used to compute queries; shape [..., D_q, N].
            key: Embeddings sequence used to compute keys; shape [..., D_k, M].
            value: Embeddings sequence used to compute values; shape [..., D_v, M].
            layer: Layer number used for value of alpha.
        Returns:
            A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [..., T', D'].
        """

        # h = number of heads
        # i = input_size + 3 * emb_size
        # j = input_size + 3 * emb_size
        # d = emb_size
        # m = number of queries
        # n = number of keys/values (categories of embedding vectors)

        # get keys and queries
        Q = torch.einsum("hid,bdm -> bhim", self.W_Q_2[layer],
                         query)  # [batch_size, num_heads, i_size + 3*e_size, query_length]
        K = torch.einsum("hid,...dn -> ...hin", self.W_K_2[layer],
                         key)  # [batch_size, num_heads, i_size + 3*e_size, key_length]

        # shape = [b, h, i, n]
        V = torch.einsum("hid,...dn -> ...hin", self.W_V_2[layer], value)

        scores = torch.einsum('bhim,...hin -> bhmn', Q, K)
        scores = F.softmax(scores, dim=-1)

        # shape = [b, h, n, m]
        attention_scores = scores

        # multiply by values
        # shape = [b, h, i, m]
        attn = torch.einsum("...hin,bhmn -> bhim", V, attention_scores)

        # multiply by projection matrix
        output = torch.einsum("hji,bhim -> bjm", self.P_2[layer], attn)

        return output

    def trans_block(self, h, nl):
        # first attention block computes updated f_i
        if not self.include_query:
            key = h[:, :, :-self.num_queries]
            value = h[:, :, :-self.num_queries]
        else:
            key = h
            value = h

        query = h

        h = h + self.self_attn(key, query, value, nl)

        # second attention block uses w_e to compute updated E(w_e)
        key = self.embedding_matrix
        value = self.embedding_matrix

        query = h

        h = h + self.cross_attn(key, query, value, nl)

        return h

    def forward(self, x):
        """Computes the transformer forward pass.

        Args:
          x: Inputs.
          is_training: Whether we're training or not.
          predict_test: Test or train prediction.
        Returns:
          Array of shape [B, T, H].
        """
        # x shape = [b, input_size + 2 * num_categories + emb_size, num_samples]
        W = self.W_e_proj.clone()
        W[self.input_size:self.input_size + self.emb_size,
        self.input_size:self.input_size + self.num_categories] = self.embedding_matrix
        W[self.input_size + self.emb_size:-self.emb_size,
        self.input_size + self.num_categories:-self.emb_size] = self.embedding_matrix

        # transform input
        # shape = [b, input_size + 3 * emb_size, num_samples]
        h = torch.matmul(W, x)

        for nl in range(self.num_layers):
            # shape = [b, input_size + 3 * emb_size, num_samples]
            h = self.trans_block(h, nl)

        # shape = [b, emb_size, 1]
        f = torch.squeeze(h[:, -self.emb_size:, :])

        # embedding_matrix shape = [emb_size, num_categories]
        logits = torch.einsum("bdn,dc -> bnc", f, self.embedding_matrix)

        # shape = [b, num_samples, num_categories
        predictions = F.softmax(logits, dim=-1)

        return logits, predictions

    def train_step(self, features, labels):
        # get query labels
        # shape = [batch_size, num_queries]
        query_labels = labels[:, -self.num_queries:]

        # get transformer output
        # shape = [batch_size, num_samples, num_categories]
        logits, predictions = self.forward(features)

        # shape = [batch_size, num_queries, num_categories]
        query_logits = logits[:, -self.num_queries:, :]
        query_preds = predictions[:, -self.num_queries:, :]

        query_logits = torch.reshape(query_logits, (-1, self.num_categories))
        query_labels = torch.reshape(query_labels, (-1,))

        # compute loss
        loss_fn = nn.CrossEntropyLoss()

        query_loss = loss_fn(query_logits, query_labels)

        # compute accuracy
        # shape = [batch_size, num_queries]
        pred_cats = query_logits.argmax(dim=-1)
        query_accuracy = (pred_cats == query_labels).float().mean()

        return query_loss, query_accuracy


class TransformerGDLinearApprox(nn.Module):
    """A flexible Transformer implementation."""

    def __init__(
            self,
            num_layers: int = 3,
            input_size: int = 2,
            num_categories: int = 1,
            emb_size: int = 5,
            init_scale: float = 0.02,
            include_query: bool = False,
            kernel: str = 'linear',
            num_queries: int = 1,
            device: str = None,
            init_seed: int = 0
    ):

        """
        Initialises the module.

        Args:
            num_layers: Number of transformer layers.
            num_ff_layers: Number of FF layers.
            num_categories: Number of categories.
            emb_size: Embedding dimension of transformer.
            init_scale: Weight initialization scale.
            include_query: Include query vector in computation.
            kernel: Kernel to use in first self-attention module.
            num_queries: Number of queries in each contextual dataset.
        """

        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.num_categories = num_categories  # number of categories
        self.emb_size = emb_size
        self.init_scale = init_scale
        self.include_query = include_query
        self.kernel = kernel
        self.num_queries = num_queries
        self.device = device

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        gen = torch.Generator().manual_seed(init_seed)

        # Initialize learnable parameters
        self.alpha = nn.Parameter(torch.randn((self.num_layers, self.emb_size), generator=gen).to(self.device))
        self.kernel_param = nn.Parameter(torch.ones(1).to(self.device))

        # Initialize initial embedding matrix
        self.embedding_matrix = nn.Parameter(
            torch.randn((self.emb_size, self.num_categories), generator=gen).to(self.device))

        self.M = nn.Parameter(torch.randn((self.emb_size, self.emb_size), generator=gen).to(self.device))

        # Initialize query and key matrices
        self.W_Q = torch.zeros((1, self.input_size + 2 * self.emb_size, self.input_size + 2 * self.emb_size)).to(
            self.device)
        self.W_K = torch.zeros((1, self.input_size + 2 * self.emb_size, self.input_size + 2 * self.emb_size)).to(
            self.device)

        self.W_Q[0, :self.input_size, :self.input_size] = torch.eye(self.input_size).to(self.device)
        self.W_K[0, :self.input_size, :self.input_size] = torch.eye(self.input_size).to(self.device)

        # Initialize value matrix
        self.W_V = torch.zeros((1, self.input_size + 2 * self.emb_size, self.input_size + 2 * self.emb_size)).to(
            self.device)

        # Initialize projection matrix
        self.P = torch.zeros((1, self.input_size + 2 * self.emb_size, self.input_size + 2 * self.emb_size)).to(
            self.device)
        self.P[0, -self.emb_size:, self.input_size:-self.emb_size] = torch.eye(self.emb_size).to(self.device)

        self.W_e_proj = torch.zeros(
            (self.input_size + 2 * self.emb_size, self.input_size + self.num_categories + self.emb_size)).to(
            self.device)
        self.W_e_proj[:self.input_size, :self.input_size] = torch.eye(self.input_size).to(self.device)
        self.W_e_proj[-self.emb_size:, -self.emb_size:] = torch.eye(self.emb_size).to(self.device)

    def attn_block(self, key, query, value, layer):
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more 'batch-like' leading dimensions.
        Args:
            query: Embeddings sequence used to compute queries; shape [..., D_q, N].
            key: Embeddings sequence used to compute keys; shape [..., D_k, M].
            value: Embeddings sequence used to compute values; shape [..., D_v, M].
            layer: Layer number used for value of alpha.
        Returns:
            A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [..., T', D'].
        """
        n_context = key.size(-1)

        # h = number of heads
        # i = input_size + 3 * emb_size
        # j = input_size + 3 * emb_size
        # d = D_q = D_k = D_v
        # m = number of queries
        # n = number of keys/values (context length)

        # get keys and queries
        Q = torch.einsum("hid,bdm -> bhim", self.W_Q, query)  # [batch_size, 1, i_size + 2*e_size, query_length]
        K = torch.einsum("hid,bdn -> bhin", self.W_K, key)  # [batch_size, 1, i_size + 2*e_size, key_length]

        # shape = [b, h, i, n]
        W_V = self.W_V.clone()
        W_V[0, self.input_size:-self.emb_size, self.input_size:-self.emb_size] = torch.diag_embed(self.alpha[layer]).to(
            self.device)

        V = torch.einsum("hid,bdn -> bhin", W_V, value)

        if self.kernel == 'linear':
            scores = self.kernel_param * torch.einsum('bim,bin -> bmn', Q[:, 0], K[:, 0])
        elif self.kernel == 'rbf':
            diff = Q[:, 0].unsqueeze(-1) - K[:, 0].unsqueeze(-2)
            sqdist = torch.sum(diff ** 2, dim=1)
            scores = torch.exp(-sqdist / (2 * self.kernel_param ** 2))
        elif self.kernel == 'softmax':  # default to scaled dot-product
            scores = self.kernel_param * torch.einsum('bim,bin -> bmn', Q[:, 0], K[:, 0])
            scores = scores / math.sqrt(Q.size(-2))
            scores = F.softmax(scores, dim=-1)

        # shape = [b, h, n, m]
        attention_scores = scores

        # multiply by values
        # shape = [b, h, i, m]
        attn = torch.einsum("bhin,bhmn -> bhim", V, attention_scores)
        if self.kernel != 'softmax':
            attn = attn * 1 / n_context

        P = self.P.clone()
        P[0, self.input_size:-self.emb_size, self.input_size:-self.emb_size] = self.M

        output = torch.einsum("hji,bhim -> bjm", P, attn)

        return output

    def trans_block(self, h, nl):
        # first attention block computes updated f_i
        if not self.include_query:
            key = h[:, :, :-self.num_queries]
            value = h[:, :, :-self.num_queries]
        else:
            key = h
            value = h

        query = h

        h = h + self.attn_block(key, query, value, nl)

        return h

    def forward(self, x):
        """Computes the transformer forward pass.

        Args:
          x: Inputs.
          is_training: Whether we're training or not.
          predict_test: Test or train prediction.
        Returns:
          Array of shape [B, T, H].
        """
        # x shape = [b, input_size + num_categories + emb_size, num_samples]

        # transform input
        # shape = [b, input_size + 2 * emb_size, num_samples]
        W = self.W_e_proj.clone()
        W[self.input_size:-self.emb_size, self.input_size:-self.num_categories] = self.embedding_matrix

        h = torch.matmul(W, x)

        for nl in range(self.num_layers):
            # shape = [b, input_size + 3 * emb_size, num_samples]
            h = self.trans_block(h, nl)

        # shape = [b, emb_size, 1]
        f = torch.squeeze(h[:, -self.emb_size:, :])

        # embedding_matrix shape = [emb_size, num_categories]
        logits = torch.einsum("bdn,dc -> bnc", f, self.embedding_matrix)

        # shape = [b, num_samples, num_categories
        predictions = F.softmax(logits, dim=-1)

        return logits, predictions

    def train_step(self, features, labels):
        # get query labels
        # shape = [batch_size, num_queries]
        query_labels = labels[:, -self.num_queries:]

        # get transformer output
        # shape = [batch_size, num_samples, num_categories]
        logits, predictions = self.forward(features)

        # shape = [batch_size, num_queries, num_categories]
        query_logits = logits[:, -self.num_queries:, :]
        query_preds = predictions[:, -self.num_queries:, :]

        query_logits = torch.reshape(query_logits, (-1, self.num_categories))
        query_labels = torch.reshape(query_labels, (-1,))

        # compute loss
        loss_fn = nn.CrossEntropyLoss()

        query_loss = loss_fn(query_logits, query_labels)

        # compute accuracy
        # shape = [batch_size, num_queries]
        pred_cats = query_logits.argmax(dim=-1)
        query_accuracy = (pred_cats == query_labels).float().mean()

        return query_loss, query_accuracy


class TransformerLinearApprox(nn.Module):
    """A flexible Transformer implementation."""

    def __init__(
            self,
            num_layers: int = 3,
            input_size: int = 2,
            num_categories: int = 1,
            emb_size: int = 5,
            init_scale: float = 0.02,
            include_query: bool = False,
            kernel: str = 'linear',
            num_queries: int = 1,
            device: str = None,
            init_seed: int = 0
    ):

        """
        Initialises the module.

        Args:
            num_layers: Number of transformer layers, usually one due DEQ behaviour.
            num_categories: Number of categories.
            emb_size: Embedding dimension of transformer.
            init_scale: Weight initialization scale.
            include_query: Include query vector in computation.
            kernel: Kernel to use in first self-attention module.
            num_queries: Number of queries in each contextual dataset.
        """

        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.num_categories = num_categories  # number of categories
        self.emb_size = emb_size
        self.init_scale = init_scale
        self.include_query = include_query
        self.kernel = kernel
        self.num_queries = num_queries
        self.device = device

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        gen = torch.Generator().manual_seed(init_seed)

        # Initialize initial embedding matrix
        self.embedding_matrix = nn.Parameter(
            torch.randn((self.emb_size, self.num_categories), generator=gen).to(self.device))

        self.W_Q = nn.Parameter(
            torch.randn((self.num_layers, 1, self.input_size + 2 * self.emb_size, self.input_size + 2 * self.emb_size),
                        generator=gen).to(
                self.device) * self.init_scale)
        self.W_K = nn.Parameter(
            torch.randn((self.num_layers, 1, self.input_size + 2 * self.emb_size, self.input_size + 2 * self.emb_size),
                        generator=gen).to(
                self.device) * self.init_scale)

        # Initialize value matrix for layer 1
        self.W_V = nn.Parameter(
            torch.randn((self.num_layers, 1, self.input_size + 2 * self.emb_size, self.input_size + 2 * self.emb_size),
                        generator=gen).to(
                self.device) * self.init_scale)

        # Initialize projection matrix for layer 1
        self.P = nn.Parameter(
            torch.randn((self.num_layers, 1, self.input_size + 2 * self.emb_size, self.input_size + 2 * self.emb_size),
                        generator=gen).to(
                self.device) * self.init_scale)

        self.W_e_proj = torch.zeros(
            (self.input_size + 2 * self.emb_size, self.input_size + self.num_categories + self.emb_size)).to(
            self.device)
        self.W_e_proj[:self.input_size, :self.input_size] = torch.eye(self.input_size).to(self.device)
        self.W_e_proj[-self.emb_size:, -self.emb_size:] = torch.eye(self.emb_size).to(self.device)

    def attn_block(self, key, query, value, layer):
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more 'batch-like' leading dimensions.
        Args:
            query: Embeddings sequence used to compute queries; shape [..., D_q, N].
            key: Embeddings sequence used to compute keys; shape [..., D_k, M].
            value: Embeddings sequence used to compute values; shape [..., D_v, M].
            layer: Layer number used for value of alpha.
        Returns:
            A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [..., T', D'].
        """
        n_context = key.size(-1)

        # h = number of heads
        # i = input_size + 2 * emb_size
        # j = input_size + 2 * emb_size
        # d = D_q = D_k = D_v
        # m = number of queries
        # n = number of keys/values (context length)

        # get keys and queries
        Q = torch.einsum("hid,bdm -> bhim", self.W_Q[layer],
                         query)  # [batch_size, heads, i_size + 2*e_size, query_length]
        K = torch.einsum("hid,bdn -> bhin", self.W_K[layer], key)  # [batch_size, heads, i_size + 2*e_size, key_length]

        # get value
        V = torch.einsum("hid,bdn -> bhin", self.W_V[layer],
                         value)  # [batch_size, heads, i_size + 2*e_size, key_length]

        # get attention scores
        if self.kernel == 'linear':
            scores = self.kernel_param * torch.einsum('bim,bin -> bmn', Q[:, 0], K[:, 0])
        elif self.kernel == 'rbf':
            diff = Q[:, 0].unsqueeze(-1) - K[:, 0].unsqueeze(-2)
            sqdist = torch.sum(diff ** 2, dim=1)
            scores = torch.exp(-sqdist / (2 * self.kernel_param ** 2))
        elif self.kernel == 'softmax':  # default to scaled dot-product
            scores = self.kernel_param * torch.einsum('bim,bin -> bmn', Q[:, 0], K[:, 0])
            scores = scores / math.sqrt(Q.size(-2))
            scores = F.softmax(scores, dim=-1)

        # shape = [b, h, n, m]
        attention_scores = scores

        # multiply by values
        # shape = [b, h, i, m]
        attn = torch.einsum("bhin,bhmn -> bhim", V, attention_scores)
        if self.kernel != 'softmax':
            attn = attn * 1 / n_context

        output = torch.einsum("hji,bhim -> bjm", self.P[layer], attn)

        return output

    def trans_block(self, h, nl):
        if not self.include_query:
            key = h[:, :, :-self.num_queries]
            value = h[:, :, :-self.num_queries]
        else:
            key = h
            value = h

        query = h

        h = h + self.attn_block(key, query, value, nl)

        return h

    def forward(self, x):
        """Computes the transformer forward pass.

        Args:
          x: Inputs.
          is_training: Whether we're training or not.
          predict_test: Test or train prediction.
        Returns:
          Array of shape [B, T, H].
        """
        # x shape = [b, input_size + num_categories + emb_size, num_samples]

        # transform input
        # shape = [b, input_size + 2 * emb_size, num_samples]
        W = self.W_e_proj.clone()
        W[self.input_size:-self.emb_size, self.input_size:-self.num_categories] = self.embedding_matrix

        h = torch.matmul(W, x)

        for nl in range(self.num_layers):
            # shape = [b, input_size + 3 * emb_size, num_samples]
            h = self.trans_block(h, nl)

        # shape = [b, emb_size, 1]
        f = torch.squeeze(h[:, -self.emb_size:, :])

        # embedding_matrix shape = [emb_size, num_categories]
        logits = torch.einsum("bdn,dc -> bnc", f, self.embedding_matrix)

        # shape = [b, num_samples, num_categories
        predictions = F.softmax(logits, dim=-1)

        return logits, predictions

    def train_step(self, features, labels):
        # get query labels
        # shape = [batch_size, num_queries]
        query_labels = labels[:, -self.num_queries:]

        # get transformer output
        # shape = [batch_size, num_samples, num_categories]
        logits, predictions = self.forward(features)

        # shape = [batch_size, num_queries, num_categories]
        query_logits = logits[:, -self.num_queries:, :]
        query_preds = predictions[:, -self.num_queries:, :]

        query_logits = torch.reshape(query_logits, (-1, self.num_categories))
        query_labels = torch.reshape(query_labels, (-1,))

        # compute loss
        loss_fn = nn.CrossEntropyLoss()

        query_loss = loss_fn(query_logits, query_labels)

        # compute accuracy
        # shape = [batch_size, num_queries]
        pred_cats = query_logits.argmax(dim=-1)
        query_accuracy = (pred_cats == query_labels).float().mean()

        return query_loss, query_accuracy

