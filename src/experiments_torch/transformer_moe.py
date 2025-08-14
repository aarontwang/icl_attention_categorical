import dataclasses
import sys
from typing import Optional, Tuple, Union, Any, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerGDMOE(nn.Module):
    """A flexible Transformer implementation."""

    def __init__(
            self,
            num_layers: int = 3,
            num_ff_layers: int = 2,
            num_heads: int = 1,
            num_mlps: int = 1,
            input_size: int = 2,
            num_categories: int = 1,
            emb_size: int = 5,
            ff_hidden_size: int = 10,
            k: int = 1,
            init_scale: float = 0.02,
            include_query: bool = False,
            kernel: str = 'linear',
            num_queries: int = 1,
            shared_ff: bool = False,
            device: str = None,
            init_seed: int = 0
    ):

        """
        Initialises the module.

        Args:
            num_layers: Number of transformer layers, usually one due DEQ behaviour.
            num_heads: Number of non-erasure heads in each attention layer.
            num_mlps: Number of MLPs to use in MOE.
            num_categories: Number of categories.
            emb_size: Embedding dimension of transformer.
            k: Number of softmax components to keep after each MOE layer.
            init_scale: Weight initialization scale.
            include_query: Include query vector in computation.
            kernel: Kernel to use in first self-attention module.
            num_queries: Number of queries in each contextual dataset.
            use_mlp: Whether to include FF element.
        """

        super().__init__()
        self.num_layers = num_layers
        self.num_ff_layers = num_ff_layers
        self.num_heads = num_heads
        self.num_mlps = num_mlps
        self.input_size = input_size
        self.num_categories = num_categories  # number of categories
        self.emb_size = emb_size
        self.ff_hidden_size = ff_hidden_size
        self.k = k
        self.init_scale = init_scale
        self.include_query = include_query
        self.kernel = kernel
        self.num_queries = num_queries
        self.shared_ff = shared_ff
        self.device = device

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        gen = torch.Generator().manual_seed(init_seed)

        # Initialize learnable parameters
        self.alpha = nn.Parameter(
            torch.randn((self.num_layers, self.num_heads, self.emb_size), generator=gen).to(self.device))
        self.kernel_param = nn.Parameter(torch.ones(self.num_heads).to(self.device))

        # Initialize initial embedding matrix
        self.embedding_matrix = nn.Parameter(
            torch.randn((self.emb_size, self.num_categories), generator=gen).to(self.device))

        # Initialize query and key matrices for self-attention
        self.W_Q = torch.zeros(
            (self.num_heads + 1, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size)).to(
            self.device)
        self.W_K = torch.zeros(
            (self.num_heads + 1, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size)).to(
            self.device)

        for i in range(self.num_heads):
            self.W_Q[i, :self.input_size, :self.input_size] = torch.eye(self.input_size).to(self.device)
            self.W_K[i, :self.input_size, :self.input_size] = torch.eye(self.input_size).to(self.device)

        self.lam = 1e5
        self.W_Q[-1, :self.input_size, :self.input_size] = self.lam * torch.eye(self.input_size).to(self.device)
        self.W_K[-1, :self.input_size, :self.input_size] = self.lam * torch.eye(self.input_size).to(self.device)

        # Initialize value matrix for self-attention
        self.W_V = torch.zeros(
            (self.num_heads + 1, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size)).to(
            self.device)
        self.W_V[-1, self.input_size + self.emb_size:-self.emb_size,
        self.input_size + self.emb_size:-self.emb_size] = torch.eye(self.emb_size).to(self.device)

        # Initialize projection matrix for self-attention
        self.P = torch.zeros(
            (self.num_heads + 1, self.input_size + 3 * self.emb_size, self.input_size + 3 * self.emb_size)).to(
            self.device)

        for i in range(self.num_heads):
            self.P[i] = torch.eye(self.input_size + 3 * self.emb_size).to(self.device)

        self.P[-1] = -1 * torch.eye(self.input_size + 3 * self.emb_size).to(self.device)

        self.W_e_proj = torch.zeros(
            (self.input_size + 3 * self.emb_size, self.input_size + 2 * self.num_categories + self.emb_size)).to(
            self.device)
        self.W_e_proj[:self.input_size, :self.input_size] = torch.eye(self.input_size).to(self.device)
        self.W_e_proj[-self.emb_size:, -self.emb_size:] = torch.eye(self.emb_size).to(self.device)

        # Initialize FF weights
        ff_weight_list = []
        if self.shared_ff:
            # create V MLPs to be shared across all layers
            moe_list = []
            for i in range(self.num_mlps):
                # input layer
                layer_list = [nn.Linear(self.emb_size, self.ff_hidden_size, bias=True)]

                # hidden layers
                for _ in range(self.num_ff_layers - 1):
                    layer_list.append(nn.Linear(self.ff_hidden_size, self.ff_hidden_size, bias=True))

                # output layer
                layer_list.append(nn.Linear(self.ff_hidden_size, self.emb_size, bias=True))

                moe_list.append(nn.ModuleList(layer_list))

            ff_weight_list.append(nn.ModuleList(moe_list))
        else:
            for ff_layer in range(self.num_layers):
                # create V MLPs per layer
                moe_list = []
                for i in range(self.num_mlps):
                    # input layer
                    layer_list = [nn.Linear(self.emb_size, self.ff_hidden_size, bias=True)]

                    # hidden layers
                    for _ in range(self.num_ff_layers - 1):
                        layer_list.append(nn.Linear(self.ff_hidden_size, self.ff_hidden_size, bias=True))

                    # output layer
                    layer_list.append(nn.Linear(self.ff_hidden_size, self.emb_size, bias=True))

                    moe_list.append(nn.ModuleList(layer_list))

                ff_weight_list.append(nn.ModuleList(moe_list))

        self.ff_weights = nn.ModuleList(ff_weight_list)

        # initialize MOE components
        self.xi = nn.Parameter(torch.randn((self.num_mlps, self.emb_size), generator=gen).to(self.device))

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
        Q = torch.einsum("hid,bdm -> bhim", self.W_Q, query)  # [batch_size, 2, i_size + 3*e_size, query_length]
        K = torch.einsum("hid,bdn -> bhin", self.W_K, key)  # [batch_size, 2, i_size + 3*e_size, key_length]

        # shape = [b, h, i, n]
        W_V = self.W_V.clone()
        # W_V[0, -self.emb_size:, self.input_size:self.input_size + self.emb_size] = torch.diag_embed(
        #     self.alpha[layer]).to(
        #     self.device)
        # W_V[0, -self.emb_size:, self.input_size + self.emb_size:-self.emb_size] = -1 * torch.diag_embed(
        #     self.alpha[layer]).to(self.device)

        # Vectorized assignment for W_V[0:num_heads]
        diag = torch.diag_embed(self.alpha[layer])  # shape: (num_heads, emb_size, emb_size)

        # First block: positive diagonal
        W_V[:self.num_heads, -self.emb_size:, self.input_size:self.input_size + self.emb_size] = diag

        # Second block: negative diagonal
        W_V[:self.num_heads, -self.emb_size:, self.input_size + self.emb_size:-self.emb_size] = -diag

        V = torch.einsum("hid,bdn -> bhin", W_V, value)

        # get attention scores
        if self.kernel == 'linear':
            scores_1 = self.kernel_param.view(1, -1, 1, 1) * torch.einsum('bhim,bhin -> bhmn', Q[:, 0:self.num_heads],
                                                                          K[:, 0:self.num_heads])
        elif self.kernel == 'rbf':
            diff = Q[:, 0:self.num_heads].unsqueeze(-1) - K[:, 0:self.num_heads].unsqueeze(-2)  # bhim1 - bhi1n = bhimn
            sqdist = torch.sum(diff ** 2, dim=-3)  # bhmn
            scores_1 = torch.exp(-sqdist / (2 * self.kernel_param.view(1, -1, 1, 1) ** 2))
        elif self.kernel == 'softmax':  # default to scaled dot-product
            scores_1 = self.kernel_param.view(1, -1, 1, 1) * torch.einsum('bhim,bhin -> bhmn', Q[:, 0:self.num_heads],
                                                                          K[:, 0:self.num_heads])
            scores_1 = scores_1 / math.sqrt(Q.size(-2))
            scores_1 = F.softmax(scores_1, dim=-1)

        # for erasure component, always use softmax
        scores_2 = torch.einsum('bim,bin -> bmn', Q[:, -1], K[:, -1])
        scores_2 = scores_2 / math.sqrt(Q.size(-2))
        scores_2 = F.softmax(scores_2, dim=-1)

        # shape = [b, h, n, m]
        attention_scores = torch.cat([scores_1, scores_2.unsqueeze(-3)], dim=1)

        # multiply by values
        # shape = [b, h, i, m]
        attn = torch.einsum("bhin,bhmn -> bhim", V, attention_scores)
        if self.kernel != 'softmax':
            attn = attn * 1 / n_context

        output = torch.einsum("hji,bhim -> bjm", self.P, attn)

        return output

    def feedforward(self, h, layer):
        # from shape (batch_size, emb_size, n + 1) to (batch_size, n + 1, emb_size)
        h_out = h.permute(0, 2, 1)

        context_length = h.shape[-1]
        batch_size = h.shape[0]

        # a = xi (num_mlps, e_size) @ h (batch_size, e_size, n + 1) -> (batch_size, num_mlps, n+1)
        a = torch.matmul(self.xi, h)

        # print(a.shape)

        # softmax(a) = (batch_size, num_mlps, n+1) across num_mlps (dim 1)
        a_sm = F.softmax(a, dim=1)

        # zero out bottom v-k in each column
        # get top-k indices
        _, indices = torch.topk(a_sm, k=self.k, dim=1)

        # make mask of top-k
        mask = torch.ones_like(a_sm, dtype=torch.bool)

        batch_idx = torch.arange(batch_size, device=h.device).view(-1, 1, 1).expand(-1, self.k, context_length)
        feat_idx = torch.arange(context_length, device=h.device).view(1, 1, -1).expand(batch_size, self.k, -1)

        # Set mask to False at the largest-k entries per column
        mask[batch_idx, indices, feat_idx] = False

        # Apply mask - zero out all but top-k
        xi_masked = a_sm.masked_fill(mask, 0.0)

        # print(xi_masked.shape)

        # reshape to (num_mlps, batch_size, n+1)
        xi_final = xi_masked.permute(1, 0, 2)

        # print(xi_final.shape)

        # there are num_ff_layers+1 total layers (including input/final output layer)
        moe_outputs = []
        if self.shared_ff:
            for i in range(self.num_mlps):
                for ff_layer in range(self.num_ff_layers):
                    h_out_temp = self.ff_weights[0][i][ff_layer](h_out)
                    h_out_temp = F.gelu(h_out_temp)

                h_out_temp = self.ff_weights[0][i][-1](h_out_temp)

                moe_outputs.append(h_out_temp)
        else:
            for i in range(self.num_mlps):
                for ff_layer in range(self.num_ff_layers):
                    h_out_temp = self.ff_weights[layer][i][ff_layer](h_out)
                    h_out_temp = F.gelu(h_out_temp)

                h_out_temp = self.ff_weights[layer][i][-1](h_out_temp)

                moe_outputs.append(h_out_temp)

        # convert moe outputs to tensor of shape (num_mlps, batch_size, n+1, emb_size)
        moe_outputs = torch.stack(moe_outputs, dim=0)

        # print(moe_outputs.shape)

        # multiply moe outputs with softmax weighting and sum across mlps (batch_size, n+1, emb_size)
        h_final = torch.sum(xi_final.unsqueeze(-1) * moe_outputs, dim=0)

        # from shape (batch_size, n + 1, emb_size) to (batch_size, emb_size, n + 1)
        return torch.permute(h_final, (0, 2, 1))

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

        # feedforward uses f_i and w_e to compute updated E(w_e)
        # ff_output = self.feedforward(torch.cat([h[:, self.input_size:self.input_size + self.emb_size, :],
        #                                         h[:, -self.emb_size:, :]], dim=1),
        #                              nl)

        # only use f_i in MLP
        ff_output = self.feedforward(h[:, -self.emb_size:, :], layer=nl)

        h_output = h.clone()
        h_output[:, self.input_size + self.emb_size:-self.emb_size, :] = h_output[:,
                                                                         self.input_size + self.emb_size:-self.emb_size,
                                                                         :] + ff_output

        return h_output

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


class TransformerMOE(nn.Module):
    def __init__(
            self,
            num_layers: int = 3,
            num_ff_layers: int = 2,
            num_heads: int = 1,
            num_mlps: int = 1,
            input_size: int = 2,
            num_categories: int = 1,
            emb_size: int = 5,
            ff_hidden_size: int = 10,
            k: int = 1,
            init_scale: float = 0.02,
            include_query: bool = False,
            kernel: str = 'linear',
            num_queries: int = 1,
            shared_ff: bool = False,
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
        self.num_ff_layers = num_ff_layers
        self.num_heads = num_heads
        self.num_mlps = num_mlps
        self.input_size = input_size
        self.num_categories = num_categories  # number of categories
        self.emb_size = emb_size
        self.ff_hidden_size = ff_hidden_size
        self.k = k
        self.init_scale = init_scale
        self.include_query = include_query
        self.kernel = kernel
        self.num_queries = num_queries
        self.shared_ff = shared_ff
        self.device = device

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        gen = torch.Generator().manual_seed(init_seed)

        # Initialize query and key matrices for self-attention
        self.W_Q = nn.Parameter(
            torch.randn((self.num_layers, self.num_heads + 1, self.input_size + 3 * self.emb_size,
                         self.input_size + 3 * self.emb_size),
                        generator=gen).to(
                self.device) * self.init_scale)
        self.W_K = nn.Parameter(
            torch.randn((self.num_layers, self.num_heads + 1, self.input_size + 3 * self.emb_size,
                         self.input_size + 3 * self.emb_size),
                        generator=gen).to(
                self.device) * self.init_scale)

        # Initialize value matrix for self-attention
        self.W_V = nn.Parameter(
            torch.randn((self.num_layers, self.num_heads + 1, self.input_size + 3 * self.emb_size,
                         self.input_size + 3 * self.emb_size),
                        generator=gen).to(
                self.device) * self.init_scale)

        # Initialize projection matrix for self-attention
        self.P = nn.Parameter(
            torch.randn((self.num_layers, self.num_heads + 1, self.input_size + 3 * self.emb_size,
                         self.input_size + 3 * self.emb_size),
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

        # Initialize FF weights
        ff_weight_list = []
        if self.shared_ff:
            # create V MLPs to be shared across all layers
            moe_list = []
            for i in range(self.num_mlps):
                # input layer
                layer_list = [nn.Linear(self.emb_size, self.ff_hidden_size, bias=True)]

                # hidden layers
                for _ in range(self.num_ff_layers - 1):
                    layer_list.append(nn.Linear(self.ff_hidden_size, self.ff_hidden_size, bias=True))

                # output layer
                layer_list.append(nn.Linear(self.ff_hidden_size, self.emb_size, bias=True))

                moe_list.append(nn.ModuleList(layer_list))

            ff_weight_list.append(nn.ModuleList(moe_list))
        else:
            for ff_layer in range(self.num_layers):
                # create V MLPs per layer
                moe_list = []
                for i in range(self.num_mlps):
                    # input layer
                    layer_list = [nn.Linear(self.emb_size, self.ff_hidden_size, bias=True)]

                    # hidden layers
                    for _ in range(self.num_ff_layers - 1):
                        layer_list.append(nn.Linear(self.ff_hidden_size, self.ff_hidden_size, bias=True))

                    # output layer
                    layer_list.append(nn.Linear(self.ff_hidden_size, self.emb_size, bias=True))

                    moe_list.append(nn.ModuleList(layer_list))

                ff_weight_list.append(nn.ModuleList(moe_list))

        self.ff_weights = nn.ModuleList(ff_weight_list)

        # initialize MOE components
        self.xi = nn.Parameter(torch.randn((self.num_mlps, self.emb_size), generator=gen).to(self.device))

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
        Q = torch.einsum("hid,bdm -> bhim", self.W_Q[layer], query)  # [batch_size, 2, i_size + 3*e_size, query_length]
        K = torch.einsum("hid,bdn -> bhin", self.W_K[layer], key)  # [batch_size, 2, i_size + 3*e_size, key_length]

        # shape = [b, h, i, n]
        V = torch.einsum("hid,bdn -> bhin", self.W_V[layer], value)

        # get attention scores
        if self.kernel == 'linear':
            scores_1 = torch.einsum('bhim,bhin -> bhmn', Q[:, 0:self.num_heads], K[:, 0:self.num_heads])
        elif self.kernel == 'rbf':
            diff = Q[:, 0:self.num_heads].unsqueeze(-1) - K[:, 0:self.num_heads].unsqueeze(-2)
            sqdist = torch.sum(diff ** 2, dim=-3)
            scores_1 = torch.exp(-sqdist / (2 * 1))
        elif self.kernel == 'softmax':  # default to scaled dot-product
            scores_1 = torch.einsum('bhim,bhin -> bhmn', Q[:, 0:self.num_heads], K[:, 0:self.num_heads])
            scores_1 = scores_1 / math.sqrt(Q.size(-2))
            scores_1 = F.softmax(scores_1, dim=-1)

        # for erasure component, always use softmax
        scores_2 = torch.einsum('bim,bin -> bmn', Q[:, -1], K[:, -1])
        scores_2 = scores_2 / math.sqrt(Q.size(-2))
        scores_2 = F.softmax(scores_2, dim=-1)

        # shape = [b, h, n, m]
        attention_scores = torch.cat([scores_1, scores_2.unsqueeze(-3)], dim=1)

        # multiply by values
        # shape = [b, h, i, m]
        attn = torch.einsum("bhin,bhmn -> bhim", V, attention_scores)
        if self.kernel != 'softmax':
            attn = attn * 1 / n_context

        output = torch.einsum("hji,bhim -> bjm", self.P[layer], attn)

        return output

    def feedforward(self, h, layer):
        # from shape (batch_size, 2 * emb_size, n + 1) to (batch_size, n + 1, 2 * emb_size)
        h_out = h.permute(0, 2, 1)

        context_length = h.shape[-1]
        batch_size = h.shape[0]

        # a = xi (num_mlps, e_size) @ h (batch_size, e_size, n + 1) -> (batch_size, num_mlps, n+1)
        a = torch.matmul(self.xi, h)

        # softmax(a) = (batch_size, num_mlps, n+1) across num_mlps (dim 1)
        a_sm = F.softmax(a, dim=1)

        # zero out bottom v-k in each column
        # get top-k indices
        _, indices = torch.topk(a_sm, k=self.k, dim=1)

        # make mask of top-k
        mask = torch.ones_like(a_sm, dtype=torch.bool)

        batch_idx = torch.arange(batch_size, device=h.device).view(-1, 1, 1).expand(-1, self.k, context_length)
        feat_idx = torch.arange(context_length, device=h.device).view(1, 1, -1).expand(batch_size, self.k, -1)

        # Set mask to False at the largest-k entries per column
        mask[batch_idx, indices, feat_idx] = False

        # Apply mask - zero out all but top-k
        xi_masked = a_sm.masked_fill(mask, 0.0)

        # reshape to (num_mlps, batch_size, n+1)
        xi_final = xi_masked.permute(1, 0, 2)

        # there are num_ff_layers+1 total layers (including input/final output layer)
        moe_outputs = []
        if self.shared_ff:
            for i in range(self.num_mlps):
                for ff_layer in range(self.num_ff_layers):
                    h_out_temp = self.ff_weights[0][i][ff_layer](h_out)
                    h_out_temp = F.gelu(h_out_temp)

                h_out_temp = self.ff_weights[0][i][-1](h_out_temp)

                moe_outputs.append(h_out_temp)
        else:
            for i in range(self.num_mlps):
                for ff_layer in range(self.num_ff_layers):
                    h_out_temp = self.ff_weights[layer][i][ff_layer](h_out)
                    h_out_temp = F.gelu(h_out_temp)

                h_out_temp = self.ff_weights[layer][i][-1](h_out_temp)

                moe_outputs.append(h_out_temp)

        # convert moe outputs to tensor of shape (num_mlps, batch_size, n+1, emb_size)
        moe_outputs = torch.stack(moe_outputs, dim=0)

        # print(moe_outputs.shape)

        # multiply moe outputs with softmax weighting and sum across mlps (batch_size, n+1, emb_size)
        h_final = torch.sum(xi_final.unsqueeze(-1) * moe_outputs, dim=0)

        # from shape (batch_size, n + 1, emb_size) to (batch_size, emb_size, n + 1)
        return torch.permute(h_final, (0, 2, 1))

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

        # feedforward uses f_i and w_e to compute updated E(w_e)
        # ff_output = self.feedforward(torch.cat([h[:, self.input_size:self.input_size + self.emb_size, :],
        #                                         h[:, -self.emb_size:, :]], dim=1),
        #                              nl)

        # only use f_i in MLP
        ff_output = self.feedforward(h[:, -self.emb_size:, :], layer=nl)

        h_output = h.clone()
        h_output[:, self.input_size + self.emb_size:-self.emb_size, :] = h_output[:,
                                                                         self.input_size + self.emb_size:-self.emb_size,
                                                                         :] + ff_output

        return h_output

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
