import torch
from torch.nn import functional as F
import math

def calculate_attn_scores(Q, K, d_embed, gamma=None, attn_fn="softmax"):
    assert attn_fn in ["softmax", "linear", "rbf"], f"Invalid attention function ({attn_fn}), must be one of ['softmax', 'linear', 'rbf']"
    assert Q.size() == K.size(), f"Q and K must have the same shape for causal attention to be applied properly (got Q={Q.shape} and K={K.shape})"
    B, _, S, E = Q.size()
    device = Q.device
    
    causal_mask = torch.tril(torch.ones(S, S, device=device), diagonal=0).view(1, S, S).bool().logical_not()
    attn_scaling = 1.0 / math.sqrt(d_embed) # Dividing by sqrt(d_embed) helps stabilize training, common in GPT models 

    if attn_fn == "linear":
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))
        attn_scores = attn_scores * attn_scaling
        attn_scores = attn_scores.masked_fill(causal_mask, 0.0)
    elif attn_fn == "rbf":
        attn_scores = torch.cdist(Q, K, p=2).pow(2)
        attn_scores = -gamma * attn_scaling * attn_scores # attn_scaling is used for consistency, but should be absorbed into gamma
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        attn_scores = torch.exp(attn_scores)
    else:
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))
        attn_scores = attn_scores * attn_scaling
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        attn_scores = F.softmax(attn_scores, dim=-1)
    
    return attn_scores