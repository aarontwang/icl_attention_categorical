import math
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from src.model_util import BaseModel, calculate_attn_scores

@dataclass
class GPTConfig:
    
    # Model Name
    model_name: str = "GPT"
    
    # Basic Model Parameters
    d_vocab: int = 50258 # Default value for GPT-2 tokenizer
    d_seq: int = 256
    d_embed: int = 512
    n_head: int = 8
    n_layer: int = 1
    
    # Model Parameters
    use_ff: bool = True
    attn_fn: str = "softmax"
    
    # Regularization and Normalization
    dropout: float = 0.1
    
    def get_name(self):
        return f"{self.model_name}_{self.d_seq}C_{self.d_embed}E_{self.n_head}H_{self.n_layer}L_{self.attn_fn}_FF={self.use_ff}"
    
    def __post_init__(self):
        assert self.attn_fn in ["softmax", "linear", "rbf"], f"Invalid attention function ({self.attn_fn}), must be one of ['softmax', 'linear', 'rbf']"
        
class Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.ln = nn.LayerNorm(config.d_embed, bias=False)

        self.W_q = nn.Parameter(torch.zeros(config.n_head, config.d_embed, config.d_embed))
        self.W_k = nn.Parameter(torch.zeros(config.n_head, config.d_embed, config.d_embed))
        self.W_v = nn.Parameter(torch.zeros(config.n_head, config.d_embed, config.d_embed))
        self.W_o = nn.Linear(config.n_head * config.d_embed, config.d_embed, bias=False)
        
        self.gamma = nn.Parameter(torch.zeros(config.n_head, 1, 1)) if config.attn_fn == "rbf" else None

        self.drop_attn = nn.Dropout(config.dropout)
        self.dropout_out = nn.Dropout(config.dropout)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_q, std=0.02)
        nn.init.normal_(self.W_k, std=0.02)
        nn.init.normal_(self.W_v, std=0.02)
        if self.gamma is not None:
            nn.init.normal_(self.gamma, mean=1.0, std=0.02)
        nn.init.normal_(self.W_o.weight, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(self, x):
        device = x.device
        B, S, E = x.size()
        
        x = self.ln(x)
        x = x.repeat(1, 1, self.config.n_head).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
        
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        attn_scores = calculate_attn_scores(Q, K, self.config.d_embed, gamma=self.gamma, attn_fn=self.config.attn_fn)
        attn_scores = self.drop_attn(attn_scores)
        
        attn_output = attn_scores @ V

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.n_head * self.config.d_embed)
        
        attn_output = self.W_o(attn_output)
        attn_output = self.dropout_out(attn_output)
        
        return attn_output


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config

        # Attention
        self.attn = Attention(config)
        
        # Feed Forward
        if config.use_ff:
            self.ff = nn.Sequential(
                nn.LayerNorm(config.d_embed, bias=False),
                nn.Linear(config.d_embed, 4 * config.d_embed, bias=False),
                nn.GELU(),
                nn.Linear(4 * config.d_embed, config.d_embed, bias=False),
                nn.Dropout(config.dropout)
            )
    
        self._init_weights()
    
    def _init_weights(self):
        if self.config.use_ff:
            nn.init.normal_(self.ff[1].weight, std=0.02)
            nn.init.normal_(self.ff[3].weight, std=0.02)

    def forward(self, x):
        x = x + self.attn(x)	
        if self.config.use_ff:
            x = x + self.ff(x)
        return x

class GPT(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        # Embedding
        self.wte = nn.Embedding(config.d_vocab, config.d_embed)
        self.wpe = nn.Embedding(config.d_seq, config.d_embed)
        
        self.x_layernorm = nn.LayerNorm(config.d_embed, bias=False)
        
        self.dropout_e = nn.Dropout(config.dropout)
        self.dropout_p = nn.Dropout(config.dropout)
        
        # Attention
        self.attn_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        
        # Output
        self.ln_out = nn.LayerNorm(config.d_embed, bias=False)
        
        self._init_weights()
        print(f"Initialized model {self.name} with {self.get_num_params_formatted()} parameters")

    def _init_weights(self):
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.02)
        
    def forward(self, x, targets=None, pad_token_id=None):
        
        B, S = x.size()
        device = x.device
        
        # Embedding
        e = self.wte(x)
        p = self.wpe(torch.arange(S, device=device))
        
        e = self.dropout_e(e)
        p = self.dropout_p(p).unsqueeze(0).expand(B, -1, -1)
        
        x = e + p
        
        x = self.x_layernorm(x)
        
        # Attention
        for block in self.attn_blocks:
            x = block(x)
        
        x = self.ln_out(x)
        
        logits = x @ self.wte.weight.transpose(-1, -2)
        
        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=pad_token_id)
        
        return logits, loss