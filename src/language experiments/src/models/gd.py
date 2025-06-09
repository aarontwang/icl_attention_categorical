import math
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from src.model_util import BaseModel, calculate_attn_scores

@dataclass
class GDConfig:
    
    # Model Name
    model_name: str = "GD"

    # Basic Model Parameters
    d_vocab: int = 50258 # Default value for GPT-2 tokenizer
    d_seq: int = 256
    d_embed: int = 512
    n_head: int = 8
    n_layer: int = 1
    
    # Model Parameters
    use_ff: bool = False
    attn_fn: str = "linear"

    # Regularization and Normalization
    dropout: float = 0.1
    
    def get_name(self):
        return f"{self.model_name}_{self.d_seq}C_{self.d_embed}E_{self.n_head}H_{self.n_layer}L_{self.attn_fn}_FF={self.use_ff}"
    
    def __post_init__(self):
        assert self.attn_fn in ["softmax", "linear", "rbf"], f"Invalid attention function ({self.attn_fn}), must be one of ['softmax', 'linear', 'rbf']"

class GD(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        # Embedding
        self.wte = nn.Embedding(config.d_vocab, config.d_embed)
        self.wpe = nn.Embedding(config.d_seq + 1, config.d_embed)
        
        self.wte_layernorm = nn.LayerNorm(config.d_embed, bias=False)
        self.wpe_layernorm = nn.LayerNorm(config.d_embed, bias=False)
        
        # Attention
        self.W_q_diag = self.W_k_diag = nn.Parameter(torch.zeros(config.n_head, config.d_embed)) # W_q = W_k and is diagonal
        
        # Gradient Descent
        
        self.W_o_list = nn.ParameterList([nn.Parameter(torch.zeros(config.d_embed * config.n_head, config.d_embed)) for _ in range(config.n_layer)])
        
        N_reg = 1.0 / torch.arange(1, config.d_seq + 1).unsqueeze(1).float()
        self.register_buffer("N_reg", N_reg)
        
        self.gamma = nn.Parameter(torch.zeros(config.n_head, 1, 1)) if config.attn_fn == "rbf" else None
        
        # Dropout
        self.drop_attn = nn.Dropout(config.dropout)
        self.drop_gd = nn.Dropout(config.dropout)
        
        # Feed Forward
        if config.use_ff:
            self.ff_list = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(config.d_embed, bias=False),
                    nn.Linear(config.d_embed, 4 * config.d_embed, bias=False),
                    nn.GELU(),
                    nn.Linear(4 * config.d_embed, config.d_embed, bias=False),
                    nn.Dropout(config.dropout)
                ) for _ in range(config.n_layer)
            ])

        # Output
        self.ln_out = nn.LayerNorm(config.d_embed, bias=False)

        self._init_weights()
        print(f"Initialized model {self.name} with {self.get_num_params_formatted()} parameters")

    def _init_weights(self):
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.02)
        nn.init.normal_(self.W_q_diag, std=0.02)
        nn.init.normal_(self.W_k_diag, std=0.02)
        if self.gamma is not None:
            nn.init.normal_(self.gamma, mean=1.0, std=0.02)
        for W_o in self.W_o_list:
            nn.init.normal_(W_o, std=0.02 / math.sqrt(2 * self.config.n_layer))
        if self.config.use_ff:
            for ff in self.ff_list:
                nn.init.normal_(ff[1].weight, std=0.02)
                nn.init.normal_(ff[3].weight, std=0.02)

    def calculate_E_wte(self, f_k):
        
        g = f_k @ self.wte.weight.transpose(-1, -2) 
        g = torch.exp(g)
        
        numerator = g @ self.wte.weight
        denominator = g.sum(dim=-1, keepdim=True)
        
        return numerator / denominator
    
    def forward(self, x, targets=None, pad_token_id=None):
        
        B, S = x.size()
        device = x.device
        
        # Embedding
        e = self.wte(x)
        p = self.wpe(torch.arange(S + 1, device=device)).repeat(B, 1, 1)
        
        e = self.wte_layernorm(e)
        p = self.wpe_layernorm(p)
        
        # Attention
        x_i = p[:, :-1, :].repeat(1, 1, self.config.n_head).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
        x_j = p[:, 1:, :].repeat(1, 1, self.config.n_head).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
        
        self.W_q = torch.diag_embed(self.W_q_diag)
        self.W_k = torch.diag_embed(self.W_k_diag)
        
        K = x_i @ self.W_k
        Q = x_j @ self.W_q
        
        attn_scores = calculate_attn_scores(Q, K, self.config.d_embed, gamma=self.gamma, attn_fn=self.config.attn_fn)
        attn_scores = self.drop_attn(attn_scores)
        
        # Gradient Descent
        
        f_k = torch.zeros((B, S + 1, self.config.d_embed), device=device) # A_0 = 0
        
        for k in range(self.config.n_layer):
            
            E_wte = self.calculate_E_wte(f_k[:, :-1, :])
            
            V = e - E_wte
            
            delta_f_k = attn_scores @ V.unsqueeze(1)
            
            delta_f_k = delta_f_k * self.N_reg[:S]
            delta_f_k = delta_f_k.transpose(1, 2).contiguous().view(B, S, self.config.n_head * self.config.d_embed) @ self.W_o_list[k]
            delta_f_k = delta_f_k.transpose(-1, -2)
            delta_f_k = self.drop_gd(delta_f_k)
        
            f_k[:, 1:, :] = f_k[:, 1:, :] + delta_f_k.transpose(1, 2)
            
            assert not torch.isnan(f_k).any(), f"NaN in f_k: \n{f_k}"
            
            if self.config.use_ff:
                f_k = f_k + self.ff_list[k](f_k)
        
        # Output
        if targets is None:
            f_k = f_k[:, [-1], :] # Only consider last token for optimized generation
        else:
            f_k = f_k[:, 1:, :]
            
        f_k = self.ln_out(f_k)
        
        logits = f_k @ self.wte_layernorm(self.wte.weight).transpose(-1, -2) # Layernorm on wte output is not strictly necessary, but it makes the layernorm on e more consistent with GD theory
        
        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=pad_token_id)
        
        return logits, loss