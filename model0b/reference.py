"""Reference model R1 (stage0b_spec.md, section 12): a minimal standard GPT.

Learned absolute position embeddings, pre-LayerNorm blocks, F.scaled_dot_product_attention with is_causal,
GELU MLP 4x. Shares no attention, position or normalization code with model0b/backbone.py, so a run that
succeeds here and fails there points at our model code rather than at the data or the loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.H = n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.H, D // self.H).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d_model=256, n_layers=2, n_heads=4, max_len=1024):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(GPTBlock(d_model, n_heads) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens, window=None):
        """`window` is accepted for interface compatibility with Backbone and ignored (full causal attention)."""
        T = tokens.shape[1]
        x = self.tok(tokens) + self.pos(torch.arange(T, device=tokens.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))
