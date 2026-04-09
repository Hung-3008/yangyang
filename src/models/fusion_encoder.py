"""
Fusion Encoder — Per-Modality Q-Former

Each modality: Linear(dim_i → d_model) → Cross-Attention Q-Former → N query vectors
Output: (B, num_modalities * num_queries, d_model)
"""

import torch
import torch.nn as nn


class ModalityQFormer(nn.Module):
    """Compress temporal sequence of one modality into query vectors.
    (B, T, dim_i) → (B, num_queries, d_model)
    """

    def __init__(self, in_dim: int, d_model: int, num_queries: int = 1,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        q = self.queries.expand(x.size(0), -1, -1)
        out, _ = self.cross_attn(query=q, key=x, value=x)
        out = self.norm1(q + out)
        out = self.norm2(out + self.ffn(out))
        return out


class FusionEncoder(nn.Module):
    """Per-modality Q-Former fusion.
    Input:  (B, T, total_dim)
    Output: (B, num_modalities * num_queries, d_model)
    """

    def __init__(self, modality_dims: list[int], d_model: int = 768,
                 num_queries: int = 1, dropout: float = 0.1):
        super().__init__()
        self.modality_dims = modality_dims
        self.d_model = d_model
        self.blocks = nn.ModuleList([
            ModalityQFormer(dim, d_model, num_queries, dropout=dropout)
            for dim in modality_dims
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        splits = torch.split(x, self.modality_dims, dim=-1)
        tokens = [block(s) for block, s in zip(self.blocks, splits)]
        return torch.cat(tokens, dim=1)
