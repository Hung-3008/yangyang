"""
Fusion Encoder — Per-Modality Bidirectional GRU
=================================================
Each modality gets its own temporal projection:
  1. Linear(dim_i → d_proj)  — project to common lower dimension
  2. Bidirectional GRU       — compress 31 frames into 1 summary vector
  3. Linear(2×d_hidden → d_model) — project to U-ViT dimension

All modalities produce one vector each → stacked as condition tokens.

Output: (B, num_modalities, d_model)  e.g. (B, 11, 768)
        11 condition tokens for U-ViT, one per modality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerModalityGRU(nn.Module):
    """Temporal projection for a single modality using Bidirectional GRU.
    
    (B, T, dim_i)  →  (B, d_model)
    
    The BiGRU reads the 31-frame sequence in both directions.
    The final hidden states (forward + backward) are concatenated
    and projected to d_model.
    """

    def __init__(self, in_dim: int, d_proj: int, d_hidden: int,
                 d_model: int, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_proj)
        self.gru = nn.GRU(
            input_size=d_proj,
            hidden_size=d_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,  # no dropout for single-layer GRU
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(2 * d_hidden),
            nn.Linear(2 * d_hidden, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, in_dim)
        Returns:
            (B, d_model)
        """
        x = self.proj(x)                    # (B, T, d_proj)

        # Ensure GRU memory is contiguous (prevents warning and improves speed)
        self.gru.flatten_parameters()

        # GRU forward: h_n shape = (2, B, d_hidden) for bidirectional
        _, h_n = self.gru(x)                
        
        # Concatenate forward and backward final hidden states
        h_fwd = h_n[0]                      # (B, d_hidden)
        h_bwd = h_n[1]                      # (B, d_hidden)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2*d_hidden)

        return self.out_proj(h_cat)          # (B, d_model)


class FusionEncoder(nn.Module):
    """
    Per-modality BiGRU fusion encoder.

    Input:  (B, T, total_dim)  — concatenated multimodal features
    Output: (B, num_modalities, d_model)  — one condition token per modality

    Parameters
    ----------
    modality_dims : list[int]
        Dimension of each modality in concatenation order.
    d_model : int
        Output embedding dimension (should match U-ViT embed_dim).
    d_proj : int
        Intermediate projection dim before GRU (controls GRU param count).
    d_hidden : int
        GRU hidden size per direction.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        modality_dims: list[int],
        d_model: int = 768,
        d_proj: int = 256,
        d_hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.num_modalities = len(modality_dims)
        self.d_model = d_model

        # Per-modality BiGRU blocks
        self.blocks = nn.ModuleList([
            PerModalityGRU(dim, d_proj, d_hidden, d_model, dropout)
            for dim in modality_dims
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, total_dim)  e.g. (B, 31, sum(modality_dims))

        Returns:
            (B, num_modalities, d_model)  e.g. (B, 11, 768)
        """
        # 1. Split concatenated features by modality
        splits = torch.split(x, self.modality_dims, dim=-1)

        # 2. Per-modality BiGRU → each outputs (B, d_model)
        modality_vectors = [block(s) for block, s in zip(self.blocks, splits)]

        # 3. Stack into condition tokens
        tokens = torch.stack(modality_vectors, dim=1)  # (B, M, d_model)

        return tokens
