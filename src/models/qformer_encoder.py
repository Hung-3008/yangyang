"""
Q-Former Fusion Encoder — Learnable Query + Cross-Attention
=============================================================
Replaces the Per-Modality BiGRU with a Q-Former architecture inspired
by BLIP-2 (Li et al., ICML 2023).

Pipeline:
  1. Linear(total_dim → d_model)     — project concatenated features
  2. + temporal positional embedding  — encode frame position
  3. Learnable Queries (N, d_model)   — fixed-size query bank
  4. L × QFormerLayer:
       a. Self-Attention among queries
       b. Cross-Attention: queries attend to projected features
       c. Feed-Forward Network
  5. LayerNorm

Output: (B, num_queries, d_model)  e.g. (B, 32, 768)
        32 condition tokens for U-ViT.

Advantages over BiGRU:
  - Queries attend to ALL modalities × ALL frames simultaneously
  - Cross-modal interaction happens naturally via cross-attention
  - Number of output tokens is decoupled from number of modalities
"""

import torch
import torch.nn as nn


class QFormerLayer(nn.Module):
    """Single Q-Former layer: Self-Attention → Cross-Attention → FFN.

    Uses pre-norm (LayerNorm before attention) for training stability.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.2):
        super().__init__()

        # Self-attention among queries
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )

        # Cross-attention: queries attend to input features
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )

        # Feed-forward network
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, N, D) — learnable query tokens
            kv:      (B, T, D) — projected multimodal features

        Returns:
            queries: (B, N, D) — updated query tokens
        """
        # 1. Self-Attention (queries interact with each other)
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]

        # 2. Cross-Attention (queries attend to input features)
        q = self.norm2(queries)
        queries = queries + self.cross_attn(q, kv, kv, need_weights=False)[0]

        # 3. FFN
        queries = queries + self.ffn(self.norm3(queries))

        return queries


class QFormerEncoder(nn.Module):
    """
    Q-Former Fusion Encoder for multimodal video features.

    Input:  (B, T, total_dim)    — concatenated multimodal features
    Output: (B, num_queries, d_model) — condition tokens for U-ViT

    Parameters
    ----------
    total_input_dim : int
        Sum of all modality dimensions (e.g. 28032).
    d_model : int
        Embedding dimension, must match U-ViT embed_dim.
    num_queries : int
        Number of learnable query tokens (= number of output condition tokens).
    num_layers : int
        Number of Q-Former layers (Self-Attn + Cross-Attn + FFN).
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    max_temporal_len : int
        Maximum number of temporal frames (for positional embedding).
    """

    def __init__(
        self,
        total_input_dim: int,
        d_model: int = 768,
        num_queries: int = 32,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
        max_temporal_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        # --- Input projection: total_dim → d_model ---
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Temporal positional embedding for input frames ---
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, max_temporal_len, d_model)
        )
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # --- Learnable query tokens ---
        self.queries = nn.Parameter(torch.zeros(1, num_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # --- Q-Former layers ---
        self.layers = nn.ModuleList([
            QFormerLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # --- Final normalization ---
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize linear weights with truncated normal."""
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, total_dim)  e.g. (B, 31, 28032)

        Returns:
            (B, num_queries, d_model)  e.g. (B, 32, 768)
        """
        B, T, _ = x.shape

        # 1. Project input features to d_model
        kv = self.input_proj(x)  # (B, T, d_model)

        # 2. Add temporal positional embedding
        kv = kv + self.temporal_pos_embed[:, :T, :]

        # 3. Expand learnable queries for batch
        queries = self.queries.expand(B, -1, -1)  # (B, N, d_model)

        # 4. Process through Q-Former layers
        for layer in self.layers:
            queries = layer(queries, kv)

        # 5. Final norm
        return self.norm(queries)
