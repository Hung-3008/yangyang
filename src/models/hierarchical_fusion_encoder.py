"""
Hierarchical Fusion Encoder
----------------------------
Compresses 31 temporal frames × 8 modalities (22,144 dims) into multi-level
latent tokens using a Q-Former / Perceiver Resampler architecture.

Motivated by MedSegDiff's multi-scale conditioning: different query sets learn
to extract features at different abstraction levels.  These are then concatenated
into a single (B, total_tokens, d_model) tensor for U-ViT token-based conditioning.

Hierarchy:
  - num_levels query sets × queries_per_level  →  low-to-high abstraction
  - 1 bottleneck query set                     →  global summary

  total_tokens = num_levels * queries_per_level + bottleneck_queries
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal dimension."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        return x + self.pe[:, : x.size(1), :]


class HierarchicalFusionEncoder(nn.Module):
    """
    Pipeline:
        1. Linear Projection  (input_dim → d_model)
        2. Temporal Transformer Encoder  (mix context across T frames)
        3. Multi-level Q-Former Decoder  (extract hierarchical latent tokens)

    Each level's query set learns to capture a different abstraction level.
    All outputs are concatenated into a flat (B, total_tokens, d_model) tensor
    that is directly compatible with U-ViT's token-concat conditioning.
    """

    def __init__(
        self,
        input_dim: int = 22144,
        d_model: int = 768,
        num_levels: int = 4,
        queries_per_level: int = 4,
        bottleneck_queries: int = 2,
        num_temporal_layers: int = 2,
        num_qformer_layers: int = 2,
        nhead: int = 12,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels
        self.queries_per_level = queries_per_level
        self.bottleneck_queries = bottleneck_queries
        self.total_tokens = num_levels * queries_per_level + bottleneck_queries

        # --- 1. Modality Projection ---
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_embed = PositionalEncoding(d_model, max_len=500)

        # --- 2. Temporal Context Mixing (self-attention over time) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_temporal_layers
        )

        # --- 3. Multi-level Query Sets ---
        # Each set captures a different abstraction level from the temporal memory.
        self.level_queries = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, queries_per_level, d_model) * 0.02)
                for _ in range(num_levels)
            ]
        )
        self.bottleneck_query = nn.Parameter(
            torch.randn(1, bottleneck_queries, d_model) * 0.02
        )

        # --- 4. Shared Q-Former Decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.q_former = nn.TransformerDecoder(
            decoder_layer, num_layers=num_qformer_layers
        )

        # --- 5. Per-level LayerNorm (output regularisation) ---
        self.level_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_levels + 1)]
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)  e.g. (B, 31, 22144)

        Returns:
            out: (B, total_tokens, d_model)
                 e.g. (B, 18, 768) with 4 levels × 4 queries + 2 bottleneck
        """
        B = x.size(0)

        # 1. Project + temporal positional encoding
        x = self.proj(x)  # (B, T, d_model)
        x = self.pos_embed(x)

        # 2. Temporal mixing
        memory = self.temporal_transformer(x)  # (B, T, d_model)

        # 3. Extract hierarchical tokens via cross-attention
        all_tokens: list[torch.Tensor] = []

        for i in range(self.num_levels):
            queries = self.level_queries[i].expand(B, -1, -1)
            level_out = self.q_former(tgt=queries, memory=memory)
            level_out = self.level_norms[i](level_out)
            all_tokens.append(level_out)

        # Bottleneck (global summary)
        bq = self.bottleneck_query.expand(B, -1, -1)
        bottleneck_out = self.q_former(tgt=bq, memory=memory)
        bottleneck_out = self.level_norms[-1](bottleneck_out)
        all_tokens.append(bottleneck_out)

        # 4. Concatenate all levels → flat tensor
        out = torch.cat(all_tokens, dim=1)  # (B, total_tokens, d_model)
        return out
