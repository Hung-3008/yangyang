"""
Fusion Encoder — Temporal-Preserving Design (TRIBEv2-inspired)

Pipeline:
  1. Per-modality Linear projection:  (B, T, D_i) → (B, T, proj_dim)
  2. Modality Dropout (training):     randomly zero entire modalities
  3. Cat on feature dim:              (B, T, M * proj_dim)
  4. Combiner MLP:                    (B, T, M * proj_dim) → (B, T, d_model)
  5. Temporal Positional Embedding:   learnable positions for T frames
  6. Temporal Dropout (training):     randomly zero individual frames
  7. Transformer Encoder:             self-attention across T (temporal modeling)

Output: (B, T, d_model) — preserves full temporal resolution
"""

import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    """Project one modality to a shared hidden dim, per-timestep.
    (B, T, D_i) → (B, T, proj_dim)
    """

    def __init__(self, in_dim: int, proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionEncoder(nn.Module):
    """Temporal-preserving multimodal fusion encoder.

    Input:  (B, T, total_dim)   — concatenated multimodal features
    Output: (B, T, d_model)     — T temporal condition tokens
    """

    def __init__(
        self,
        modality_dims: list[int],
        d_model: int = 768,
        num_encoder_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        modality_dropout: float = 0.1,
        temporal_dropout: float = 0.1,
        max_len: int = 128,
        # Legacy params accepted but ignored (backward compat with old configs)
        num_queries: int = 1,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.d_model = d_model
        self.modality_dropout_p = modality_dropout
        self.temporal_dropout_p = temporal_dropout
        num_modalities = len(modality_dims)

        # 1. Per-modality projectors: D_i → proj_dim
        proj_dim = d_model // num_modalities
        self.proj_dim = proj_dim
        self.projectors = nn.ModuleList([
            ModalityProjector(dim, proj_dim, dropout=dropout)
            for dim in modality_dims
        ])

        # 2. Combiner MLP: cat_dim → d_model (cross-modality mixing per timestep)
        cat_dim = proj_dim * num_modalities
        self.combiner = nn.Sequential(
            nn.Linear(cat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3. Temporal positional embedding
        self.temporal_pos_emb = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02
        )

        # 4. Transformer encoder (self-attention on temporal axis)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, total_dim) — concatenated multimodal features

        Returns:
            (B, T, d_model) — temporal condition tokens
        """
        # 1. Split and project each modality
        splits = torch.split(x, self.modality_dims, dim=-1)
        projected = []
        for i, (proj, s) in enumerate(zip(self.projectors, splits)):
            feat = proj(s)  # (B, T, proj_dim)

            # Modality dropout: zero entire modality for some samples
            if self.training and self.modality_dropout_p > 0:
                mask = torch.rand(feat.size(0), device=feat.device) < self.modality_dropout_p
                feat = feat.masked_fill(mask[:, None, None], 0.0)

            projected.append(feat)

        # 2. Cat on feature dim → combiner
        out = torch.cat(projected, dim=-1)      # (B, T, M * proj_dim)
        out = self.combiner(out)                 # (B, T, d_model)

        # 3. Add temporal positional embedding
        T = out.size(1)
        out = out + self.temporal_pos_emb[:, :T, :]

        # 4. Temporal dropout: zero random frames
        if self.training and self.temporal_dropout_p > 0:
            B = out.size(0)
            for b in range(B):
                frame_mask = torch.rand(T, device=out.device) < self.temporal_dropout_p
                out[b, frame_mask, :] = 0.0

        # 5. Transformer encoder (temporal self-attention)
        out = self.encoder(out)                  # (B, T, d_model)

        return out
