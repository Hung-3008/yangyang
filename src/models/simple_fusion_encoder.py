"""
Simple Fusion Encoder — Per-Modality Projection + Lightweight Transformer
==========================================================================
Inspired by TRIBE (Algonauts 2025 winner): each modality gets its own
LayerNorm + Linear projection to d_model, then all projections are SUMMED.

Why per-modality projection (not flat concat → single Linear):
  1. Per-modality LayerNorm normalizes different feature distributions
     (e.g. vjepa2 vs whisper vs LLM) before mixing — prevents scale mismatch
  2. Modality dropout: randomly zero-out entire modalities during training
     → regularization, forces model to not over-rely on any single modality
  3. Same parameter count as single Linear(total_dim → d_model), but structured

Pipeline:
  1. Split concatenated input by modality dims
  2. Per-modality: LayerNorm(dim_i) → Linear(dim_i → d_model)
  3. Sum all projected modalities → (B, T, d_model)
  4. + Learnable temporal positional embedding
  5. N-layer Transformer Encoder (self-attention only)
  6. Outputs:
     - context_tokens: (B, T, d_model) — for cross-attention in DiT blocks
     - global_cond:    (B, d_model)    — mean-pooled, for AdaLN modulation
"""

import torch
import torch.nn as nn


class ModalityProjection(nn.Module):
    """Per-modality projection: LayerNorm → Linear → d_model.
    
    Each modality is normalized independently before projection,
    preventing scale mismatch between different feature extractors.
    """

    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class FusionTransformerLayer(nn.Module):
    """Single Transformer encoder layer with pre-norm.
    
    Self-Attention → FFN (no cross-attention needed here).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-Attention (pre-norm)
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        # FFN (pre-norm)
        x = x + self.ffn(self.norm2(x))
        return x


class SimpleFusionEncoder(nn.Module):
    """
    TRIBE-style Fusion Encoder for multimodal video features.

    Input:  (B, T, total_dim)  e.g. (B, 31, 28032) — concatenated modalities
    Output: context_tokens (B, T, d_model)  — for cross-attention
            global_cond    (B, d_model)     — for AdaLN modulation

    Parameters
    ----------
    modality_dims : list[int]
        Dimension of each modality in concatenation order.
        e.g. [1408, 1280, 5120, 3584, 2048, 3072, 3584, 2048, 768, 1536, 3584]
    d_model : int
        Embedding dimension, must match velocity network embed_dim.
    num_layers : int
        Number of Transformer encoder layers.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    modality_drop_prob : float
        Probability of dropping each modality during training (0 = disabled).
    max_temporal_len : int
        Maximum number of temporal frames (for positional embedding).
    """

    def __init__(
        self,
        modality_dims: list[int],
        d_model: int = 768,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        modality_drop_prob: float = 0.1,
        max_temporal_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.modality_dims = modality_dims
        self.num_modalities = len(modality_dims)
        self.modality_drop_prob = modality_drop_prob

        # --- Per-modality projections: LN(dim_i) → Linear(dim_i → d_model) ---
        self.modality_projs = nn.ModuleList([
            ModalityProjection(dim, d_model)
            for dim in modality_dims
        ])

        # --- Post-fusion LayerNorm (stabilize after summation) ---
        self.post_fusion_norm = nn.LayerNorm(d_model)

        # --- Temporal positional embedding ---
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, max_temporal_len, d_model)
        )
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # --- Transformer encoder layers ---
        self.layers = nn.ModuleList([
            FusionTransformerLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # --- Final normalization ---
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal."""
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, total_dim)  e.g. (B, 31, 28032)

        Returns:
            context_tokens: (B, T, d_model)  — for cross-attention in DiT
            global_cond:    (B, d_model)     — mean-pooled, for AdaLN
        """
        B, T, _ = x.shape

        # 1. Split concatenated input by modality boundaries
        modality_features = torch.split(x, self.modality_dims, dim=-1)

        # 2. Project each modality independently
        projections = []
        for i, (feat, proj) in enumerate(zip(modality_features, self.modality_projs)):
            h = proj(feat)  # (B, T, d_model)
            projections.append(h)

        # 3. Modality dropout: randomly zero entire modalities (training only)
        if self.training and self.modality_drop_prob > 0:
            # Per-sample, per-modality dropout mask
            drop_mask = torch.rand(
                B, self.num_modalities, device=x.device
            ) > self.modality_drop_prob  # True = keep

            # Ensure at least 1 modality is kept per sample
            all_dropped = ~drop_mask.any(dim=1)  # (B,)
            if all_dropped.any():
                # Force-keep a random modality for those samples
                rand_idx = torch.randint(
                    self.num_modalities, (all_dropped.sum(),), device=x.device
                )
                drop_mask[all_dropped, rand_idx] = True

            # Scale kept modalities to preserve expected sum magnitude
            num_kept = drop_mask.float().sum(dim=1, keepdim=True)  # (B, 1)
            scale = self.num_modalities / num_kept  # (B, 1)

            # Apply mask: (B, 1, 1) for broadcasting over (B, T, d_model)
            for i in range(self.num_modalities):
                mask_i = drop_mask[:, i].float().view(B, 1, 1)  # (B, 1, 1)
                scale_i = scale.view(B, 1, 1)
                projections[i] = projections[i] * mask_i * scale_i

        # 4. Sum all modality projections
        x = torch.stack(projections, dim=0).sum(dim=0)  # (B, T, d_model)

        # 5. Post-fusion normalization
        x = self.post_fusion_norm(x)

        # 6. Add temporal positional embedding
        x = x + self.temporal_pos_embed[:, :T, :]

        # 7. Process through Transformer encoder layers
        for layer in self.layers:
            x = layer(x)

        # 8. Final norm
        context_tokens = self.norm(x)  # (B, T, d_model)

        # 9. Global condition via mean pooling
        global_cond = context_tokens.mean(dim=1)  # (B, d_model)

        return context_tokens, global_cond
