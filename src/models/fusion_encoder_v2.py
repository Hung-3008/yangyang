"""
Fusion Encoder V2 — Modality-Aware Hierarchical Conditioning
=============================================================
Full redesign of the condition encoder with three key improvements
over V1 (HierarchicalFusionEncoder):

  A) Per-modality linear projections instead of single Linear(22144→768).
     Each modality is projected into d_model independently, preserving
     modality-specific information before fusion.

  B) Per-level Q-Former decoders instead of a shared decoder.
     Each decoder specialises for a different abstraction level.

  C) Cross-modal attention stage that lets modalities at the same
     timestep interact before temporal mixing.

  D) Auxiliary reconstruction head that predicts fMRI directly from
     the condition tokens (MedSegDiff-inspired auxiliary loss).

Pipeline:
  1. Split input → per-modality Linear(dim_i → d_model) + modality embeddings
  2. Cross-Modal Transformer  (modalities at same timestep attend each other)
  3. Attention-weighted pooling across modalities → (B, T, d_model)
  4. Temporal Transformer Encoder  (deeper, 4 layers)
  5. Per-level Q-Former Decoders  (each with own weights)
  6. Optional auxiliary prediction head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE(nn.Module):
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
        return x + self.pe[:, : x.size(1), :]


class FusionEncoderV2(nn.Module):
    """
    Modality-aware hierarchical fusion encoder.

    Input:  (B, T, total_dim)  — concatenated multimodal features
    Output: (B, total_tokens, d_model) — hierarchical condition tokens

    If return_aux=True, also returns auxiliary fMRI prediction (B, fmri_dim).
    """

    def __init__(
        self,
        modality_dims: list[int],
        d_model: int = 768,
        num_levels: int = 4,
        queries_per_level: int = 4,
        bottleneck_queries: int = 2,
        num_cross_modal_layers: int = 2,
        num_temporal_layers: int = 4,
        num_qformer_layers_per_level: int = 1,
        nhead: int = 12,
        dropout: float = 0.2,
        fmri_dim: int = 1000,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.num_modalities = len(modality_dims)
        self.d_model = d_model
        self.num_levels = num_levels
        self.queries_per_level = queries_per_level
        self.bottleneck_queries = bottleneck_queries
        self.total_tokens = num_levels * queries_per_level + bottleneck_queries

        # ── Stage 1: Per-Modality Projection ──────────────────────────
        self.mod_projections = nn.ModuleList(
            [nn.Linear(dim, d_model) for dim in modality_dims]
        )
        # Learnable modality-type embeddings (like segment embeddings in BERT)
        self.mod_embedding = nn.Embedding(self.num_modalities, d_model)

        # ── Stage 2: Cross-Modal Transformer ──────────────────────────
        # Modalities at the same timestep attend each other
        cross_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_modal_transformer = nn.TransformerEncoder(
            cross_layer, num_layers=num_cross_modal_layers
        )

        # Attention-weighted pooling across modalities
        self.mod_attn_pool = nn.Linear(d_model, 1)

        # ── Stage 3: Temporal Context Mixing ──────────────────────────
        self.temporal_pe = SinusoidalPE(d_model, max_len=500)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer, num_layers=num_temporal_layers
        )

        # ── Stage 4: Per-Level Q-Former Decoders ──────────────────────
        self.level_queries = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, queries_per_level, d_model) * 0.02)
                for _ in range(num_levels)
            ]
        )
        self.bottleneck_query = nn.Parameter(
            torch.randn(1, bottleneck_queries, d_model) * 0.02
        )

        # Each level has its own dedicated decoder
        self.q_formers = nn.ModuleList()
        for _ in range(num_levels + 1):  # +1 for bottleneck
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.q_formers.append(
                nn.TransformerDecoder(
                    decoder_layer, num_layers=num_qformer_layers_per_level
                )
            )

        # Per-level output normalisation
        self.level_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_levels + 1)]
        )

        # ── Stage 5: Auxiliary Reconstruction Head ────────────────────
        self.aux_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, fmri_dim),
        )

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          (B, T, total_dim)  concatenated features, e.g. (B, 31, 22144)
            return_aux: if True, also return auxiliary fMRI prediction

        Returns:
            tokens:   (B, total_tokens, d_model)  e.g. (B, 18, 768)
            aux_pred: (B, fmri_dim) — only if return_aux=True
        """
        B, T, _ = x.shape

        # ── 1. Split & project per modality ───────────────────────────
        splits = torch.split(x, self.modality_dims, dim=-1)
        # Each split: (B, T, dim_i)

        projected = []
        for i, (proj, s) in enumerate(zip(self.mod_projections, splits)):
            h = proj(s)  # (B, T, d_model)
            # Add modality type embedding
            mod_emb = self.mod_embedding.weight[i]  # (d_model,)
            h = h + mod_emb
            projected.append(h)

        # Stack: (B, T, M, d_model)
        stacked = torch.stack(projected, dim=2)

        # ── 2. Cross-modal attention ──────────────────────────────────
        # Reshape so each timestep is an independent sequence of M tokens
        M, D = self.num_modalities, self.d_model
        stacked = stacked.reshape(B * T, M, D)
        fused = self.cross_modal_transformer(stacked)  # (B*T, M, D)

        # ── 3. Attention-weighted pooling across modalities ───────────
        attn_logits = self.mod_attn_pool(fused).squeeze(-1)  # (B*T, M)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B*T, M)
        fused = (fused * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B*T, D)
        fused = fused.reshape(B, T, D)

        # ── 4. Temporal context mixing ────────────────────────────────
        fused = self.temporal_pe(fused)
        memory = self.temporal_transformer(fused)  # (B, T, D)

        # ── 5. Per-level Q-Former extraction ──────────────────────────
        all_tokens: list[torch.Tensor] = []

        for i in range(self.num_levels):
            queries = self.level_queries[i].expand(B, -1, -1)
            level_out = self.q_formers[i](tgt=queries, memory=memory)
            level_out = self.level_norms[i](level_out)
            all_tokens.append(level_out)

        # Bottleneck (global summary)
        bq = self.bottleneck_query.expand(B, -1, -1)
        bottleneck_out = self.q_formers[-1](tgt=bq, memory=memory)
        bottleneck_out = self.level_norms[-1](bottleneck_out)
        all_tokens.append(bottleneck_out)

        tokens = torch.cat(all_tokens, dim=1)  # (B, total_tokens, d_model)

        # ── 6. Auxiliary reconstruction ───────────────────────────────
        if return_aux:
            # Mean-pool condition tokens → predict fMRI directly
            aux_input = tokens.mean(dim=1)  # (B, d_model)
            aux_pred = self.aux_head(aux_input)  # (B, fmri_dim)
            return tokens, aux_pred

        return tokens
