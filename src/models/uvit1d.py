"""
U-ViT 1D — Velocity Network for fMRI Flow Matching
----------------------------------------------------
Adapted from "All are Worth Words: A ViT Backbone for Diffusion Models"
(Bao et al., CVPR 2023) for 1D fMRI data (1000 Schaefer parcels).

Key features vs the previous DiT:
  1. All inputs (timestep, condition tokens, data patches) treated as tokens
     → no cross-attention, no AdaLN; conditioning is implicit via self-attention
  2. Long skip connections between encoder and decoder halves (U-Net style)
     → dramatically improves convergence and gradient flow
  3. Drop-in compatible with OT-CFM training loop (same forward signature)
"""

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """Create sinusoidal timestep embeddings (same convention as torchcfm)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


# ──────────────────────────────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────────────────────────────
class PatchEmbed1D(nn.Module):
    """Patchify 1D fMRI array.  1000 parcels → 50 patches of size 20."""

    def __init__(self, in_features: int = 1000, patch_size: int = 20,
                 embed_dim: int = 768):
        super().__init__()
        assert in_features % patch_size == 0, "in_features must be divisible by patch_size"
        self.num_patches = in_features // patch_size
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.view(B, self.num_patches, self.patch_size)  # (B, L, P)
        x = self.proj(x)  # (B, L, D)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with Flash-Attention support."""

    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, L, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each (B, H, L, D)

        # Flash Attention (PyTorch ≥ 2.0)
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """Feed-forward MLP with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int | None = None,
                 drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Block(nn.Module):
    """U-ViT Transformer Block with optional long skip connection."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0,
                 skip: bool = False, use_checkpoint: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio), drop=drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, skip, use_reentrant=False
            )
        return self._forward(x, skip)

    def _forward(self, x: torch.Tensor,
                 skip: torch.Tensor | None = None) -> torch.Tensor:
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────
class UViT1D(nn.Module):
    """
    U-ViT for 1D fMRI velocity prediction.

    Forward:
        x       : (B, 1000)  — noisy fMRI at time t
        t       : (B,)       — flow-matching timestep ∈ [0, 1]
        context : (B, N, D)  — condition tokens from HierarchicalFusionEncoder

    Returns:
        v_t     : (B, 1000)  — predicted velocity field
    """

    def __init__(
        self,
        in_features: int = 1000,
        patch_size: int = 20,
        embed_dim: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.2,
        num_cond_tokens: int = 18,
        skip: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_cond_tokens = num_cond_tokens

        # --- Patch Embedding ---
        self.patch_embed = PatchEmbed1D(in_features, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches  # e.g. 50

        # --- Time Embedding MLP ---
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        # extras = 1 (time token) + num_cond_tokens
        self.extras = 1 + num_cond_tokens

        # --- Positional Embedding (all tokens) ---
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.extras + num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- Encoder blocks (first half of depth) ---
        self.in_blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                      drop=drop_rate, skip=False,
                      use_checkpoint=use_checkpoint)
                for _ in range(depth // 2)
            ]
        )

        # --- Bottleneck ---
        self.mid_block = Block(
            embed_dim, num_heads, mlp_ratio, qkv_bias,
            drop=drop_rate, skip=False,
            use_checkpoint=use_checkpoint,
        )

        # --- Decoder blocks (second half, with long skip connections) ---
        self.out_blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                      drop=drop_rate, skip=skip,
                      use_checkpoint=use_checkpoint)
                for _ in range(depth // 2)
            ]
        )

        # --- Output head ---
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Initialise following U-ViT defaults + zero-init output."""

        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init)

        # Zero-init last projection for stable flow-matching start
        nn.init.constant_(self.decoder_pred.weight, 0)
        nn.init.constant_(self.decoder_pred.bias, 0)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, in_features) — noisy fMRI data
            t:       (B,) — timestep values in [0, 1]
            context: (B, num_cond_tokens, embed_dim) — hierarchical condition

        Returns:
            v_t:     (B, in_features) — predicted velocity
        """
        # 1. Patch embed
        x = self.patch_embed(x)  # (B, L, D)
        B, L, D = x.shape

        # 2. Time token
        time_token = self.time_embed(
            timestep_embedding(t, self.embed_dim)
        ).unsqueeze(1)  # (B, 1, D)

        # 3. Assemble sequence:  [time | condition | data_patches]
        x = torch.cat([time_token, context, x], dim=1)  # (B, extras+L, D)
        x = x + self.pos_embed

        # 4. Encoder (save activations for skip connections)
        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        # 5. Bottleneck
        x = self.mid_block(x)

        # 6. Decoder (consume skip connections in reverse)
        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        # 7. Output head → strip extras → flatten to 1D
        x = self.norm(x)
        x = self.decoder_pred(x)  # (B, extras+L, patch_size)

        # Keep only data-patch tokens
        x = x[:, self.extras :, :]  # (B, L, patch_size)
        x = x.reshape(B, -1)  # (B, in_features)

        return x
