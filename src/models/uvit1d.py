"""
U-ViT 1D — Velocity Network v_θ(x_t, t, C) for Flow Matching

Each block:
  1. Self-Attention  — AdaLN-Zero modulated by (timestep + pooled condition)
  2. Cross-Attention — data patches attend to condition tokens C
  3. FFN             — AdaLN-Zero modulated by (timestep + pooled condition)

Layout: Encoder (N blocks) → Bottleneck → Decoder (N blocks with U-Net skip)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """Sinusoidal timestep embeddings (standard diffusion convention)."""
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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    """AdaLN modulation: x * (1 + scale) + shift.  shift/scale are (B, D)."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
        x = x.view(B, self.num_patches, self.patch_size)
        return self.proj(x)


class Attention(nn.Module):
    """Multi-head self-attention with Flash-Attention support."""

    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
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
        q, k, v = qkv.unbind(0)

        if hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """Multi-head cross-attention: data tokens (Q) attend to condition tokens (KV)."""

    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, L, D) — query tokens (data patches)
            context: (B, S, D) — key/value tokens (condition)
        """
        B, L, C = x.shape
        S = context.size(1)

        q = (self.q_proj(x)
             .reshape(B, L, self.num_heads, self.head_dim)
             .permute(0, 2, 1, 3))
        kv = (self.kv_proj(context)
              .reshape(B, S, 2, self.num_heads, self.head_dim)
              .permute(2, 0, 3, 1, 4))
        k, v = kv.unbind(0)

        if hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
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


# ──────────────────────────────────────────────────────────────────────
# Transformer Block
# ──────────────────────────────────────────────────────────────────────
class Block(nn.Module):
    """
    U-ViT Block: AdaLN-Zero(timestep) + Cross-Attention(condition) + U-Net skip.

    Sub-layers:
      1. Self-Attention  — AdaLN-Zero modulated (scale, shift, gate from timestep)
      2. Cross-Attention — condition tokens as KV, zero-initialized gate
      3. FFN             — AdaLN-Zero modulated (scale, shift, gate from timestep)
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0,
                 skip: bool = False, use_checkpoint: bool = False):
        super().__init__()
        mlp_hidden = int(dim * mlp_ratio)

        # 1. Self-Attention (AdaLN-Zero modulated)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias, proj_drop=drop)

        # 2. Cross-Attention (condition injection)
        self.norm_cross = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias, proj_drop=drop)
        self.cross_gate = nn.Parameter(torch.zeros(dim))

        # 3. FFN (AdaLN-Zero modulated)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, mlp_hidden, drop)

        # AdaLN-Zero: timestep → 6 modulation parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        # U-Net skip connection (decoder blocks only)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                cond: torch.Tensor,
                skip: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, t_emb, cond, skip, use_reentrant=False
            )
        return self._forward(x, t_emb, cond, skip)

    def _forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                 cond: torch.Tensor,
                 skip: torch.Tensor | None = None) -> torch.Tensor:
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        (shift_sa, scale_sa, gate_sa,
         shift_ff, scale_ff, gate_ff) = self.adaLN_modulation(t_emb).chunk(6, dim=-1)

        # 1. Self-Attention with AdaLN-Zero
        h = modulate(self.norm1(x), shift_sa, scale_sa)
        h = self.attn(h)
        x = x + gate_sa.unsqueeze(1) * h

        # 2. Cross-Attention (condition injection, zero-init gate)
        h = self.cross_attn(self.norm_cross(x), cond)
        x = x + self.cross_gate * h

        # 3. FFN with AdaLN-Zero
        h = modulate(self.norm2(x), shift_ff, scale_ff)
        h = self.mlp(h)
        x = x + gate_ff.unsqueeze(1) * h

        return x


# ──────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────
class UViT1D(nn.Module):
    """
    U-ViT 1D with AdaLN-Zero + Cross-Attention conditioning.

    Conditioning:
      - Timestep → sinusoidal emb → MLP → per-block AdaLN-Zero
      - Condition tokens → per-block Cross-Attention (KV)

    Sequence processed = data patches only (50 tokens for 1000 parcels).
    No token concatenation needed — condition enters via cross-attention.

    Forward:
        x       : (B, 1000)           — noisy fMRI at time t
        t       : (B,)                — flow-matching timestep ∈ [0, 1]
        context : (B, N_cond, D)      — condition tokens from FusionEncoder

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
        drop_rate: float = 0.1,
        skip: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # --- Patch Embedding ---
        self.patch_embed = PatchEmbed1D(in_features, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches

        # --- Positional Embedding (data patches only, no extras) ---
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # --- Timestep → global AdaLN conditioning vector ---
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        # --- Condition → pooled vector added to timestep for AdaLN ---
        # Provides direct gradient path: loss → AdaLN → cond_embed → FusionEncoder
        # Cross-attention provides additional fine-grained per-token conditioning
        self.cond_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

        # --- Encoder blocks ---
        self.in_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                  drop=drop_rate, skip=False, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)
        ])

        # --- Bottleneck ---
        self.mid_block = Block(
            embed_dim, num_heads, mlp_ratio, qkv_bias,
            drop=drop_rate, skip=False, use_checkpoint=use_checkpoint,
        )

        # --- Decoder blocks (with long skip connections) ---
        self.out_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                  drop=drop_rate, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)
        ])

        # --- Output head: AdaLN-modulated norm → Linear ---
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True),
        )
        self.decoder_pred = nn.Linear(embed_dim, patch_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize: trunc_normal for linears, then zero-init all gates & output."""

        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)

        self.apply(_basic_init)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Zero-init AdaLN modulation in every block (all gates start closed)
        for block in list(self.in_blocks) + [self.mid_block] + list(self.out_blocks):
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-init final output AdaLN
        nn.init.constant_(self.final_adaln[-1].weight, 0)
        nn.init.constant_(self.final_adaln[-1].bias, 0)

        # Zero-init decoder prediction head
        nn.init.constant_(self.decoder_pred.weight, 0)
        nn.init.constant_(self.decoder_pred.bias, 0)

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
            context: (B, N_cond, embed_dim) — condition tokens

        Returns:
            v_t:     (B, in_features) — predicted velocity
        """
        # 1. Patch embed + positional encoding
        x = self.patch_embed(x)                                    # (B, L, D)
        B, L, D = x.shape
        x = x + self.pos_embed

        # 2. Timestep + pooled condition → combined AdaLN vector
        t_emb = self.time_embed(timestep_embedding(t * 1000.0, self.embed_dim))  # (B, D)
        c_emb = self.cond_embed(context.mean(dim=1))                    # (B, D)
        adaln_emb = t_emb + c_emb                                      # (B, D)

        # 3. Encoder (save activations for skip connections)
        skips = []
        for blk in self.in_blocks:
            x = blk(x, adaln_emb, context)
            skips.append(x)

        # 4. Bottleneck
        x = self.mid_block(x, adaln_emb, context)

        # 5. Decoder (consume skip connections in reverse)
        for blk in self.out_blocks:
            x = blk(x, adaln_emb, context, skip=skips.pop())

        # 6. Output: AdaLN-modulated norm → linear → reshape
        shift, scale = self.final_adaln(adaln_emb).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.decoder_pred(x)                                   # (B, L, patch_size)
        x = x.reshape(B, -1)                                       # (B, in_features)

        return x
