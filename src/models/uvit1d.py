"""
U-ViT 1D v2 — Velocity Network with AdaLN-Zero + Cross-Attention
------------------------------------------------------------------
Refactored from v1 "all-as-tokens" U-ViT to a DiT-style architecture
with explicit conditioning mechanisms:

  1. AdaLN-Zero: global modulation via (timestep + global_cond) →
     scale, shift, gate for normalization layers
  2. Cross-Attention: fMRI patches (Q) attend to context tokens (KV)
     from the fusion encoder — explicit information routing
  3. Long skip connections retained for U-Net-style gradient flow

Key changes vs v1:
  - No more token concatenation [time | cond | data]
  - Time enters via AdaLN modulation (not as a prepended token)
  - Condition enters via cross-attention + AdaLN (not as prepended tokens)
  - Sequence length reduced: 50 patches only (was 83 = 1+32+50)
  - Each block has: AdaLN-SelfAttn → CrossAttn → AdaLN-FFN

References:
  - DiT: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
  - U-ViT: "All are Worth Words" (Bao et al., 2023) — skip connections
  - plan.md: Cross-Attention recommended for multimodal conditioning
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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    """Apply AdaLN modulation: x * (1 + scale) + shift."""
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
        x = x.view(B, self.num_patches, self.patch_size)  # (B, L, P)
        x = self.proj(x)  # (B, L, D)
        return x


class SelfAttention(nn.Module):
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
        q, k, v = qkv.unbind(0)  # each (B, H, L, D)

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            x = torch.nn.functional.scaled_dot_product_attention(
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
    """Multi-head cross-attention: Q from fMRI, KV from context tokens."""

    def __init__(self, dim: int, context_dim: int, num_heads: int = 8,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, L, D) — fMRI patch tokens (Query)
            context: (B, S, D) — encoder context tokens (Key/Value)
        Returns:
            out:     (B, L, D)
        """
        B, L, C = x.shape
        S = context.size(1)
        H = self.num_heads
        head_dim = self.head_dim

        q = self.q_proj(x).reshape(B, L, H, head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, S, H, head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, S, H, head_dim).permute(0, 2, 1, 3)

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            scale = head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


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
# DiT Block with AdaLN-Zero + Cross-Attention + Optional Skip
# ──────────────────────────────────────────────────────────────────────
class DiTBlock(nn.Module):
    """
    DiT-style transformer block with:
      1. AdaLN-Zero modulated Self-Attention
      2. Cross-Attention (Q=fMRI patches, KV=context tokens)
      3. AdaLN-Zero modulated Feed-Forward Network
      + Optional long skip connection (U-ViT style)

    AdaLN-Zero produces 6 modulation params per block:
      (γ1, β1, α1) for self-attention
      (γ2, β2, α2) for FFN
    Cross-attention uses standard pre-norm (not AdaLN) for simplicity.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cross_attn_heads: int = 8,
        context_dim: int = 768,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        skip: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # --- Skip connection linear (U-ViT style) ---
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None

        # --- AdaLN-Zero Self-Attention ---
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=drop,
        )

        # --- Cross-Attention ---
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(
            dim, context_dim=context_dim, num_heads=cross_attn_heads,
            qkv_bias=qkv_bias, proj_drop=drop,
        )

        # --- AdaLN-Zero FFN ---
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )

        # --- AdaLN modulation projection ---
        # Produces 6 params: (γ1, β1, α1, γ2, β2, α2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: torch.Tensor,
        skip_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, c, context, skip_x, use_reentrant=False,
            )
        return self._forward(x, c, context, skip_x)

    def _forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: torch.Tensor,
        skip_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, L, D)  — fMRI patch tokens
            c:       (B, D)     — conditioning vector (time + global_cond)
            context: (B, S, D)  — context tokens from encoder
            skip_x:  (B, L, D)  — skip connection from encoder (optional)
        """
        # 0. Skip connection (U-ViT style)
        if self.skip_linear is not None and skip_x is not None:
            x = self.skip_linear(torch.cat([x, skip_x], dim=-1))

        # 1. Get AdaLN modulation parameters
        mod = self.adaLN_modulation(c)  # (B, 6*D)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # 2. AdaLN Self-Attention
        h = modulate(self.norm1(x), beta1, gamma1)
        x = x + alpha1.unsqueeze(1) * self.attn(h)

        # 3. Cross-Attention (standard pre-norm, no AdaLN)
        x = x + self.cross_attn(self.norm_cross(x), context)

        # 4. AdaLN FFN
        h = modulate(self.norm2(x), beta2, gamma2)
        x = x + alpha2.unsqueeze(1) * self.mlp(h)

        return x


# ──────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────
class UViT1D(nn.Module):
    """
    U-ViT 1D v2 — Velocity Network for fMRI Flow Matching.

    Architecture: DiT blocks with AdaLN-Zero + Cross-Attention,
    arranged in U-Net topology with long skip connections.

    Forward:
        x             : (B, 1000)  — noisy fMRI at time t
        t             : (B,)       — flow-matching timestep ∈ [0, 1]
        context_tokens: (B, S, D)  — temporal context from fusion encoder
        global_cond   : (B, D)     — global condition for AdaLN

    Returns:
        v_t           : (B, 1000)  — predicted velocity field
    """

    def __init__(
        self,
        in_features: int = 1000,
        patch_size: int = 20,
        embed_dim: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        cross_attn_heads: int = 8,
        context_dim: int = 768,
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
        num_patches = self.patch_embed.num_patches  # e.g. 50

        # --- Time + Global Condition MLP → conditioning vector ---
        # Merges: timestep_embedding(t) + global_cond → c for AdaLN
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        # --- Positional Embedding (fMRI patches only, no time/cond tokens) ---
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- Encoder blocks (first half of depth) ---
        self.in_blocks = nn.ModuleList([
            DiTBlock(
                embed_dim, num_heads, cross_attn_heads, context_dim,
                mlp_ratio, qkv_bias, drop=drop_rate, skip=False,
                use_checkpoint=use_checkpoint,
            )
            for _ in range(depth // 2)
        ])

        # --- Bottleneck ---
        self.mid_block = DiTBlock(
            embed_dim, num_heads, cross_attn_heads, context_dim,
            mlp_ratio, qkv_bias, drop=drop_rate, skip=False,
            use_checkpoint=use_checkpoint,
        )

        # --- Decoder blocks (second half, with long skip connections) ---
        self.out_blocks = nn.ModuleList([
            DiTBlock(
                embed_dim, num_heads, cross_attn_heads, context_dim,
                mlp_ratio, qkv_bias, drop=drop_rate, skip=skip,
                use_checkpoint=use_checkpoint,
            )
            for _ in range(depth // 2)
        ])

        # --- Final AdaLN + Output head ---
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim),  # γ_final, β_final
        )
        self.decoder_pred = nn.Linear(embed_dim, patch_size)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Initialise following DiT/U-ViT defaults + zero-init output."""

        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

        self.apply(_init)

        # Zero-init AdaLN modulation layers → starts as identity
        for block in list(self.in_blocks) + [self.mid_block] + list(self.out_blocks):
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-init final AdaLN
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)

        # Zero-init output projection for stable flow-matching start
        nn.init.constant_(self.decoder_pred.weight, 0)
        nn.init.constant_(self.decoder_pred.bias, 0)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context_tokens: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, in_features) — noisy fMRI data
            t:              (B,) — timestep values in [0, 1]
            context_tokens: (B, S, D) — temporal context from fusion encoder
            global_cond:    (B, D) — global condition for AdaLN

        Returns:
            v_t:            (B, in_features) — predicted velocity
        """
        # 1. Patch embed
        x = self.patch_embed(x)  # (B, L, D)
        B, L, D = x.shape

        # 2. Add positional embedding
        x = x + self.pos_embed

        # 3. Build conditioning vector: c = time_embed + cond_embed
        t_emb = self.time_embed(timestep_embedding(t, self.embed_dim))  # (B, D)
        c_emb = self.cond_embed(global_cond)  # (B, D)
        c = t_emb + c_emb  # (B, D) — merged conditioning vector

        # 4. Encoder (save activations for skip connections)
        skips = []
        for blk in self.in_blocks:
            x = blk(x, c, context_tokens)
            skips.append(x)

        # 5. Bottleneck
        x = self.mid_block(x, c, context_tokens)

        # 6. Decoder (consume skip connections in reverse)
        for blk in self.out_blocks:
            x = blk(x, c, context_tokens, skip_x=skips.pop())

        # 7. Final AdaLN + Output head
        mod = self.final_adaLN(c)
        gamma, beta = mod.chunk(2, dim=-1)
        x = modulate(self.final_norm(x), beta, gamma)

        x = self.decoder_pred(x)  # (B, L, patch_size)
        x = x.reshape(B, -1)  # (B, in_features)

        return x
