import math
import torch
import torch.nn as nn

def modulate(x, shift, scale):
    """ Modulate LayerNorm outputs via Shift and Scale. """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class PatchEmbed1D(nn.Module):
    """ 
    Patchifies the 1D fMRI array.
    Example: 1000 parcels -> 50 patches of size 20.
    """
    def __init__(self, in_features=1000, patch_size=20, hidden_size=768):
        super().__init__()
        assert in_features % patch_size == 0, "in_features must be divisible by patch_size"
        self.num_patches = in_features // patch_size
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, hidden_size)

    def forward(self, x):
        # x shape: (B, 1000)
        B = x.size(0)
        x = x.view(B, self.num_patches, self.patch_size) # (B, 50, 20)
        x = self.proj(x) # (B, 50, hidden_size)
        return x

class DiTBlock(nn.Module):
    """
    A DiT block containing AdaLN, Self-Attention, Cross-Attention (for condition), and MLP.
    """
    def __init__(self, hidden_size, num_heads, context_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Nhan biet context co the la dim khac so voi hidden size. Pytorch ho tro kdim, vdim.
        self.attn2 = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads, 
            kdim=context_dim, 
            vdim=context_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        
        # AdaLN modulation mapping for 3 sub-blocks: Self-Attn, Cross-Attn, FFN.
        # Each needs 3 terms: shift, scale, gate. -> Total = 9 parameters per layer.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, t_emb, context):
        """
        x: (B, L, hidden_size)
        t_emb: (B, hidden_size)
        context: (B, seq_ctx, context_dim)
        """
        # (B, 9 * hidden_size) -> 9 chunks of (B, hidden_size)
        shift_msa, scale_msa, gate_msa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(9, dim=1)
            
        # 1. Modulated Self-Attention
        mod1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn1(mod1, mod1, mod1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # 2. Modulated Cross-Attention
        mod2 = modulate(self.norm2(x), shift_ca, scale_ca)
        # Context is used as Key and Value
        cross_out, _ = self.attn2(mod2, context, context)
        x = x + gate_ca.unsqueeze(1) * cross_out
        
        # 3. Modulated FFN
        mod3 = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(mod3)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT. Unpatchifies and produces final output scaling.
    """
    def __init__(self, hidden_size, patch_size, out_features):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x) # (B, L, hidden_size) -> (B, L, patch_size)
        
        # Flatten back to 1D
        B = x.size(0)
        x = x.view(B, -1) # -> (B, L * patch_size)
        return x

class DiTConditional(nn.Module):
    """
    Diffusion Model for 1D Array (fMRI) with AdaLN timestep and Cross Attention Context.
    """
    def __init__(
        self,
        in_features=1000,
        patch_size=20,
        hidden_size=768,
        depth=8,
        num_heads=12,
        context_dim=1024,
        dropout=0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.patch_size = patch_size
        
        self.x_embedder = PatchEmbed1D(in_features, patch_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Absolute positional embeddings for patches
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=True)
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, context_dim, dropout=dropout) for _ in range(depth)
        ])
        
        # Final block to map back to 1000 dims
        self.final_layer = FinalLayer(hidden_size, patch_size, in_features)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize DiT blocks properly (Zero-init for gates is standard for stability)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize final layer to output zeros (good for Flow Matching start)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, context):
        """
        x: (B, 1000) - Noisy fMRI Data
        t: (B,) - Timesteps array
        context: (B, num_latents, context_dim) - Context from Fusion Encoder
        
        Returns:
            v_t: (B, 1000) - The Vector field prediction
        """
        # Patchify and add position embed
        x = self.x_embedder(x) + self.pos_embed  # (B, L, hidden_size)
        
        # Embed timestep
        t_emb = self.t_embedder(t)              # (B, hidden_size)
        
        # Run through blocks
        for block in self.blocks:
            x = block(x, t_emb, context)
            
        # Final Unpatchify
        x = self.final_layer(x, t_emb)          # (B, 1000)
        
        return x
