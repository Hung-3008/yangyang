"""
Multimodal Fusion Encoder
-------------------------
Nén 31 frames dữ liệu thời gian của 8 modalities (22,144 dims) thành một tập
hợp các latent tokens nhỏ gọn (ví dụ: 8 tokens x 1024 dims) bằng kiến trúc
giống Q-Former / Perceiver Resampler.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor, shape [B, T, D]
        """
        return x + self.pe[:, :x.size(1), :]


class MultimodalFusionEncoder(nn.Module):
    """
    1. Linear Projection (22144 -> 1024)
    2. Temporal Transformer Encoder (Mix context across 31 frames)
    3. Q-Former / Transformer Decoder (Compress to `num_latents` tokens)
    """
    def __init__(
        self,
        input_dim: int = 22144,
        d_model: int = 1024,
        num_latents: int = 8,
        num_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_latents = num_latents
        
        # 1. Modality Projection
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_embed = PositionalEncoding(d_model, max_len=500)
        
        # 2. Temporal Context Mixing (Self-Attention over time)
        # Using 2 layers for local temporal mixing before cross-attention.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. Querying / Latent Resampling (Cross-Attention for compression)
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, d_model))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        # The decoder acts as the Q-Former: 
        # Target (tgt) = latent_queries
        # Memory = temporal context
        self.q_former = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shape (B, T, input_dim)
               e.g. B = batch_size, T = 31 (window_size), input_dim = 22144
               
        Returns:
            out: Tensor shape (B, num_latents, d_model)
                 e.g. B = batch_size, num_latents = 8, d_model = 1024
        """
        B, T, _ = x.shape
        
        # 1. Project into d_model space & add temporal positioning
        x = self.proj(x)                  # (B, T, d_model)
        x = self.pos_embed(x)             # (B, T, d_model)
        
        # 2. Mix temporal information (let frames talk to each other)
        memory = self.temporal_transformer(x)    # (B, T, d_model)
        
        # 3. Extract latents using Cross-Attention (Q-Former logic)
        queries = self.latent_queries.expand(B, -1, -1)  # (B, num_latents, d_model)
        
        out = self.q_former(tgt=queries, memory=memory)  # (B, num_latents, d_model)
        
        return out
