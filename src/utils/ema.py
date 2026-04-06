"""
Exponential Moving Average (EMA) for model parameters.

Maintains a shadow copy of model weights updated as:
    θ_ema ← β * θ_ema + (1 - β) * θ_train

The EMA weights are used for validation/inference and typically 
produce smoother, higher quality predictions than raw training weights.

OOM-safe: avoids unnecessary .cpu() copies and .clone() of full param sets.
"""

import torch
import torch.nn as nn


class EMAModel:
    """
    Exponential Moving Average wrapper for one or more nn.Module models.
    
    Features:
    - Supports multiple models (e.g., Fusion Encoder + U-ViT)
    - Warmup: gradually increases decay from a lower value to target
    - Context manager for temporarily swapping EMA weights into models
    - Memory-safe: in-place swap avoids cloning entire parameter sets
    
    Usage:
        ema = EMAModel([encoder, uvit], decay=0.999)
        
        # After each optimizer.step():
        ema.update()
        
        # For validation:
        with ema.apply():
            val_loss = validate()
    """
    
    def __init__(
        self, 
        models: list[nn.Module], 
        decay: float = 0.999,
        warmup_steps: int = 1000,
        warmup_start_decay: float = 0.9,
    ):
        self.models = models
        self.target_decay = decay
        self.warmup_steps = warmup_steps
        self.warmup_start_decay = warmup_start_decay
        self.step_count = 0
        
        # Create shadow copies (detached, on same device as model params)
        self.shadow_params = []
        for model in models:
            shadow = [p.clone().detach() for p in model.parameters()]
            self.shadow_params.append(shadow)
    
    @property
    def current_decay(self) -> float:
        """Get current decay value, accounting for warmup."""
        if self.step_count >= self.warmup_steps:
            return self.target_decay
        progress = self.step_count / self.warmup_steps
        return self.warmup_start_decay + (self.target_decay - self.warmup_start_decay) * progress
    
    @torch.no_grad()
    def update(self):
        """Update EMA parameters. Call after each optimizer.step()."""
        decay = self.current_decay
        self.step_count += 1
        
        for model, shadow in zip(self.models, self.shadow_params):
            for s_param, m_param in zip(shadow, model.parameters()):
                s_param.mul_(decay).add_(m_param.data, alpha=1.0 - decay)
    
    @torch.no_grad()
    def apply(self):
        """Context manager that temporarily swaps EMA weights into models.
        
        Memory-safe: uses in-place swap (model ↔ shadow) instead of
        cloning the entire parameter set for backup.
        
        Usage:
            with ema.apply():
                # models now use EMA weights
                output = model(input)
            # models restored to training weights
        """
        return _EMAContext(self)
    
    def state_dict(self) -> dict:
        """Return EMA state for checkpointing.
        
        Note: shadow_params are stored as direct references.
        torch.save() handles GPU→CPU serialisation automatically,
        avoiding the OOM caused by explicit .cpu() copies.
        """
        return {
            "shadow_params": [
                [p.data for p in shadow] for shadow in self.shadow_params
            ],
            "step_count": self.step_count,
            "target_decay": self.target_decay,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load EMA state from checkpoint."""
        self.step_count = state_dict["step_count"]
        self.target_decay = state_dict["target_decay"]
        for shadow, saved_shadow in zip(self.shadow_params, state_dict["shadow_params"]):
            for s_param, saved_param in zip(shadow, saved_shadow):
                s_param.copy_(saved_param.to(s_param.device))


class _EMAContext:
    """Context manager for temporarily applying EMA weights.
    
    Uses in-place data swap between model params and shadow params,
    avoiding the memory overhead of cloning the entire parameter set.
    """
    
    def __init__(self, ema: EMAModel):
        self.ema = ema
    
    def __enter__(self):
        # In-place swap: model.data ↔ shadow.data
        # No .clone() needed — just swap the underlying storage
        for model, shadow in zip(self.ema.models, self.ema.shadow_params):
            for m_param, s_param in zip(model.parameters(), shadow):
                tmp = m_param.data
                m_param.data = s_param.data
                s_param.data = tmp
        return self
    
    def __exit__(self, *args):
        # Swap back: shadow.data ↔ model.data (same operation restores)
        for model, shadow in zip(self.ema.models, self.ema.shadow_params):
            for m_param, s_param in zip(model.parameters(), shadow):
                tmp = m_param.data
                m_param.data = s_param.data
                s_param.data = tmp
