"""
Exponential Moving Average (EMA) for model parameters.

Maintains a shadow copy of model weights updated as:
    θ_ema ← β * θ_ema + (1 - β) * θ_train

The EMA weights are used for validation/inference and typically 
produce smoother, higher quality predictions than raw training weights.
"""

import copy
import torch
import torch.nn as nn


class EMAModel:
    """
    Exponential Moving Average wrapper for one or more nn.Module models.
    
    Features:
    - Supports multiple models (e.g., Fusion Encoder + DiT)
    - Warmup: gradually increases decay from a lower value to target
    - Context manager for temporarily swapping EMA weights into models
    
    Usage:
        ema = EMAModel([encoder, dit], decay=0.999)
        
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
        """
        Parameters
        ----------
        models : list[nn.Module]
            List of models to track EMA for
        decay : float
            Target EMA decay rate (0.999 = slow update, 0.99 = fast update)
        warmup_steps : int
            Number of steps to linearly ramp decay from warmup_start_decay to decay
        warmup_start_decay : float
            Initial decay value during warmup
        """
        self.models = models
        self.target_decay = decay
        self.warmup_steps = warmup_steps
        self.warmup_start_decay = warmup_start_decay
        self.step_count = 0
        
        # Create shadow copies (detached, on same device)
        self.shadow_params = []
        for model in models:
            shadow = [p.clone().detach() for p in model.parameters()]
            self.shadow_params.append(shadow)
    
    @property
    def current_decay(self) -> float:
        """Get current decay value, accounting for warmup."""
        if self.step_count >= self.warmup_steps:
            return self.target_decay
        # Linear warmup
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
        
        Usage:
            with ema.apply():
                # models now use EMA weights
                output = model(input)
            # models restored to training weights
        """
        return _EMAContext(self)
    
    def state_dict(self) -> dict:
        """Return EMA state for checkpointing."""
        return {
            "shadow_params": [
                [p.cpu() for p in shadow] for shadow in self.shadow_params
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
    """Context manager for temporarily applying EMA weights."""
    
    def __init__(self, ema: EMAModel):
        self.ema = ema
        self.backup_params = []
    
    def __enter__(self):
        # Backup current training weights and load EMA weights
        for model, shadow in zip(self.ema.models, self.ema.shadow_params):
            backup = []
            for m_param, s_param in zip(model.parameters(), shadow):
                backup.append(m_param.data.clone())
                m_param.data.copy_(s_param)
            self.backup_params.append(backup)
        return self
    
    def __exit__(self, *args):
        # Restore training weights
        for model, backup in zip(self.ema.models, self.backup_params):
            for m_param, b_param in zip(model.parameters(), backup):
                m_param.data.copy_(b_param)
        self.backup_params.clear()
