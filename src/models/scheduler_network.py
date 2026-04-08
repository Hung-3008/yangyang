"""
Scheduler Network — Condition-Dependent Interpolation Schedule
================================================================
Generates dynamic interpolation coefficients α_t(C) and σ_t(C)
that replace the linear interpolation X_t = tX₁ + (1-t)X₀ with
a condition-dependent path X_t = α_t(C)X₁ + σ_t(C)X₀.

Parameterization (boundary-safe, analytic derivatives):
    g_α(C), g_σ(C) = MLP(C_pooled)            ∈ [-1, 1]  (Tanh)

    α_t(C) = t      + t(1-t) · g_α(C)         α(0)=0, α(1)=1  ✓
    σ_t(C) = (1-t)  + t(1-t) · g_σ(C)         σ(0)=1, σ(1)=0  ✓

    α̇_t(C) = 1   + (1-2t) · g_α(C)           analytic ✓
    σ̇_t(C) = -1  + (1-2t) · g_σ(C)           analytic ✓

Zero-init trick: last Linear layer initialized to zeros, so the
network starts as standard linear interpolation and gradually
learns condition-dependent adjustments.

Reference:
    Plan.md — Phương pháp 2: Condition-Dependent Interpolant
"""

import torch
import torch.nn as nn


class SchedulerNetwork(nn.Module):
    """Condition-dependent interpolation schedule.

    Generates α_t(C), σ_t(C) and their time-derivatives from
    a pooled condition vector and timestep t.

    Parameters
    ----------
    d_cond : int
        Dimension of the pooled condition vector (e.g. 768 = d_model).
    d_hidden : int
        Hidden dimension of the MLP.
    """

    def __init__(self, d_cond: int = 768, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_cond, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 2),   # 2 outputs: g_alpha, g_sigma
            nn.Tanh(),                # bound to [-1, 1] for stability
        )

        # Zero-init last Linear → starts as linear interpolation
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(
        self, C_pooled: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            C_pooled: (B, d_cond) — mean-pooled condition from Q-Former
            t:        (B,)       — timestep ∈ [0, 1]

        Returns:
            alpha_t:   (B, 1) — signal coefficient
            sigma_t:   (B, 1) — noise coefficient
            alpha_dot: (B, 1) — ∂α_t/∂t
            sigma_dot: (B, 1) — ∂σ_t/∂t
        """
        g = self.net(C_pooled)            # (B, 2)
        g_alpha = g[:, 0:1]              # (B, 1)
        g_sigma = g[:, 1:2]              # (B, 1)

        t_ = t.unsqueeze(-1)             # (B, 1)
        residual = t_ * (1.0 - t_)       # (B, 1)  — = 0 at t=0 and t=1

        # Interpolation coefficients
        alpha_t = t_ + residual * g_alpha
        sigma_t = (1.0 - t_) + residual * g_sigma

        # Analytic time-derivatives
        alpha_dot = 1.0 + (1.0 - 2.0 * t_) * g_alpha
        sigma_dot = -1.0 + (1.0 - 2.0 * t_) * g_sigma

        return alpha_t, sigma_t, alpha_dot, sigma_dot
