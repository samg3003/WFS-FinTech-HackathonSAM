"""
AdvisorIQ — Layer A: VolSSM (S5-based Volatility Forecaster)

Architecture (unchanged from validated notebook):
    Linear(P→D) → 3× [Pre-LN residual S5 + GELU + Dropout] → final token → MLP → softplus

Input:  (batch, T=252, P=8)  raw daily features
Output: (batch,)             predicted 30-day annualised HV (strictly positive)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────
# HiPPO Initialisation
# ─────────────────────────────────────────────────────────────────────

def hippo_init(N, H):
    """
    Standard S4D/S5 HiPPO Initialization.
    Returns Lambda (N/2 complex), B (N/2, H complex), C (H, N/2 complex)
    """
    n = torch.arange(1, N // 2 + 1)
    Lambda = -0.5 + 1j * np.pi * n

    B = torch.ones((N // 2, H), dtype=torch.cfloat)
    C = (torch.randn(H, N // 2, dtype=torch.cfloat)) / (N**0.5)

    return Lambda, B, C


# ─────────────────────────────────────────────────────────────────────
# S5 Layer — Core SSM with Conv/RNN Duality
# ─────────────────────────────────────────────────────────────────────

class S5Layer(nn.Module):
    def __init__(self, d_model: int, state_size: int, seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        self.half_state = state_size // 2

        Lambda, B, C = hippo_init(state_size, d_model)

        self.log_Lambda_real = nn.Parameter(torch.log(-Lambda.real.clamp(max=-1e-4)))
        self.Lambda_imag = nn.Parameter(Lambda.imag)

        self.B = nn.Parameter(B)
        self.C = nn.Parameter(C)
        self.D = nn.Parameter(torch.randn(d_model, d_model) / (d_model**0.5))

        self.log_Delta = nn.Parameter(torch.tensor([-3.0]))

    def get_kernel(self, T: int):
        """Helper for visualization: returns the impulse response of the SSM."""
        with torch.no_grad():
            A_bar, B_bar, _, _ = self._discretize()
            t_idx = torch.arange(T, device=A_bar.device)
            log_A_bar = torch.log(A_bar)
            k_steps = torch.exp(log_A_bar.unsqueeze(1) * t_idx.unsqueeze(0))
            v = torch.einsum('np,n->np', self.B, k_steps[:, 0])
            kernel = 2.0 * torch.einsum('pn,nt->pt', self.C, k_steps).real
            return kernel.mean(dim=0)

    def _discretize(self):
        """Bilinear Transform (Tustin's method) for superior stability."""
        Delta = torch.exp(self.log_Delta)
        Lambda = -torch.exp(self.log_Lambda_real.clamp(min=-10, max=10)) + 1j * self.Lambda_imag

        denom = 1.0 - (Delta / 2.0) * Lambda
        A_bar = (1.0 + (Delta / 2.0) * Lambda) / denom
        B_bar = (Delta / denom).unsqueeze(-1) * self.B

        return A_bar, B_bar, Delta, Lambda

    def _conv_forward(self, u: torch.Tensor) -> torch.Tensor:
        B_sz, T, P = u.shape
        A_bar, B_bar, _, _ = self._discretize()

        t_idx = torch.arange(T, device=u.device)
        log_A_bar = torch.log(A_bar)
        k = torch.exp(log_A_bar.unsqueeze(1) * t_idx.unsqueeze(0))

        u_c = u.to(torch.cfloat)
        v = torch.einsum('np,btp->bnt', B_bar, u_c)

        fft_size = 2 * T
        V_f = torch.fft.fft(v, n=fft_size, dim=-1)
        K_f = torch.fft.fft(k, n=fft_size, dim=-1)
        conv = torch.fft.ifft(V_f * K_f.unsqueeze(0), dim=-1)[..., :T]

        y = 2.0 * torch.einsum('pn,bnt->btp', self.C, conv).real
        return y + (u @ self.D.T)

    def _rnn_forward(self, u: torch.Tensor) -> torch.Tensor:
        B_sz, T, P = u.shape
        A_bar, B_bar, _, _ = self._discretize()

        x = torch.zeros(B_sz, self.half_state, dtype=torch.cfloat, device=u.device)
        u_c = u.to(torch.cfloat)
        outputs = []

        for t in range(T):
            u_t = u_c[:, t, :]
            x = A_bar.unsqueeze(0) * x + torch.einsum('np,bp->bn', B_bar, u_t)
            y_t = 2.0 * torch.einsum('pn,bn->bp', self.C, x).real + (u[:, t, :] @ self.D.T)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def forward(self, u: torch.Tensor, mode: str = 'conv') -> torch.Tensor:
        return self._conv_forward(u) if mode == 'conv' else self._rnn_forward(u)


# ─────────────────────────────────────────────────────────────────────
# VolSSM — Full Model
# ─────────────────────────────────────────────────────────────────────

class VolSSM(nn.Module):
    """
    Volatility forecasting model using stacked S5 blocks.

    Architecture:
        Linear(P -> D) -> 3x [Pre-LN residual S5 + GELU + Dropout] -> final token -> MLP -> softplus

    Input: (batch, T=252, P=8) raw daily features
    Output: (batch,) predicted 30-day annualised HV (strictly positive via softplus)
    """

    def __init__(
        self,
        input_channels: int = 8,
        d_model: int = 128,
        state_size: int = 64,
        seq_len: int = 252,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.d_model = d_model
        self.seq_len = seq_len

        self.input_proj = nn.Linear(input_channels, d_model)

        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(S5Layer(d_model, state_size, seq_len))
            self.norms.append(nn.LayerNorm(d_model))

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, u: torch.Tensor, mode: str = 'conv') -> torch.Tensor:
        assert u.ndim == 3, f"Expected 3D input, got {u.ndim}D"
        assert u.shape[1] <= self.seq_len, f"T={u.shape[1]} > seq_len={self.seq_len}"
        assert u.shape[2] == self.input_channels, \
            f"Expected {self.input_channels} channels, got {u.shape[2]}"

        x = self.input_proj(u)

        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x), mode=mode)
            x = F.gelu(x)
            x = self.dropout(x)

        x = x[:, -1, :]
        x = self.head(x)
        x = F.softplus(x)
        return x.squeeze(-1)


# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────

def seed_all(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model):
    """Count real degrees of freedom (complex params count as 2x)."""
    total = 0
    for p in model.parameters():
        n = p.numel()
        if p.is_complex():
            n *= 2
        total += n
    return total


def make_optimizer(model, ssm_lr=1e-4, other_lr=1e-3, weight_decay=1e-4):
    """Create Adam optimizer with separate SSM and non-SSM param groups."""
    ssm_param_names = set()
    for i, _ in enumerate(model.blocks):
        for pname in ['log_Lambda_real', 'Lambda_imag', 'B', 'C', 'log_Delta']:
            ssm_param_names.add(f'blocks.{i}.{pname}')

    ssm_params, other_params = [], []
    for name, p in model.named_parameters():
        if name in ssm_param_names:
            ssm_params.append(p)
        else:
            other_params.append(p)

    return torch.optim.Adam([
        {'params': ssm_params, 'lr': ssm_lr},
        {'params': other_params, 'lr': other_lr, 'weight_decay': weight_decay},
    ])
