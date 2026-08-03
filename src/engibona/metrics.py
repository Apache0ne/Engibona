from __future__ import annotations

import torch


def activation_diagonal(
    activations: torch.Tensor,
    group_size: int,
    ridge: float = 1.0e-4,
) -> torch.Tensor:
    """Estimate diagonal input Hessian blocks from calibration activations.

    activations: [..., in_features]
    returns: [num_input_groups, group_size]
    """
    x = activations.detach().float().reshape(-1, activations.shape[-1])
    pad = (-x.shape[-1]) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    xg = x.reshape(x.shape[0], -1, group_size)
    diag = xg.square().mean(dim=0)
    mean_diag = diag.mean(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return diag + ridge * mean_diag


def activation_covariance(
    activations: torch.Tensor,
    group_size: int,
    ridge: float = 1.0e-4,
    center: bool = True,
) -> torch.Tensor:
    """Estimate full within-group activation covariance/Hessian blocks.

    For a linear layer y = xW, output-reconstruction MSE is weighted by
    H = X^T X. This function returns the g128 diagonal blocks of H.
    """
    x = activations.detach().float().reshape(-1, activations.shape[-1])
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    pad = (-x.shape[-1]) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    xg = x.reshape(x.shape[0], -1, group_size).transpose(0, 1).contiguous()
    cov = torch.einsum("gtr,gts->grs", xg, xg) / max(x.shape[0], 1)
    mean_diag = cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1.0e-12)
    eye = torch.eye(group_size, device=cov.device, dtype=cov.dtype)
    return cov + ridge * mean_diag[:, None, None] * eye


def normalize_log_sensitivity(trace_values: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Hestia-style log-standardized sigmoid sensitivity normalization."""
    h = trace_values.detach().float().clamp_min(1.0e-30).log()
    return torch.sigmoid(gain * (h - h.mean()) / (h.std(unbiased=False) + 1.0e-8))


def hutchinson_trace(
    loss: torch.Tensor,
    parameter: torch.Tensor,
    samples: int = 4,
) -> torch.Tensor:
    """Small reference Hutchinson Hessian-trace estimator.

    This is intentionally simple. Large-model use should replace it with a
    sharded Hutch++ implementation.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    grad = torch.autograd.grad(loss, parameter, create_graph=True, retain_graph=True)[0]
    estimates = []
    for _ in range(samples):
        v = torch.empty_like(parameter).bernoulli_(0.5).mul_(2).sub_(1)
        hv = torch.autograd.grad(
            (grad * v).sum(), parameter, retain_graph=True, create_graph=False
        )[0]
        estimates.append((v * hv).sum())
    return torch.stack(estimates).mean()
