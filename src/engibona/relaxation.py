from __future__ import annotations

import math

import torch

from .config import QuantMode


def categorical_relaxation(
    normalized_weight: torch.Tensor,
    mode: QuantMode | str,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    """Hestia-style differentiable categorical quantization.

    Returns the expectation over {-1,+1} or {-1,0,+1} under a Softmax of
    negative squared distance.
    """
    mode = QuantMode(mode)
    states = (
        torch.tensor([-1.0, 1.0], device=normalized_weight.device, dtype=normalized_weight.dtype)
        if mode == QuantMode.BINARY
        else torch.tensor([-1.0, 0.0, 1.0], device=normalized_weight.device, dtype=normalized_weight.dtype)
    )
    tau = torch.as_tensor(temperature, device=normalized_weight.device, dtype=normalized_weight.dtype)
    tau = tau.clamp_min(torch.finfo(normalized_weight.dtype).eps)
    while tau.ndim < normalized_weight.ndim:
        tau = tau.unsqueeze(-1)
    distance = (normalized_weight.unsqueeze(-1) - states).square()
    probabilities = torch.softmax(-distance / tau.unsqueeze(-1), dim=-1)
    return (probabilities * states).sum(dim=-1)


def catq_ternary_relaxation(
    normalized_weight: torch.Tensor,
    sharpness: float | torch.Tensor,
    threshold: float | torch.Tensor = 0.5,
) -> torch.Tensor:
    """CAT-Q smooth ternary transition.

    f(w;s,delta) = [tanh(s(w-delta)) + tanh(s(w+delta))] / [2 tanh(s)]
    """
    s = torch.as_tensor(sharpness, device=normalized_weight.device, dtype=normalized_weight.dtype)
    d = torch.as_tensor(threshold, device=normalized_weight.device, dtype=normalized_weight.dtype)
    s = s.clamp_min(torch.finfo(normalized_weight.dtype).eps)
    return (
        torch.tanh(s * (normalized_weight - d))
        + torch.tanh(s * (normalized_weight + d))
    ) / (2.0 * torch.tanh(s))


def hard_codes(normalized_weight: torch.Tensor, mode: QuantMode | str, threshold: float = 0.5) -> torch.Tensor:
    mode = QuantMode(mode)
    if mode == QuantMode.BINARY:
        return torch.where(normalized_weight >= 0, 1.0, -1.0)
    return torch.where(
        normalized_weight > threshold,
        1.0,
        torch.where(normalized_weight < -threshold, -1.0, 0.0),
    )


def compression_pressure(step: int, total_steps: int, compression_fraction: float) -> float:
    """Hestia-style convex transfer from dense to quantized responsibility."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if compression_fraction <= 0:
        return 1.0
    return min(1.0, step / max(compression_fraction * total_steps, 1.0))


def hessian_guided_temperature(
    step: int,
    total_steps: int,
    initial_temperature: float,
    sensitivity: torch.Tensor | float,
    alpha: float = 1.5,
    compression_fraction: float = 0.30,
) -> torch.Tensor:
    """Hestia-style tensor-specific cosine temperature schedule."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    compression_end = int(total_steps * compression_fraction)
    if step <= compression_end:
        base = initial_temperature
    else:
        progress = (step - compression_end) / max(total_steps - compression_end, 1)
        base = 0.5 * initial_temperature * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    s = torch.as_tensor(sensitivity, dtype=torch.float32)
    return torch.as_tensor(base, dtype=torch.float32) * torch.exp(alpha * s)


def effective_weight(
    latent_weight: torch.Tensor,
    quantized_surrogate: torch.Tensor,
    pressure: float,
) -> torch.Tensor:
    p = min(max(float(pressure), 0.0), 1.0)
    return (1.0 - p) * latent_weight + p * quantized_surrogate
