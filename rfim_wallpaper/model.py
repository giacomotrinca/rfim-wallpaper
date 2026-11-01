"""Random Field Ising Model dynamics on the GPU (PyTorch/CUDA).

Hamiltonian
    H = -J Σ_<ij> s_i s_j - Σ_i h_i s_i - H_ext Σ_i s_i,   with h_i ~ N(0, Δ).

Two zero- vs finite-temperature dynamics are provided:

* :func:`avalanche` — T = 0 field ramp. Spins start all-down and the external
  field H_ext is ramped upward; each step is relaxed to the unique ground state
  via synchronous down→up flips. Because the ferromagnetic process is monotone,
  the no-passing (Middleton) theorem guarantees the fixed point is independent
  of update order. The bursts of flips between field steps are the avalanches
  (Barkhausen crackling), and each site records log(1 + avalanche size).

* :func:`thermal` — finite-temperature Metropolis Monte Carlo, vectorized on the
  checkerboard sublattice. Δ = 0 and T = T_c ≈ 2.269 reproduces the clean 2D
  Ising critical point; Δ > 0 pins and roughens the domain walls (Imry–Ma
  disorder in d = 2).
"""
from __future__ import annotations

import math

import torch

T_C = 2.0 / math.log(1.0 + math.sqrt(2.0))  # ≈ 2.269185


def _neighbor_sum(s: torch.Tensor) -> torch.Tensor:
    """Sum of the 4 nearest neighbours (periodic BC) via torch.roll."""
    return (
        torch.roll(s, 1, 0)
        + torch.roll(s, -1, 0)
        + torch.roll(s, 1, 1)
        + torch.roll(s, -1, 1)
    )


def _relax(
    s: torch.Tensor,
    h: torch.Tensor,
    h_ext: float,
    j: float,
    max_iter: int,
) -> tuple[int, torch.Tensor]:
    """Zero-temperature relaxation at fixed external field; mutates ``s`` in place.

    Returns ``(num_flips, cascade_mask)`` for a single avalanche.
    """
    total = 0
    cascade = torch.zeros_like(s, dtype=torch.bool)
    for _ in range(max_iter):
        f = j * _neighbor_sum(s) + h + h_ext
        unstable = (s < 0) & (f > 0)
        if not unstable.any():
            break
        cascade |= unstable
        s[unstable] = 1.0
        total += int(unstable.sum().item())
    return total, cascade


def avalanche(
    shape: tuple[int, int],
    disorder: float,
    target_m: float,
    ramp_steps: int,
    seed: int,
    device: torch.device,
    j: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """T = 0 field ramp; returns the spin configuration at m ≈ ``target_m``.

    Returns:
        s:        spin configuration (float32, {-1, +1}).
        activity: log(1 + avalanche size) per flipped site, 0 elsewhere.
    """
    if disorder <= 0:
        raise ValueError("disorder must be > 0 in avalanche mode")

    H, W = shape

    # Quenched field is generated on CPU (deterministic even on GPU).
    gen = torch.Generator().manual_seed(seed)
    h = torch.randn((H, W), generator=gen, dtype=torch.float32) * disorder
    h = h.to(device)

    s = -torch.ones((H, W), dtype=torch.float32, device=device)
    activity = torch.zeros((H, W), dtype=torch.float32, device=device)

    margin = 4.0 * j + 6.0 * disorder
    h_ext = -margin
    dh = (2.0 * margin) / ramp_steps
    max_iter = H + W  # lattice diameter: safe bound for front propagation

    best_s = s.clone()
    best_err = float("inf")

    for _ in range(ramp_steps):
        h_ext += dh
        n, mask = _relax(s, h, h_ext, j, max_iter)
        if n > 0:
            activity[mask] = math.log1p(n)
        m = float(s.mean().item())
        err = abs(m - target_m)
        if err < best_err:
            best_err = err
            best_s = s.clone()
        if m >= target_m:
            break

    return best_s, activity


def thermal(
    shape: tuple[int, int],
    temperature: float,
    disorder: float,
    field: float,
    sweeps: int,
    seed: int,
    device: torch.device,
    j: float = 1.0,
) -> torch.Tensor:
    """Finite-temperature Metropolis Monte Carlo (checkerboard, GPU).

    Returns:
        s: final spin configuration (float32, {-1, +1}).
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use avalanche mode for T = 0")

    H, W = shape
    gen = torch.Generator(device=device).manual_seed(seed)

    # Quenched field (deterministic CPU generator), then moved to the device.
    cpu_gen = torch.Generator().manual_seed(seed)
    h = torch.randn((H, W), generator=cpu_gen, dtype=torch.float32) * disorder
    h = h.to(device)

    s = (torch.rand((H, W), generator=gen, device=device) < 0.5).float() * 2 - 1

    grid = (
        torch.arange(H, device=device)[:, None]
        + torch.arange(W, device=device)[None, :]
    )
    even = (grid % 2 == 0)
    odd = ~even

    beta = 1.0 / temperature
    for _ in range(sweeps):
        for parity in (even, odd):
            dE = 2.0 * s * (j * _neighbor_sum(s) + h + field)
            r = torch.rand((H, W), generator=gen, device=device)
            accept = (dE <= 0) | (r < torch.exp(-torch.clamp(dE, 0, None) * beta))
            flip = accept & parity
            s[flip] *= -1
    return s
