"""Rendering.

The ``texture`` style turns a spin configuration into a soft "mottled" ember
texture: domain walls are Gaussian-blurred on the GPU, junctions are highlighted
as hot spots, and (in avalanche mode) the crackling structure adds a subtle
organic layer. The ``material`` style is a static low-frequency aurora backdrop.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def boundary(mask: torch.Tensor) -> torch.Tensor:
    """Per-cell boundary intensity (0..4) of a boolean mask."""
    b = torch.zeros_like(mask, dtype=torch.float32)
    dh = mask[:, 1:] != mask[:, :-1]
    b[:, :-1] += dh.float()
    b[:, 1:] += dh.float()
    dv = mask[1:, :] != mask[:-1, :]
    b[:-1, :] += dv.float()
    b[1:, :] += dv.float()
    return b


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur (replicate padding) on a 2D field."""
    if sigma <= 0:
        return x
    r = max(1, int(round(3 * sigma)))
    ax = torch.arange(-r, r + 1, dtype=torch.float32, device=x.device)
    g = torch.exp(-(ax * ax) / (2 * sigma * sigma))
    g = g / g.sum()
    x4 = x.view(1, 1, x.shape[0], x.shape[1])
    x4 = F.pad(x4, (r, r, r, r), mode="replicate")
    x4 = F.conv2d(x4, g.view(1, 1, 1, -1))  # horizontal
    x4 = F.conv2d(x4, g.view(1, 1, -1, 1))  # vertical
    return x4.view(x.shape[0], x.shape[1])


def _norm(t: torch.Tensor) -> torch.Tensor:
    m = t.max()
    return t / m if m > 0 else t


def render_texture(
    s: torch.Tensor,
    activity: torch.Tensor,
    bg: torch.Tensor,
    accent: torch.Tensor,
    hi: torch.Tensor,
    grain_gen: torch.Generator,
) -> np.ndarray:
    """Mottled texture from a spin configuration.

    ``bg``, ``accent``, ``hi`` are per-pixel (H, W, 3) color fields in [0, 1]
    (see :mod:`rfim_wallpaper.colors`); pass constant fields for a solid theme.
    """
    H, W = s.shape
    device = s.device

    walls = boundary(s > 0)
    junction = (walls >= 2).float()

    fil = _norm(gaussian_blur(walls, 1.4)) ** 1.1   # main line
    glow = _norm(gaussian_blur(walls, 6.0))         # soft halo
    hot = _norm(gaussian_blur(junction, 2.0))       # junctions / corners
    mottled = _norm(gaussian_blur(activity, 9.0))   # crackling structure

    # warm vertical base gradient (brighter at the top)
    yy = torch.linspace(1.12, 0.84, H, dtype=torch.float32, device=device)
    base = bg * yy.view(H, 1, 1)

    img = base.clone()
    t = torch.clamp(fil + 0.28 * glow + 0.30 * mottled, 0, 1)
    img += (accent - bg) * t.unsqueeze(-1)
    img += (hi - accent) * torch.clamp(hot, 0, 1).unsqueeze(-1) * 0.55

    # radial vignette
    xx = torch.linspace(-1, 1, W, dtype=torch.float32, device=device).view(1, W)
    yy2 = torch.linspace(-1, 1, H, dtype=torch.float32, device=device).view(H, 1)
    r = torch.sqrt(xx * xx + yy2 * yy2)
    vig = 1.0 - 0.32 * torch.clamp(r - 0.35, 0, 1) ** 2
    img *= vig.unsqueeze(-1)

    # subtle grain (deterministic CPU generator)
    grain = torch.randn((H, W, 1), generator=grain_gen, dtype=torch.float32).to(device)
    img += grain * 0.008

    img = torch.clamp(img, 0, 1)
    return (img * 255).round().to(torch.uint8).cpu().numpy()


def render_material(
    size: tuple[int, int], colors: dict[str, tuple[int, int, int]], seed: int
) -> np.ndarray:
    """Static "aurora blobs" backdrop (low frequency, readable behind panels)."""
    W, H = size
    rng = np.random.default_rng(seed)

    top = np.array(colors["background"], np.float32) / 255.0
    bot = np.array(colors["surface_dim"], np.float32) / 255.0
    yy = np.linspace(0, 1, H, dtype=np.float32)
    img = top[None, None, :] + (bot[None, None, :] - top[None, None, :]) * yy[:, None, None]
    img = np.broadcast_to(img, (H, W, 3)).copy()

    blobs = [
        ("accent", 0.16, (0.80, 0.22), 0.30),
        ("accent_dim", 0.20, (0.14, 0.32), 0.42),
        ("accent_hi", 0.10, (0.70, 0.78), 0.36),
        ("accent", 0.08, (0.42, 0.50), 0.60),
    ]
    gw, gh = 256, 144
    gx = np.linspace(-1, 1, gw, dtype=np.float32)[None, :]
    gy = np.linspace(-1, 1, gh, dtype=np.float32)[:, None]

    for name, alpha, (cx, cy), sigma in blobs:
        c = np.array(colors[name], np.float32) / 255.0
        x0, y0 = cx * 2 - 1, cy * 2 - 1
        g = np.exp(-((gx - x0) ** 2) / (2 * sigma ** 2)) * np.exp(-((gy - y0) ** 2) / (2 * sigma ** 2))
        g = g / (g.max() + 1e-6)
        mask = np.asarray(
            Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8), "L").resize((W, H), Image.BILINEAR),
            np.float32,
        ) / 255.0
        img += c[None, None, :] * (mask[..., None] * alpha)

    xx = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    yy2 = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    r = np.sqrt(xx * xx + yy2 * yy2)
    vig = 1.0 - 0.22 * np.clip(r - 0.5, 0, 1) ** 2
    img *= vig[..., None]
    img += rng.normal(0, 1, (H, W, 1)).astype(np.float32) * 0.003

    img = np.clip(img, 0, 1)
    return (img * 255).round().astype(np.uint8)
