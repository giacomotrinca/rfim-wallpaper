"""Color themes and spatial color blending.

Themes are declared in ``colors.toml`` (edit it to add your own). The ``--colors``
CLI flag accepts a single theme name or a comma-separated list; multiple names
are blended across the image by a smooth spatial field ("zones" of each color).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
COLORS_FILE = HERE / "colors.toml"

MIX_MODES = ("clouds", "gradient", "radial", "angular", "zones")
ROLES = ("background", "accent", "accent_hi")


def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def load_themes() -> dict[str, dict[str, tuple[int, int, int]]]:
    data = tomllib.loads(COLORS_FILE.read_text())
    return {
        name: {k: _rgb(v) for k, v in theme.items()}
        for name, theme in data.items()
    }


def theme_colors(name: str) -> dict[str, tuple[int, int, int]]:
    """Full mini-palette of a single theme (for the material style)."""
    themes = load_themes()
    if name not in themes:
        raise ValueError(f"unknown color theme: {name!r}")
    return themes[name]


def _mix_field(shape: tuple[int, int], mode: str, seed: int, device: torch.device) -> torch.Tensor:
    """Spatial scalar field t(x,y) in [0, 1] that drives the color blend."""
    H, W = shape
    yy = torch.linspace(0.0, 1.0, H, device=device)
    xx = torch.linspace(0.0, 1.0, W, device=device)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")

    if mode == "gradient":
        t = (X + Y) / 2.0
    elif mode == "radial":
        t = torch.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) * 1.41421356
    elif mode == "angular":
        t = torch.atan2(Y - 0.5, X - 0.5) / (2.0 * torch.pi) + 0.5
    elif mode == "zones":
        n = 8
        g = torch.Generator().manual_seed(seed)
        grid = torch.rand((n, n), generator=g).to(device)
        t = F.interpolate(grid.view(1, 1, n, n), size=(H, W), mode="nearest").view(H, W)
    else:  # "clouds": smooth multi-octave organic blobs (default)
        t = torch.zeros((H, W), device=device)
        for i, (gy, gx) in enumerate(((6, 6), (14, 14), (28, 28))):
            g = torch.Generator().manual_seed(seed + i)
            low = torch.rand((gy, gx), generator=g).to(device)
            t += F.interpolate(
                low.view(1, 1, gy, gx), size=(H, W), mode="bilinear", align_corners=False
            ).view(H, W)
        t /= 3.0
    return torch.clamp(t.to(device), 0.0, 1.0)


def resolve_color_fields(
    names: list[str],
    shape: tuple[int, int],
    mix_mode: str,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-pixel color fields (background, accent, accent_hi), each (H, W, 3) in [0, 1].

    A single name yields constant fields; N names are interpolated along the
    spatial field, with ``zones`` snapping to the nearest theme instead of
    blending.
    """
    themes = load_themes()
    for n in names:
        if n not in themes:
            raise ValueError(f"unknown color theme: {n!r}. Available: {', '.join(themes)}")

    H, W = shape
    t = _mix_field(shape, mix_mode, seed, device)
    N = len(names)
    smooth = mix_mode != "zones" and N > 1

    p = t * (N - 1) if N > 1 else torch.zeros_like(t)
    if smooth:
        k = p.floor().long().clamp(0, N - 2)
        frac = (p - k.float()).clamp(0.0, 1.0)
    else:
        k = p.round().long().clamp(0, N - 1)
        frac = torch.zeros_like(p)

    idx = k.reshape(-1)

    def role_field(role: str) -> torch.Tensor:
        pal = torch.tensor(
            [themes[n][role] for n in names], dtype=torch.float32, device=device
        ) / 255.0  # (N, 3)
        left = pal[idx]  # (H*W, 3)
        right = pal[(idx + 1).clamp(max=N - 1)]
        f = left + (right - left) * frac.reshape(-1, 1)
        return f.reshape(H, W, 3)

    return role_field("background"), role_field("accent"), role_field("accent_hi")
