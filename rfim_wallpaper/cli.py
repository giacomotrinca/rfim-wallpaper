"""Command-line interface for RFIM Wallpaper."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import __version__, colors, model, presets, render

RESOLUTIONS = {
    "hd": (1280, 720),
    "fhd": (1920, 1080),
    "qhd": (2560, 1440),
    "uhd": (3840, 2160),
}
MIN_HD = (1280, 720)

MODES = ("thermal", "avalanche")


def detect_native() -> tuple[int, int]:
    """Resolution of the active monitor (minimum HD); FHD fallback."""
    try:
        raw = subprocess.run(
            ["hyprctl", "monitors", "-j"],
            check=True, capture_output=True, text=True,
        ).stdout
        monitors = json.loads(raw)
        if monitors:
            m = monitors[0]
            return max(int(m["width"]), MIN_HD[0]), max(int(m["height"]), MIN_HD[1])
    except Exception:
        pass
    return RESOLUTIONS["fhd"]


def resolve_resolution(name: str, width: int | None, height: int | None) -> tuple[int, int]:
    if width and height:
        return width, height
    if name != "native":
        return RESOLUTIONS[name]
    return detect_native()


def _swatch(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"\x1b[48;2;{r};{g};{b}m   \x1b[0m"


def _list_colors() -> None:
    print("Color themes (edit rfim_wallpaper/colors.toml):")
    for name, t in colors.load_themes().items():
        print(f"  {name:<10} {t['accent']}  bg {_swatch(t['background'])} acc {_swatch(t['accent'])} hi {_swatch(t['accent_hi'])}")


def _list_presets() -> None:
    print("Simulation presets (edit rfim_wallpaper/presets.toml):")
    for name, p in presets.load_presets().items():
        params = ", ".join(f"{k}={v}" for k, v in p.items())
        print(f"  {name:<10} {params}")


def _save(img: np.ndarray, out: str, W: int, H: int, label: str) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, "RGB").save(out_path)
    print(f"✓ {out_path}  ({W}x{H}) · {label}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="rfim-wallpaper",
        description="RFIM Wallpaper — GPU-accelerated Random Field Ising Model art.",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--preset", type=str, default=None, help="simulation preset from presets.toml")
    ap.add_argument("--list-presets", action="store_true", help="list presets and exit")
    ap.add_argument("--colors", type=str, default="braci", help="color theme(s), comma-separated (e.g. 'blue' or 'blue,lime')")
    ap.add_argument("--mix", choices=colors.MIX_MODES, default="clouds", help="how to blend multiple colors")
    ap.add_argument("--list-colors", action="store_true", help="list color themes and exit")
    ap.add_argument("--resolution", choices=list(RESOLUTIONS) + ["native"], default="native")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--style", choices=["texture", "material"], default="texture")
    ap.add_argument("--mode", choices=MODES, default=None, help="thermal | avalanche")
    ap.add_argument("--disorder", type=float, default=None, help="Δ of the quenched field (0 = clean Ising)")
    ap.add_argument("--temperature", type=float, default=None, help="T in units of J (thermal only)")
    ap.add_argument("--field", type=float, default=None, help="uniform external field (thermal only)")
    ap.add_argument("--sweeps", type=int, default=None, help="Monte Carlo sweeps (thermal only)")
    ap.add_argument("--ramp-steps", type=int, default=None, help="field-ramp steps (avalanche only)")
    ap.add_argument("--target-m", type=float, default=None, help="target magnetization (avalanche only)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cpu", action="store_true", help="force CPU instead of GPU")
    ap.add_argument("--out", type=str, default=str(Path.home() / "rfim-wallpaper.png"))
    args = ap.parse_args(argv)

    if args.list_colors:
        _list_colors()
        return 0
    if args.list_presets:
        _list_presets()
        return 0

    # --- merge: preset (default or explicit) + explicit CLI flags ---
    pname = args.preset or presets.DEFAULT_PRESET
    try:
        p = presets.get_preset(pname)
    except ValueError as exc:
        ap.error(str(exc))

    mode = args.mode or p.get("mode", "thermal")
    if mode not in MODES:
        ap.error(f"unknown mode: {mode}")

    disorder = args.disorder if args.disorder is not None else float(p.get("disorder", 0.0))
    temperature = args.temperature if args.temperature is not None else float(p.get("temperature", 2.0))
    field = args.field if args.field is not None else float(p.get("field", 0.0))
    sweeps = args.sweeps if args.sweeps is not None else int(p.get("sweeps", 2000))
    ramp_steps = args.ramp_steps if args.ramp_steps is not None else int(p.get("ramp_steps", 256))
    target_m = args.target_m if args.target_m is not None else float(p.get("target_m", 0.0))

    color_names = [n.strip() for n in args.colors.split(",") if n.strip()]
    if not color_names:
        ap.error("--colors needs at least one theme name")

    W, H = resolve_resolution(args.resolution, args.width, args.height)

    if args.style == "material":
        try:
            c = colors.theme_colors(color_names[0])
        except ValueError as exc:
            ap.error(str(exc))
        img = render.render_material((W, H), c, args.seed)
        _save(img, args.out, W, H, f"material · {color_names[0]}")
        return 0

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    print(f"Device: {device} · resolution {W}x{H} ({W * H / 1e6:.1f} M spins) · preset '{pname}' · {mode}")

    t0 = time.time()
    if mode == "thermal":
        s = model.thermal((H, W), temperature, disorder, field, sweeps, args.seed, device)
        activity = torch.zeros((H, W), dtype=torch.float32, device=device)
        label = f"thermal T={temperature} Δ={disorder}"
        print(f"Simulation: {time.time() - t0:.2f}s · T = {temperature} · Δ = {disorder} · {sweeps} sweeps")
    else:
        s, activity = model.avalanche((H, W), disorder, target_m, ramp_steps, args.seed, device)
        label = f"avalanche Δ={disorder}"
        print(f"Simulation: {time.time() - t0:.2f}s · Δ = {disorder} · target m = {target_m}")

    print(f"Magnetization <m> = {float(s.mean()):+.4f}")

    try:
        bg, accent, hi = colors.resolve_color_fields(color_names, (H, W), args.mix, args.seed, device)
    except ValueError as exc:
        ap.error(str(exc))

    grain_gen = torch.Generator().manual_seed(args.seed + 999)
    img = render.render_texture(s, activity, bg, accent, hi, grain_gen)
    _save(img, args.out, W, H, f"{label} · {','.join(color_names)}")
    return 0
