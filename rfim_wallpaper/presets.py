"""Simulation presets loaded from ``presets.toml``.

Presets bundle a ``mode`` with its parameters, so you can experiment with named
recipes (e.g. ``--preset crackle``) and override any single value on the CLI.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRESETS_FILE = HERE / "presets.toml"

DEFAULT_PRESET = "fumo"


def load_presets() -> dict[str, dict]:
    with PRESETS_FILE.open("rb") as f:
        return tomllib.load(f)


def get_preset(name: str) -> dict:
    presets = load_presets()
    if name not in presets:
        raise ValueError(f"unknown preset: {name!r}. Available: {', '.join(presets)}")
    return dict(presets[name])
