"""Filesystem helpers that work both from source and from a frozen binary."""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent


def data_dir() -> Path:
    """Directory containing the package data files (``colors.toml``, ``presets.toml``).

    Under a PyInstaller ``onefile`` build, ``sys._MEIPASS`` points at the temporary
    extraction directory (where the bundled ``.toml`` files live); otherwise the
    package directory on disk is used.
    """
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        return Path(frozen) / "rfim_wallpaper"
    return _THIS
