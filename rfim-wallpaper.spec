# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build configuration for rfim-wallpaper.

Produces a single-file executable (``onefile``). Build with::

    pyinstaller rfim-wallpaper.spec

For a CUDA build, install the matching CUDA PyTorch wheel first; the resulting
binary bundles the CUDA runtime libraries, but the target machine still needs
the NVIDIA driver (``libcuda.so.1``).
"""

from PyInstaller.utils.hooks import collect_all

datas = [
    ("rfim_wallpaper/colors.toml", "rfim_wallpaper"),
    ("rfim_wallpaper/presets.toml", "rfim_wallpaper"),
]
binaries = []
hiddenimports = []

# torch loads its C extensions and CUDA/CPU shared libraries dynamically, so
# gather submodules, dynamic libraries and data files explicitly.
torch_datas, torch_binaries, torch_hidden = collect_all("torch")
datas += torch_datas
binaries += torch_binaries
hiddenimports += torch_hidden

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="rfim-wallpaper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
