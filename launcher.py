"""PyInstaller entry point.

A standalone top-level script avoids the relative-import issues that arise when
packaging ``rfim_wallpaper/__main__.py`` directly as the entry point.
"""
from rfim_wallpaper.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
