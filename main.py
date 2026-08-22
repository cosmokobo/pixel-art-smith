#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PixelArtSmith - AI Sprite Sheet to Grid-Perfect Pixel Art Engine (GUI + CLI)."""

import sys
from pathlib import Path

# Ensure local package importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pixel_art_smith.cli.runner import main_cli
from pixel_art_smith.gui.app import main_gui


def main():
    # If no arguments or explicitly requesting --gui, launch GUI mode
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == "--gui"):
        try:
            main_gui()
            return 0
        except Exception as e:
            print(f"[WARN] GUI failed to start: {e}. Falling back to CLI help.", file=sys.stderr)
            return main_cli(["--help"])

    # Otherwise execute headless CLI
    return main_cli(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
