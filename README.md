# 🎨 PixelArtSmith (픽셀아트 스미스)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Platform: Cross-Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)]()

> **PixelArtSmith** transforms AI-generated character sprite sheets (Stable Diffusion, Midjourney, ComfyUI, etc.) into **authentic, grid-perfect retro pixel art** with CIELAB palette snapping, AI background matting, bottom-center foot alignment, and standardized game engine sheet/atlas packing.

---

## 🌟 Key Features

1. **AI Background Matting & Hard Alpha Binary Thresholding**:
   - `rembg` (IS-Net / BiRefNet) powered AI foreground extraction.
   - Strict Binary Alpha ($A \in \{0, 255\}$) enforcement to eliminate translucent fuzzy fringes.
   - Automatic edge defringing and background color decontamination.
2. **Grid-Perfect Downsampling (Mixels & Subpixels Elimination)**:
   - Area/Box downsampling to target logical resolutions ($32\times32$, $48\times64$, $64\times64$, etc.).
   - Nearest-Neighbor integer upscale multipliers ($1\times, 2\times, 3\times, 4\times$) for razor-sharp rendering in modern game engines.
3. **Perceptual CIELAB Palette Snapping**:
   - **`endesga-32` (EDG32)**: Iconic 32-color fantasy RPG master palette.
   - **`dawnbringer-32` (DB32)** & **`dawnbringer-16` (DB16)**: Classical retro RPG color ramps.
   - **`pico-8`**: 16-color vivid retro fantasy console palette.
   - **`resurrect-64`**: 64-color extended palette for rich characters and gear.
   - **`gameboy-classic`**: 4-color monochrome nostalgia.
   - **`adaptive-16 / adaptive-24 / adaptive-32`**: K-Means clustering in CIELAB space for character-adaptive color extraction.
   - **`none`**: Maintain original colors while enforcing spatial grid perfection.
4. **Sprite Cell Slicing & Bottom-Center Ground Anchor Alignment**:
   - Automatic connected-component contour detection and spatial sorting of multi-pose character frames.
   - Bottom-center ground line normalization to prevent character jitter during in-game walking/running animations.
5. **Dual Mode Support**:
   - **Modern Desktop GUI**: Side-by-side Before/After preview canvas, interactive sliders, theme support, and one-click export.
   - **Headless CLI**: High-throughput batch processing for CI/CD and automated pipelines.
6. **Game Engine Ready Exports**:
   - Packed horizontal sprite sheet (`.png`).
   - Standard JSON metadata / atlas file (`.json`) compatible with Godot 4 `SpriteFrames` and Unity.
   - Individual frame sequence PNGs.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10, 3.11, or 3.12

### 1. Interactive Desktop GUI
```bash
# macOS / Linux
./run.sh

# Windows (PowerShell)
.\run.ps1
```
*(The launcher automatically sets up a dedicated virtual environment `.venv` and installs dependencies on first run).*

### 2. Headless CLI Processing
```bash
# Process a single sprite sheet with Endesga-32 palette and 48x64 cell size
./run.sh /path/to/character.png -o ./output

# Batch process an entire directory: 32x32 cell, PICO-8 palette, 2x integer upscale
./run.sh /path/to/sprites_dir/ -c 32x32 -p pico-8 -s 2 -o ./output

# Skip background removal if the image already has transparent alpha
./run.sh /path/to/transparent_sheet.png --no-remove-bg -o ./output
```

### CLI Command Options
```
usage: pixel-art-smith [-h] [-o OUTPUT_DIR] [-c CELL_SIZE] [-s SCALE]
                       [-p {endesga-32,dawnbringer-32,dawnbringer-16,pico-8,resurrect-64,gameboy-classic,adaptive-16,adaptive-24,adaptive-32,none}]
                       [--no-remove-bg] [--no-clean] [--no-export-frames]
                       [--model MODEL]
                       input

positional arguments:
  input                 Input image file or directory path.

options:
  -h, --help            Show this help message and exit.
  -o, --output-dir      Output directory path (default: ./output).
  -c, --cell-size       Target logical cell size WxH (default: 48x64).
  -s, --scale           Nearest-neighbor integer upscale factor (default: 1).
  -p, --palette         Palette preset to snap colors to (default: endesga-32).
  --no-remove-bg        Skip AI background removal.
  --no-clean            Skip 1-pixel orphan noise cleanup.
  --no-export-frames    Do not save individual frame PNGs.
  --model MODEL         Rembg AI model name (default: isnet-general-use).
```

---

## 🏗️ Standalone Binary Packaging (PyInstaller)

Build a standalone, single-executable binary that requires no Python installation:

```bash
# macOS / Linux (creates ./dist/pixel-art-smith or build/pixel-art-smith/pixel-art-smith)
./build.sh

# Windows
.\build.ps1
```

---

## 📁 Repository Structure

```
pixel-art-smith/
├── LICENSE                     # GNU General Public License v3.0
├── README.md                   # Documentation & Usage Guide
├── requirements.txt            # Python dependencies
├── main.py                     # Entry point dispatcher (GUI / CLI)
├── run.sh / run.ps1            # Auto-venv launcher scripts
├── build.sh / build.ps1 / build.bat # Standalone PyInstaller builder
└── pixel_art_smith/
    ├── core/
    │   ├── bg_remover.py       # AI matting, binary alpha & defringing
    │   ├── sprite_isolator.py  # Contour frame detection & spatial sorting
    │   ├── grid_detector.py    # Box downsampling & integer upscale
    │   ├── palette.py          # CIELAB color snapping & K-Means quantization
    │   ├── cleaner.py          # Orphan 1px noise & outline cleaner
    │   └── packer.py           # Ground anchor alignment & sheet packing
    ├── cli/
    │   └── runner.py           # Headless batch CLI runner
    └── gui/
        └── app.py              # CustomTkinter / Tkinter desktop application
```

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the [LICENSE](./LICENSE) file for details.
