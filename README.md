# 🎨 PixelArtSmith (픽셀아트 스미스)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Platform: Cross-Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)]()

> **PixelArtSmith** transforms AI-generated character sprite sheets (Stable Diffusion, Midjourney, ComfyUI, etc.) into **authentic, grid-perfect 1px retro pixel art** with CIELAB master palette snapping, AI background matting, 2D Motion Matrix (Rows=Motions, Columns=Frames) preservation, and Agentic AI/Game Engine ready metadata.

---

## 🌟 Key Features & True-Grid Architecture

1. **True-Grid Mode Pooling (100% Mixels Elimination)**:
   - **Auto-Pitch Detection**: Analyzes pseudo-pixel block frequencies ($P = 6, 8, 10, 12, 16\text{px}$) via edge autocorrelation.
   - **Mode (Majority Color) Pooling**: Replaces arithmetic linear blurring with statistical color mode selection per block, keeping solid pixel art colors and 1px crisp outlines.
   - **Integer Scaling**: Downscales full sheets to native logical grids ($1024\times1024 \to 128\times128$) and upscales cleanly with Nearest-Neighbor integer multipliers ($1\times \sim 8\times$).
2. **2D Motion Matrix Preservation (Rows = Motions, Columns = Frames)**:
   - Preserves the character sprite sheet structure where each **Row represents a distinct Motion/Direction** (Walk-Down, Walk-Left, Walk-Right, Walk-Up) and each **Column represents an animation frame**.
   - Standardizes every cell with **Bottom-Center Foot Grounding Alignment** to eliminate in-game vertical jitter.
3. **Extensive Retro Console & Master Palette Library**:
   - **Retro Consoles**:
     - `nes-54` (Nintendo Famicom / NES 54-color hardware palette)
     - `snes-classic` (Super Nintendo 32-color ramp)
     - `sega-genesis` (MegaDrive 64-color palette)
     - `gameboy-classic` (Original DMG-01 4-color olive green)
     - `gameboy-pocket` (4-color true grayscale)
     - `gameboy-color` (GBC 32-color master selection)
     - `pico-8` (16-color fantasy console)
     - `c64-commodore` (Commodore 64 16-color)
   - **Curated Artist Master Palettes**:
     - `endesga-32` (EDG32 fantasy RPG master)
     - `endesga-64` (EDG64 high-detail palette)
     - `dawnbringer-32` (DB32 classic RPG)
     - `dawnbringer-16` (DB16 compact RPG)
     - `resurrect-64` (Rich color ramps)
     - `sweetie-16` (Vibrant pastel)
   - **Adaptive & Dynamic**:
     - `adaptive-8 / 16 / 24 / 32 / 64` (CIELAB K-Means extraction)
     - `none` (True-Grid downsampling only)
4. **Agentic AI & Game Engine Metadata (Schema 2.0)**:
   - Clean, structured JSON with motion row IDs, frame bounding rects, color palette swatches, and cell dimensions.
5. **Interactive Desktop GUI Studio**:
   - Side-by-side Before/After preview.
   - **Live Animation Player** with real-time FPS control ($1\sim16$ FPS) and motion row selector.
   - Dynamic parameter sliders and one-click multi-format export.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10, 3.11, or 3.12

### 1. Interactive Desktop GUI Studio
```bash
# macOS / Linux
./run.sh

# Windows (PowerShell)
.\run.ps1
```

### 2. Headless CLI Processing
```bash
# Auto-detect pitch, snap to Endesga 32, and export 4x scaled Matrix Sheet + JSON
./run.sh /path/to/character.png -p endesga-32 -s 4 -o ./output

# Batch process with NES 54-color palette and custom 8px pitch
./run.sh /path/to/sprites_dir/ -p nes-54 -P 8 -s 2 -o ./output

# Adaptive 24-color K-Means palette on transparent sheet
./run.sh /path/to/transparent_sheet.png -p adaptive-24 --no-remove-bg -o ./output
```

---

## 🏗️ Standalone Binary Packaging (PyInstaller)

```bash
# Build standalone binary -> build/pixel-art-smith/pixel-art-smith
./build.sh      # macOS/Linux
.\build.ps1     # Windows
```

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the [LICENSE](./LICENSE) file for details.
