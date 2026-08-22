#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless CLI Runner for PixelArtSmith with Grid Modes, Core Sampling & Semantic Quantization."""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

from ..core.bg_remover import BackgroundRemover
from ..core.grid_detector import GridDetector
from ..core.sprite_isolator import SpriteIsolator, FrameItem
from ..core.palette import PaletteQuantizer, PALETTES, hex_to_rgb
from ..core.cleaner import PixelCleaner
from ..core.packer import SpritePacker
from ..core.posterizer import PixelPosterizer


def parse_cell_size(size_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse 'WxH' or 'N' into (width, height), or None for auto."""
    if not size_str or size_str.lower() in ("auto", "none", "0"):
        return None
    if 'x' in size_str.lower():
        parts = size_str.lower().split('x')
        return int(parts[0]), int(parts[1])
    n = int(size_str)
    return n, n


def process_single_image(
    input_path: Path,
    output_dir: Path,
    pitch: Optional[int] = None,
    cell_size: Optional[Tuple[int, int]] = None,
    grid_mode: str = "auto-fit",
    scale: int = 4,
    palette_name: str = "snapper-13",
    max_colors: int = 13,
    remove_bg: bool = True,
    clean_orphans: bool = True,
    export_frames: bool = False,
    bg_remover: Optional[BackgroundRemover] = None
) -> dict:
    """Execute full True-Grid post-processing pipeline on a sprite sheet with Semantic Palette Quantization."""
    print(f"\n[INFO] Processing: {input_path.name}")
    raw_img = Image.open(input_path).convert("RGBA")

    # 1. Background removal
    if remove_bg:
        print("  [1/5] Removing background with AI matting & defringing...")
        if bg_remover is None:
            bg_remover = BackgroundRemover()
        clean_bg_img = bg_remover.remove_background(raw_img, alpha_threshold=128, defringe=True)
    else:
        print("  [1/5] Skipping AI background removal (using existing alpha)...")
        clean_bg_img = PixelCleaner.cleanup_transparency_halos(raw_img)

    # 2. Pitch Detection & Core Sub-Block Sampling (Zero-Bleed)
    if pitch is None or pitch <= 0:
        detected_pitch = GridDetector.estimate_pixel_pitch(clean_bg_img)
        print(f"  [2/5] Auto-detected pseudo-pixel block pitch: {detected_pitch}px")
    else:
        detected_pitch = pitch
        print(f"  [2/5] Using configured pixel block pitch: {detected_pitch}px")

    margin = 1 if detected_pitch >= 6 else 0
    print(f"        Sampling Core Sub-Block ({raw_img.width}x{raw_img.height} -> {raw_img.width // detected_pitch}x{raw_img.height // detected_pitch}, Margin: {margin}px)...")
    grid_img = GridDetector.core_subblock_downsample(
        clean_bg_img,
        pitch=detected_pitch,
        margin=margin
    )

    # 3. Clean orphan pixels & Semantic Palette Quantization
    if clean_orphans:
        grid_img = PixelCleaner.remove_orphan_pixels(grid_img)

    print(f"  [3/5] Applying Chroma-Weighted Semantic Quantization (Palette: '{palette_name}', Max Colors: {max_colors})...")
    palette_colors: List[str] = []

    if palette_name.startswith("snapper") or palette_name == "default" or palette_name.startswith("adaptive"):
        n_c = int(palette_name.split("-")[1]) if "-" in palette_name else max_colors
        grid_img, palette_colors = PixelPosterizer.process_snapper_pipeline(
            grid_img,
            max_colors=n_c,
            w_chroma=2.0
        )
    elif palette_name in PALETTES:
        hex_list = PALETTES[palette_name]
        # Always include black outline if not present
        if "#000000" not in hex_list and "#000000" not in [h.lower() for h in hex_list]:
            hex_list = ["#000000"] + hex_list
        palette_rgb = np.array([hex_to_rgb(h) for h in hex_list], dtype=np.uint8)
        grid_img, palette_colors = PixelPosterizer.quantize_chroma_weighted(
            grid_img,
            palette_rgb=palette_rgb,
            w_chroma=2.0
        )
    else:
        pass

    # 4. 2D Matrix Slicing (Rows = Motions, Cols = Frames)
    print("  [4/5] Detecting 2D motion matrix (Rows=Motions, Cols=Frames)...")
    isolator = SpriteIsolator(min_area=12, padding=1)
    matrix = isolator.isolate_matrix(grid_img)

    n_rows = len(matrix)
    row_counts = [len(r) for r in matrix]
    total_frames = sum(row_counts)
    print(f"  --> Identified {n_rows} motion row(s) with frames: {row_counts} (Total {total_frames} frames).")

    # Determine standardized cell size using Grid Mode
    if cell_size is not None:
        final_cell_size = cell_size
        print(f"        Using explicit logical cell size: {final_cell_size[0]}x{final_cell_size[1]}px")
    else:
        final_cell_size = SpritePacker.resolve_cell_size(matrix, grid_mode=grid_mode)
        print(f"        Resolved cell size ({grid_mode}): {final_cell_size[0]}x{final_cell_size[1]}px")

    # 5. Pack Matrix Sprite Sheet & Export
    print(f"  [5/5] Assembling Matrix Sprite Sheet (Grid Mode: {grid_mode}, Scale: {scale}x)...")
    packed_sheet, metadata, std_grid = SpritePacker.pack_matrix_sheet(
        matrix=matrix,
        cell_size=final_cell_size,
        scale=scale,
        palette_name=palette_name,
        palette_colors=palette_colors,
        grid_mode=grid_mode
    )

    stem = input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save primary packed sprite sheet (clean 1 file)
    sheet_path = output_dir / f"{stem}_pixel_sheet.png"
    packed_sheet.save(sheet_path)
    print(f"  [SUCCESS] Output Sprite Sheet: {sheet_path}")

    # Save Agentic AI JSON metadata if requested
    json_path = output_dir / f"{stem}_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save individual motion frames only if explicitly requested
    if export_frames:
        frames_dir = output_dir / f"{stem}_frames"
        frames_dir.mkdir(exist_ok=True)
        for r_idx, row in enumerate(std_grid):
            for c_idx, frame_img in enumerate(row):
                frame_path = frames_dir / f"motion_{r_idx:02d}_frame_{c_idx:02d}.png"
                frame_img.save(frame_path)
        print(f"  [INFO] Exported individual frames to: {frames_dir}/")

    return {
        "status": "success",
        "input": str(input_path),
        "sheet": str(sheet_path),
        "rows": n_rows,
        "total_frames": total_frames,
        "cell_size": f"{final_cell_size[0] * scale}x{final_cell_size[1] * scale}",
        "grid_mode": grid_mode
    }


def main_cli(args: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="PixelArtSmith: Convert AI/SD Character Sheets into Authentic Grid-Perfect Pixel Art."
    )
    parser.add_argument("input", type=str, help="Input image file or directory path.")
    parser.add_argument("-o", "--output-dir", type=str, default="./output", help="Output directory path (default: ./output).")
    parser.add_argument("-P", "--pitch", type=int, default=8, help="Pixel block pitch (default: 8 for 32px retro, 4 for 64px RPG, 0 for auto).")
    parser.add_argument("-g", "--grid-mode", type=str, default="auto-fit",
                        choices=["auto-fit", "fixed-32", "fixed-48", "fixed-64"],
                        help="Grid packaging mode: auto-fit (default), fixed-32 (32x32 standard), fixed-48, fixed-64.")
    parser.add_argument("-c", "--cell-size", type=str, default="auto", help="Override logical cell size WxH (default: auto).")
    parser.add_argument("-s", "--scale", type=int, default=4, help="Nearest-neighbor integer upscale factor (default: 4).")
    parser.add_argument("-p", "--palette", type=str, default="snapper-13",
                        choices=[
                            "snapper-13", "snapper-16", "dawnbringer-16", "sweetie-16", "pico-8", "nes-54",
                            "gameboy-classic", "gameboy-pocket", "gameboy-color", "snes-classic",
                            "sega-genesis", "c64-commodore", "endesga-32", "endesga-64", "resurrect-64",
                            "adaptive-8", "adaptive-12", "adaptive-16", "adaptive-24", "adaptive-32", "none"
                        ],
                        help="Palette preset (default: snapper-13 for 13-color clean retro).")
    parser.add_argument("-k", "--max-colors", type=int, default=13, help="Maximum discrete colors per character (default: 13).")
    parser.add_argument("--no-remove-bg", action="store_true", help="Skip AI background removal.")
    parser.add_argument("--no-clean", action="store_true", help="Skip 1-pixel orphan noise cleanup.")
    parser.add_argument("--split-frames", action="store_true", help="Also export individual sliced frame PNG files.")
    parser.add_argument("--model", type=str, default="isnet-general-use", help="Rembg AI model name.")

    parsed = parser.parse_args(args)

    input_target = Path(parsed.input)
    if not input_target.exists():
        print(f"[ERROR] Input target does not exist: {input_target}", file=sys.stderr)
        return 1

    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_size = parse_cell_size(parsed.cell_size)
    bg_remover = BackgroundRemover(model_name=parsed.model) if not parsed.no_remove_bg else None

    # Collect images (ignore sub-outputs or already processed sheets)
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if input_target.is_file():
        files = [input_target]
    else:
        files = [
            p for p in input_target.iterdir()
            if p.is_file() and p.suffix.lower() in image_exts and not p.stem.endswith("_pixel_sheet") and not p.stem.endswith("_true_grid") and not "snapper" in p.stem.lower()
        ]

    if not files:
        print(f"[WARN] No supported image files found in {input_target}", file=sys.stderr)
        return 0

    print("========================================================================")
    print(" 🎨 PixelArtSmith: True-Grid AI Sprite Sheet -> Pixel Art Engine")
    print(f" Pitch: {parsed.pitch}px | Grid Mode: {parsed.grid_mode} | Palette: {parsed.palette} | Max Colors: {parsed.max_colors} | Scale: {parsed.scale}x")
    print(f" Found {len(files)} image(s) to process.")
    print("========================================================================")

    for file_path in files:
        process_single_image(
            input_path=file_path,
            output_dir=output_dir,
            pitch=parsed.pitch,
            cell_size=cell_size,
            grid_mode=parsed.grid_mode,
            scale=parsed.scale,
            palette_name=parsed.palette,
            max_colors=parsed.max_colors,
            remove_bg=not parsed.no_remove_bg,
            clean_orphans=not parsed.no_clean,
            export_frames=parsed.split_frames,
            bg_remover=bg_remover
        )

    print("\n[SUCCESS] All processing completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
