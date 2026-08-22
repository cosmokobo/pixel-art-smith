#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless CLI Runner for PixelArtSmith."""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image

from ..core.bg_remover import BackgroundRemover
from ..core.grid_detector import GridDetector
from ..core.sprite_isolator import SpriteIsolator
from ..core.palette import PaletteQuantizer, PALETTES
from ..core.cleaner import PixelCleaner
from ..core.packer import SpritePacker


def parse_cell_size(size_str: str) -> Tuple[int, int]:
    """Parse 'WxH' or 'N' into (width, height)."""
    if 'x' in size_str.lower():
        parts = size_str.lower().split('x')
        return int(parts[0]), int(parts[1])
    n = int(size_str)
    return n, n


def process_single_image(
    input_path: Path,
    output_dir: Path,
    cell_size: Tuple[int, int] = (48, 64),
    scale: int = 1,
    palette_name: str = "endesga-32",
    remove_bg: bool = True,
    clean_orphans: bool = True,
    export_frames: bool = True,
    bg_remover: Optional[BackgroundRemover] = None
) -> dict:
    """Execute full 6-stage post-processing pipeline on a single sprite sheet image."""
    print(f"\n[INFO] Processing: {input_path.name}")
    raw_img = Image.open(input_path).convert("RGBA")

    # 1. Background removal
    if remove_bg:
        print("  [1/5] Removing background with AI matting...")
        if bg_remover is None:
            bg_remover = BackgroundRemover()
        clean_bg_img = bg_remover.remove_background(raw_img, alpha_threshold=128, defringe=True)
    else:
        print("  [1/5] Skipping background removal (using raw alpha)...")
        clean_bg_img = PixelCleaner.cleanup_transparency_halos(raw_img)

    # 2. Isolate character frames
    print("  [2/5] Detecting character poses and isolating frames...")
    isolator = SpriteIsolator(min_area=300, padding=2)
    detected_frames = isolator.isolate_frames(clean_bg_img)
    print(f"  --> Found {len(detected_frames)} character pose(s).")

    # 3. Grid-Perfect Downsampling & Palette Quantization per frame
    print(f"  [3/5] Applying Grid-Perfect downsampling to {cell_size[0]}x{cell_size[1]} and palette '{palette_name}'...")
    
    # Initialize Palette Quantizer
    quantizer = None
    if palette_name.startswith("adaptive-"):
        n_colors = int(palette_name.split("-")[1])
        adaptive_colors = PaletteQuantizer.extract_adaptive_palette(clean_bg_img, n_colors=n_colors)
        quantizer = PaletteQuantizer(custom_colors=adaptive_colors)
    elif palette_name in PALETTES:
        quantizer = PaletteQuantizer(palette_name=palette_name)

    processed_frames: List[Image.Image] = []
    for i, (frame_raw, bbox) in enumerate(detected_frames):
        # Determine target frame bounding size while maintaining aspect ratio
        fw, fh = frame_raw.size
        ratio = min(cell_size[0] / max(1, fw), cell_size[1] / max(1, fh))
        logical_w = max(1, int(fw * ratio))
        logical_h = max(1, int(fh * ratio))

        # Downsample with Box filter (Box/Area eliminates mixels)
        grid_frame = GridDetector.downsample_to_grid(frame_raw, (logical_w, logical_h))

        # Clean orphan pixels
        if clean_orphans:
            grid_frame = PixelCleaner.remove_orphan_pixels(grid_frame)

        # Apply palette snapping
        if quantizer is not None:
            grid_frame = quantizer.quantize(grid_frame)

        # Standardize frame canvas (bottom-center anchor)
        std_frame = SpritePacker.standardize_frame(grid_frame, cell_size)

        # Upscale if requested
        if scale > 1:
            std_frame = GridDetector.upscale_nearest(std_frame, scale=scale)

        processed_frames.append(std_frame)

    # 4. Pack into unified sprite sheet
    print("  [4/5] Standardizing bottom-center anchors and packing sprite sheet...")
    scaled_cell = (cell_size[0] * scale, cell_size[1] * scale)
    packed_sheet, metadata = SpritePacker.pack_horizontal_sheet(processed_frames, scaled_cell)

    # 5. Export results
    print("  [5/5] Exporting output files...")
    stem = input_path.stem
    char_out_dir = output_dir / stem
    char_out_dir.mkdir(parents=True, exist_ok=True)

    # Save packed sheet
    sheet_path = char_out_dir / f"{stem}_pixel_sheet.png"
    packed_sheet.save(sheet_path)
    print(f"  [SUCCESS] Saved Packed Sheet: {sheet_path}")

    # Save JSON metadata
    json_path = char_out_dir / f"{stem}_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save individual frames if requested
    if export_frames:
        frames_dir = char_out_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(processed_frames):
            frame_path = frames_dir / f"frame_{i:02d}.png"
            frame.save(frame_path)
        print(f"  [SUCCESS] Exported {len(processed_frames)} individual frames to {frames_dir}/")

    return {
        "status": "success",
        "input": str(input_path),
        "sheet": str(sheet_path),
        "frame_count": len(processed_frames),
        "cell_size": f"{scaled_cell[0]}x{scaled_cell[1]}"
    }


def main_cli(args: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="PixelArtSmith: Convert SD AI Character Sheets into Authentic Grid-Perfect Pixel Art."
    )
    parser.add_argument("input", type=str, help="Input image file or directory path.")
    parser.add_argument("-o", "--output-dir", type=str, default="./output", help="Output directory path (default: ./output).")
    parser.add_argument("-c", "--cell-size", type=str, default="48x64", help="Target logical cell size WxH (default: 48x64).")
    parser.add_argument("-s", "--scale", type=int, default=1, help="Nearest-neighbor integer upscale factor (default: 1).")
    parser.add_argument("-p", "--palette", type=str, default="endesga-32",
                        choices=list(PALETTES.keys()) + ["adaptive-16", "adaptive-24", "adaptive-32", "none"],
                        help="Palette preset to snap colors to (default: endesga-32).")
    parser.add_argument("--no-remove-bg", action="store_true", help="Skip AI background removal.")
    parser.add_argument("--no-clean", action="store_true", help="Skip 1-pixel orphan noise cleanup.")
    parser.add_argument("--no-export-frames", action="store_true", help="Do not save individual frame PNGs.")
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

    # Collect images
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if input_target.is_file():
        files = [input_target]
    else:
        files = [p for p in input_target.iterdir() if p.is_file() and p.suffix.lower() in image_exts]

    if not files:
        print(f"[WARN] No supported image files found in {input_target}", file=sys.stderr)
        return 0

    print("========================================================================")
    print(" PixelArtSmith: SD Sprite Sheet -> Grid-Perfect Pixel Art Engine")
    print(f" Target cell: {cell_size[0]}x{cell_size[1]} | Scale: {parsed.scale}x | Palette: {parsed.palette}")
    print(f" Found {len(files)} image(s) to process.")
    print("========================================================================")

    for file_path in files:
        process_single_image(
            input_path=file_path,
            output_dir=output_dir,
            cell_size=cell_size,
            scale=parsed.scale,
            palette_name=parsed.palette,
            remove_bg=not parsed.no_remove_bg,
            clean_orphans=not parsed.no_clean,
            export_frames=not parsed.no_export_frames,
            bg_remover=bg_remover
        )

    print("\n[SUCCESS] All processing completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
