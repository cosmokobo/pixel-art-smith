#!/usr/bin/env python3
"""Headless CLI Runner for PixelArtSmith with Snapper-Parity Sampling & Semantic Quantization."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..core.cleaner import PixelCleaner
from ..core.grid_detector import GridDetector
from ..core.packer import SpritePacker
from ..core.palette import PALETTES, hex_to_rgb
from ..core.posterizer import PixelPosterizer
from ..core.sprite_isolator import SpriteIsolator


def parse_cell_size(size_str: str | None) -> tuple[int, int] | None:
    """Parse 'WxH' or 'N' into (width, height), or None for auto."""
    if not size_str or size_str.lower() in ("auto", "none", "0"):
        return None
    if "x" in size_str.lower():
        parts = size_str.lower().split("x")
        return int(parts[0]), int(parts[1])
    n = int(size_str)
    return n, n


def process_single_image(
    input_path: Path,
    output_dir: Path,
    pitch: int | None = None,
    cell_size: tuple[int, int] | None = None,
    grid_mode: str = "auto-fit",
    scale: int = 4,
    palette_name: str = "snapper-16",
    max_colors: int = 16,
    remove_bg: bool = True,
    clean_orphans: bool = False,
    export_frames: bool = False,
    expected_rows: int = 4,
    expected_cols: int = 4,
) -> dict:
    """Execute Snapper-Parity True-Grid post-processing pipeline on a sprite sheet."""
    print(f"\n[INFO] Processing: {input_path.name}")
    raw_img = Image.open(input_path).convert("RGB")

    # 1. Pitch Detection & Center-Subblock Downsampling (Zero-Bleed, 100% Full RGB Retention)
    if pitch is None or pitch <= 0:
        detected_pitch = GridDetector.estimate_pixel_pitch(raw_img)
        print(f"  [1/4] Auto-detected pseudo-pixel block pitch: {detected_pitch}px")
    else:
        detected_pitch = pitch
        print(f"  [1/4] Using configured pixel block pitch: {detected_pitch}px")

    grid_img = GridDetector.core_subblock_downsample(raw_img, pitch=detected_pitch)
    print(f"        Sampled Core Grid: {raw_img.width}x{raw_img.height} -> {grid_img.width}x{grid_img.height}")

    grid_arr = np.array(grid_img)
    target_h, target_w = grid_arr.shape[:2]

    # 2. Strict Non-Leaking Spatial Background Segmentation (4-connected floodfill from corners)
    if remove_bg:
        print("  [2/4] Detecting background perimeter without leaking into character interior...")
        mask = np.zeros((target_h + 2, target_w + 2), np.uint8)
        bgr = cv2.cvtColor(grid_arr, cv2.COLOR_RGB2BGR)
        flags = cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 4  # 4-connectivity prevents diagonal leakage
        diff = (5, 5, 5)

        # 4 corners and borders
        cv2.floodFill(bgr.copy(), mask, (0, 0), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (target_w - 1, 0), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (0, target_h - 1), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (target_w - 1, target_h - 1), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (target_w // 2, 0), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (target_w // 2, target_h - 1), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (0, target_h // 2), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (target_w - 1, target_h // 2), 255, diff, diff, flags=flags)

        bg_mask = mask[1 : target_h + 1, 1 : target_w + 1] == 255
        fg_mask = ~bg_mask
    else:
        bg_mask = np.zeros((target_h, target_w), dtype=bool)
        fg_mask = np.ones((target_h, target_w), dtype=bool)

    # 3. Dedicated Foreground Semantic Palette Quantization (Full 16 Colors for Character)
    print(
        f"  [3/4] Applying Chroma-Weighted Semantic Quantization (Palette: '{palette_name}', Max Colors: {max_colors})..."
    )
    palette_colors: list[str] = []

    # Extract & quantize foreground
    fg_img = Image.fromarray(grid_arr[fg_mask].reshape(-1, 1, 3))
    if palette_name.startswith("snapper") or palette_name in ("default", "adaptive", "none"):
        n_c = int(palette_name.split("-")[1]) if "-" in palette_name else max_colors
        quant_fg_img, palette_colors = PixelPosterizer.process_snapper_pipeline(fg_img, max_colors=n_c, w_chroma=2.0)
    elif palette_name in PALETTES:
        hex_list = PALETTES[palette_name]
        if "#000000" not in hex_list and "#000000" not in [h.lower() for h in hex_list]:
            hex_list = ["#000000"] + hex_list
        palette_rgb = np.array([hex_to_rgb(h) for h in hex_list], dtype=np.uint8)
        quant_fg_img, palette_colors = PixelPosterizer.quantize_chroma_weighted(
            fg_img, palette_rgb=palette_rgb, w_chroma=2.0
        )
    else:
        quant_fg_img, palette_colors = PixelPosterizer.process_snapper_pipeline(fg_img, max_colors=max_colors)

    quant_fg_arr = np.array(quant_fg_img).reshape(-1, 3)

    # Assemble final RGBA image
    clean_arr = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    clean_arr[fg_mask, :3] = quant_fg_arr
    clean_arr[fg_mask, 3] = 255
    clean_arr[bg_mask, :3] = grid_arr[bg_mask]
    clean_arr[bg_mask, 3] = 0 if remove_bg else 255

    clean_img = Image.fromarray(clean_arr, "RGBA")

    if clean_orphans:
        clean_img = PixelCleaner.remove_orphan_pixels(clean_img)

    # 4. 2D Matrix Slicing (Rows = Motions, Cols = Frames) & Packing
    print("  [4/4] Segmenting & Assembling Matrix Sprite Sheet...")
    isolator = SpriteIsolator(min_area=12, padding=1)
    matrix = isolator.isolate_matrix(clean_img, expected_rows=expected_rows, expected_cols=expected_cols)

    n_rows = len(matrix)
    row_counts = [len(r) for r in matrix]
    total_frames = sum(row_counts)
    print(f"  --> Identified {n_rows} motion row(s) with frames: {row_counts} (Total {total_frames} frames).")

    if cell_size is not None:
        final_cell_size = cell_size
    else:
        final_cell_size = SpritePacker.resolve_cell_size(matrix, grid_mode=grid_mode)
    print(f"        Resolved cell size ({grid_mode}): {final_cell_size[0]}x{final_cell_size[1]}px")

    packed_sheet, metadata, std_grid = SpritePacker.pack_matrix_sheet(
        matrix=matrix,
        cell_size=final_cell_size,
        scale=scale,
        palette_name=palette_name,
        palette_colors=palette_colors,
        grid_mode=grid_mode,
    )

    stem = input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save primary packed sprite sheet (clean 1 file)
    sheet_path = output_dir / f"{stem}_pixel_sheet.png"
    packed_sheet.save(sheet_path)
    print(f"  [SUCCESS] Output Sprite Sheet: {sheet_path}")

    # Save Agentic AI JSON metadata
    json_path = output_dir / f"{stem}_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

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
        "grid_mode": grid_mode,
    }


def main_cli(args: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="PixelArtSmith: Convert AI/SD Character Sheets into Authentic Grid-Perfect Pixel Art."
    )
    parser.add_argument("input", type=str, help="Input image file or directory path.")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="./output", help="Output directory path (default: ./output)."
    )
    parser.add_argument(
        "-P",
        "--pitch",
        type=int,
        default=8,
        help="Pixel block pitch (default: 8 for 32px retro, 4 for 64px RPG, 0 for auto).",
    )
    parser.add_argument(
        "-g",
        "--grid-mode",
        type=str,
        default="auto-fit",
        help="Grid layout mode: auto-fit (default), fixed-32, fixed-48, fixed-64, preserve-sheet, or WxH.",
    )
    parser.add_argument(
        "-c", "--cell-size", type=str, default=None, help="Explicit cell size 'WxH' or 'N' (overrides --grid-mode)."
    )
    parser.add_argument(
        "-p", "--palette", type=str, default="snapper-16", help="Palette preset name (default: 'snapper-16')."
    )
    parser.add_argument(
        "-k", "--max-colors", type=int, default=16, help="Max discrete colors per character (default: 16)."
    )
    parser.add_argument("-s", "--scale", type=int, default=4, help="Integer upscale factor for display (default: 4).")
    parser.add_argument("--no-bg-remove", action="store_true", help="Keep solid background without transparency.")
    parser.add_argument("--clean-orphans", action="store_true", help="Clean isolated single-pixel noise dots.")
    parser.add_argument("--export-frames", action="store_true", help="Export individual standardized frame PNGs.")

    parsed = parser.parse_args(args)

    input_path = Path(parsed.input)
    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        return 1

    output_dir = Path(parsed.output_dir)
    explicit_cell = parse_cell_size(parsed.cell_size)

    # Collect images
    if input_path.is_file():
        image_files = [input_path]
    else:
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        image_files = sorted(
            [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts and "pixel_sheet" not in p.name]
        )

    if not image_files:
        print(f"[ERROR] No valid images found at: {input_path}", file=sys.stderr)
        return 1

    print("========================================================================")
    print(" 🎨 PixelArtSmith: True-Grid AI Sprite Sheet -> Pixel Art Engine")
    print(
        f" Pitch: {parsed.pitch}px | Grid Mode: {parsed.grid_mode} | Palette: {parsed.palette} | Max Colors: {parsed.max_colors} | Scale: {parsed.scale}x"
    )
    print(f" Found {len(image_files)} image(s) to process.")
    print("========================================================================")

    success_count = 0
    for img_p in image_files:
        try:
            res = process_single_image(
                input_path=img_p,
                output_dir=output_dir,
                pitch=parsed.pitch,
                cell_size=explicit_cell,
                grid_mode=parsed.grid_mode,
                scale=parsed.scale,
                palette_name=parsed.palette,
                max_colors=parsed.max_colors,
                remove_bg=not parsed.no_bg_remove,
                clean_orphans=parsed.clean_orphans,
                export_frames=parsed.export_frames,
            )
            if res.get("status") == "success":
                success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {img_p.name}: {e}", file=sys.stderr)

    print("\n[SUCCESS] All processing completed successfully!")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main_cli())
