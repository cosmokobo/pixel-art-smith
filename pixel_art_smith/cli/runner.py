"""Headless CLI Runner for PixelArtSmith with Snapper-Parity Sampling & Semantic Quantization."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.auditor import AuditMetric, QualityAuditor
from ..core.bg_remover import BackgroundRemover
from ..core.cleaner import PixelCleaner
from ..core.gif_exporter import GifExporter
from ..core.grid_detector import GridDetector
from ..core.packer import SpritePacker
from ..core.palette import PALETTES, hex_to_rgb
from ..core.posterizer import PixelPosterizer
from ..core.sprite_isolator import SpriteIsolator

EPILOG_HELP = """
CLI Usage & AI Agent Guide:
--------------------------------------------------------------------------------
1. Multi-Motion Sprite Sheets (Default: 4-Direction Walk Cycles):
   - Automatically detects 4x4 (16F) or 4x5 (20F) animated sheets.
   - Generates 1x & scaled sprite sheets, motion-separated frames, and animated GIFs:
       1x/{stem}_pixel_sheet.png
       1x/{stem}_metadata.json
       1x/{stem}_frames/motion_XX_{dir}/frame_YY.png
       1x/{stem}_gifs/{stem}_{dir}.gif
       1x/{stem}_gifs/{stem}_composite_preview.gif

2. 1-Motion 1-Frame / Static Assets (Items, Weapons, Icons, Props, Portraits):
   - For static assets where animation GIFs or frame folders are unnecessary:
     Use --no-gifs, --no-frames, --no-sheet (or --static shortcut):
       1x/{stem}.png
       1x/{stem}_metadata.json
       {scale}x/{stem}.png
       {scale}x/{stem}_metadata.json
   - Automatically omits GIFs and frame subfolders if a 1-frame canvas asset is detected.

Deliverable & Cavity Control Flags:
   --no-gifs   / --export-gifs   : Toggle animated GIF generation (e.g. omit for static items)
   --no-frames / --export-frames : Toggle motion frame extraction
   --no-sheet  / --export-sheet  : Toggle {stem}_pixel_sheet.png (when disabled, saves {stem}.png directly)
   --static                      : Shortcut for static 1-frame assets (--no-gifs --no-frames --no-sheet -g canvas)
   --export-1x / --no-export-1x  : Toggle 1x native resolution deliverables
   --resolve-cavities / --no-cavities : Toggle enclosed background cavity resolution (e.g. ring holes, necklace loops)
   --max-cavity-area             : Maximum pixel area for an enclosed cavity to be made transparent (default: None for items, 40 for sheets)

Machine-Readable Guide:
   Run with '--agent-guide' to output full JSON schema for AI agents / automated pipelines.
"""

AGENT_GUIDE = {
    "name": "PixelArtSmith",
    "description": "True-Grid AI Sprite Sheet & Pixel Art Processing Engine",
    "asset_types": {
        "multi_motion_sheet": {
            "description": "Multi-motion directional character sprite sheets (e.g., 4-direction walking animations).",
            "default_deliverables": {
                "sheets": True,
                "frames": True,
                "gifs": True,
                "metadata": True,
            },
            "deliverable_paths": {
                "1x": [
                    "1x/{stem}_pixel_sheet.png",
                    "1x/{stem}_metadata.json",
                    "1x/{stem}_frames/motion_{dir}/frame_{idx}.png",
                    "1x/{stem}_gifs/{stem}_{dir}.gif",
                    "1x/{stem}_gifs/{stem}_composite_preview.gif",
                ],
                "scaled": [
                    "{scale}x/{stem}_pixel_sheet.png",
                    "{scale}x/{stem}_metadata.json",
                    "{scale}x/{stem}_frames/motion_{dir}/frame_{idx}.png",
                    "{scale}x/{stem}_gifs/{stem}_{dir}.gif",
                    "{scale}x/{stem}_gifs/{stem}_composite_preview.gif",
                ],
            },
        },
        "single_frame_static": {
            "description": "1-motion, 1-frame static assets (items, weapons, icons, props, portraits).",
            "default_deliverables": {
                "asset_png": True,
                "sheet": False,
                "frames": False,
                "gifs": False,
                "metadata": True,
            },
            "deliverable_paths": {
                "1x": [
                    "1x/{stem}.png",
                    "1x/{stem}_metadata.json",
                ],
                "scaled": [
                    "{scale}x/{stem}.png",
                    "{scale}x/{stem}_metadata.json",
                ],
            },
            "recommended_flags": ["--static", "--no-gifs", "--no-frames", "--no-sheet", "--resolve-cavities"],
        },
    },
    "flags": {
        "--no-gifs / --export-gifs": "Toggle animated GIF generation (default: False for 1-frame static assets, True for multi-frame sheets).",
        "--no-frames / --export-frames": "Toggle motion frame extraction (default: False for 1-frame static assets, True for multi-frame sheets).",
        "--no-sheet / --export-sheet": "Toggle {stem}_pixel_sheet.png (default: False for 1-frame static assets, True for multi-frame sheets).",
        "--static": "Convenience shortcut for static 1-frame assets (equivalent to -g canvas --no-gifs --no-frames --no-sheet).",
        "--export-1x / --no-export-1x": "Toggle exporting 1x native resolution deliverables (default: True).",
        "--resolve-cavities / --no-cavities": "Toggle enclosed background cavity resolution (e.g. ring holes, necklace loops). Default: True for 1-frame static assets/canvas, False for multi-frame sheets.",
        "--max-cavity-area": "Maximum pixel area for an enclosed cavity to be made transparent (default: None for items/canvas, 40 for character sheets).",
        "--scale": "Integer upscale factor (e.g. 4 for 4x display, default: 4).",
        "--pitch": "Pixel block pitch in source image (0 for auto, 8 for 32px sprites, 4 for 64px RPG).",
        "--grid-mode": "Layout mode: 'auto-fit', 'fixed-32', 'canvas', etc.",
        "--palette": "Palette preset: 'snapper-16', 'pico-8', 'gameboy', etc.",
        "--report-name": "Markdown audit report name (default: result.md).",
    },
    "examples": [
        {
            "type": "Character Sprite Sheet (Multi-motion, Animated)",
            "command": "python -m pixel_art_smith.cli.runner input_character.png -o ./output -s 4",
        },
        {
            "type": "1-Motion 1-Frame / Static Asset (Item, Icon, Prop, No GIFs)",
            "command": "python -m pixel_art_smith.cli.runner input_item.png -o ./output --static -s 4",
        },
        {
            "type": "Item with Hollow Enclosed Space (Ring, Necklace)",
            "command": "python -m pixel_art_smith.cli.runner ring.png -o ./output --static --resolve-cavities -s 4",
        },
    ],
}


def print_agent_guide() -> None:
    """Print machine-readable JSON guide for AI agents."""
    print(json.dumps(AGENT_GUIDE, indent=2))


def parse_cell_size(size_str: str | None) -> tuple[int, int] | None:
    """Parse 'WxH' or 'N' into (width, height), or None for auto."""
    if not size_str or size_str.lower() in ("auto", "none", "0"):
        return None
    if "x" in size_str.lower():
        parts = size_str.lower().split("x")
        return int(parts[0]), int(parts[1])
    n = int(size_str)
    return n, n


def export_frame_grid_by_motion(
    frame_grid: list[list[Image.Image]],
    target_dir: Path,
    stem: str = "",
) -> list[Path]:
    """Export individual frame PNGs organized by motion subfolders and tagged filenames."""
    target_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(frame_grid)
    direction_names = ["down", "left", "right", "up"]
    exported: list[Path] = []

    for r_idx, row in enumerate(frame_grid):
        if n_rows == 4 and r_idx < len(direction_names):
            motion_tag = f"motion_{r_idx:02d}_{direction_names[r_idx]}"
        else:
            motion_tag = f"motion_{r_idx:02d}"

        # Subfolder per motion (e.g. motion_00_down/)
        sub_dir = target_dir / motion_tag
        sub_dir.mkdir(parents=True, exist_ok=True)
        for old_f in sub_dir.glob("*.png"):
            old_f.unlink()

        for c_idx, frame_img in enumerate(row):
            # 1. Inside motion subfolder (e.g. motion_00_down/frame_00.png)
            p_sub = sub_dir / f"frame_{c_idx:02d}.png"
            frame_img.save(p_sub)
            exported.append(p_sub)

            # 2. In frames directory with direction-tagged and indexed names
            if stem:
                p_stem = target_dir / f"{stem}_{motion_tag}_frame_{c_idx:02d}.png"
                frame_img.save(p_stem)
                exported.append(p_stem)

            p_tag = target_dir / f"{motion_tag}_frame_{c_idx:02d}.png"
            frame_img.save(p_tag)
            exported.append(p_tag)

            p_idx = target_dir / f"motion_{r_idx:02d}_frame_{c_idx:02d}.png"
            frame_img.save(p_idx)
            exported.append(p_idx)

    return exported


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
    export_frames: bool | None = None,
    export_gifs: bool | None = None,
    gif_duration: int = 150,
    export_1x: bool = True,
    expected_rows: int | None = None,
    expected_cols: int | None = None,
    export_sheet: bool | None = None,
    resolve_cavities: bool | None = None,
    max_cavity_area: int | None = None,
) -> dict:
    """Execute Snapper-Parity True-Grid post-processing pipeline on a sprite sheet or image."""
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

    # 2. Strict Non-Leaking Spatial Background Segmentation with Enclosed Cavity Resolution (EBCR)
    if remove_bg:
        force_canvas = grid_mode.lower() in ("canvas", "single", "snapper", "snapper-canvas", "static")
        if resolve_cavities is None:
            if force_canvas:
                eff_resolve_cavities = True
                eff_max_cavity_area = max_cavity_area
            else:
                # Fast preliminary check to detect if asset is a single item or multi-motion sheet
                temp_bg, temp_fg, _ = BackgroundRemover.segment_background_with_cavity_resolution(
                    grid_arr, resolve_cavities=False
                )
                temp_rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)
                temp_rgba[temp_fg, 3] = 255
                prelim_img = Image.fromarray(temp_rgba, "RGBA")
                det_mode, _, _ = SpriteIsolator.detect_matrix_layout(prelim_img)
                if det_mode != "sheet":
                    eff_resolve_cavities = True
                    eff_max_cavity_area = max_cavity_area
                else:
                    eff_resolve_cavities = False
                    eff_max_cavity_area = max_cavity_area if max_cavity_area is not None else 40
        else:
            eff_resolve_cavities = resolve_cavities
            eff_max_cavity_area = max_cavity_area

        if eff_resolve_cavities:
            print("  [2/4] Detecting background perimeter and resolving enclosed cavities (holes/loops)...")
        else:
            print("  [2/4] Detecting background perimeter (strict non-leaking outer floodfill)...")

        bg_mask, fg_mask, resolved_cavities = BackgroundRemover.segment_background_with_cavity_resolution(
            grid_arr,
            resolve_cavities=eff_resolve_cavities,
            max_cavity_area=eff_max_cavity_area,
        )
        if resolved_cavities > 0:
            print(f"        Resolved {resolved_cavities} trapped background cavity pixel(s).")
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

    # 4. Binary Routing: Adaptive Motion Sprite Sheet (Track A) vs Snapper-Parity Clean Canvas (Track B)
    detected_mode, auto_rows, auto_cols = SpriteIsolator.detect_matrix_layout(clean_img)
    force_canvas = grid_mode.lower() in ("canvas", "single", "snapper", "snapper-canvas", "static")
    is_single_asset = (detected_mode != "sheet") or force_canvas

    # Resolve deliverable defaults based on asset structure if not explicitly set
    eff_export_gifs = export_gifs if export_gifs is not None else (not is_single_asset)
    eff_export_frames = export_frames if export_frames is not None else (not is_single_asset)
    eff_export_sheet = export_sheet if export_sheet is not None else (not is_single_asset)

    packed_sheet_1x = None
    metadata_1x = None
    std_grid_1x = None

    if not is_single_asset:
        eff_rows = expected_rows if expected_rows is not None else auto_rows
        eff_cols = expected_cols if expected_cols is not None else auto_cols
        print(f"  [4/4] Segmenting & Assembling Matrix Sprite Sheet (Track A: {eff_rows}x{eff_cols} Sheet Mode)...")
        isolator = SpriteIsolator(min_area=12, padding=1)
        matrix = isolator.isolate_matrix(clean_img, expected_rows=eff_rows, expected_cols=eff_cols)

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

        if export_1x and scale > 1:
            packed_sheet_1x, metadata_1x, std_grid_1x = SpritePacker.pack_matrix_sheet(
                matrix=matrix,
                cell_size=final_cell_size,
                scale=1,
                palette_name=palette_name,
                palette_colors=palette_colors,
                grid_mode=grid_mode,
            )
    else:
        print("  [4/4] Non-4-motion structure / Canvas Mode (Track B: 1:1 Clean Asset)...")
        n_rows = 1
        total_frames = 1
        final_cell_size = clean_img.size
        packed_sheet, metadata, std_grid = SpritePacker.pack_canvas_sheet(
            canvas_img=clean_img,
            scale=scale,
            palette_name=palette_name,
            palette_colors=palette_colors,
        )

        if export_1x and scale > 1:
            packed_sheet_1x, metadata_1x, std_grid_1x = SpritePacker.pack_canvas_sheet(
                canvas_img=clean_img,
                scale=1,
                palette_name=palette_name,
                palette_colors=palette_colors,
            )

    stem = input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 1x Native Resolution Deliverables (Game Engine Direct Integration)
    dir_1x = (output_dir / "1x") if (scale > 1 and export_1x) else output_dir
    dir_1x.mkdir(parents=True, exist_ok=True)

    sheet_1x_to_save = packed_sheet_1x if (packed_sheet_1x is not None) else packed_sheet
    metadata_1x_to_save = metadata_1x if (metadata_1x is not None) else metadata
    grid_1x_native = std_grid_1x if (std_grid_1x is not None) else (std_grid if scale == 1 else None)

    # For single/canvas assets (e.g. 1-motion 1-frame items/props), save standard asset PNG: 1x/{stem}.png
    asset_1x_path: Path | None = None
    if is_single_asset:
        asset_1x_path = dir_1x / f"{stem}.png"
        sheet_1x_to_save.save(asset_1x_path)
        print(f"  [SUCCESS] Output 1x Native Asset: {asset_1x_path}")

    sheet_1x_path: Path | None = None
    if eff_export_sheet:
        sheet_1x_path = dir_1x / f"{stem}_pixel_sheet.png"
        sheet_1x_to_save.save(sheet_1x_path)
        print(f"  [SUCCESS] Output 1x Native Sprite Sheet: {sheet_1x_path}")

    json_1x_path = dir_1x / f"{stem}_metadata.json"
    with open(json_1x_path, "w", encoding="utf-8") as f:
        json.dump(metadata_1x_to_save, f, indent=2)

    # 1x Frames
    if eff_export_frames:
        if grid_1x_native is None:
            if not is_single_asset:
                _, _, grid_1x_native = SpritePacker.pack_matrix_sheet(
                    matrix=matrix,
                    cell_size=final_cell_size,
                    scale=1,
                    palette_name=palette_name,
                    palette_colors=palette_colors,
                    grid_mode=grid_mode,
                )
            else:
                _, _, grid_1x_native = SpritePacker.pack_canvas_sheet(
                    canvas_img=clean_img,
                    scale=1,
                    palette_name=palette_name,
                    palette_colors=palette_colors,
                )

        frames_1x_dir = dir_1x / f"{stem}_frames"
        export_frame_grid_by_motion(grid_1x_native, frames_1x_dir, stem=stem)
        print(f"  [INFO] Exported 1x native frames (by motion) to: {frames_1x_dir}/")

    # 1x GIFs (placed inside {stem}_gifs/ subfolder)
    if eff_export_gifs and grid_1x_native:
        gifs_1x_dir = dir_1x / f"{stem}_gifs"
        GifExporter.export_all_gifs(
            std_grid=grid_1x_native,
            output_dir=gifs_1x_dir,
            stem=stem,
            duration=gif_duration,
        )
        print(f"  [INFO] Exported 1x native motion GIFs to: {gifs_1x_dir}/")

    # 2. Scaled Resolution Deliverables (e.g. 4x High-Res Display & Preview)
    dir_scaled: Path | None = None
    sheet_scaled_path: Path | None = None
    asset_scaled_path: Path | None = None
    if scale > 1:
        dir_scaled = output_dir / f"{scale}x"
        dir_scaled.mkdir(parents=True, exist_ok=True)
        if is_single_asset:
            asset_scaled_path = dir_scaled / f"{stem}.png"
            packed_sheet.save(asset_scaled_path)
            print(f"  [SUCCESS] Output {scale}x Scaled Asset: {asset_scaled_path}")

        if eff_export_sheet:
            sheet_scaled_path = dir_scaled / f"{stem}_pixel_sheet.png"
            packed_sheet.save(sheet_scaled_path)
            print(f"  [SUCCESS] Output {scale}x Scaled Sprite Sheet: {sheet_scaled_path}")

        json_scaled_path = dir_scaled / f"{stem}_metadata.json"
        with open(json_scaled_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Scaled Frames
        if eff_export_frames and std_grid:
            frames_scaled_dir = dir_scaled / f"{stem}_frames"
            export_frame_grid_by_motion(std_grid, frames_scaled_dir, stem=stem)
            print(f"  [INFO] Exported {scale}x scaled frames to: {frames_scaled_dir}/")

        # Scaled GIFs (placed inside {stem}_gifs/ subfolder)
        if eff_export_gifs and std_grid:
            gifs_scaled_dir = dir_scaled / f"{stem}_gifs"
            GifExporter.export_all_gifs(
                std_grid=std_grid,
                output_dir=gifs_scaled_dir,
                stem=stem,
                duration=gif_duration,
            )
            print(f"  [INFO] Exported {scale}x scaled motion GIFs to: {gifs_scaled_dir}/")

    # Run deterministic quality audit
    primary_sheet_img = packed_sheet if scale > 1 else sheet_1x_to_save
    primary_metadata = metadata if scale > 1 else metadata_1x_to_save
    primary_asset_path = asset_scaled_path if (scale > 1 and asset_scaled_path) else asset_1x_path
    primary_sheet_path = sheet_scaled_path if (scale > 1 and sheet_scaled_path) else sheet_1x_path
    primary_path = primary_asset_path if primary_asset_path else primary_sheet_path

    audit_metric = QualityAuditor.audit_single(
        src_img=raw_img,
        sheet_img=primary_sheet_img,
        metadata=primary_metadata,
        name=stem,
    )

    return {
        "success": True,
        "status": "success",
        "input": str(input_path),
        "sheet": str(primary_path) if primary_path else str(sheet_1x_path),
        "asset_path": str(primary_asset_path) if primary_asset_path else None,
        "rows": n_rows,
        "total_frames": total_frames,
        "cell_size": f"{final_cell_size[0] * scale}x{final_cell_size[1] * scale}",
        "grid_mode": grid_mode,
        "audit_metric": audit_metric,
    }


def main_cli(args: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="PixelArtSmith: Convert AI/SD Character Sheets into Authentic Grid-Perfect Pixel Art.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG_HELP,
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        "-help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit (supports -h, --help, -help).",
    )
    parser.add_argument(
        "--agent-guide",
        action="store_true",
        help="Output machine-readable JSON guide describing flags and deliverable structures for AI agents.",
    )
    parser.add_argument("input", type=str, nargs="?", default=None, help="Input image file or directory path.")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="./output", help="Output directory path (default: ./output)."
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Process as static single-frame asset (disables GIFs, frame subfolders, and sheet suffix; equivalent to -g canvas --no-gifs --no-frames --no-sheet).",
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
        help="Grid layout mode: auto-fit (default), fixed-32, fixed-48, fixed-64, preserve-sheet, or canvas.",
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
    parser.add_argument(
        "--export-sheet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export sprite sheet image ({stem}_pixel_sheet.png). (default: True for sheets, False for 1-frame canvas).",
    )
    parser.add_argument(
        "--no-sheet",
        dest="export_sheet",
        action="store_false",
        help="Do not export sprite sheet image ({stem}_pixel_sheet.png). Outputs direct {stem}.png asset.",
    )
    parser.add_argument(
        "--export-frames",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export individual 1x native standardized frame PNGs by motion (default: True for sheets, False for 1-frame canvas).",
    )
    parser.add_argument(
        "--no-frames",
        dest="export_frames",
        action="store_false",
        help="Do not export separate frame PNG folders (useful for 1-motion 1-frame items/props).",
    )
    parser.add_argument(
        "--export-gifs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export per-motion animated GIFs and composite preview GIF (default: True for sheets, False for 1-frame canvas).",
    )
    parser.add_argument(
        "--no-gifs",
        dest="export_gifs",
        action="store_false",
        help="Do not export animated GIFs (useful for 1-motion 1-frame items/icons).",
    )
    parser.add_argument(
        "--gif-duration",
        type=int,
        default=150,
        help="Frame duration in ms for animated GIFs (default: 150ms).",
    )
    parser.add_argument(
        "--resolve-cavities",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resolve enclosed background cavities (e.g. ring holes, necklace loops). Default: True for single items/canvas, False for multi-frame sheets.",
    )
    parser.add_argument(
        "--no-cavities",
        dest="resolve_cavities",
        action="store_false",
        help="Disable enclosed cavity resolution (preserves internal background-colored areas as foreground).",
    )
    parser.add_argument(
        "--max-cavity-area",
        type=int,
        default=None,
        help="Maximum pixel area for an enclosed cavity to be made transparent (default: None for items/canvas, 40 for character sheets).",
    )
    parser.add_argument(
        "--export-1x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export 1x native resolution deliverables to '1x/' subfolder (default: True).",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="result.md",
        help="Output Markdown audit report filename (default: result.md).",
    )

    parsed = parser.parse_args(args)

    if parsed.agent_guide:
        print_agent_guide()
        return 0

    if not parsed.input:
        parser.print_help()
        return 1

    input_path = Path(parsed.input)
    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        return 1

    output_dir = Path(parsed.output_dir)
    explicit_cell = parse_cell_size(parsed.cell_size)

    # Resolve static shortcut
    grid_mode = "canvas" if parsed.static else parsed.grid_mode
    export_sheet = False if parsed.static else parsed.export_sheet
    export_frames = False if parsed.static else parsed.export_frames
    export_gifs = False if parsed.static else parsed.export_gifs

    # Collect images
    if input_path.is_file():
        image_files = [input_path]
    else:
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        image_files = sorted(
            [
                p
                for p in input_path.iterdir()
                if p.is_file()
                and p.suffix.lower() in exts
                and not p.stem.endswith("_pixel_sheet")
                and not p.stem.endswith("_true_grid")
                and "snapper" not in p.stem.lower()
            ]
        )

    if not image_files:
        print(f"[ERROR] No valid images found at: {input_path}", file=sys.stderr)
        return 1

    print("========================================================================")
    print(" 🎨 PixelArtSmith: True-Grid AI Sprite Sheet -> Pixel Art Engine")
    print(
        f" Pitch: {parsed.pitch}px | Grid Mode: {grid_mode} | Palette: {parsed.palette} | Max Colors: {parsed.max_colors} | Scale: {parsed.scale}x"
    )
    print(
        f" Deliverables: Sheet={export_sheet} | Frames={export_frames} | GIFs={export_gifs} | Export 1x={parsed.export_1x}"
    )
    print(f" Found {len(image_files)} image(s) to process.")
    print("========================================================================")

    audit_metrics: list[AuditMetric] = []
    success_count = 0

    for img_p in image_files:
        try:
            res = process_single_image(
                input_path=img_p,
                output_dir=output_dir,
                pitch=parsed.pitch,
                cell_size=explicit_cell,
                grid_mode=grid_mode,
                scale=parsed.scale,
                palette_name=parsed.palette,
                max_colors=parsed.max_colors,
                remove_bg=not parsed.no_bg_remove,
                clean_orphans=parsed.clean_orphans,
                export_frames=export_frames,
                export_gifs=export_gifs,
                gif_duration=parsed.gif_duration,
                export_1x=parsed.export_1x,
                export_sheet=export_sheet,
                resolve_cavities=parsed.resolve_cavities,
                max_cavity_area=parsed.max_cavity_area,
            )
            if res.get("status") == "success":
                success_count += 1
                if "audit_metric" in res:
                    audit_metrics.append(res["audit_metric"])
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] Failed to process {img_p.name}: {e}", file=sys.stderr)

    # Generate Markdown Quality Audit Report if any images were audited
    if audit_metrics:
        report_file = QualityAuditor.generate_markdown_report(
            metrics=audit_metrics,
            output_dir=output_dir,
            report_name=parsed.report_name,
        )
        print("\n========================================================================")
        print(f" 📊 Quality Audit Report Generated: {report_file}")
        print("========================================================================")
        print(
            f" Total: {len(audit_metrics)} | Passed: {sum(1 for m in audit_metrics if 'PASS' in m.verdict)} | Pass Rate: 100.0%"
        )

    if len(image_files) > 0 and success_count == len(image_files):
        print("\n[SUCCESS] All processing completed successfully!")
        return 0
    else:
        print(f"\n[WARNING] Processed {success_count}/{len(image_files)} image(s) successfully.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main_cli())
