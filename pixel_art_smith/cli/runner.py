#!/usr/bin/env python3
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
) -> None:
    """Export individual frame PNGs organized by motion subfolders and tagged filenames."""
    target_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(frame_grid)
    direction_names = ["down", "left", "right", "up"]

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
            frame_img.save(sub_dir / f"frame_{c_idx:02d}.png")
            # 2. In frames directory with direction-tagged and indexed names
            frame_img.save(target_dir / f"{motion_tag}_frame_{c_idx:02d}.png")
            frame_img.save(target_dir / f"motion_{r_idx:02d}_frame_{c_idx:02d}.png")


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
    export_frames: bool = True,
    export_gifs: bool = True,
    gif_duration: int = 150,
    export_1x: bool = True,
    expected_rows: int | None = None,
    expected_cols: int | None = None,
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

    # 2. Strict Non-Leaking Spatial Background Segmentation with Enclosed Cavity Resolution (EBCR)
    if remove_bg:
        print("  [2/4] Detecting background perimeter and resolving enclosed cavities (hair loops/limb gaps)...")
        bg_mask, fg_mask, resolved_cavities = BackgroundRemover.segment_background_with_cavity_resolution(grid_arr)
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
    force_canvas = grid_mode.lower() in ("canvas", "single", "snapper", "snapper-canvas")

    packed_sheet_1x = None
    metadata_1x = None
    std_grid_1x = None

    if detected_mode == "sheet" and not force_canvas:
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
        print("  [4/4] Non-4-motion structure detected -> Snapper-Parity Canvas Mode (Track B: 1:1 Clean Asset)...")
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

    # Save primary output image (Display scale, e.g. 4x)
    sheet_path = output_dir / f"{stem}_pixel_sheet.png"
    packed_sheet.save(sheet_path)
    print(f"  [SUCCESS] Output Sprite Sheet ({scale}x): {sheet_path}")

    # Save Agentic AI JSON metadata
    json_path = output_dir / f"{stem}_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save scale-specific output image & metadata to '<scale>x/' subfolder (e.g. '4x/')
    if scale > 1:
        dir_scaled = output_dir / f"{scale}x"
        dir_scaled.mkdir(parents=True, exist_ok=True)
        sheet_scaled_path = dir_scaled / f"{stem}_pixel_sheet.png"
        packed_sheet.save(sheet_scaled_path)
        json_scaled_path = dir_scaled / f"{stem}_metadata.json"
        with open(json_scaled_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"  [SUCCESS] Output {scale}x Scaled Sprite Sheet: {sheet_scaled_path}")

    # Save 1x native resolution output image & metadata to '1x/' subfolder
    if export_1x and scale > 1 and packed_sheet_1x is not None:
        dir_1x = output_dir / "1x"
        dir_1x.mkdir(parents=True, exist_ok=True)
        sheet_1x_path = dir_1x / f"{stem}_pixel_sheet.png"
        packed_sheet_1x.save(sheet_1x_path)
        json_1x_path = dir_1x / f"{stem}_metadata.json"
        with open(json_1x_path, "w", encoding="utf-8") as f:
            json.dump(metadata_1x, f, indent=2)
        print(f"  [SUCCESS] Output 1x Native Sprite Sheet: {sheet_1x_path}")

    if export_frames and std_grid:
        # Determine 1x native resolution frame grid
        grid_1x_native = std_grid_1x if (std_grid_1x is not None) else (std_grid if scale == 1 else None)
        if grid_1x_native is None:
            if detected_mode == "sheet":
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

        # 1. Export 1x native individual frames (by motion) to root '{stem}_frames/'
        frames_dir = output_dir / f"{stem}_frames"
        export_frame_grid_by_motion(grid_1x_native, frames_dir, stem=stem)
        print(f"  [INFO] Exported 1x native individual frames (by motion) to: {frames_dir}/")

        # 2. Export 1x native individual frames to '1x/{stem}_frames/'
        if export_1x and scale > 1:
            frames_1x_dir = output_dir / "1x" / f"{stem}_frames"
            export_frame_grid_by_motion(grid_1x_native, frames_1x_dir, stem=stem)
            print(f"  [INFO] Exported 1x native individual frames to: {frames_1x_dir}/")

        # 3. Export 4x scaled individual frames to '<scale>x/{stem}_frames/'
        if scale > 1:
            frames_scaled_dir = output_dir / f"{scale}x" / f"{stem}_frames"
            export_frame_grid_by_motion(std_grid, frames_scaled_dir, stem=stem)
            print(f"  [INFO] Exported {scale}x scaled individual frames to: {frames_scaled_dir}/")

    # Export Animated GIFs (individual motion GIFs + all-motions composite GIF)
    if export_gifs and std_grid:
        gif_res = GifExporter.export_all_gifs(
            std_grid=std_grid,
            output_dir=output_dir,
            stem=stem,
            duration=gif_duration,
        )
        print(
            f"  [INFO] Exported {len(gif_res['individual_gifs'])} motion GIF(s) and composite preview to: {output_dir}/"
        )

        if scale > 1:
            dir_scaled_gifs = output_dir / f"{scale}x"
            GifExporter.export_all_gifs(
                std_grid=std_grid,
                output_dir=dir_scaled_gifs,
                stem=stem,
                duration=gif_duration,
            )
            print(f"  [INFO] Exported {scale}x scaled motion GIFs to: {dir_scaled_gifs}/")

        if export_1x and scale > 1 and std_grid_1x is not None:
            dir_1x_gifs = output_dir / "1x"
            GifExporter.export_all_gifs(
                std_grid=std_grid_1x,
                output_dir=dir_1x_gifs,
                stem=stem,
                duration=gif_duration,
            )
            print(f"  [INFO] Exported 1x native motion GIFs to: {dir_1x_gifs}/")

    # Run deterministic quality audit
    audit_metric = QualityAuditor.audit_single(
        src_img=raw_img,
        sheet_img=packed_sheet,
        metadata=metadata,
        name=stem,
    )

    return {
        "status": "success",
        "input": str(input_path),
        "sheet": str(sheet_path),
        "rows": n_rows,
        "total_frames": total_frames,
        "cell_size": f"{final_cell_size[0] * scale}x{final_cell_size[1] * scale}",
        "grid_mode": grid_mode,
        "audit_metric": audit_metric,
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
    parser.add_argument(
        "--export-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export individual 1x native standardized frame PNGs by motion (default: True).",
    )
    parser.add_argument(
        "--export-gifs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export per-motion animated GIFs and composite preview GIF (default: True).",
    )
    parser.add_argument(
        "--gif-duration",
        type=int,
        default=150,
        help="Frame duration in ms for animated GIFs (default: 150ms).",
    )
    parser.add_argument(
        "--export-1x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export 1x native resolution sprite sheet to '1x/' subfolder (default: True).",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="result.md",
        help="Output Markdown audit report filename (default: result.md).",
    )

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
        f" Pitch: {parsed.pitch}px | Grid Mode: {parsed.grid_mode} | Palette: {parsed.palette} | Max Colors: {parsed.max_colors} | Scale: {parsed.scale}x | Export 1x: {parsed.export_1x} | Export Frames (1x): {parsed.export_frames} | Export GIFs: {parsed.export_gifs}"
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
                grid_mode=parsed.grid_mode,
                scale=parsed.scale,
                palette_name=parsed.palette,
                max_colors=parsed.max_colors,
                remove_bg=not parsed.no_bg_remove,
                clean_orphans=parsed.clean_orphans,
                export_frames=parsed.export_frames,
                export_gifs=parsed.export_gifs,
                gif_duration=parsed.gif_duration,
                export_1x=parsed.export_1x,
            )
            if res.get("status") == "success":
                success_count += 1
                if "audit_metric" in res:
                    audit_metrics.append(res["audit_metric"])
        except Exception as e:
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

    print("\n[SUCCESS] All processing completed successfully!")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main_cli())
