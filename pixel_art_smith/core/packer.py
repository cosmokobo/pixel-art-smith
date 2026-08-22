#!/usr/bin/env python3
"""Sprite Sheet Matrix Packer, Ground Alignment, Grid Modes, and Agentic AI Metadata."""

from typing import Any

from PIL import Image

from .grid_detector import GridDetector
from .sprite_isolator import FrameItem


class SpritePacker:
    """Aligns frames to bottom-center anchor and packs into 2D Matrix (Rows=Motions, Cols=Frames) Sheets."""

    @staticmethod
    def standardize_frame(sprite: Image.Image, cell_size: tuple[int, int], bottom_margin: int = 1) -> Image.Image:
        """Place sprite onto a fixed-size transparent canvas with bottom-center ground anchor."""
        cell_w, cell_h = cell_size
        canvas = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))

        sw, sh = sprite.size
        # Fit into cell if larger
        if sw > cell_w or sh > cell_h:
            ratio = min(cell_w / sw, cell_h / sh)
            new_w, new_h = max(1, int(sw * ratio)), max(1, int(sh * ratio))
            sprite = sprite.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
            sw, sh = sprite.size

        offset_x = max(0, (cell_w - sw) // 2)
        offset_y = max(0, cell_h - sh - bottom_margin)

        canvas.paste(sprite, (offset_x, offset_y), sprite)
        return canvas

    @staticmethod
    def calculate_optimal_cell_size(matrix: list[list[FrameItem]], min_w: int = 24, min_h: int = 32) -> tuple[int, int]:
        """Determine optimal standardized cell size based on maximum sprite dimensions across all frames."""
        max_w = min_w
        max_h = min_h
        for row in matrix:
            for item in row:
                w, h = item.image.size
                if w > max_w:
                    max_w = w
                if h > max_h:
                    max_h = h

        # Add 2px margin for breathing room and round up to multiple of 2
        cell_w = ((max_w + 3) // 2) * 2
        cell_h = ((max_h + 3) // 2) * 2
        return cell_w, cell_h

    @staticmethod
    def resolve_cell_size(matrix: list[list[FrameItem]], grid_mode: str = "auto-fit") -> tuple[int, int]:
        """Resolve cell size based on grid mode."""
        mode = grid_mode.lower().strip()
        if mode in ("auto-fit", "auto", "fit", "none", "0"):
            return SpritePacker.calculate_optimal_cell_size(matrix)
        elif mode in ("preserve-sheet", "snapper", "original") or mode in ("fixed-32", "32", "32x32"):
            return 32, 32
        elif mode in ("fixed-48", "48", "48x48"):
            return 48, 48
        elif mode in ("fixed-64", "64", "64x64"):
            return 64, 64
        elif "x" in mode:
            parts = mode.replace("fixed-", "").split("x")
            try:
                return int(parts[0]), int(parts[1])
            except Exception:
                pass
        return SpritePacker.calculate_optimal_cell_size(matrix)

    @staticmethod
    def pack_matrix_sheet(
        matrix: list[list[FrameItem]],
        cell_size: tuple[int, int] | None = None,
        scale: int = 1,
        palette_name: str = "snapper-13",
        palette_colors: list[str] | None = None,
        grid_mode: str = "auto-fit",
    ) -> tuple[Image.Image, dict[str, Any], list[list[Image.Image]]]:
        """Pack 2D matrix into an M (Rows/Motions) x N (Columns/Frames) Sprite Sheet.

        Returns:
            (Packed_Sheet_Image, Agentic_Metadata_Dict, Standardized_Frame_Grid)
        """
        if cell_size is None:
            cell_size = SpritePacker.resolve_cell_size(matrix, grid_mode=grid_mode)

        n_rows = len(matrix)
        max_cols = max(len(row) for row in matrix) if n_rows > 0 else 0

        cell_w, cell_h = cell_size
        scaled_w = cell_w * scale
        scaled_h = cell_h * scale

        sheet_w = max_cols * scaled_w
        sheet_h = n_rows * scaled_h

        sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
        std_grid: list[list[Image.Image]] = []

        animations_meta: dict[str, Any] = {}
        total_frame_count = 0

        for r_idx, row in enumerate(matrix):
            std_row: list[Image.Image] = []
            motion_id = f"motion_{r_idx}"
            frames_list: list[dict[str, Any]] = []

            for c_idx, item in enumerate(row):
                # 1. Standardize frame canvas
                std_frame = SpritePacker.standardize_frame(item.image, cell_size)

                # 2. Integer upscale if scale > 1
                if scale > 1:
                    disp_frame = GridDetector.upscale_nearest(std_frame, scale=scale)
                else:
                    disp_frame = std_frame

                std_row.append(disp_frame)

                # 3. Paste into sheet
                pos_x = c_idx * scaled_w
                pos_y = r_idx * scaled_h
                sheet.paste(disp_frame, (pos_x, pos_y), disp_frame)

                frames_list.append(
                    {
                        "frame_index": c_idx,
                        "rect": {"x": pos_x, "y": pos_y, "w": scaled_w, "h": scaled_h},
                        "anchor": "bottom-center",
                    }
                )
                total_frame_count += 1

            std_grid.append(std_row)
            animations_meta[motion_id] = {"row_index": r_idx, "frame_count": len(row), "frames": frames_list}

        # Clean, concise metadata structure optimized for Agentic AI and Game Engines
        metadata: dict[str, Any] = {
            "schema_version": "2.0",
            "sprite_sheet": {
                "format": "RGBA8888",
                "width": sheet_w,
                "height": sheet_h,
                "grid_layout": {
                    "grid_mode": grid_mode,
                    "rows": n_rows,
                    "columns": max_cols,
                    "cell_size": {"width": scaled_w, "height": scaled_h},
                    "logical_cell_size": {"width": cell_w, "height": cell_h},
                    "scale_factor": scale,
                },
                "total_frames": total_frame_count,
            },
            "palette": {
                "name": palette_name,
                "color_count": len(palette_colors) if palette_colors else 0,
                "colors": palette_colors or [],
            },
            "animations": animations_meta,
        }

        return sheet, metadata, std_grid

    @staticmethod
    def pack_canvas_sheet(
        canvas_img: Image.Image,
        scale: int = 1,
        palette_name: str = "snapper-16",
        palette_colors: list[str] | None = None,
    ) -> tuple[Image.Image, dict[str, Any], list[list[Image.Image]]]:
        """Pack a single-asset / non-4-motion canvas into Snapper-Parity 1:1 Pixel Art.

        Returns:
            (Packed_Canvas_Image, Metadata_Dict, Single_Frame_Grid)
        """
        orig_w, orig_h = canvas_img.size
        if scale > 1:
            scaled_img = GridDetector.upscale_nearest(canvas_img, scale=scale)
        else:
            scaled_img = canvas_img.copy()

        scaled_w, scaled_h = scaled_img.size

        metadata: dict[str, Any] = {
            "schema_version": "2.0",
            "sprite_sheet": {
                "format": "RGBA8888",
                "width": scaled_w,
                "height": scaled_h,
                "grid_layout": {
                    "grid_mode": "snapper-canvas",
                    "rows": 1,
                    "columns": 1,
                    "cell_size": {"width": scaled_w, "height": scaled_h},
                    "logical_cell_size": {"width": orig_w, "height": orig_h},
                    "scale_factor": scale,
                },
                "total_frames": 1,
            },
            "palette": {
                "name": palette_name,
                "color_count": len(palette_colors) if palette_colors else 0,
                "colors": palette_colors or [],
            },
            "animations": {
                "single_frame": {
                    "row_index": 0,
                    "frame_count": 1,
                    "frames": [
                        {
                            "frame_index": 0,
                            "rect": {"x": 0, "y": 0, "w": scaled_w, "h": scaled_h},
                            "anchor": "center",
                        }
                    ],
                }
            },
        }

        return scaled_img, metadata, [[scaled_img]]

