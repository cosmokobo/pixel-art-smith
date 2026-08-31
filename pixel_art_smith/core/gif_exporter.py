#!/usr/bin/env python3
"""Animated GIF Exporter for Sprite Sheet Motions and Composite Previews."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class GifExporter:
    """Exports standardized sprite frames into per-motion and composite animated GIFs."""

    MOTION_NAMES_4DIR = ["down", "left", "right", "up"]

    @staticmethod
    def _convert_to_gif_frame(img: Image.Image, transparent: bool = True) -> Image.Image:
        """Convert RGBA image into an optimized palette frame for GIF animation."""
        if not transparent or img.mode != "RGBA":
            return img.convert("RGB")

        alpha = img.getchannel("A")
        # If no transparent pixels, simple conversion
        if np.all(np.array(alpha) == 255):
            return img.convert("RGB")

        # Create RGB base and paste RGBA with mask
        rgb_img = Image.new("RGB", img.size, (0, 0, 0))
        rgb_img.paste(img, mask=alpha)

        # Quantize to 255 colors (saving index 255 for transparency)
        p_img = rgb_img.quantize(colors=255, method=Image.Resampling.NEAREST if hasattr(Image, "Resampling") else 0)
        p_arr = np.array(p_img)
        p_arr[np.array(alpha) == 0] = 255

        p_out = Image.fromarray(p_arr, "P")
        palette = p_img.getpalette()[: 255 * 3] + [0, 0, 0]
        p_out.putpalette(palette)
        p_out.info["transparency"] = 255
        p_out.info["disposal"] = 2
        return p_out

    @classmethod
    def export_motion_gif(
        cls,
        frames: list[Image.Image],
        output_path: Path,
        duration: int = 150,
        transparent: bool = True,
    ) -> Path:
        """Save a list of frames as a looping animated GIF."""
        if not frames:
            raise ValueError("No frames provided for GIF export.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed = [cls._convert_to_gif_frame(f, transparent=transparent) for f in frames]

        first = processed[0]
        kwargs: dict[str, Any] = {
            "save_all": True,
            "append_images": processed[1:] if len(processed) > 1 else [],
            "duration": duration,
            "loop": 0,
            "disposal": 2,
        }
        if transparent and first.mode == "P" and "transparency" in first.info:
            kwargs["transparency"] = first.info["transparency"]

        first.save(output_path, **kwargs)
        return output_path

    @classmethod
    def export_composite_preview_gif(
        cls,
        std_grid: list[list[Image.Image]],
        output_path: Path,
        duration: int = 150,
        bg_color: tuple[int, int, int] = (30, 34, 42),
    ) -> Path:
        """Create a side-by-side composite animated GIF displaying all motions simultaneously."""
        if not std_grid or not std_grid[0]:
            raise ValueError("Empty sprite grid provided for composite GIF.")

        n_rows = len(std_grid)
        max_cols = max(len(row) for row in std_grid)
        cell_w, cell_h = std_grid[0][0].size

        composite_frames: list[Image.Image] = []

        for c_idx in range(max_cols):
            comp_img = Image.new("RGB", (cell_w * n_rows, cell_h), bg_color)
            for r_idx in range(n_rows):
                row = std_grid[r_idx]
                frame = row[c_idx % len(row)]
                if frame.mode == "RGBA":
                    bg = Image.new("RGBA", (cell_w, cell_h), (*bg_color, 255))
                    char_frame = Image.alpha_composite(bg, frame).convert("RGB")
                else:
                    char_frame = frame.convert("RGB")
                comp_img.paste(char_frame, (r_idx * cell_w, 0))
            composite_frames.append(comp_img)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        composite_frames[0].save(
            output_path,
            save_all=True,
            append_images=composite_frames[1:] if len(composite_frames) > 1 else [],
            duration=duration,
            loop=0,
        )
        return output_path

    @classmethod
    def export_all_gifs(
        cls,
        std_grid: list[list[Image.Image]],
        output_dir: Path,
        stem: str,
        duration: int = 150,
        export_individual: bool = True,
        export_composite: bool = True,
    ) -> dict[str, Any]:
        """Export both individual motion GIFs and all-motions composite GIF."""
        output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = {
            "individual_gifs": [],
            "composite_gif": None,
        }

        n_rows = len(std_grid)
        if export_individual:
            for r_idx, row in enumerate(std_grid):
                if n_rows == 4 and r_idx < len(cls.MOTION_NAMES_4DIR):
                    motion_tag = f"motion_{r_idx:02d}_{cls.MOTION_NAMES_4DIR[r_idx]}"
                else:
                    motion_tag = f"motion_{r_idx:02d}"

                gif_path = output_dir / f"{stem}_{motion_tag}.gif"
                cls.export_motion_gif(row, gif_path, duration=duration, transparent=True)
                results["individual_gifs"].append(str(gif_path))

        if export_composite and n_rows > 0:
            comp_path = output_dir / f"{stem}_all_motions.gif"
            cls.export_composite_preview_gif(std_grid, comp_path, duration=duration)
            results["composite_gif"] = str(comp_path)

        return results
