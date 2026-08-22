#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sprite Sheet Packer, Ground Alignment, and Metadata Exporter."""

import json
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image


class SpritePacker:
    """Aligns frames to bottom-center anchor and packs them into standard sprite sheets."""

    @staticmethod
    def standardize_frame(
        sprite: Image.Image,
        cell_size: Tuple[int, int],
        bottom_margin: int = 1
    ) -> Image.Image:
        """Place sprite onto a fixed-size transparent canvas with bottom-center anchor."""
        cell_w, cell_h = cell_size
        canvas = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))

        sw, sh = sprite.size
        # Fit into cell if larger
        if sw > cell_w or sh > cell_h:
            ratio = min(cell_w / sw, cell_h / sh)
            new_w, new_h = max(1, int(sw * ratio)), max(1, int(sh * ratio))
            sprite = sprite.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
            sw, sh = sprite.size

        offset_x = (cell_w - sw) // 2
        offset_y = cell_h - sh - bottom_margin
        offset_y = max(0, offset_y)

        canvas.paste(sprite, (offset_x, offset_y), sprite)
        return canvas

    @staticmethod
    def pack_horizontal_sheet(
        frames: List[Image.Image],
        cell_size: Tuple[int, int]
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Pack a list of standardized frames into a horizontal 1xN sprite sheet."""
        cell_w, cell_h = cell_size
        n = len(frames)
        if n == 0:
            return Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0)), {}

        sheet = Image.new("RGBA", (cell_w * n, cell_h), (0, 0, 0, 0))
        metadata: Dict[str, Any] = {
            "meta": {
                "format": "RGBA8888",
                "size": {"w": cell_w * n, "h": cell_h},
                "scale": 1,
                "cell_size": {"w": cell_w, "h": cell_h},
                "frame_count": n
            },
            "frames": {}
        }

        for i, frame in enumerate(frames):
            sheet.paste(frame, (i * cell_w, 0), frame)
            metadata["frames"][f"frame_{i}"] = {
                "frame": {"x": i * cell_w, "y": 0, "w": cell_w, "h": cell_h},
                "rotated": False,
                "trimmed": False,
                "spriteSourceSize": {"x": 0, "y": 0, "w": cell_w, "h": cell_h},
                "sourceSize": {"w": cell_w, "h": cell_h}
            }

        return sheet, metadata
