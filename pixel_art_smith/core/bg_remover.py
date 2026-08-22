#!/usr/bin/env python3
"""AI & Deterministic Background Removal with Zero-Leakage 4-Connected Quantized FloodFill."""

import cv2
import numpy as np
from PIL import Image


class BackgroundRemover:
    """Removes background and produces crisp 1-bit alpha pixel art edges without leaking into character interior."""

    @staticmethod
    def remove_background_quantized(quant_img: Image.Image) -> Image.Image:
        """Apply strict 4-connected floodfill on the quantized 128x128 grid.

        Because the quantized image has solid, discrete palette colors and crisp outlines,
        4-connected floodfill with tolerance=0 cleanly removes ONLY the outer background perimeter
        and CANNOT leak through diagonally connected outlines into faces, hair, or eye-whites.
        """
        arr = np.array(quant_img.convert("RGB"))
        h, w = arr.shape[:2]

        mask = np.zeros((h + 2, w + 2), np.uint8)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        # 4-connectivity (no diagonal jumping) + tolerance=0 (exact background color match only)
        flags = cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 4
        diff = (0, 0, 0)

        # Floodfill from all 4 corners
        cv2.floodFill(bgr.copy(), mask, (0, 0), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (w - 1, 0), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (0, h - 1), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (w - 1, h - 1), 255, diff, diff, flags=flags)

        # Edge centers
        cv2.floodFill(bgr.copy(), mask, (w // 2, 0), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (w // 2, h - 1), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (0, h // 2), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (w - 1, h // 2), 255, diff, diff, flags=flags)

        bg_mask = mask[1 : h + 1, 1 : w + 1] == 255

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = arr
        rgba[:, :, 3] = np.where(bg_mask, 0, 255).astype(np.uint8)

        return Image.fromarray(rgba, "RGBA")
