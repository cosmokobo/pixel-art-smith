#!/usr/bin/env python3
"""Pixel art heuristic cleanup: 8-connectivity non-destructive orphan pixel cleanup."""

import cv2
import numpy as np
from PIL import Image


class PixelCleaner:
    """Safe pixel art cleaner that preserves thin diagonal hair strands, veil tips, and outlines."""

    @staticmethod
    def remove_orphan_pixels(img: Image.Image) -> Image.Image:
        """Remove only truly isolated 1x1 noise pixels with zero neighbors in all 8 directions.

        Preserves diagonal 1-pixel hair strands, pointed veil tips, and dagger edges.
        """
        arr = np.array(img.convert("RGBA"))
        alpha = arr[:, :, 3]

        mask = (alpha > 0).astype(np.uint8)

        # Full 8-connected kernel (checks all orthogonal and diagonal neighbors)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

        neighbor_count = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # Truly isolated dot: opaque (mask == 1) but has 0 neighbors in all 8 directions
        orphan_mask = (mask == 1) & (neighbor_count == 0)

        arr[orphan_mask, 3] = 0
        return Image.fromarray(arr, "RGBA")

    @staticmethod
    def cleanup_transparency_halos(img: Image.Image) -> Image.Image:
        """Ensure all pixels are either completely transparent (A=0) or completely opaque (A=255)."""
        arr = np.array(img.convert("RGBA"))
        alpha = arr[:, :, 3]
        arr[:, :, 3] = np.where(alpha >= 128, 255, 0).astype(np.uint8)
        return Image.fromarray(arr, "RGBA")
