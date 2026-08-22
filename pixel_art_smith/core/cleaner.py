#!/usr/bin/env python3
"""Pixel art heuristic cleanup: orphan pixel removal and outline smoothing."""

import cv2
import numpy as np
from PIL import Image


class PixelCleaner:
    """Heuristic pixel art cleaner for removing single-pixel noise and jagged contours."""

    @staticmethod
    def remove_orphan_pixels(img: Image.Image) -> Image.Image:
        """Remove or absorb 1-pixel orphan dots that have zero orthogonal opaque neighbors."""
        arr = np.array(img.convert("RGBA"))
        alpha = arr[:, :, 3]
        h, w = alpha.shape

        # Binary mask
        mask = (alpha > 0).astype(np.uint8)

        # 4-connected kernel (orthogonal neighbors)
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)

        neighbor_count = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # Isolated pixel: mask == 1 but neighbor_count == 0
        orphan_mask = (mask == 1) & (neighbor_count == 0)

        # Clear orphan pixels to transparent
        arr[orphan_mask, 3] = 0
        return Image.fromarray(arr, "RGBA")

    @staticmethod
    def cleanup_transparency_halos(img: Image.Image) -> Image.Image:
        """Ensure all pixels are either completely transparent (A=0) or completely opaque (A=255)."""
        arr = np.array(img.convert("RGBA"))
        alpha = arr[:, :, 3]
        arr[:, :, 3] = np.where(alpha >= 128, 255, 0).astype(np.uint8)
        return Image.fromarray(arr, "RGBA")
