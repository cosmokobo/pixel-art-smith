#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pixel scale auto-detection and True-Grid Mode Pooling Downsampler."""

from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image


class GridDetector:
    """Detects pseudo-pixel pitch and normalizes images into strict 1:1 integer pixel grids."""

    @staticmethod
    def estimate_pixel_pitch(img: Image.Image, min_pitch: int = 4, max_pitch: int = 24) -> int:
        """Estimate the pseudo-pixel block pitch (in raw pixels) using edge autocorrelation."""
        gray = np.array(img.convert("L"))
        h, w = gray.shape

        # Horizontal gradient differences
        diff_x = np.abs(gray[:, 1:].astype(float) - gray[:, :-1].astype(float))
        edge_x = np.mean(diff_x > 25, axis=0)

        # Autocorrelation
        norm_x = edge_x - np.mean(edge_x)
        ac_x = np.correlate(norm_x, norm_x, mode='full')
        ac_x = ac_x[len(norm_x) - 1:]

        # Find local peaks in autocorrelation
        candidates = []
        for lag in range(min_pitch, min(max_pitch + 1, len(ac_x) - 1)):
            if ac_x[lag] > ac_x[lag - 1] and ac_x[lag] > ac_x[lag + 1]:
                candidates.append((lag, float(ac_x[lag])))

        if candidates:
            # Sort by score
            candidates.sort(key=lambda c: c[1], reverse=True)
            best_pitch = candidates[0][0]
            # If pitch is multiple of a smaller pitch (e.g. 16 instead of 8), check if 8 is also strong
            for p, score in candidates:
                if p < best_pitch and best_pitch % p == 0 and score > 0.6 * candidates[0][1]:
                    best_pitch = p
            return int(best_pitch)

        return 8  # Standard default for 512/1024 SD pixel art

    @staticmethod
    def mode_downsample_global(
        img: Image.Image,
        pitch: int,
        alpha_threshold: float = 0.25
    ) -> Image.Image:
        """Downsample full sprite sheet using Mode (Majority Color) Pooling based on native pitch.
        
        This maps each PxP pseudo-pixel block into exactly 1 logical pixel without color blurring or mixels.
        """
        if pitch <= 1:
            return img.copy()

        arr = np.array(img.convert("RGBA"))
        h, w, _ = arr.shape

        target_w = max(1, w // pitch)
        target_h = max(1, h // pitch)

        mode_arr = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        block_area = pitch * pitch
        min_opaque_count = max(1, int(block_area * alpha_threshold))

        for y in range(target_h):
            y_start = y * pitch
            y_end = min(h, (y + 1) * pitch)
            for x in range(target_w):
                x_start = x * pitch
                x_end = min(w, (x + 1) * pitch)

                block = arr[y_start:y_end, x_start:x_end]
                if block.size == 0:
                    continue

                opaque_mask = block[:, :, 3] > 0
                opaque_count = np.sum(opaque_mask)

                if opaque_count >= min_opaque_count:
                    fg_pixels = block[opaque_mask]
                    # Find mode (most frequent RGBA color)
                    unique_colors, counts = np.unique(fg_pixels, axis=0, return_counts=True)
                    mode_color = unique_colors[np.argmax(counts)]
                    mode_arr[y, x] = mode_color
                    mode_arr[y, x, 3] = 255  # Strict solid alpha
                else:
                    mode_arr[y, x] = [0, 0, 0, 0]  # Fully transparent

        return Image.fromarray(mode_arr, "RGBA")

    @staticmethod
    def upscale_nearest(img: Image.Image, scale: int = 1) -> Image.Image:
        """Upscale image by an exact integer factor using Nearest-Neighbor interpolation."""
        if scale <= 1:
            return img
        w, h = img.size
        return img.resize((w * scale, h * scale), resample=Image.Resampling.NEAREST)
