#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pixel scale auto-detection and Feature-Preserving True-Grid Downsampler."""

from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image


class GridDetector:
    """Detects pseudo-pixel pitch and normalizes images into strict 1:1 integer pixel grids with feature preservation."""

    @staticmethod
    def estimate_pixel_pitch(img: Image.Image, min_pitch: int = 3, max_pitch: int = 16) -> int:
        """Estimate the optimal pseudo-pixel block pitch (in raw pixels) using edge autocorrelation."""
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
            # Default to 4px for high-detail pixel characters if peak is found near 4 or 8
            for p, score in candidates:
                if p == 4 or (p % 4 == 0 and p <= 8):
                    return p
            return int(candidates[0][0])

        return 4  # Standard default for high-detail 48x64 / 64x64 pixel sprites

    @staticmethod
    def feature_preserving_downsample(
        img: Image.Image,
        pitch: int,
        alpha_threshold: float = 0.20,
        outline_boost: float = 2.2,
        contrast_boost: float = 1.5
    ) -> Image.Image:
        """Downsample sprite sheet using Saliency-Weighted Mode Pooling.
        
        Preserves thin 1px dark outlines, eye pupils, eye whites, highlights,
        and fine clothing trims from being swallowed by flat skin/hair background areas.
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
                    unique_colors, counts = np.unique(fg_pixels, axis=0, return_counts=True)

                    # Calculate saliency weights for each unique color in block
                    weights = counts.astype(float)
                    for i, c in enumerate(unique_colors):
                        r, g, b = float(c[0]), float(c[1]), float(c[2])
                        lum = 0.299 * r + 0.587 * g + 0.114 * b

                        # 1. Outline boost: dark lines (L < 55) must be preserved
                        if lum < 55:
                            weights[i] *= outline_boost

                        # 2. Chromatic / Contrast boost: vivid features (pupil, lips, gems, highlights)
                        chroma = max(r, g, b) - min(r, g, b)
                        if chroma > 50:
                            weights[i] *= contrast_boost

                    best_idx = np.argmax(weights)
                    mode_color = unique_colors[best_idx]
                    mode_arr[y, x] = mode_color
                    mode_arr[y, x, 3] = 255  # Strict binary solid alpha
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
