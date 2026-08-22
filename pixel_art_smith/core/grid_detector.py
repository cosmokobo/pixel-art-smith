#!/usr/bin/env python3
"""Pixel scale auto-detection and Core-Subblock Sampling Downsampler (Zero-Bleed)."""

import numpy as np
from PIL import Image


class GridDetector:
    """Detects pseudo-pixel pitch and normalizes images into strict 1:1 integer pixel grids."""

    @staticmethod
    def estimate_pixel_pitch(img: Image.Image, min_pitch: int = 4, max_pitch: int = 16) -> int:
        """Estimate the pseudo-pixel block pitch (in raw pixels) using edge autocorrelation."""
        gray = np.array(img.convert("L"))
        h, w = gray.shape

        # Horizontal gradient differences
        diff_x = np.abs(gray[:, 1:].astype(float) - gray[:, :-1].astype(float))
        edge_x = np.mean(diff_x > 25, axis=0)

        # Autocorrelation
        norm_x = edge_x - np.mean(edge_x)
        ac_x = np.correlate(norm_x, norm_x, mode="full")
        ac_x = ac_x[len(norm_x) - 1 :]

        # Find local peaks in autocorrelation
        candidates = []
        for lag in range(min_pitch, min(max_pitch + 1, len(ac_x) - 1)):
            if ac_x[lag] > ac_x[lag - 1] and ac_x[lag] > ac_x[lag + 1]:
                candidates.append((lag, float(ac_x[lag])))

        if candidates:
            candidates.sort(key=lambda c: c[1], reverse=True)
            for p, _score in candidates:
                if p in (8, 4):
                    return p
            return int(candidates[0][0])

        return 8  # Standard default for 32x32 retro pixel sprites

    @staticmethod
    def core_subblock_downsample(
        img: Image.Image, pitch: int = 8, margin: int = 1, alpha_threshold: float = 0.25
    ) -> Image.Image:
        """Downsample image by sampling the pure center core of each PxP block.

        Samples the inner sub-block of each pixel cell to eliminate edge blur and color bleeding.
        Preserves all RGB channels intact.
        """
        if pitch <= 1:
            return img.copy()

        has_alpha = img.mode == "RGBA"
        arr = np.array(img.convert("RGBA" if has_alpha else "RGB"))
        h, w = arr.shape[:2]

        target_w = max(1, w // pitch)
        target_h = max(1, h // pitch)

        if has_alpha:
            mode_arr = np.zeros((target_h, target_w, 4), dtype=np.uint8)
            for y in range(target_h):
                cy = y * pitch + pitch // 2
                for x in range(target_w):
                    cx = x * pitch + pitch // 2
                    block = arr[max(0, cy - 1) : min(h, cy + 1), max(0, cx - 1) : min(w, cx + 1)]
                    if block.size == 0:
                        continue
                    opaque_mask = block[:, :, 3] > 0
                    if np.sum(opaque_mask) >= max(1, block.shape[0] * block.shape[1] * alpha_threshold):
                        fg = block[opaque_mask]
                        mode_arr[y, x, :3] = fg[:, :3].mean(axis=0).astype(np.uint8)
                        mode_arr[y, x, 3] = 255
                    else:
                        mode_arr[y, x] = [0, 0, 0, 0]
            return Image.fromarray(mode_arr, "RGBA")
        else:
            mode_arr = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            for y in range(target_h):
                cy = y * pitch + pitch // 2
                for x in range(target_w):
                    cx = x * pitch + pitch // 2
                    block = arr[max(0, cy - 1) : min(h, cy + 1), max(0, cx - 1) : min(w, cx + 1)]
                    if block.size > 0:
                        mode_arr[y, x] = block.mean(axis=(0, 1)).astype(np.uint8)
            return Image.fromarray(mode_arr, "RGB")

    @staticmethod
    def upscale_nearest(img: Image.Image, scale: int = 1) -> Image.Image:
        """Upscale image by an exact integer factor using Nearest-Neighbor interpolation."""
        if scale <= 1:
            return img
        w, h = img.size
        return img.resize((w * scale, h * scale), resample=Image.Resampling.NEAREST)
