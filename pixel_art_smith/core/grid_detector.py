#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pixel scale auto-detection and Grid-Perfect normalization."""

from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image


class GridDetector:
    """Detects pseudo-pixel pitch and normalizes images into strict integer pixel grids."""

    @staticmethod
    def estimate_pixel_scale(img: Image.Image) -> float:
        """Estimate the pseudo-pixel scale factor in the AI-generated sprite using gradient autocorrelation."""
        gray = np.array(img.convert("L"))
        h, w = gray.shape

        # Calculate horizontal and vertical gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)

        # Autocorrelation along horizontal profile
        row_profile = np.mean(mag, axis=0)
        profile_norm = row_profile - np.mean(row_profile)
        autocorr = np.correlate(profile_norm, profile_norm, mode='full')
        autocorr = autocorr[len(profile_norm) - 1:]

        # Find first peak between lag 2 and 16
        peaks = []
        for lag in range(2, min(16, len(autocorr) - 1)):
            if autocorr[lag] > autocorr[lag - 1] and autocorr[lag] > autocorr[lag + 1]:
                peaks.append((lag, autocorr[lag]))

        if peaks:
            best_lag = max(peaks, key=lambda p: p[1])[0]
            return float(best_lag)

        return 4.0  # Common default for SD 512/1024 pixel art generations

    @staticmethod
    def downsample_to_grid(
        img: Image.Image,
        target_size: Tuple[int, int],
        resample_filter: Image.Resampling = Image.Resampling.BOX
    ) -> Image.Image:
        """Downsample image to target logical resolution (e.g. 32x32, 48x48, 64x64).
        
        Box / Area filtering averages subpixel blur into a clean logical pixel value.
        """
        downscaled = img.resize(target_size, resample=resample_filter)
        
        # Binary alpha cleanup on downsampled image
        arr = np.array(downscaled)
        if arr.shape[2] == 4:
            arr[:, :, 3] = np.where(arr[:, :, 3] >= 128, 255, 0).astype(np.uint8)
        return Image.fromarray(arr, "RGBA")

    @staticmethod
    def upscale_nearest(
        img: Image.Image,
        scale: int = 1
    ) -> Image.Image:
        """Upscale image by an exact integer factor using Nearest-Neighbor interpolation."""
        if scale <= 1:
            return img
        w, h = img.size
        return img.resize((w * scale, h * scale), resample=Image.Resampling.NEAREST)
