#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pixel Art Color Posterization, Ramp Consolidation, and 1px Solid Outline Snapping."""

from typing import List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans


class PixelPosterizer:
    """Consolidates AI continuous color gradients into discrete, clean 1~3 step retro pixel art color ramps."""

    @staticmethod
    def consolidate_color_ramps(
        img: Image.Image,
        max_colors: int = 14,
        enforce_black_outline: bool = True,
        outline_lum_thresh: float = 45.0,
        flat_median_passes: int = 1
    ) -> Tuple[Image.Image, List[str]]:
        """Consolidate image colors into max_colors discrete tones in CIELAB space.
        
        Args:
            img: Downsampled RGBA PIL Image.
            max_colors: Target number of distinct character colors (10~16 typical for authentic pixel art).
            enforce_black_outline: Whether to snap all dark boundary pixels to pure solid #000000.
            outline_lum_thresh: Luminance threshold for outline detection (0~255).
            flat_median_passes: Number of median filter passes to smooth internal gradient noise.
            
        Returns:
            (Posterized_RGBA_Image, List_Of_Hex_Colors)
        """
        arr = np.array(img.convert("RGBA"))
        h, w, _ = arr.shape
        alpha = arr[:, :, 3]
        opaque_mask = alpha > 0

        if not np.any(opaque_mask):
            return img, []

        rgb = arr[:, :, :3]

        # 1. Optional Flat Surface Noise Reduction (removes scattered 1-pixel color noise)
        if flat_median_passes > 0:
            # Apply median blur only on opaque areas to prevent edge bleeding
            blurred = cv2.medianBlur(rgb, 3)
            rgb[opaque_mask] = blurred[opaque_mask]

        # 2. Separate Outlines from Interior Surfaces if enforce_black_outline is True
        opaque_pixels = rgb[opaque_mask]
        lum = 0.299 * opaque_pixels[:, 0] + 0.587 * opaque_pixels[:, 1] + 0.114 * opaque_pixels[:, 2]
        
        outline_mask_in_opaque = lum < outline_lum_thresh
        interior_mask_in_opaque = ~outline_mask_in_opaque

        # 3. K-Means Clustering in CIELAB space on interior surface pixels
        interior_pixels = opaque_pixels[interior_mask_in_opaque]
        n_interior_colors = max(2, max_colors - (1 if enforce_black_outline else 0))

        if len(interior_pixels) == 0:
            # Entire character is dark
            res_arr = arr.copy()
            res_arr[opaque_mask, :3] = 0
            return Image.fromarray(res_arr, "RGBA"), ["#000000"]

        if len(np.unique(interior_pixels, axis=0)) <= n_interior_colors:
            quant_interior_rgb = interior_pixels
            unique_centers = np.unique(interior_pixels, axis=0)
        else:
            lab_interior = cv2.cvtColor(interior_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
            kmeans = KMeans(n_clusters=n_interior_colors, n_init=5, random_state=42)
            kmeans.fit(lab_interior)
            
            centers_lab = kmeans.cluster_centers_.astype(np.uint8)
            quant_lab = centers_lab[kmeans.labels_]
            quant_interior_rgb = cv2.cvtColor(quant_lab.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
            unique_centers = cv2.cvtColor(centers_lab.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)

        # 4. Reassemble output image
        output_opaque_rgb = np.zeros_like(opaque_pixels)
        
        if enforce_black_outline:
            output_opaque_rgb[outline_mask_in_opaque] = [0, 0, 0]
        else:
            # Quantize outlines along with the rest
            output_opaque_rgb[outline_mask_in_opaque] = opaque_pixels[outline_mask_in_opaque]

        output_opaque_rgb[interior_mask_in_opaque] = quant_interior_rgb

        output_arr = arr.copy()
        output_arr[opaque_mask, :3] = output_opaque_rgb

        # Collect hex color list
        all_unique = np.unique(output_opaque_rgb, axis=0)
        hex_colors = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in all_unique]

        return Image.fromarray(output_arr, "RGBA"), hex_colors
