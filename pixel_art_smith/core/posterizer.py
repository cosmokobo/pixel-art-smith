#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure Pixel Art Mode-Peak Quantization & Crisp 1px Outline Locking (Zero Blurring)."""

from typing import List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import MiniBatchKMeans


class PixelPosterizer:
    """Consolidates continuous AI colors into crisp, vibrant discrete pixel art tones with zero spatial blurring."""

    @staticmethod
    def consolidate_color_ramps(
        img: Image.Image,
        max_colors: int = 14,
        enforce_black_outline: bool = True,
        outline_lum_thresh: float = 45.0,
        flat_median_passes: int = 0  # Default 0 to prevent watercolor bleeding
    ) -> Tuple[Image.Image, List[str]]:
        """Consolidate image colors into max_colors discrete tones using Medoid/Mode Peak Extraction.
        
        Zero spatial blurring is used to maintain 100% razor-sharp pixel edges.
        
        Args:
            img: Downsampled RGBA PIL Image.
            max_colors: Target number of distinct character colors (12~16 typical).
            enforce_black_outline: Whether to lock all dark boundary pixels to pure #000000.
            outline_lum_thresh: Luminance threshold for outline detection (0~255).
            flat_median_passes: Must be 0 to prevent watercolor bleeding across borders.
            
        Returns:
            (Consolidated_RGBA_Image, List_Of_Hex_Colors)
        """
        arr = np.array(img.convert("RGBA"))
        h, w, _ = arr.shape
        alpha = arr[:, :, 3]
        opaque_mask = alpha > 0

        if not np.any(opaque_mask):
            return img, []

        fg_rgb = arr[opaque_mask, :3]

        if len(fg_rgb) == 0:
            return img, []

        # 1. CIELAB K-Means Clustering on Opaque Foreground
        lab_fg = cv2.cvtColor(fg_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        n_clusters = max(2, min(max_colors, len(np.unique(fg_rgb, axis=0))))

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=2048,
            random_state=42,
            n_init=5
        )
        kmeans.fit(lab_fg)

        # 2. Extract the EXACT Mode/Medoid color of each cluster (NOT the arithmetic mean)
        # Arithmetic mean creates muddy watercolor desaturation; Mode selects the true vibrant pixel.
        cluster_centers_rgb = np.zeros((n_clusters, 3), dtype=np.uint8)
        labels = kmeans.labels_

        for k in range(n_clusters):
            k_mask = labels == k
            if np.any(k_mask):
                k_pixels = fg_rgb[k_mask]
                unique_k, counts_k = np.unique(k_pixels, axis=0, return_counts=True)
                # Select the most frequent pure color in this cluster
                cluster_centers_rgb[k] = unique_k[np.argmax(counts_k)]
            else:
                cluster_centers_rgb[k] = [0, 0, 0]

        # 3. Outline Locking
        lums = 0.299 * cluster_centers_rgb[:, 0] + 0.587 * cluster_centers_rgb[:, 1] + 0.114 * cluster_centers_rgb[:, 2]
        darkest_k = int(np.argmin(lums))

        if enforce_black_outline:
            cluster_centers_rgb[darkest_k] = [0, 0, 0]

        # 4. Nearest Neighbor Mapping in CIELAB space
        centers_lab = cv2.cvtColor(cluster_centers_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
        
        # Calculate Delta-E squared in Lab space
        diff = lab_fg[:, np.newaxis, :] - centers_lab[np.newaxis, :, :]
        # Weight L channel higher for contrast separation
        dists = (1.5 * diff[:, :, 0])**2 + diff[:, :, 1]**2 + diff[:, :, 2]**2

        nearest_k = np.argmin(dists, axis=1)

        # Force dark edge pixels (L < outline_lum_thresh) to darkest outline slot
        if enforce_black_outline:
            is_dark = lab_fg[:, 0] < outline_lum_thresh
            nearest_k[is_dark] = darkest_k

        quant_fg_rgb = cluster_centers_rgb[nearest_k]

        # 5. Assemble final RGBA image
        output_arr = arr.copy()
        output_arr[opaque_mask, :3] = quant_fg_rgb

        # Collect unique hex colors
        unique_colors = np.unique(quant_fg_rgb, axis=0)
        hex_colors = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in unique_colors]

        return Image.fromarray(output_arr, "RGBA"), hex_colors
