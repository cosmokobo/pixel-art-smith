#!/usr/bin/env python3
"""Pure Semantic Palette Extraction & Chroma-Weighted CIELAB Quantization."""

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


class PixelPosterizer:
    """Consolidates continuous AI colors into discrete, crisp pixel art tones with semantic eye/skin/outline preservation."""

    @staticmethod
    def extract_semantic_palette(
        img: Image.Image, max_colors: int = 13, white_hex: str = "#ececec", black_hex: str = "#000000"
    ) -> np.ndarray:
        """Extract a high-contrast semantic palette with dedicated Black outline, White highlight, and Material Medoids."""
        arr = np.array(img.convert("RGBA"))
        alpha = arr[:, :, 3]
        opaque_mask = alpha > 0

        if not np.any(opaque_mask):
            return np.array([[0, 0, 0], [236, 236, 236]], dtype=np.uint8)

        pixels = arr[opaque_mask, :3]
        lums = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]

        # Filter out extreme dark (outlines) and extreme bright (whites) from clustering
        fg_mask = (lums > 35) & (lums < 240)
        fg_pixels = pixels[fg_mask]

        palette_list: list[list[int]] = [
            [0, 0, 0],  # 0: Dedicated Pure Black Outline
            [236, 236, 236],  # 1: Dedicated Pure White / Eye White / Highlight
        ]

        n_interior = max(2, max_colors - 2)

        if len(fg_pixels) > 0:
            if len(np.unique(fg_pixels, axis=0)) <= n_interior:
                for c in np.unique(fg_pixels, axis=0):
                    palette_list.append(c.tolist())
            else:
                lab_fg = cv2.cvtColor(fg_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
                kmeans = KMeans(n_clusters=n_interior, n_init=5, random_state=42)
                kmeans.fit(lab_fg)

                labels = kmeans.labels_
                for k in range(n_interior):
                    k_mask = labels == k
                    if np.any(k_mask):
                        k_colors = fg_pixels[k_mask]
                        unique_k, counts_k = np.unique(k_colors, axis=0, return_counts=True)
                        palette_list.append(unique_k[np.argmax(counts_k)].tolist())

        return np.array(palette_list, dtype=np.uint8)

    @staticmethod
    def quantize_chroma_weighted(
        img: Image.Image, palette_rgb: np.ndarray, w_chroma: float = 2.0, outline_lum_thresh: float = 35.0
    ) -> tuple[Image.Image, list[str]]:
        """Quantize image into the given palette using Chroma-Weighted CIELAB Delta-E.

        Args:
            img: Downsampled RGBA Image.
            palette_rgb: Palette array of shape (N, 3).
            w_chroma: Weight on A & B color channels (higher = vivid saturated colors are preserved).
            outline_lum_thresh: Luminance threshold to snap directly to Black slot 0.

        Returns:
            (Quantized_RGBA_Image, List_Of_Hex_Colors)
        """
        arr = np.array(img.convert("RGBA"))
        alpha = arr[:, :, 3]
        opaque_mask = alpha > 0

        if not np.any(opaque_mask):
            return img, []

        pixels_rgb = arr[opaque_mask, :3]
        pixels_lab = cv2.cvtColor(pixels_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
        palette_lab = cv2.cvtColor(palette_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)

        diff_L = pixels_lab[:, np.newaxis, 0] - palette_lab[np.newaxis, :, 0]
        diff_A = pixels_lab[:, np.newaxis, 1] - palette_lab[np.newaxis, :, 1]
        diff_B = pixels_lab[:, np.newaxis, 2] - palette_lab[np.newaxis, :, 2]

        dists = diff_L**2 + (w_chroma**2) * (diff_A**2 + diff_B**2)

        # Force dark pixels (L < 35) to slot 0 (Pure Black #000000)
        is_dark = pixels_lab[:, 0] < outline_lum_thresh
        nearest_idx = np.argmin(dists, axis=-1)
        nearest_idx[is_dark] = 0

        quant_rgb = palette_rgb[nearest_idx]

        output_arr = arr.copy()
        output_arr[opaque_mask, :3] = quant_rgb

        unique_colors = np.unique(quant_rgb, axis=0)
        hex_colors = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in unique_colors]

        return Image.fromarray(output_arr, "RGBA"), hex_colors

    @staticmethod
    def process_snapper_pipeline(
        img: Image.Image, max_colors: int = 13, w_chroma: float = 2.0
    ) -> tuple[Image.Image, list[str]]:
        """Complete Snapper-style semantic color quantization pipeline."""
        palette_rgb = PixelPosterizer.extract_semantic_palette(img, max_colors=max_colors)
        return PixelPosterizer.quantize_chroma_weighted(img, palette_rgb, w_chroma=w_chroma)
