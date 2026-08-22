#!/usr/bin/env python3
"""Pure Semantic Palette Extraction & Chroma-Weighted CIELAB Quantization matching Snapper."""

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


class PixelPosterizer:
    """Consolidates continuous AI colors into discrete, crisp pixel art tones with semantic eye/skin/outline preservation."""

    @staticmethod
    def extract_semantic_palette(
        img: Image.Image, max_colors: int = 16, white_hex: str = "#ececec", black_hex: str = "#000000"
    ) -> np.ndarray:
        """Extract a high-contrast semantic palette with dedicated Black outline, White highlight, and Material Medoids."""
        has_alpha = img.mode == "RGBA"
        arr = np.array(img.convert("RGBA" if has_alpha else "RGB"))

        if has_alpha:
            alpha = arr[:, :, 3]
            opaque_mask = alpha > 0
            if not np.any(opaque_mask):
                return np.array([[0, 0, 0], [236, 236, 236]], dtype=np.uint8)
            pixels = arr[opaque_mask, :3]
        else:
            pixels = arr.reshape(-1, 3)

        lums = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]

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
        """Quantize image into the given palette using Chroma-Weighted CIELAB Delta-E."""
        has_alpha = img.mode == "RGBA"
        arr = np.array(img.convert("RGBA" if has_alpha else "RGB"))

        if has_alpha:
            alpha = arr[:, :, 3]
            opaque_mask = alpha > 0
            if not np.any(opaque_mask):
                return img, []
            pixels_rgb = arr[opaque_mask, :3]
        else:
            opaque_mask = np.ones((arr.shape[0], arr.shape[1]), dtype=bool)
            pixels_rgb = arr.reshape(-1, 3)

        pixels_lab = cv2.cvtColor(pixels_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
        palette_lab = cv2.cvtColor(palette_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)

        diff_L = pixels_lab[:, np.newaxis, 0] - palette_lab[np.newaxis, :, 0]
        diff_A = pixels_lab[:, np.newaxis, 1] - palette_lab[np.newaxis, :, 1]
        diff_B = pixels_lab[:, np.newaxis, 2] - palette_lab[np.newaxis, :, 2]

        dists = diff_L**2 + (w_chroma**2) * (diff_A**2 + diff_B**2)

        is_dark = pixels_lab[:, 0] < outline_lum_thresh
        nearest_idx = np.argmin(dists, axis=-1)
        nearest_idx[is_dark] = 0

        quant_rgb = palette_rgb[nearest_idx]

        output_arr = arr.copy()
        output_arr[opaque_mask, :3] = quant_rgb

        unique_colors = np.unique(quant_rgb, axis=0)
        palette_hex = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in unique_colors]

        return Image.fromarray(output_arr, "RGBA" if has_alpha else "RGB"), palette_hex

    @staticmethod
    def process_snapper_pipeline(
        img: Image.Image, max_colors: int = 16, w_chroma: float = 2.0
    ) -> tuple[Image.Image, list[str]]:
        """Adaptive Semantic Quantization matching commercial Pixel Snapper fidelity."""
        has_alpha = img.mode == "RGBA"
        arr = np.array(img.convert("RGBA" if has_alpha else "RGB"))

        if has_alpha:
            alpha = arr[:, :, 3]
            opaque_mask = alpha > 0
            if not np.any(opaque_mask):
                return img, []
            pixels = arr[opaque_mask, :3].astype(float)
        else:
            opaque_mask = np.ones((arr.shape[0], arr.shape[1]), dtype=bool)
            pixels = arr.reshape(-1, 3).astype(float)

        km = KMeans(n_clusters=min(max_colors, len(np.unique(pixels, axis=0))), random_state=42, n_init=5)
        km.fit(pixels)
        centers = np.uint8(np.clip(km.cluster_centers_, 0, 255))
        labels = km.predict(pixels)

        quant_rgb = centers[labels]
        output_arr = arr.copy()
        output_arr[opaque_mask, :3] = quant_rgb

        palette_hex = [
            f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
            for c in sorted(centers, key=lambda c: (c[0] * 0.299 + c[1] * 0.587 + c[2] * 0.114))
        ]

        return Image.fromarray(output_arr, "RGBA" if has_alpha else "RGB"), palette_hex
