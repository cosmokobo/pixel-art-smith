#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pixel art color palettes and CIELAB color quantization."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans

# -----------------------------------------------------------------------------
# Curated Master Palettes (Hex)
# -----------------------------------------------------------------------------
PALETTES: Dict[str, List[str]] = {
    # 32-Color iconic master fantasy RPG palette by Endesga
    "endesga-32": [
        "#be4a2f", "#d77643", "#ead4aa", "#e4a672", "#b86f50", "#733e39", "#3e2731", "#a22633",
        "#e43b44", "#f77622", "#feae34", "#fee761", "#63c74d", "#3e8948", "#265c42", "#193c3e",
        "#124e89", "#0099db", "#2ce8f5", "#ffffff", "#c0cbdc", "#8b9bb4", "#5a6988", "#3a4466",
        "#262b44", "#181425", "#ff0044", "#68386c", "#b55088", "#f6757a", "#e8b796", "#c28569"
    ],
    # 32-Color classic RPG palette by DawnBringer
    "dawnbringer-32": [
        "#000000", "#222034", "#45283c", "#663931", "#8f563b", "#df7126", "#d9a066", "#eec39a",
        "#fbf236", "#99e550", "#6abe30", "#37946e", "#4b692f", "#524b24", "#323c39", "#3f3f74",
        "#306082", "#5b6ee1", "#639bff", "#5fcde4", "#cbdbfc", "#ffffff", "#9badb7", "#847e87",
        "#696a6a", "#595652", "#76428a", "#ac3232", "#d95763", "#d77643", "#8f974a", "#8a6f30"
    ],
    # 16-Color standard palette by DawnBringer
    "dawnbringer-16": [
        "#140c1c", "#442434", "#30346d", "#4e4a4e", "#854c30", "#346524", "#d04648", "#757161",
        "#597dce", "#d27d2c", "#8595a1", "#6daa2c", "#d2aa99", "#6caa96", "#d4d06a", "#ffffff"
    ],
    # 16-Color iconic PICO-8 fantasy console palette
    "pico-8": [
        "#000000", "#1d2b53", "#7e2553", "#008751", "#ab5236", "#5f574f", "#c2c3c7", "#fff1e8",
        "#ff004d", "#ffa300", "#ffec27", "#00e436", "#29adff", "#83769c", "#ff77a8", "#ffccaa"
    ],
    # 64-Color rich expansion palette (Resurrect 64)
    "resurrect-64": [
        "#2e040e", "#431021", "#591c34", "#702b49", "#893d5d", "#a35272", "#bd6a88", "#d7849e",
        "#f0a0b6", "#2c1e74", "#40318e", "#5648a7", "#6d61c0", "#857cd7", "#9f99ed", "#bbb6ff",
        "#0b2b28", "#12423d", "#1d5c54", "#2a776c", "#3a9385", "#4db09f", "#63cdba", "#7cebd5",
        "#3c2415", "#55361d", "#704b27", "#8c6233", "#aa7b41", "#c89551", "#e7b164", "#ffce7a",
        "#380000", "#5c0e0e", "#801d1d", "#a43030", "#c84747", "#eb6262", "#ff8080", "#ffa1a1",
        "#0d2030", "#16334a", "#224a66", "#306284", "#407da3", "#5399c2", "#68b7e2", "#80d6ff",
        "#1a1c23", "#282b37", "#393d4e", "#4d5267", "#646a82", "#7d849e", "#99a0bb", "#b6bdd8",
        "#000000", "#1f1f1f", "#3e3e3e", "#616161", "#888888", "#b1b1b1", "#dedede", "#ffffff"
    ],
    # 4-Color classic GameBoy monochrome
    "gameboy-classic": [
        "#0f380f", "#306230", "#8bac0f", "#9bbc0f"
    ]
}


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex color string to (R, G, B) tuple."""
    hex_str = hex_str.lstrip('#')
    return int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert (R, G, B) tuple to hex color string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


class PaletteQuantizer:
    """CIELAB-based color quantization and palette snapping engine."""

    def __init__(self, palette_name: str = "endesga-32", custom_colors: Optional[List[str]] = None):
        self.palette_name = palette_name.lower()
        self._palette_rgb: Optional[np.ndarray] = None
        self._palette_lab: Optional[np.ndarray] = None

        if custom_colors:
            self.set_custom_palette(custom_colors)
        elif self.palette_name in PALETTES:
            self.set_custom_palette(PALETTES[self.palette_name])

    def set_custom_palette(self, hex_list: List[str]) -> None:
        """Set active palette from a list of hex strings."""
        rgb_list = [hex_to_rgb(h) for h in hex_list]
        self._palette_rgb = np.array(rgb_list, dtype=np.uint8)
        # Convert RGB palette to CIELAB for perceptual color matching
        rgb_reshaped = self._palette_rgb.reshape(-1, 1, 3)
        self._palette_lab = cv2.cvtColor(rgb_reshaped, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    def quantize(self, img: Image.Image, dithering: bool = False) -> Image.Image:
        """Quantize an RGBA PIL image to the active palette.
        
        Preserves alpha channel transparency (A=0 stays transparent).
        """
        if self._palette_rgb is None or len(self._palette_rgb) == 0:
            return img

        arr = np.array(img.convert("RGBA"))
        h, w, _ = arr.shape
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]

        mask = alpha > 0
        if not np.any(mask):
            return img

        opaque_pixels = rgb[mask]
        
        # Convert opaque pixels to CIELAB
        lab_pixels = cv2.cvtColor(opaque_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

        # Vectorized Euclidean distance in CIELAB space (approximates Delta-E)
        # dist shape: (N_pixels, N_palette_colors)
        dists = np.sum((lab_pixels[:, np.newaxis, :] - self._palette_lab[np.newaxis, :, :]) ** 2, axis=2)
        nearest_indices = np.argmin(dists, axis=1)

        quantized_rgb = self._palette_rgb[nearest_indices]

        output_arr = arr.copy()
        output_arr[mask, :3] = quantized_rgb
        return Image.fromarray(output_arr, "RGBA")

    @staticmethod
    def extract_adaptive_palette(img: Image.Image, n_colors: int = 24) -> List[str]:
        """Extract an adaptive K-Means palette from the image in CIELAB color space."""
        arr = np.array(img.convert("RGBA"))
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]

        mask = alpha > 0
        pixels = rgb[mask]
        if len(pixels) == 0:
            return ["#000000"]

        if len(pixels) <= n_colors:
            unique_rgb = np.unique(pixels, axis=0)
            return [rgb_to_hex(tuple(c)) for c in unique_rgb]

        # Cluster in CIELAB space for perceptual clustering
        lab_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_colors, n_init=4, random_state=42)
        kmeans.fit(lab_pixels)
        centers_lab = kmeans.cluster_centers_.astype(np.uint8)

        centers_rgb = cv2.cvtColor(centers_lab.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
        return [rgb_to_hex(tuple(c)) for c in centers_rgb]
