#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pixel art color palettes and CIELAB color quantization."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans

# -----------------------------------------------------------------------------
# Curated Master Palettes (Hex) - Categorized into Consoles, Master, and Special
# -----------------------------------------------------------------------------
PALETTES: Dict[str, List[str]] = {
    # -------------------------------------------------------------------------
    # 🎮 Retro Gaming Consoles (Nintendo, Sega, Commodore)
    # -------------------------------------------------------------------------
    # NES / Famicom 54-Color Hardware Palette
    "nes-54": [
        "#7c7c7c", "#0000fc", "#0000bc", "#4428bc", "#940084", "#a80020", "#a81000", "#881400",
        "#503000", "#007800", "#006800", "#005800", "#004058", "#000000", "#bcbcbc", "#0078f8",
        "#0058f8", "#6844fc", "#d800cc", "#e40058", "#f83800", "#e45c10", "#ac7c00", "#00b800",
        "#00a800", "#00a844", "#008888", "#000000", "#f8f8f8", "#3cbcfc", "#6888fc", "#9878f8",
        "#f878f8", "#f85898", "#f87858", "#fca044", "#f8b800", "#b8f818", "#58d854", "#58f898",
        "#00e8d8", "#787878", "#fcbcfc", "#fcfcfc", "#a4e4fc", "#b8b8f8", "#d8b8f8", "#f8b8f8",
        "#f8a4c0", "#f0d0b0", "#fce0a8", "#f8d878", "#d8f878", "#b8f8b8"
    ],
    # Original GameBoy DMG-01 (4 Shades of Olive Green)
    "gameboy-classic": [
        "#0f380f", "#306230", "#8bac0f", "#9bbc0f"
    ],
    # GameBoy Pocket (4 Shades of Pure Grayscale)
    "gameboy-pocket": [
        "#2b2b2b", "#545454", "#878787", "#c2c2c2"
    ],
    # GameBoy Color (GBC Master 32-Color Selection)
    "gameboy-color": [
        "#000000", "#181010", "#282020", "#403030", "#604848", "#886868", "#b89090", "#e8c0c0",
        "#ffffff", "#f83800", "#f87800", "#f8b800", "#f8f800", "#78f800", "#00f800", "#00f878",
        "#00f8f8", "#0078f8", "#0000f8", "#7800f8", "#f800f8", "#f80078", "#880000", "#884000",
        "#888800", "#008800", "#008888", "#000088", "#880088", "#505050", "#909090", "#d0d0d0"
    ],
    # Super Nintendo (SNES 32-Color Master Ramps)
    "snes-classic": [
        "#000000", "#191919", "#323232", "#4b4b4b", "#646464", "#7d7d7d", "#969696", "#afafaf",
        "#c8c8c8", "#e1e1e1", "#ffffff", "#800000", "#b02020", "#e04040", "#ff7070", "#402000",
        "#704010", "#a06020", "#d08030", "#ffa040", "#004010", "#107020", "#20a030", "#40d050",
        "#70ff80", "#002040", "#104070", "#2060a0", "#3080d0", "#50a0ff", "#500050", "#801080"
    ],
    # Sega Genesis / MegaDrive 64-Color Palette
    "sega-genesis": [
        "#000000", "#000033", "#000066", "#000099", "#0000cc", "#0000ff", "#003300", "#003333",
        "#003366", "#003399", "#0033cc", "#0033ff", "#006600", "#006633", "#006666", "#006699",
        "#0066cc", "#0066ff", "#009900", "#009933", "#009966", "#009999", "#0099cc", "#0099ff",
        "#00cc00", "#00cc33", "#00cc66", "#00cc99", "#00cccc", "#00ccff", "#00ff00", "#00ff33",
        "#00ff66", "#00ff99", "#00ffcc", "#00ffff", "#330000", "#660000", "#990000", "#cc0000",
        "#ff0000", "#ff3300", "#ff6600", "#ff9900", "#ffcc00", "#ffff00", "#ff00ff", "#ffffff",
        "#333333", "#666666", "#999999", "#cccccc", "#4d2600", "#804000", "#b35900", "#e67300",
        "#2b1400", "#592c00", "#8c4600", "#bf5f00", "#f27900", "#663300", "#994c00", "#cc6600"
    ],
    # PICO-8 (16-Color Fantasy Console)
    "pico-8": [
        "#000000", "#1d2b53", "#7e2553", "#008751", "#ab5236", "#5f574f", "#c2c3c7", "#fff1e8",
        "#ff004d", "#ffa300", "#ffec27", "#00e436", "#29adff", "#83769c", "#ff77a8", "#ffccaa"
    ],
    # Commodore 64 (16 Colors)
    "c64-commodore": [
        "#000000", "#ffffff", "#880000", "#aaffee", "#cc44cc", "#00cc55", "#0000aa", "#eeee77",
        "#dd8855", "#664400", "#ff7777", "#333333", "#777777", "#aaff66", "#0088ff", "#bbbbbb"
    ],

    # -------------------------------------------------------------------------
    # 🎨 Master Pixel Artist Palettes
    # -------------------------------------------------------------------------
    # Endesga 32 (EDG32) - Industry standard for fantasy pixel art
    "endesga-32": [
        "#be4a2f", "#d77643", "#ead4aa", "#e4a672", "#b86f50", "#733e39", "#3e2731", "#a22633",
        "#e43b44", "#f77622", "#feae34", "#fee761", "#63c74d", "#3e8948", "#265c42", "#193c3e",
        "#124e89", "#0099db", "#2ce8f5", "#ffffff", "#c0cbdc", "#8b9bb4", "#5a6988", "#3a4466",
        "#262b44", "#181425", "#ff0044", "#68386c", "#b55088", "#f6757a", "#e8b796", "#c28569"
    ],
    # Endesga 64 (EDG64) - 64-Color expanded master palette
    "endesga-64": [
        "#ff0044", "#f77622", "#feae34", "#fee761", "#63c74d", "#3e8948", "#265c42", "#193c3e",
        "#124e89", "#0099db", "#2ce8f5", "#ffffff", "#c0cbdc", "#8b9bb4", "#5a6988", "#3a4466",
        "#262b44", "#181425", "#be4a2f", "#d77643", "#ead4aa", "#e4a672", "#b86f50", "#733e39",
        "#3e2731", "#a22633", "#e43b44", "#68386c", "#b55088", "#f6757a", "#e8b796", "#c28569",
        "#141b1b", "#1a2a22", "#284433", "#3d6e49", "#569b59", "#78c366", "#a6e477", "#d8f88c",
        "#1b1424", "#2e1e3b", "#492b57", "#6b3b77", "#935099", "#be6ebb", "#e495db", "#fbc3f4",
        "#201217", "#381c24", "#5a2a33", "#823e47", "#ad575e", "#d47879", "#f2a19b", "#fecfbe",
        "#111822", "#1b283b", "#273f5a", "#385c7f", "#4e80a8", "#6da8d1", "#96d2f3", "#c6f2fe"
    ],
    # DawnBringer 32 (DB32) - Iconic RPG palette
    "dawnbringer-32": [
        "#000000", "#222034", "#45283c", "#663931", "#8f563b", "#df7126", "#d9a066", "#eec39a",
        "#fbf236", "#99e550", "#6abe30", "#37946e", "#4b692f", "#524b24", "#323c39", "#3f3f74",
        "#306082", "#5b6ee1", "#639bff", "#5fcde4", "#cbdbfc", "#ffffff", "#9badb7", "#847e87",
        "#696a6a", "#595652", "#76428a", "#ac3232", "#d95763", "#d77643", "#8f974a", "#8a6f30"
    ],
    # DawnBringer 16 (DB16) - Classical 16-color RPG
    "dawnbringer-16": [
        "#140c1c", "#442434", "#30346d", "#4e4a4e", "#854c30", "#346524", "#d04648", "#757161",
        "#597dce", "#d27d2c", "#8595a1", "#6daa2c", "#d2aa99", "#6caa96", "#d4d06a", "#ffffff"
    ],
    # Resurrect 64 - Rich, balanced 64-color palette
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
    # Sweetie 16 - Soft, vibrant pastel palette
    "sweetie-16": [
        "#1a1c2b", "#57294b", "#9b4153", "#d06e59", "#f7a768", "#ffe37a", "#a2d966", "#459c52",
        "#1a5959", "#247385", "#40b0a6", "#73eff7", "#ffffff", "#a4b2c6", "#68728a", "#3b3d54"
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
    """CIELAB-based perceptual color quantization and palette snapping engine."""

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

    def get_colors_hex(self) -> List[str]:
        """Return list of active palette colors as hex strings."""
        if self._palette_rgb is None:
            return []
        return [rgb_to_hex(tuple(c)) for c in self._palette_rgb]

    def quantize(self, img: Image.Image) -> Image.Image:
        """Quantize an RGBA PIL image to the active palette in CIELAB space.
        
        Preserves alpha channel transparency (A=0 stays transparent).
        """
        if self._palette_rgb is None or len(self._palette_rgb) == 0:
            return img

        arr = np.array(img.convert("RGBA"))
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]

        mask = alpha > 0
        if not np.any(mask):
            return img

        opaque_pixels = rgb[mask]
        
        # Convert opaque pixels to CIELAB
        lab_pixels = cv2.cvtColor(opaque_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

        # Vectorized Euclidean distance in CIELAB space (approximates Delta-E 1976)
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
