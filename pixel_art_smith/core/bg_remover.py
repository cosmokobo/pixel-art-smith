#!/usr/bin/env python3
"""AI & Deterministic Background Removal, Hard Alpha Thresholding, and Edge Defringing."""

import cv2
import numpy as np
from PIL import Image


class BackgroundRemover:
    """Removes background and produces crisp 1-bit alpha pixel art edges."""

    def __init__(self, model_name: str = "isnet-general-use"):
        self.model_name = model_name
        self._session = None

    def _get_session(self):
        if self._session is None:
            from rembg import new_session

            self._session = new_session(self.model_name)
        return self._session

    def remove_background(
        self,
        img: Image.Image,
        alpha_threshold: int = 128,
        defringe: bool = True,
        bg_color_hint: tuple[int, int, int] | None = (255, 255, 255),
        method: str = "auto",
    ) -> Image.Image:
        """Remove background using deterministic FloodFill (for sprite sheets) or AI matting.

        Args:
            img: Input PIL Image.
            alpha_threshold: Cutoff for alpha transparency (0 or 255).
            defringe: Whether to clean up background color bleed on outer edges.
            bg_color_hint: RGB color hint of the background.
            method: 'auto' (detect solid background and use floodfill, else AI), 'floodfill', or 'ai'.
        """
        rgba = img.convert("RGBA")
        arr = np.array(rgba)

        # Check if corners are uniform (indicating solid background like white background in SD)
        corners = np.array([arr[0, 0, :3], arr[0, -1, :3], arr[-1, 0, :3], arr[-1, -1, :3]])
        corner_std = np.std(corners, axis=0).mean()

        if method == "floodfill" or (method == "auto" and corner_std < 15.0):
            return self.remove_background_floodfill(img, tolerance=15, defringe=defringe, bg_color_hint=bg_color_hint)

        # Fallback to AI matting
        try:
            from rembg import remove

            session = self._get_session()
            matted = remove(
                rgba,
                session=session,
                alpha_matting=False,
            )
            arr = np.array(matted)
            alpha = arr[:, :, 3]
            binary_alpha = np.where(alpha >= alpha_threshold, 255, 0).astype(np.uint8)
            arr[:, :, 3] = binary_alpha

            if defringe and bg_color_hint is not None:
                arr = self._defringe(arr, bg_color_hint)

            return Image.fromarray(arr, "RGBA")
        except Exception:
            return self.remove_background_floodfill(img, tolerance=20, defringe=defringe, bg_color_hint=bg_color_hint)

    @classmethod
    def remove_background_floodfill(
        cls,
        img: Image.Image,
        tolerance: int = 15,
        defringe: bool = True,
        bg_color_hint: tuple[int, int, int] | None = (255, 255, 255),
    ) -> Image.Image:
        """Deterministic, 100% loss-free background removal via border floodfill."""
        arr = np.array(img.convert("RGBA"))
        h, w = arr.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
        diff = (tolerance, tolerance, tolerance)

        # Floodfill from all 4 borders & corners
        flags = cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 8

        # 4 corners
        cv2.floodFill(bgr.copy(), mask, (0, 0), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (w - 1, 0), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (0, h - 1), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (w - 1, h - 1), 255, diff, diff, flags=flags)

        # Mid-border edge points to ensure complete perimeter floodfill
        cv2.floodFill(bgr.copy(), mask, (w // 2, 0), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (w // 2, h - 1), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (0, h // 2), 255, diff, diff, flags=flags)
        cv2.floodFill(bgr.copy(), mask, (w - 1, h // 2), 255, diff, diff, flags=flags)

        # Extract interior mask
        bg_mask = mask[1 : h + 1, 1 : w + 1] == 255
        arr[:, :, 3] = np.where(bg_mask, 0, 255).astype(np.uint8)

        # Apply defringing
        if defringe:
            hint = bg_color_hint or tuple(int(c) for c in np.median(arr[0, :5, :3], axis=0))
            arr = cls._defringe(arr, hint)

        return Image.fromarray(arr, "RGBA")

    @staticmethod
    def _defringe(arr: np.ndarray, bg_hint: tuple[int, int, int], tolerance: int = 40) -> np.ndarray:
        """Decontaminate edge pixels where the background color bled into character outline."""
        rgb = arr[:, :, :3].astype(np.int32)
        alpha = arr[:, :, 3]

        opaque = alpha > 0
        if not np.any(opaque):
            return arr

        diff = np.abs(rgb - np.array(bg_hint, dtype=np.int32))
        is_bg_tint = np.all(diff < tolerance, axis=2) & opaque

        clean_opaque = opaque & (~is_bg_tint)
        if np.any(clean_opaque):
            inpaint_mask = is_bg_tint.astype(np.uint8) * 255
            bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
            repaired_bgr = cv2.inpaint(bgr, inpaint_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
            repaired_rgb = cv2.cvtColor(repaired_bgr, cv2.COLOR_BGR2RGB)
            arr[is_bg_tint, :3] = repaired_rgb[is_bg_tint]

        return arr
