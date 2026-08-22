#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI Background Removal, Hard Alpha Thresholding, and Edge Defringing."""

from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session


class BackgroundRemover:
    """Removes background and produces crisp 1-bit alpha pixel art edges."""

    def __init__(self, model_name: str = "isnet-general-use"):
        self.model_name = model_name
        self._session = None

    def _get_session(self):
        if self._session is None:
            self._session = new_session(self.model_name)
        return self._session

    def remove_background(
        self,
        img: Image.Image,
        alpha_threshold: int = 128,
        defringe: bool = True,
        bg_color_hint: Optional[Tuple[int, int, int]] = (255, 255, 255)
    ) -> Image.Image:
        """Remove background using AI matting, apply binary alpha, and defringe edge bleed."""
        rgba = img.convert("RGBA")
        
        # 1. AI Matting using rembg
        session = self._get_session()
        matted = remove(
            rgba,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=15,
            alpha_matting_erode_size=5
        )
        
        arr = np.array(matted)
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]

        # 2. Strict Binary Alpha Thresholding (0 or 255)
        binary_alpha = np.where(alpha >= alpha_threshold, 255, 0).astype(np.uint8)
        arr[:, :, 3] = binary_alpha

        # 3. Edge Defringing (Color Decontamination)
        if defringe and bg_color_hint is not None:
            arr = self._defringe(arr, bg_color_hint)

        return Image.fromarray(arr, "RGBA")

    @staticmethod
    def _defringe(arr: np.ndarray, bg_hint: Tuple[int, int, int], tolerance: int = 40) -> np.ndarray:
        """Decontaminate edge pixels where the background color bled into character outline."""
        rgb = arr[:, :, :3].astype(np.int32)
        alpha = arr[:, :, 3]

        # Mask of opaque pixels
        opaque = alpha > 0
        if not np.any(opaque):
            return arr

        # Identify pixels close to the background hint color
        diff = np.abs(rgb - np.array(bg_hint, dtype=np.int32))
        is_bg_tint = np.all(diff < tolerance, axis=2) & opaque

        # Dilate clean opaque mask to find nearest character interior color
        clean_opaque = opaque & (~is_bg_tint)
        if np.any(clean_opaque):
            # Inpaint the contaminated pixels using nearest neighbor interpolation
            inpaint_mask = is_bg_tint.astype(np.uint8) * 255
            bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
            repaired_bgr = cv2.inpaint(bgr, inpaint_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
            repaired_rgb = cv2.cvtColor(repaired_bgr, cv2.COLOR_BGR2RGB)
            arr[is_bg_tint, :3] = repaired_rgb[is_bg_tint]

        return arr
