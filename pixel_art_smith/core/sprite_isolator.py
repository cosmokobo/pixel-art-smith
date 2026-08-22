#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sprite Cell Isolation and Bounding Detection."""

from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image


class SpriteIsolator:
    """Isolates individual character poses and motion frames from a sprite sheet."""

    def __init__(self, min_area: int = 400, padding: int = 2):
        self.min_area = min_area
        self.padding = padding

    def isolate_frames(self, img: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """Detect individual sprite bounding boxes and return a list of (Cropped_Image, (x, y, w, h))."""
        arr = np.array(img.convert("RGBA"))
        alpha = arr[:, :, 3]

        # Find external contours based on binary alpha channel
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        img_h, img_w = alpha.shape

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < self.min_area:
                continue
            
            # Apply slight padding
            x1 = max(0, x - self.padding)
            y1 = max(0, y - self.padding)
            x2 = min(img_w, x + w + self.padding)
            y2 = min(img_h, y + h + self.padding)
            boxes.append((x1, y1, x2 - x1, y2 - y1))

        if not boxes:
            # Fallback: treat whole image as single frame
            return [(img, (0, 0, img.width, img.height))]

        # Sort spatially: Cluster by rows (Y), then sort by columns (X)
        # Calculate average height to determine row grouping tolerance
        avg_h = sum(b[3] for b in boxes) / len(boxes)
        row_tol = max(20, int(avg_h * 0.5))

        # Sort primarily by Y (binned into rows), then by X
        boxes.sort(key=lambda b: (b[1] // row_tol, b[0]))

        results = []
        for x, y, w, h in boxes:
            crop_arr = arr[y:y+h, x:x+w]
            crop_img = Image.fromarray(crop_arr, "RGBA")
            results.append((crop_img, (x, y, w, h)))

        return results
