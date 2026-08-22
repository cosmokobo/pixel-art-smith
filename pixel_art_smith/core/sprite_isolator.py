#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sprite Frame Isolation, 2D Matrix (Row/Column) Clustering and Motion Structuring."""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image


class FrameItem:
    """Represents a single isolated sprite frame."""
    def __init__(self, image: Image.Image, bbox: Tuple[int, int, int, int], row: int = 0, col: int = 0):
        self.image = image
        self.bbox = bbox  # (x, y, w, h) in logical pixels
        self.row = row    # Motion / Animation index
        self.col = col    # Frame sequence index in motion


class SpriteIsolator:
    """Isolates character frames and organizes them into 2D (Row=Motion, Col=Frame) matrix structures."""

    def __init__(self, min_area: int = 16, padding: int = 1):
        self.min_area = min_area
        self.padding = padding

    def isolate_matrix(
        self,
        grid_img: Image.Image,
        expected_rows: Optional[int] = None,
        expected_cols: Optional[int] = None
    ) -> List[List[FrameItem]]:
        """Isolate frames from a downsampled True-Grid image and cluster into Rows x Columns."""
        arr = np.array(grid_img.convert("RGBA"))
        alpha = arr[:, :, 3]
        img_h, img_w = alpha.shape

        # Find external contours based on binary alpha channel
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < self.min_area:
                continue

            x1 = max(0, x - self.padding)
            y1 = max(0, y - self.padding)
            x2 = min(img_w, x + w + self.padding)
            y2 = min(img_h, y + h + self.padding)
            raw_boxes.append((x1, y1, x2 - x1, y2 - y1))

        if not raw_boxes:
            # Single frame fallback
            crop = grid_img.copy()
            item = FrameItem(crop, (0, 0, grid_img.width, grid_img.height), row=0, col=0)
            return [[item]]

        # Cluster boxes into rows by Y coordinate
        # Sort all boxes by Y center
        raw_boxes.sort(key=lambda b: b[1] + b[3] / 2.0)

        # Determine row threshold
        avg_h = sum(b[3] for b in raw_boxes) / len(raw_boxes)
        row_gap_threshold = max(4.0, avg_h * 0.45)

        rows_clustered: List[List[Tuple[int, int, int, int]]] = []
        current_row: List[Tuple[int, int, int, int]] = []
        current_y_center = None

        for b in raw_boxes:
            y_center = b[1] + b[3] / 2.0
            if current_y_center is None:
                current_y_center = y_center
                current_row.append(b)
            elif abs(y_center - current_y_center) < row_gap_threshold:
                # Same row: update running average center
                current_y_center = (current_y_center * len(current_row) + y_center) / (len(current_row) + 1)
                current_row.append(b)
            else:
                # New row
                rows_clustered.append(current_row)
                current_row = [b]
                current_y_center = y_center

        if current_row:
            rows_clustered.append(current_row)

        # Sort each row horizontally by X (left to right)
        matrix: List[List[FrameItem]] = []
        for r_idx, row_boxes in enumerate(rows_clustered):
            row_boxes.sort(key=lambda b: b[0])
            row_items: List[FrameItem] = []
            for c_idx, (x, y, w, h) in enumerate(row_boxes):
                crop_arr = arr[y:y+h, x:x+w]
                crop_img = Image.fromarray(crop_arr, "RGBA")
                item = FrameItem(crop_img, (x, y, w, h), row=r_idx, col=c_idx)
                row_items.append(item)
            matrix.append(row_items)

        return matrix
