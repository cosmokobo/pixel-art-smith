#!/usr/bin/env python3
"""Sprite Frame Isolation, 2D Matrix (Row/Column) Structuring, and True-Grid Cell Segmentation."""

import cv2
import numpy as np
from PIL import Image


class FrameItem:
    """Represents a single isolated sprite frame."""

    def __init__(self, image: Image.Image, bbox: tuple[int, int, int, int], row: int = 0, col: int = 0):
        self.image = image
        self.bbox = bbox  # (x, y, w, h) in logical pixels
        self.row = row  # Motion / Animation index
        self.col = col  # Frame sequence index in motion


class SpriteIsolator:
    """Isolates character frames and organizes them into 2D (Row=Motion, Col=Frame) matrix structures."""

    def __init__(self, min_area: int = 16, padding: int = 1):
        self.min_area = min_area
        self.padding = padding

    def isolate_matrix(
        self, grid_img: Image.Image, expected_rows: int | None = 4, expected_cols: int | None = 4
    ) -> list[list[FrameItem]]:
        """Isolate frames from a downsampled True-Grid image.

        By default, utilizes Grid-Cell Segmentation (4 rows x 4 cols = 16 frames)
        which keeps detached wands, floating particles, and hair ribbons safely bound to each frame.
        """
        arr = np.array(grid_img.convert("RGBA"))
        img_h, img_w = arr.shape[:2]

        # Use regular Grid-Cell Slicing for standard matrix sprite sheets
        if expected_rows and expected_cols and expected_rows > 0 and expected_cols > 0:
            return self.isolate_grid_cells(grid_img, n_rows=expected_rows, n_cols=expected_cols)

        # Fallback to contour-based detection for non-standard or arbitrary layouts
        return self._isolate_by_contours(grid_img)

    def isolate_grid_cells(self, grid_img: Image.Image, n_rows: int = 4, n_cols: int = 4) -> list[list[FrameItem]]:
        """Slice the grid image into a clean N_rows x N_cols matrix of frames."""
        arr = np.array(grid_img.convert("RGBA"))
        img_h, img_w = arr.shape[:2]
        cell_w = img_w // n_cols
        cell_h = img_h // n_rows

        matrix: list[list[FrameItem]] = []
        for r in range(n_rows):
            row_items: list[FrameItem] = []
            for c in range(n_cols):
                x1, y1 = c * cell_w, r * cell_h
                x2, y2 = min(img_w, (c + 1) * cell_w), min(img_h, (r + 1) * cell_h)
                cell_arr = arr[y1:y2, x1:x2]

                alpha = cell_arr[:, :, 3]
                ys, xs = np.where(alpha > 0)
                if len(xs) > 0 and len(ys) > 0:
                    bx1, bx2 = max(0, xs.min() - self.padding), min(cell_arr.shape[1], xs.max() + 1 + self.padding)
                    by1, by2 = max(0, ys.min() - self.padding), min(cell_arr.shape[0], ys.max() + 1 + self.padding)
                    tight_crop = Image.fromarray(cell_arr[by1:by2, bx1:bx2], "RGBA")
                    bbox = (x1 + bx1, y1 + by1, bx2 - bx1, by2 - by1)
                else:
                    tight_crop = Image.fromarray(cell_arr, "RGBA")
                    bbox = (x1, y1, cell_w, cell_h)

                item = FrameItem(tight_crop, bbox, row=r, col=c)
                row_items.append(item)
            matrix.append(row_items)

        return matrix

    def _isolate_by_contours(self, grid_img: Image.Image) -> list[list[FrameItem]]:
        """Contour clustering fallback for non-grid layouts."""
        arr = np.array(grid_img.convert("RGBA"))
        alpha = arr[:, :, 3]
        img_h, img_w = alpha.shape

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
            crop = grid_img.copy()
            item = FrameItem(crop, (0, 0, grid_img.width, grid_img.height), row=0, col=0)
            return [[item]]

        raw_boxes.sort(key=lambda b: b[1] + b[3] / 2.0)
        avg_h = sum(b[3] for b in raw_boxes) / len(raw_boxes)
        row_gap_threshold = max(4.0, avg_h * 0.45)

        rows_clustered: list[list[tuple[int, int, int, int]]] = []
        current_row: list[tuple[int, int, int, int]] = []
        current_y_center = None

        for b in raw_boxes:
            y_center = b[1] + b[3] / 2.0
            if current_y_center is None:
                current_y_center = y_center
                current_row.append(b)
            elif abs(y_center - current_y_center) < row_gap_threshold:
                current_y_center = (current_y_center * len(current_row) + y_center) / (len(current_row) + 1)
                current_row.append(b)
            else:
                rows_clustered.append(current_row)
                current_row = [b]
                current_y_center = y_center

        if current_row:
            rows_clustered.append(current_row)

        matrix: list[list[FrameItem]] = []
        for r_idx, row_boxes in enumerate(rows_clustered):
            row_boxes.sort(key=lambda b: b[0])
            row_items: list[FrameItem] = []
            for c_idx, (x, y, w, h) in enumerate(row_boxes):
                crop_arr = arr[y : y + h, x : x + w]
                crop_img = Image.fromarray(crop_arr, "RGBA")
                item = FrameItem(crop_img, (x, y, w, h), row=r_idx, col=c_idx)
                row_items.append(item)
            matrix.append(row_items)

        return matrix
