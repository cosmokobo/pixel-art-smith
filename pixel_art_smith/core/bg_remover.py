#!/usr/bin/env python3
"""AI & Deterministic Background Removal with Zero-Leakage 4-Connected Quantized FloodFill."""

import cv2
import numpy as np
from PIL import Image


class BackgroundRemover:
    """Removes background and produces crisp 1-bit alpha pixel art edges without leaking into character interior."""

    @staticmethod
    def segment_background_with_cavity_resolution(
        grid_arr: np.ndarray,
        bg_diff_thresh: int = 6,
        max_cavity_area: int = 40,
        resolve_cavities: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Segment background using non-leaking 4-connected floodfill from perimeter edges.

        1. Samples canvas background color dynamically from perimeter medians.
        2. Applies 4-connected floodfill from border edges (outer perimeter background).
        3. Preserves internal character elements (white clothing, aprons, frills, eyes, skin).

        Returns:
            (bg_mask, fg_mask, resolved_cavity_pixel_count)
        """
        target_h, target_w = grid_arr.shape[:2]

        # 1. Background color reference (median of outer border pixels)
        border_pixels = np.vstack([grid_arr[0, :], grid_arr[-1, :], grid_arr[:, 0], grid_arr[:, -1]])
        bg_color = np.median(border_pixels, axis=0)

        # 2. Outer floodfill from perimeter (4-connected prevents diagonal leakage)
        mask = np.zeros((target_h + 2, target_w + 2), np.uint8)
        bgr = cv2.cvtColor(grid_arr, cv2.COLOR_RGB2BGR)
        flags = cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 4
        diff = (5, 5, 5)

        perimeter_points = [
            (0, 0),
            (target_w - 1, 0),
            (0, target_h - 1),
            (target_w - 1, target_h - 1),
            (target_w // 2, 0),
            (target_w // 2, target_h - 1),
            (0, target_h // 2),
            (target_w - 1, target_h // 2),
            (target_w // 4, 0),
            ((3 * target_w) // 4, 0),
            (target_w // 4, target_h - 1),
            ((3 * target_w) // 4, target_h - 1),
        ]

        for pt in perimeter_points:
            cv2.floodFill(bgr.copy(), mask, pt, 255, diff, diff, flags=flags)

        outer_bg_mask = mask[1 : target_h + 1, 1 : target_w + 1] == 255
        enclosed_bg_mask = np.zeros((target_h, target_w), dtype=bool)

        # 3. Optional resolution of enclosed background cavities (only if explicitly enabled)
        if resolve_cavities:
            color_diff = np.max(np.abs(grid_arr.astype(int) - bg_color.astype(int)), axis=-1)
            is_pure_bg_color = color_diff <= bg_diff_thresh
            candidate_trapped = (~outer_bg_mask) & is_pure_bg_color

            trapped_u8 = (candidate_trapped.astype(np.uint8)) * 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(trapped_u8, connectivity=8)

            for i in range(1, num_labels):
                comp_mask = labels == i
                area = stats[i, cv2.CC_STAT_AREA]

                dilated = cv2.dilate(comp_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
                perimeter = dilated & (~comp_mask)
                perimeter_rgbs = grid_arr[perimeter]

                if len(perimeter_rgbs) == 0:
                    continue

                perim_diffs = np.max(np.abs(perimeter_rgbs.astype(int) - bg_color.astype(int)), axis=-1)
                non_bg_fraction = np.mean(perim_diffs > 10)

                if non_bg_fraction > 0.85 and area <= max_cavity_area:
                    enclosed_bg_mask |= comp_mask

        final_bg_mask = outer_bg_mask | enclosed_bg_mask
        final_fg_mask = ~final_bg_mask
        resolved_count = int(np.sum(enclosed_bg_mask))

        return final_bg_mask, final_fg_mask, resolved_count

    @staticmethod
    def remove_background_quantized(quant_img: Image.Image) -> Image.Image:
        """Apply strict 4-connected floodfill on the quantized 128x128 grid."""
        arr = np.array(quant_img.convert("RGB"))
        h, w = arr.shape[:2]

        bg_mask, fg_mask, _ = BackgroundRemover.segment_background_with_cavity_resolution(arr)

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = arr
        rgba[:, :, 3] = np.where(bg_mask, 0, 255).astype(np.uint8)

        return Image.fromarray(rgba, "RGBA")
