#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modern Desktop GUI Studio for PixelArtSmith with Animation Player & Chroma-Weighted Quantization."""

import os
import sys
import threading
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageTk, ImageDraw

try:
    import customtkinter as ctk
    GUI_BACKEND = "customtkinter"
except ImportError:
    import tkinter as ctk
    import tkinter.ttk as ttk
    GUI_BACKEND = "tkinter"

import tkinter as tk
from tkinter import filedialog, messagebox

from ..core.bg_remover import BackgroundRemover
from ..core.grid_detector import GridDetector
from ..core.sprite_isolator import SpriteIsolator, FrameItem
from ..core.palette import PaletteQuantizer, PALETTES, hex_to_rgb
from ..core.cleaner import PixelCleaner
from ..core.packer import SpritePacker
from ..core.posterizer import PixelPosterizer


class PixelArtSmithApp:
    """Desktop GUI Studio Application for PixelArtSmith."""

    def __init__(self, root):
        self.root = root
        self.root.title("🎨 PixelArtSmith Studio - True-Grid AI Pixel Art & Animation Lab")
        self.root.geometry("1240x840")
        self.root.minsize(1020, 720)

        # State
        self.current_image_path: Optional[Path] = None
        self.raw_image: Optional[Image.Image] = None
        self.processed_sheet: Optional[Image.Image] = None
        self.std_grid: List[List[Image.Image]] = []
        self.metadata: Dict[str, Any] = {}
        self.bg_remover = BackgroundRemover()

        # Animation Player State
        self.anim_running = False
        self.anim_motion_idx = 0
        self.anim_frame_idx = 0
        self.anim_timer_id = None

        # GUI Setup
        self._build_ui()

    def _build_ui(self):
        # Configure grid
        self.root.grid_columnconfigure(0, weight=0)  # Sidebar
        self.root.grid_columnconfigure(1, weight=1)  # Main display tabs
        self.root.grid_rowconfigure(0, weight=1)

        # ---------------------------------------------------------------------
        # Left Control Sidebar
        # ---------------------------------------------------------------------
        if GUI_BACKEND == "customtkinter":
            self.sidebar = ctk.CTkScrollableFrame(self.root, width=320, corner_radius=0)
        else:
            self.sidebar = tk.Frame(self.root, width=320, bg="#242424")

        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Title
        if GUI_BACKEND == "customtkinter":
            title_lbl = ctk.CTkLabel(self.sidebar, text="🎨 PixelArtSmith Studio", font=ctk.CTkFont(size=18, weight="bold"))
        else:
            title_lbl = tk.Label(self.sidebar, text="🎨 PixelArtSmith Studio", font=("Helvetica", 15, "bold"), fg="white", bg="#242424")
        title_lbl.pack(padx=15, pady=(15, 10), anchor="w")

        # Open File Button
        if GUI_BACKEND == "customtkinter":
            self.btn_open = ctk.CTkButton(self.sidebar, text="📁 Open Sprite Image", command=self._on_open_file)
        else:
            self.btn_open = tk.Button(self.sidebar, text="📁 Open Sprite Image", command=self._on_open_file)
        self.btn_open.pack(padx=15, pady=(0, 12), fill="x")

        # Resolution Preset
        self._create_label("Resolution Preset:")
        res_options = [
            "Retro Snapper (32x32px - Pitch 8px)",
            "High-Detail RPG (64x64px - Pitch 4px)",
            "Medium Console (48x48px - Pitch 6px)"
        ]
        self.var_resolution = tk.StringVar(value=res_options[0])
        if GUI_BACKEND == "customtkinter":
            self.opt_resolution = ctk.CTkOptionMenu(self.sidebar, values=res_options, variable=self.var_resolution)
        else:
            self.opt_resolution = ttk.Combobox(self.sidebar, values=res_options, textvariable=self.var_resolution, state="readonly")
        self.opt_resolution.pack(padx=15, pady=(0, 8), fill="x")

        # Palette Selector
        self._create_label("Color Palette & Ramp Mode:")
        palette_options = [
            "snapper-13 (Clean 13-Color Semantic Ramps)",
            "endesga-32 (Fantasy RPG)",
            "dawnbringer-16 (Compact DB16)",
            "pico-8 (16-Color Retro)",
            "sweetie-16 (Pastel Vibrant)",
            "nes-54 (Nintendo Famicom)",
            "snes-classic (Super Nintendo)",
            "sega-genesis (MegaDrive 64)",
            "gameboy-classic (DMG-01 4-Color)",
            "gameboy-pocket (Grayscale)",
            "gameboy-color (GBC 32-Color)",
            "c64-commodore (Commodore 16)",
            "endesga-64 (Rich RPG)",
            "adaptive-12 (K-Means 12)",
            "adaptive-16 (K-Means 16)",
            "none (Raw Grid Only)"
        ]
        self.var_palette_display = tk.StringVar(value=palette_options[0])
        if GUI_BACKEND == "customtkinter":
            self.opt_palette = ctk.CTkOptionMenu(self.sidebar, values=palette_options, variable=self.var_palette_display)
        else:
            self.opt_palette = ttk.Combobox(self.sidebar, values=palette_options, textvariable=self.var_palette_display, state="readonly")
        self.opt_palette.pack(padx=15, pady=(0, 8), fill="x")

        # Max Discrete Colors Slider (Posterization)
        self._create_label("Max Colors per Character (10~16):")
        self.var_max_colors = tk.IntVar(value=13)
        if GUI_BACKEND == "customtkinter":
            self.slider_colors = ctk.CTkSlider(self.sidebar, from_=8, to=24, number_of_steps=16, variable=self.var_max_colors)
            self.lbl_colors_val = ctk.CTkLabel(self.sidebar, textvariable=self.var_max_colors)
        else:
            self.slider_colors = tk.Scale(self.sidebar, from_=8, to=24, orient="horizontal", variable=self.var_max_colors, bg="#242424", fg="white")
            self.lbl_colors_val = tk.Label(self.sidebar, textvariable=self.var_max_colors, bg="#242424", fg="white")
        self.slider_colors.pack(padx=15, fill="x")
        self.lbl_colors_val.pack(padx=15, anchor="e")

        # Scale Factor
        self._create_label("Export Display Scale:")
        scale_options = ["4x (Recommended)", "1x (Raw 1:1 Grid)", "2x", "3x", "6x", "8x"]
        self.var_scale = tk.StringVar(value="4x (Recommended)")
        if GUI_BACKEND == "customtkinter":
            self.opt_scale = ctk.CTkOptionMenu(self.sidebar, values=scale_options, variable=self.var_scale)
        else:
            self.opt_scale = ttk.Combobox(self.sidebar, values=scale_options, textvariable=self.var_scale, state="readonly")
        self.opt_scale.pack(padx=15, pady=(0, 8), fill="x")

        # Toggles
        self.var_remove_bg = tk.BooleanVar(value=True)
        self.var_clean = tk.BooleanVar(value=True)

        if GUI_BACKEND == "customtkinter":
            self.chk_bg = ctk.CTkCheckBox(self.sidebar, text="AI Background Removal", variable=self.var_remove_bg)
            self.chk_clean = ctk.CTkCheckBox(self.sidebar, text="Clean 1px Noise / Orphans", variable=self.var_clean)
        else:
            self.chk_bg = tk.Checkbutton(self.sidebar, text="AI Background Removal", variable=self.var_remove_bg, bg="#242424", fg="white", selectcolor="#444")
            self.chk_clean = tk.Checkbutton(self.sidebar, text="Clean 1px Noise / Orphans", variable=self.var_clean, bg="#242424", fg="white", selectcolor="#444")

        self.chk_bg.pack(padx=15, pady=4, anchor="w")
        self.chk_clean.pack(padx=15, pady=4, anchor="w")

        # Action Buttons
        if GUI_BACKEND == "customtkinter":
            self.btn_process = ctk.CTkButton(self.sidebar, text="⚡ Process True-Grid", fg_color="#1f6aa5", height=36, command=self._on_process)
            self.btn_export = ctk.CTkButton(self.sidebar, text="💾 Export Sprite Sheet + Meta", fg_color="#2e7d32", height=36, command=self._on_export)
        else:
            self.btn_process = tk.Button(self.sidebar, text="⚡ Process True-Grid", bg="#1f6aa5", fg="white", height=2, command=self._on_process)
            self.btn_export = tk.Button(self.sidebar, text="💾 Export Sprite Sheet + Meta", bg="#2e7d32", fg="white", height=2, command=self._on_export)

        self.btn_process.pack(padx=15, pady=(16, 8), fill="x")
        self.btn_export.pack(padx=15, pady=4, fill="x")

        # Status Box
        if GUI_BACKEND == "customtkinter":
            self.lbl_status = ctk.CTkLabel(self.sidebar, text="Ready. Open an AI sprite sheet to start.", wraplength=280)
        else:
            self.lbl_status = tk.Label(self.sidebar, text="Ready. Open an AI sprite sheet to start.", wraplength=280, bg="#242424", fg="#aaa")
        self.lbl_status.pack(padx=15, pady=(12, 10), anchor="w")

        # ---------------------------------------------------------------------
        # Right Preview Tabs (Comparison View / Animation Player)
        # ---------------------------------------------------------------------
        if GUI_BACKEND == "customtkinter":
            self.tabview = ctk.CTkTabview(self.root)
            self.tabview.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
            self.tab_compare = self.tabview.add("🔍 Comparison View")
            self.tab_anim = self.tabview.add("🎬 Live Animation Player")
        else:
            self.tabview = ttk.Notebook(self.root)
            self.tabview.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
            self.tab_compare = tk.Frame(self.tabview, bg="#1e1e1e")
            self.tab_anim = tk.Frame(self.tabview, bg="#1e1e1e")
            self.tabview.add(self.tab_compare, text="🔍 Comparison View")
            self.tabview.add(self.tab_anim, text="🎬 Live Animation Player")

        self._build_comparison_tab()
        self._build_animation_tab()

    def _build_comparison_tab(self):
        self.tab_compare.grid_columnconfigure(0, weight=1)
        self.tab_compare.grid_columnconfigure(1, weight=1)
        self.tab_compare.grid_rowconfigure(1, weight=1)

        if GUI_BACKEND == "customtkinter":
            lbl_orig = ctk.CTkLabel(self.tab_compare, text="Original AI Sprite Sheet", font=ctk.CTkFont(size=13, weight="bold"))
            lbl_proc = ctk.CTkLabel(self.tab_compare, text="True-Grid Matrix Pixel Art (Semantic)", font=ctk.CTkFont(size=13, weight="bold"))
        else:
            lbl_orig = tk.Label(self.tab_compare, text="Original AI Sprite Sheet", font=("Helvetica", 11, "bold"), fg="white", bg="#1e1e1e")
            lbl_proc = tk.Label(self.tab_compare, text="True-Grid Matrix Pixel Art (Semantic)", font=("Helvetica", 11, "bold"), fg="white", bg="#1e1e1e")

        lbl_orig.grid(row=0, column=0, padx=8, pady=(6, 2), sticky="w")
        lbl_proc.grid(row=0, column=1, padx=8, pady=(6, 2), sticky="w")

        self.canvas_orig = tk.Canvas(self.tab_compare, bg="#141414", highlightthickness=1, highlightbackground="#333")
        self.canvas_proc = tk.Canvas(self.tab_compare, bg="#141414", highlightthickness=1, highlightbackground="#333")

        self.canvas_orig.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.canvas_proc.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)

    def _build_animation_tab(self):
        self.tab_anim.grid_columnconfigure(0, weight=1)
        self.tab_anim.grid_rowconfigure(1, weight=1)

        # Animation Controls Bar
        if GUI_BACKEND == "customtkinter":
            ctrl_bar = ctk.CTkFrame(self.tab_anim)
        else:
            ctrl_bar = tk.Frame(self.tab_anim, bg="#2a2a2a")
        ctrl_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=8)

        # Motion Dropdown
        self.var_motion_select = tk.StringVar(value="Motion Row 0")
        self.opt_motion = ctk.CTkOptionMenu(ctrl_bar, values=["Motion Row 0"], variable=self.var_motion_select, command=self._on_motion_change) if GUI_BACKEND == "customtkinter" else ttk.Combobox(ctrl_bar, values=["Motion Row 0"], textvariable=self.var_motion_select)
        self.opt_motion.pack(side="left", padx=10, pady=6)

        # FPS Slider
        self.var_fps = tk.IntVar(value=6)
        if GUI_BACKEND == "customtkinter":
            lbl_fps = ctk.CTkLabel(ctrl_bar, text="FPS:")
            self.slider_fps = ctk.CTkSlider(ctrl_bar, from_=1, to=16, number_of_steps=15, variable=self.var_fps, width=120)
            self.lbl_fps_val = ctk.CTkLabel(ctrl_bar, textvariable=self.var_fps)
        else:
            lbl_fps = tk.Label(ctrl_bar, text="FPS:", bg="#2a2a2a", fg="white")
            self.slider_fps = tk.Scale(ctrl_bar, from_=1, to=16, orient="horizontal", variable=self.var_fps, bg="#2a2a2a", fg="white")
            self.lbl_fps_val = tk.Label(ctrl_bar, textvariable=self.var_fps, bg="#2a2a2a", fg="white")

        lbl_fps.pack(side="left", padx=(15, 2))
        self.slider_fps.pack(side="left", padx=4)
        self.lbl_fps_val.pack(side="left", padx=4)

        # Play / Pause Button
        if GUI_BACKEND == "customtkinter":
            self.btn_play = ctk.CTkButton(ctrl_bar, text="▶ Play Animation", width=120, command=self._toggle_animation)
        else:
            self.btn_play = tk.Button(ctrl_bar, text="▶ Play Animation", command=self._toggle_animation)
        self.btn_play.pack(side="left", padx=15)

        # Animation Display Canvas
        self.canvas_anim = tk.Canvas(self.tab_anim, bg="#111111", highlightthickness=1, highlightbackground="#333")
        self.canvas_anim.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

    def _create_label(self, text: str):
        if GUI_BACKEND == "customtkinter":
            lbl = ctk.CTkLabel(self.sidebar, text=text, font=ctk.CTkFont(size=12))
        else:
            lbl = tk.Label(self.sidebar, text=text, bg="#242424", fg="#ddd", font=("Helvetica", 10))
        lbl.pack(padx=15, pady=(6, 2), anchor="w")
        return lbl

    def _on_open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.webp;*.bmp")]
        )
        if not file_path:
            return

        self.current_image_path = Path(file_path)
        self.raw_image = Image.open(self.current_image_path).convert("RGBA")
        self._show_on_canvas(self.canvas_orig, self.raw_image)
        self.lbl_status.configure(text=f"Loaded: {self.current_image_path.name}")

    def _get_selected_palette_name(self) -> str:
        raw_val = self.var_palette_display.get()
        return raw_val.split()[0].lower()

    def _get_pitch_from_res_preset(self) -> int:
        val = self.var_resolution.get()
        if "32x32" in val or "Pitch 8" in val:
            return 8
        elif "64x64" in val or "Pitch 4" in val:
            return 4
        elif "48x48" in val or "Pitch 6" in val:
            return 6
        return 8

    def _on_process(self):
        if self.raw_image is None:
            messagebox.showwarning("Warning", "Please open an image first.")
            return

        self.lbl_status.configure(text="Processing True-Grid... please wait.")
        self.btn_process.configure(state="disabled")

        def task():
            try:
                palette_name = self._get_selected_palette_name()
                scale_str = self.var_scale.get()
                scale = int(scale_str[0]) if scale_str[0].isdigit() else 4
                remove_bg = self.var_remove_bg.get()
                clean_orphans = self.var_clean.get()
                max_colors = self.var_max_colors.get()
                pitch = self._get_pitch_from_res_preset()

                # 1. Background removal
                if remove_bg:
                    clean_bg_img = self.bg_remover.remove_background(self.raw_image, alpha_threshold=128, defringe=True)
                else:
                    clean_bg_img = PixelCleaner.cleanup_transparency_halos(self.raw_image)

                # 2. Core sub-block sampling (zero-bleed)
                margin = 1 if pitch >= 6 else 0
                grid_img = GridDetector.core_subblock_downsample(
                    clean_bg_img,
                    pitch=pitch,
                    margin=margin
                )

                # 3. Clean orphan pixels
                if clean_orphans:
                    grid_img = PixelCleaner.remove_orphan_pixels(grid_img)

                # 4. Chroma-Weighted Semantic Quantization
                palette_colors: List[str] = []
                if palette_name.startswith("snapper") or palette_name.startswith("adaptive"):
                    grid_img, palette_colors = PixelPosterizer.process_snapper_pipeline(
                        grid_img,
                        max_colors=max_colors,
                        w_chroma=2.0
                    )
                elif palette_name in PALETTES:
                    hex_list = PALETTES[palette_name]
                    if "#000000" not in hex_list and "#000000" not in [h.lower() for h in hex_list]:
                        hex_list = ["#000000"] + hex_list
                    palette_rgb = np.array([hex_to_rgb(h) for h in hex_list], dtype=np.uint8)
                    grid_img, palette_colors = PixelPosterizer.quantize_chroma_weighted(
                        grid_img,
                        palette_rgb=palette_rgb,
                        w_chroma=2.0
                    )

                # 5. 2D Matrix isolation
                isolator = SpriteIsolator(min_area=12, padding=1)
                matrix = isolator.isolate_matrix(grid_img)

                cell_size = SpritePacker.calculate_optimal_cell_size(matrix)

                packed_sheet, meta, std_grid = SpritePacker.pack_matrix_sheet(
                    matrix=matrix,
                    cell_size=cell_size,
                    scale=scale,
                    palette_name=palette_name,
                    palette_colors=palette_colors
                )

                self.processed_sheet = packed_sheet
                self.std_grid = std_grid
                self.metadata = meta

                self.root.after(0, lambda: self._on_process_finished(len(matrix), sum(len(r) for r in matrix), pitch, cell_size, scale, len(palette_colors)))
            except Exception as ex:
                self.root.after(0, lambda: self._on_process_error(str(ex)))

        threading.Thread(target=task, daemon=True).start()

    def _on_process_finished(self, n_rows: int, total_frames: int, pitch: int, cell_size: Tuple[int, int], scale: int, n_colors: int):
        self.btn_process.configure(state="normal")
        if self.processed_sheet:
            self._show_on_canvas(self.canvas_proc, self.processed_sheet)

        motion_options = [f"Motion Row {i} ({len(self.std_grid[i])} frames)" for i in range(len(self.std_grid))]
        if motion_options:
            if GUI_BACKEND == "customtkinter":
                self.opt_motion.configure(values=motion_options)
                self.var_motion_select.set(motion_options[0])
            else:
                self.opt_motion["values"] = motion_options
                self.var_motion_select.set(motion_options[0])

        self.lbl_status.configure(
            text=f"Done! Pitch: {pitch}px | {n_colors} Colors | {n_rows} Motions ({total_frames} frames) | Cell: {cell_size[0]}x{cell_size[1]}px ({scale}x)"
        )

    def _on_process_error(self, err_msg: str):
        self.btn_process.configure(state="normal")
        self.lbl_status.configure(text=f"Error: {err_msg}")
        messagebox.showerror("Processing Error", f"Failed to process sprite sheet:\n{err_msg}")

    def _on_motion_change(self, val: str):
        try:
            self.anim_motion_idx = int(val.split()[2])
        except Exception:
            self.anim_motion_idx = 0
        self.anim_frame_idx = 0
        self._render_current_anim_frame()

    def _toggle_animation(self):
        if not self.std_grid:
            messagebox.showinfo("Notice", "Process an image first to preview animation.")
            return

        if self.anim_running:
            self.anim_running = False
            self.btn_play.configure(text="▶ Play Animation")
            if self.anim_timer_id:
                self.root.after_cancel(self.anim_timer_id)
        else:
            self.anim_running = True
            self.btn_play.configure(text="⏸ Pause Animation")
            self._anim_tick()

    def _anim_tick(self):
        if not self.anim_running or not self.std_grid:
            return

        current_row = self.std_grid[min(self.anim_motion_idx, len(self.std_grid) - 1)]
        if not current_row:
            return

        self._render_current_anim_frame()
        self.anim_frame_idx = (self.anim_frame_idx + 1) % len(current_row)

        fps = max(1, self.var_fps.get())
        interval_ms = int(1000 / fps)
        self.anim_timer_id = self.root.after(interval_ms, self._anim_tick)

    def _render_current_anim_frame(self):
        if not self.std_grid:
            return
        current_row = self.std_grid[min(self.anim_motion_idx, len(self.std_grid) - 1)]
        if not current_row:
            return

        frame = current_row[min(self.anim_frame_idx, len(current_row) - 1)]
        self._show_on_canvas(self.canvas_anim, frame, scale_zoom=3.0)

    def _on_export(self):
        if self.processed_sheet is None:
            messagebox.showwarning("Warning", "No processed sprite sheet to export.")
            return

        out_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            initialfile=f"{self.current_image_path.stem if self.current_image_path else 'sprite'}_pixel_sheet.png"
        )
        if not out_path:
            return

        out_path = Path(out_path)
        self.processed_sheet.save(out_path)

        meta_path = out_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        messagebox.showinfo(
            "Export Success",
            f"Successfully exported:\n- Matrix Sheet: {out_path.name}\n- Agentic Metadata: {meta_path.name}"
        )

    def _show_on_canvas(self, canvas: tk.Canvas, pil_img: Image.Image, scale_zoom: float = 1.0):
        canvas.update_idletasks()
        cw = max(100, canvas.winfo_width())
        ch = max(100, canvas.winfo_height())

        iw, ih = pil_img.size
        ratio = min(cw / iw, ch / ih) * scale_zoom
        disp_w = max(1, int(iw * ratio))
        disp_h = max(1, int(ih * ratio))

        resized = pil_img.resize((disp_w, disp_h), resample=Image.Resampling.NEAREST)
        photo = ImageTk.PhotoImage(resized)

        canvas.delete("all")
        canvas.image = photo
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")


def main_gui():
    """Launch the GUI application."""
    if GUI_BACKEND == "customtkinter":
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        root = ctk.CTk()
    else:
        root = tk.Tk()

    app = PixelArtSmithApp(root)
    root.mainloop()


if __name__ == "__main__":
    main_gui()
