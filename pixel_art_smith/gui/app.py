#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modern Desktop GUI for PixelArtSmith."""

import os
import sys
import threading
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image, ImageTk

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
from ..core.sprite_isolator import SpriteIsolator
from ..core.palette import PaletteQuantizer, PALETTES
from ..core.cleaner import PixelCleaner
from ..core.packer import SpritePacker


class PixelArtSmithApp:
    """Desktop GUI Application for PixelArtSmith."""

    def __init__(self, root):
        self.root = root
        self.root.title("PixelArtSmith - SD Sprite Sheet to Grid-Perfect Pixel Art Studio")
        self.root.geometry("1180x800")
        self.root.minsize(960, 680)

        # State
        self.current_image_path: Optional[Path] = None
        self.raw_image: Optional[Image.Image] = None
        self.processed_sheet: Optional[Image.Image] = None
        self.processed_frames: List[Image.Image] = []
        self.metadata: dict = {}
        self.bg_remover = BackgroundRemover()

        # GUI Setup
        self._build_ui()

    def _build_ui(self):
        # Configure grid
        self.root.grid_columnconfigure(0, weight=0)  # Sidebar
        self.root.grid_columnconfigure(1, weight=1)  # Main display
        self.root.grid_rowconfigure(0, weight=1)

        # ---------------------------------------------------------------------
        # Left Control Sidebar
        # ---------------------------------------------------------------------
        if GUI_BACKEND == "customtkinter":
            self.sidebar = ctk.CTkFrame(self.root, width=320, corner_radius=0)
        else:
            self.sidebar = tk.Frame(self.root, width=320, bg="#2b2b2b")

        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_propagate(False)

        # Title
        if GUI_BACKEND == "customtkinter":
            title_lbl = ctk.CTkLabel(self.sidebar, text="🎨 PixelArtSmith", font=ctk.CTkFont(size=20, weight="bold"))
        else:
            title_lbl = tk.Label(self.sidebar, text="🎨 PixelArtSmith", font=("Helvetica", 16, "bold"), fg="white", bg="#2b2b2b")
        title_lbl.pack(padx=20, pady=(20, 15), anchor="w")

        # Open File Button
        if GUI_BACKEND == "customtkinter":
            self.btn_open = ctk.CTkButton(self.sidebar, text="📁 Open Sprite Image", command=self._on_open_file)
        else:
            self.btn_open = tk.Button(self.sidebar, text="📁 Open Sprite Image", command=self._on_open_file)
        self.btn_open.pack(padx=20, pady=(0, 15), fill="x")

        # Cell Width Slider
        self._create_label("Logical Cell Width (px):")
        self.var_cell_w = tk.IntVar(value=48)
        if GUI_BACKEND == "customtkinter":
            self.slider_w = ctk.CTkSlider(self.sidebar, from_=16, to=128, number_of_steps=14, variable=self.var_cell_w)
            self.lbl_val_w = ctk.CTkLabel(self.sidebar, textvariable=self.var_cell_w)
        else:
            self.slider_w = tk.Scale(self.sidebar, from_=16, to=128, resolution=8, orient="horizontal", variable=self.var_cell_w, bg="#2b2b2b", fg="white")
            self.lbl_val_w = tk.Label(self.sidebar, textvariable=self.var_cell_w, bg="#2b2b2b", fg="white")
        self.slider_w.pack(padx=20, fill="x")
        self.lbl_val_w.pack(padx=20, anchor="e")

        # Cell Height Slider
        self._create_label("Logical Cell Height (px):")
        self.var_cell_h = tk.IntVar(value=64)
        if GUI_BACKEND == "customtkinter":
            self.slider_h = ctk.CTkSlider(self.sidebar, from_=16, to=128, number_of_steps=14, variable=self.var_cell_h)
            self.lbl_val_h = ctk.CTkLabel(self.sidebar, textvariable=self.var_cell_h)
        else:
            self.slider_h = tk.Scale(self.sidebar, from_=16, to=128, resolution=8, orient="horizontal", variable=self.var_cell_h, bg="#2b2b2b", fg="white")
            self.lbl_val_h = tk.Label(self.sidebar, textvariable=self.var_cell_h, bg="#2b2b2b", fg="white")
        self.slider_h.pack(padx=20, fill="x")
        self.lbl_val_h.pack(padx=20, anchor="e")

        # Palette Selector
        self._create_label("Palette Snapping:")
        palette_options = list(PALETTES.keys()) + ["adaptive-16", "adaptive-24", "adaptive-32", "none"]
        self.var_palette = tk.StringVar(value="endesga-32")
        if GUI_BACKEND == "customtkinter":
            self.opt_palette = ctk.CTkOptionMenu(self.sidebar, values=palette_options, variable=self.var_palette)
        else:
            self.opt_palette = ttk.Combobox(self.sidebar, values=palette_options, textvariable=self.var_palette, state="readonly")
        self.opt_palette.pack(padx=20, pady=(0, 10), fill="x")

        # Scale Factor
        self._create_label("Upscale Multiplier (Nearest):")
        scale_options = ["1x (Raw Grid)", "2x", "3x", "4x"]
        self.var_scale = tk.StringVar(value="2x")
        if GUI_BACKEND == "customtkinter":
            self.opt_scale = ctk.CTkOptionMenu(self.sidebar, values=scale_options, variable=self.var_scale)
        else:
            self.opt_scale = ttk.Combobox(self.sidebar, values=scale_options, textvariable=self.var_scale, state="readonly")
        self.opt_scale.pack(padx=20, pady=(0, 10), fill="x")

        # Checkboxes
        self.var_remove_bg = tk.BooleanVar(value=True)
        self.var_clean = tk.BooleanVar(value=True)

        if GUI_BACKEND == "customtkinter":
            self.chk_bg = ctk.CTkCheckBox(self.sidebar, text="AI Background Removal", variable=self.var_remove_bg)
            self.chk_clean = ctk.CTkCheckBox(self.sidebar, text="Clean 1px Orphan Noise", variable=self.var_clean)
        else:
            self.chk_bg = tk.Checkbutton(self.sidebar, text="AI Background Removal", variable=self.var_remove_bg, bg="#2b2b2b", fg="white", selectcolor="#444")
            self.chk_clean = tk.Checkbutton(self.sidebar, text="Clean 1px Orphan Noise", variable=self.var_clean, bg="#2b2b2b", fg="white", selectcolor="#444")

        self.chk_bg.pack(padx=20, pady=5, anchor="w")
        self.chk_clean.pack(padx=20, pady=5, anchor="w")

        # Action Buttons
        if GUI_BACKEND == "customtkinter":
            self.btn_process = ctk.CTkButton(self.sidebar, text="⚡ Process & Preview", fg_color="#1f6aa5", command=self._on_process)
            self.btn_export = ctk.CTkButton(self.sidebar, text="💾 Export Sprite Sheet", fg_color="#2e7d32", command=self._on_export)
        else:
            self.btn_process = tk.Button(self.sidebar, text="⚡ Process & Preview", bg="#1f6aa5", fg="white", command=self._on_process)
            self.btn_export = tk.Button(self.sidebar, text="💾 Export Sprite Sheet", bg="#2e7d32", fg="white", command=self._on_export)

        self.btn_process.pack(padx=20, pady=(20, 10), fill="x")
        self.btn_export.pack(padx=20, pady=5, fill="x")

        # Status Label
        if GUI_BACKEND == "customtkinter":
            self.lbl_status = ctk.CTkLabel(self.sidebar, text="Ready. Open an image to start.", wraplength=280)
        else:
            self.lbl_status = tk.Label(self.sidebar, text="Ready. Open an image to start.", wraplength=280, bg="#2b2b2b", fg="#aaa")
        self.lbl_status.pack(padx=20, pady=(15, 10), anchor="w")

        # ---------------------------------------------------------------------
        # Right Preview Area
        # ---------------------------------------------------------------------
        if GUI_BACKEND == "customtkinter":
            self.main_frame = ctk.CTkFrame(self.root)
        else:
            self.main_frame = tk.Frame(self.root, bg="#1e1e1e")

        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # Header labels
        if GUI_BACKEND == "customtkinter":
            lbl_orig_title = ctk.CTkLabel(self.main_frame, text="Original Image", font=ctk.CTkFont(size=14, weight="bold"))
            lbl_proc_title = ctk.CTkLabel(self.main_frame, text="Grid-Perfect Pixel Art Preview", font=ctk.CTkFont(size=14, weight="bold"))
        else:
            lbl_orig_title = tk.Label(self.main_frame, text="Original Image", font=("Helvetica", 12, "bold"), fg="white", bg="#1e1e1e")
            lbl_proc_title = tk.Label(self.main_frame, text="Grid-Perfect Pixel Art Preview", font=("Helvetica", 12, "bold"), fg="white", bg="#1e1e1e")

        lbl_orig_title.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        lbl_proc_title.grid(row=0, column=1, padx=10, pady=(10, 5), sticky="w")

        # Canvases for side-by-side comparison
        self.canvas_orig = tk.Canvas(self.main_frame, bg="#181818", highlightthickness=1, highlightbackground="#333")
        self.canvas_proc = tk.Canvas(self.main_frame, bg="#181818", highlightthickness=1, highlightbackground="#333")

        self.canvas_orig.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.canvas_proc.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)

    def _create_label(self, text: str):
        if GUI_BACKEND == "customtkinter":
            lbl = ctk.CTkLabel(self.sidebar, text=text, font=ctk.CTkFont(size=12))
        else:
            lbl = tk.Label(self.sidebar, text=text, bg="#2b2b2b", fg="#ddd", font=("Helvetica", 10))
        lbl.pack(padx=20, pady=(8, 2), anchor="w")

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

    def _on_process(self):
        if self.raw_image is None:
            messagebox.showwarning("Warning", "Please open an image first.")
            return

        self.lbl_status.configure(text="Processing... please wait.")
        self.btn_process.configure(state="disabled")

        def task():
            try:
                cell_w = self.var_cell_w.get()
                cell_h = self.var_cell_h.get()
                scale_str = self.var_scale.get()
                scale = int(scale_str[0]) if scale_str[0].isdigit() else 1
                palette_name = self.var_palette.get()
                remove_bg = self.var_remove_bg.get()
                clean_orphans = self.var_clean.get()

                # Step 1: Remove background
                if remove_bg:
                    clean_bg_img = self.bg_remover.remove_background(self.raw_image, alpha_threshold=128, defringe=True)
                else:
                    clean_bg_img = PixelCleaner.cleanup_transparency_halos(self.raw_image)

                # Step 2: Detect frames
                isolator = SpriteIsolator(min_area=300, padding=2)
                detected = isolator.isolate_frames(clean_bg_img)

                # Step 3: Quantizer setup
                quantizer = None
                if palette_name.startswith("adaptive-"):
                    n = int(palette_name.split("-")[1])
                    colors = PaletteQuantizer.extract_adaptive_palette(clean_bg_img, n_colors=n)
                    quantizer = PaletteQuantizer(custom_colors=colors)
                elif palette_name in PALETTES:
                    quantizer = PaletteQuantizer(palette_name=palette_name)

                # Step 4: Process frames
                processed = []
                for frame_raw, bbox in detected:
                    fw, fh = frame_raw.size
                    ratio = min(cell_w / max(1, fw), cell_h / max(1, fh))
                    lw = max(1, int(fw * ratio))
                    lh = max(1, int(fh * ratio))

                    grid_frame = GridDetector.downsample_to_grid(frame_raw, (lw, lh))
                    if clean_orphans:
                        grid_frame = PixelCleaner.remove_orphan_pixels(grid_frame)
                    if quantizer:
                        grid_frame = quantizer.quantize(grid_frame)

                    std_frame = SpritePacker.standardize_frame(grid_frame, (cell_w, cell_h))
                    if scale > 1:
                        std_frame = GridDetector.upscale_nearest(std_frame, scale=scale)
                    processed.append(std_frame)

                scaled_cell = (cell_w * scale, cell_h * scale)
                packed_sheet, meta = SpritePacker.pack_horizontal_sheet(processed, scaled_cell)

                self.processed_sheet = packed_sheet
                self.processed_frames = processed
                self.metadata = meta

                self.root.after(0, lambda: self._on_process_finished(len(processed), scaled_cell))
            except Exception as ex:
                self.root.after(0, lambda: self._on_process_error(str(ex)))

        threading.Thread(target=task, daemon=True).start()

    def _on_process_finished(self, frame_count: int, scaled_cell: Tuple[int, int]):
        self.btn_process.configure(state="normal")
        if self.processed_sheet:
            self._show_on_canvas(self.canvas_proc, self.processed_sheet)
        self.lbl_status.configure(text=f"Done! {frame_count} frames ({scaled_cell[0]}x{scaled_cell[1]}).")

    def _on_process_error(self, err_msg: str):
        self.btn_process.configure(state="normal")
        self.lbl_status.configure(text=f"Error: {err_msg}")
        messagebox.showerror("Processing Error", f"Failed to process image:\n{err_msg}")

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

        # Save metadata JSON adjacent
        meta_path = out_path.with_suffix(".json")
        import json
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        # Export individual frames folder
        frames_dir = out_path.parent / f"{out_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(self.processed_frames):
            frame.save(frames_dir / f"frame_{i:02d}.png")

        messagebox.showinfo("Export Success", f"Exported:\n- Sheet: {out_path.name}\n- Meta: {meta_path.name}\n- Frames: {frames_dir.name}/")

    def _show_on_canvas(self, canvas: tk.Canvas, pil_img: Image.Image):
        canvas.update_idletasks()
        cw = max(100, canvas.winfo_width())
        ch = max(100, canvas.winfo_height())

        # Fit with aspect ratio
        iw, ih = pil_img.size
        ratio = min(cw / iw, ch / ih)
        disp_w = max(1, int(iw * ratio))
        disp_h = max(1, int(ih * ratio))

        # Use nearest neighbor for crisp preview
        resized = pil_img.resize((disp_w, disp_h), resample=Image.Resampling.NEAREST)
        photo = ImageTk.PhotoImage(resized)

        canvas.delete("all")
        canvas.image = photo  # Keep reference
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
