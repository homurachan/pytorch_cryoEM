#!/usr/bin/env python3
"""
v2lite.py
A lightweight EMAN1-v2-style 3-D MRC slice viewer using only:
    tkinter, matplotlib, mrcfile, numpy
(no PyQt/PySide, no scipy)

Controls
--------
Mouse wheel                  : slice +/- 1
Shift + mouse wheel          : slice +/- fast_step (default 5)
Ctrl + mouse wheel           : zoom around cursor
Up / PageUp                  : slice +1
Down / PageDown              : slice -1
Left mouse drag              : pan
Middle mouse click           : display menu (Scale / Brightness / Contrast)
Right mouse drag left/right  : contrast
Right mouse drag up/down     : brightness
X                            : YZ plane, fixed X
Y                            : XZ plane, fixed Y
Z                            : XY plane, fixed Z
Home                         : center slice
A                            : automatic contrast from current slice
R                            : reset contrast, gamma, scale and view
G                            : enter gamma value

Coordinate convention
---------------------
MRC data are indexed directly as data[z, y, x].  No centered or reversed Z
remapping is applied.  The status bar reports raw voxel indices, so the center
of an even-sized (nx, ny, nz) volume is shown as (nx/2, ny/2, nz/2).
"""

from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
import mrcfile

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import PowerNorm
from matplotlib.figure import Figure


class VolumeSource:
    """Full-RAM or mmap-backed access to a 3-D MRC volume."""

    def __init__(self, path: str, load_mode: str = "full", cache_slices: int = 7):
        self.path = str(path)
        self.load_mode = load_mode
        self.cache_slices = max(1, int(cache_slices))
        self._mrc = None
        self._data = None
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

        if load_mode == "full":
            with mrcfile.open(self.path, mode="r", permissive=True) as m:
                if m.data.ndim != 3:
                    raise ValueError(f"Expected a 3-D MRC volume, got {m.data.shape}")
                self.shape = tuple(int(v) for v in m.data.shape)  # (nz, ny, nx)
                self.voxel_size = self._read_voxel_size(m)
                self._data = np.array(m.data, copy=True)
        elif load_mode == "cache":
            self._mrc = mrcfile.mmap(self.path, mode="r", permissive=True)
            if self._mrc.data.ndim != 3:
                shape = self._mrc.data.shape
                self._mrc.close()
                self._mrc = None
                raise ValueError(f"Expected a 3-D MRC volume, got {shape}")
            self.shape = tuple(int(v) for v in self._mrc.data.shape)
            self.voxel_size = self._read_voxel_size(self._mrc)
        else:
            raise ValueError("load_mode must be 'full' or 'cache'")

        self.nz, self.ny, self.nx = self.shape

    @staticmethod
    def _read_voxel_size(mrc):
        try:
            vals = [
                float(mrc.voxel_size.x),
                float(mrc.voxel_size.y),
                float(mrc.voxel_size.z),
            ]
        except Exception:
            vals = [1.0, 1.0, 1.0]
        return tuple(v if np.isfinite(v) and v > 0 else 1.0 for v in vals)

    @property
    def data(self):
        return self._data if self.load_mode == "full" else self._mrc.data

    def axis_length(self, axis: str) -> int:
        return {"X": self.nx, "Y": self.ny, "Z": self.nz}[axis.upper()]

    def plane_name(self, axis: str) -> str:
        return {"X": "YZ", "Y": "XZ", "Z": "XY"}[axis.upper()]

    def plane_aspect(self, axis: str) -> float:
        """Displayed vertical voxel size / displayed horizontal voxel size."""
        vx, vy, vz = self.voxel_size
        return {
            "X": vz / vy,  # horizontal y, vertical z
            "Y": vz / vx,  # horizontal x, vertical z
            "Z": vy / vx,  # horizontal x, vertical y
        }[axis.upper()]

    def _extract_view(self, axis: str, index: int):
        """Extract a plane directly from MRC/NumPy data[z, y, x]."""
        axis = axis.upper()
        if axis == "Z":      # XY plane, fixed z
            return self.data[index, :, :]
        if axis == "Y":      # XZ plane, fixed y; rows are z
            return self.data[:, index, :]
        if axis == "X":      # YZ plane, fixed x; rows are z
            return self.data[:, :, index]
        raise ValueError(axis)

    def get_slice(self, axis: str, index: int) -> np.ndarray:
        axis = axis.upper()
        n = self.axis_length(axis)
        index = int(index)
        if not 0 <= index < n:
            raise IndexError(index)

        if self.load_mode == "full":
            return np.asarray(self._extract_view(axis, index))

        key = (axis, index)
        if key in self._cache:
            arr = self._cache.pop(key)
            self._cache[key] = arr
            return arr

        arr = np.asarray(self._extract_view(axis, index), dtype=np.float32).copy()
        self._cache[key] = arr
        while len(self._cache) > self.cache_slices:
            self._cache.popitem(last=False)
        return arr

    def estimate_global_contrast(self):
        z_indices = np.linspace(0, self.nz - 1, min(9, self.nz), dtype=int)
        sy = max(1, math.ceil(self.ny / 256))
        sx = max(1, math.ceil(self.nx / 256))
        chunks = []

        for z in z_indices:
            a = self.get_slice("Z", int(z))[::sy, ::sx].ravel()
            a = a[np.isfinite(a)]
            if a.size:
                chunks.append(a)

        if not chunks:
            return 0.0, 1.0

        vals = np.concatenate(chunks)
        lo, hi = np.percentile(vals, [0.5, 99.5])
        lo, hi = float(lo), float(hi)

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals))
            if not np.isfinite(std) or std <= 0:
                std = 1.0
            lo, hi = mean - 2 * std, mean + 2 * std
        return lo, hi

    def close(self):
        self._cache.clear()
        if self._mrc is not None:
            self._mrc.close()
            self._mrc = None


class V2LiteViewer:
    def __init__(self, root, volume, initial_plane="Z", fast_step=5):
        self.root = root
        self.volume = volume
        self.fast_step = int(fast_step)

        self.axis = initial_plane.upper()
        self.indices = {
            "X": volume.nx // 2,
            "Y": volume.ny // 2,
            "Z": volume.nz // 2,
        }
        self.current_slice: Optional[np.ndarray] = None

        self.initial_vmin, self.initial_vmax = volume.estimate_global_contrast()
        self.vmin, self.vmax = self.initial_vmin, self.initial_vmax
        # Reference display window used by the numeric brightness/contrast menu.
        # A resets this reference to the current-slice auto contrast; R restores
        # the original global reference.
        self.ref_vmin, self.ref_vmax = self.initial_vmin, self.initial_vmax
        self.gamma = 1.0

        # Display scale: horizontal screen pixels per displayed voxel.
        # 1.0 is true native 1:1 display scale.
        self.scale = 1.0
        self._resize_job = None

        # Drag state
        self.drag_mode = None
        self.drag_start_px = (0.0, 0.0)
        self.drag_start_xlim = None
        self.drag_start_ylim = None
        self.drag_start_vmin = self.vmin
        self.drag_start_vmax = self.vmax

        self._build_ui()
        self._connect_events()
        self._show_current_slice(center_image=True)

        # Tk must finish geometry negotiation before exact 1:1 scaling can be
        # calculated from the real Axes pixel size.
        self.root.after_idle(self._initialize_native_scale)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    @property
    def index(self):
        return self.indices[self.axis]

    def _initial_slice_geometry(self):
        """Return displayed (height, width) in voxels for the initial plane."""
        if self.axis == "Z":
            return self.volume.ny, self.volume.nx
        if self.axis == "Y":
            return self.volume.nz, self.volume.nx
        if self.axis == "X":
            return self.volume.nz, self.volume.ny
        raise ValueError(self.axis)

    def _build_ui(self):
        # Grid instead of packing an expanding canvas before the status bar.
        # Row 0 may grow/shrink; row 1 is always reserved for status.
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0, minsize=28)
        self.root.grid_columnconfigure(0, weight=1)

        main = ttk.Frame(self.root)
        main.grid(row=0, column=0, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # Start near the native 1:1 image size instead of an arbitrary 800x800
        # Matplotlib window.  Small volumes therefore do not open surrounded by
        # a huge empty border, while very large volumes are capped to the screen.
        h_vox, w_vox = self._initial_slice_geometry()
        plane_aspect = max(float(self.volume.plane_aspect(self.axis)), 1.0e-12)
        desired_canvas_w = max(240, int(round(w_vox)))
        desired_canvas_h = max(180, int(round(h_vox * plane_aspect)))

        screen_w = max(int(self.root.winfo_screenwidth()), 640)
        screen_h = max(int(self.root.winfo_screenheight()), 480)
        max_canvas_w = max(320, int(screen_w * 0.80))
        max_canvas_h = max(240, int(screen_h * 0.78) - 32)
        canvas_w = min(desired_canvas_w, max_canvas_w)
        canvas_h = min(desired_canvas_h, max_canvas_h)

        dpi = 100
        self.figure = Figure(
            figsize=(canvas_w / dpi, canvas_h / dpi),
            dpi=dpi,
            facecolor="black",
        )
        # Let the image axes fill the entire canvas.  Any region outside the
        # actual image is black rather than the distracting Matplotlib white.
        self.figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.set_facecolor("black")
        # We explicitly control x/y data limits to implement exact screen scale.
        self.ax.set_aspect("auto")

        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")
        self.canvas_widget.configure(takefocus=True, background="black",
                                     highlightthickness=0)

        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=1, column=0, sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)

        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            anchor="w",
            padding=(6, 4),
        )
        self.status_label.grid(row=0, column=0, sticky="ew")

        # Explicit initial client size: the canvas follows the volume instead
        # of Matplotlib's old fixed 8-inch default.  The status bar remains in
        # its own reserved grid row.
        self.root.geometry(f"{canvas_w}x{canvas_h + 30}")

        # Middle-button context menu.  Labels are refreshed before each popup
        # so they also act as a compact readout of the current display settings.
        self.display_menu = tk.Menu(self.root, tearoff=False)
        self.display_menu.add_command(label="Scale...", command=self._ask_scale)
        self.display_menu.add_separator()
        self.display_menu.add_command(
            label="Brightness...", command=self._ask_brightness
        )
        self.display_menu.add_command(
            label="Contrast...", command=self._ask_contrast
        )

        self.image_artist = None

    def _connect_events(self):
        c = self.canvas.mpl_connect
        c("scroll_event", self._on_scroll)
        c("key_press_event", self._on_key)
        c("motion_notify_event", self._on_motion)
        c("button_press_event", self._on_button_press)
        c("button_release_event", self._on_button_release)
        c("resize_event", self._on_canvas_resize)

        self.canvas_widget.bind("<Enter>", lambda _e: self.canvas_widget.focus_set())
        self.canvas_widget.bind(
            "<Button-1>", lambda _e: self.canvas_widget.focus_set(), add="+"
        )

    def _norm(self):
        if not np.isfinite(self.vmin):
            self.vmin = 0.0
        if not np.isfinite(self.vmax) or self.vmax <= self.vmin:
            self.vmax = self.vmin + 1.0
        if not np.isfinite(self.gamma) or self.gamma <= 0:
            self.gamma = 1.0
        return PowerNorm(
            gamma=self.gamma, vmin=self.vmin, vmax=self.vmax, clip=True
        )

    def _image_center(self):
        if self.current_slice is None:
            return 0.0, 0.0
        h, w = self.current_slice.shape
        return (w - 1) / 2.0, (h - 1) / 2.0

    def _current_view_center(self):
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    def _initialize_native_scale(self):
        self.canvas.draw()
        self._apply_scale(1.0, center=self._image_center(), redraw=True)

    def _show_current_slice(self, center_image=False):
        self.current_slice = self.volume.get_slice(self.axis, self.index)

        if self.image_artist is None:
            self.image_artist = self.ax.imshow(
                self.current_slice,
                cmap="gray",
                origin="lower",
                interpolation="nearest",
                norm=self._norm(),
                aspect="auto",
            )
            center_image = True
        else:
            self.image_artist.set_data(self.current_slice)
            self.image_artist.set_norm(self._norm())

        self._update_title()
        self._update_status(None)

        # Maintain the chosen numeric scale on every plane/slice change.  Plane
        # switches re-center on the new image; ordinary slice changes preserve pan.
        center = self._image_center() if center_image else self._current_view_center()
        self._apply_scale(self.scale, center=center, redraw=False)
        self.canvas.draw_idle()

    def _apply_scale(self, scale, center=None, redraw=True):
        """
        Apply an exact numeric display scale.

        scale=1 means one screen pixel per horizontal image voxel.  For
        anisotropic MRC voxels, the vertical screen scale is multiplied by the
        physical voxel-size ratio of the selected plane.
        """
        try:
            scale = float(scale)
        except Exception:
            return
        if not np.isfinite(scale) or scale <= 0:
            return

        self.scale = scale
        if center is None:
            center = self._current_view_center()
        cx, cy = center

        # Axes bbox is expressed in display pixels after a draw/resize.
        bbox = self.ax.bbox
        pixel_w = max(float(bbox.width), 1.0)
        pixel_h = max(float(bbox.height), 1.0)

        aspect = max(float(self.volume.plane_aspect(self.axis)), 1.0e-12)
        visible_w = pixel_w / self.scale
        visible_h = pixel_h / (self.scale * aspect)

        self.ax.set_xlim(cx - visible_w / 2.0, cx + visible_w / 2.0)
        self.ax.set_ylim(cy - visible_h / 2.0, cy + visible_h / 2.0)

        self._update_title()
        if redraw:
            self.canvas.draw_idle()

    def _on_canvas_resize(self, _event):
        # Keep screen pixels/voxel constant while only the amount of visible
        # image changes.  after_idle coalesces a burst of resize callbacks.
        if self._resize_job is not None:
            try:
                self.root.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.root.after_idle(self._reapply_scale_after_resize)

    def _reapply_scale_after_resize(self):
        self._resize_job = None
        if not self.root.winfo_exists():
            return
        center = self._current_view_center()
        self._apply_scale(self.scale, center=center, redraw=True)

    def _update_title(self):
        filename = Path(self.volume.path).name
        plane = self.volume.plane_name(self.axis)
        mode = (
            "full"
            if self.volume.load_mode == "full"
            else f"cache:{self.volume.cache_slices}"
        )
        self.root.title(
            f"v2lite — {filename} — {self.axis}/{plane} "
            f"{self.index}/{self.volume.axis_length(self.axis)} — "
            f"scale {self.scale:.3g}x — {mode}"
        )

    def _pixel_size_text(self):
        vx, vy, vz = self.volume.voxel_size
        if max(abs(vx - vy), abs(vx - vz), abs(vy - vz)) < 1e-6:
            return f"{vx:.4g} Å/pix"
        return f"{vx:.4g}×{vy:.4g}×{vz:.4g} Å/vox"

    def _display_to_voxel(self, ix, iy):
        """Return raw MRC voxel indices (x, y, z)."""
        if self.axis == "Z":      # XY, fixed z
            return ix, iy, self.index
        if self.axis == "Y":      # XZ: horizontal x, vertical z
            return ix, self.index, iy
        if self.axis == "X":      # YZ: horizontal y, vertical z
            return self.index, ix, iy
        raise ValueError(self.axis)

    def _update_status(self, event):
        n = self.volume.axis_length(self.axis)
        prefix = f"{self.axis} {self.index}/{n}"
        tail = self._pixel_size_text()

        if (
            event is None
            or event.inaxes is not self.ax
            or event.xdata is None
            or event.ydata is None
            or self.current_slice is None
        ):
            self.status_var.set(
                f"{prefix}    x=-- y=-- z=--    density=--    "
                f"scale={self.scale:.3g}x    {tail}"
            )
            return

        ix = int(math.floor(float(event.xdata) + 0.5))
        iy = int(math.floor(float(event.ydata) + 0.5))
        h, w = self.current_slice.shape
        if not (0 <= ix < w and 0 <= iy < h):
            self.status_var.set(
                f"{prefix}    x=-- y=-- z=--    density=--    "
                f"scale={self.scale:.3g}x    {tail}"
            )
            return

        x, y, z = self._display_to_voxel(ix, iy)
        density = float(self.current_slice[iy, ix])
        dtext = f"{density:.6g}" if np.isfinite(density) else str(density)

        self.status_var.set(
            f"{prefix}    "
            f"x={x} y={y} z={z}    "
            f"density={dtext}    scale={self.scale:.3g}x    {tail}"
        )

    def _change_slice(self, delta):
        n = self.volume.axis_length(self.axis)
        new_idx = int(np.clip(self.index + int(delta), 0, n - 1))
        if new_idx != self.index:
            self.indices[self.axis] = new_idx
            self._show_current_slice(center_image=False)

    @staticmethod
    def _mods(key):
        return "" if key is None else str(key).lower()

    def _on_scroll(self, event):
        mods = self._mods(event.key)
        if "control" in mods or "ctrl" in mods:
            self._zoom_at_cursor(event)
            return

        step = self.fast_step if "shift" in mods else 1
        direction = 1 if float(event.step) > 0 else -1
        self._change_slice(direction * step)

    def _on_key(self, event):
        key = "" if event.key is None else str(event.key).lower()

        if key in ("up", "pageup"):
            self._change_slice(+1)
        elif key in ("down", "pagedown"):
            self._change_slice(-1)
        elif key in ("x", "y", "z"):
            self._switch_axis(key.upper())
        elif key == "home":
            self.indices[self.axis] = self.volume.axis_length(self.axis) // 2
            self._show_current_slice(center_image=True)
        elif key == "a":
            self._auto_contrast()
        elif key == "r":
            self._reset_display()
        elif key == "g":
            self._ask_gamma()

    def _switch_axis(self, axis):
        if axis != self.axis:
            self.axis = axis
            self._show_current_slice(center_image=True)

    def _auto_contrast(self):
        vals = np.asarray(self.current_slice)
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            return
        lo, hi = np.percentile(vals, [0.5, 99.5])
        lo, hi = float(lo), float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals))
            if not np.isfinite(std) or std <= 0:
                std = 1.0
            lo, hi = mean - 2 * std, mean + 2 * std

        self.vmin, self.vmax = lo, hi
        self.ref_vmin, self.ref_vmax = lo, hi
        self.image_artist.set_norm(self._norm())
        self.canvas.draw_idle()

    def _reset_display(self):
        self.vmin, self.vmax = self.initial_vmin, self.initial_vmax
        self.ref_vmin, self.ref_vmax = self.initial_vmin, self.initial_vmax
        self.gamma = 1.0
        self.image_artist.set_norm(self._norm())
        self._apply_scale(1.0, center=self._image_center(), redraw=True)

    def _ask_gamma(self):
        value = simpledialog.askfloat(
            "Gamma",
            "Gamma (> 0):",
            initialvalue=self.gamma,
            minvalue=0.01,
            parent=self.root,
        )
        if value is not None:
            self.gamma = float(value)
            self.image_artist.set_norm(self._norm())
            self.canvas.draw_idle()
            self.canvas_widget.focus_set()

    def _reference_window(self):
        center = 0.5 * (self.ref_vmin + self.ref_vmax)
        width = max(self.ref_vmax - self.ref_vmin, np.finfo(float).eps)
        return center, width

    def _brightness_percent(self):
        ref_center, ref_width = self._reference_window()
        current_center = 0.5 * (self.vmin + self.vmax)
        # Positive means visually brighter: shift the density window downward.
        return (ref_center - current_center) / ref_width * 100.0

    def _contrast_factor(self):
        _, ref_width = self._reference_window()
        current_width = max(self.vmax - self.vmin, np.finfo(float).eps)
        # >1 means stronger contrast (narrower density window).
        return ref_width / current_width

    def _ask_scale(self):
        value = simpledialog.askfloat(
            "Display scale",
            "Scale (1.0 = one screen pixel per voxel):",
            initialvalue=self.scale,
            minvalue=0.01,
            parent=self.root,
        )
        if value is not None:
            self._apply_scale(
                float(value), center=self._current_view_center(), redraw=True
            )
            self._update_status(None)
        self.canvas_widget.focus_set()

    def _ask_brightness(self):
        value = simpledialog.askfloat(
            "Brightness",
            "Brightness (%)\n"
            "0 = automatic/reference level\n"
            "positive = brighter, negative = darker:",
            initialvalue=self._brightness_percent(),
            parent=self.root,
        )
        if value is not None:
            ref_center, ref_width = self._reference_window()
            current_width = max(self.vmax - self.vmin, np.finfo(float).eps)
            center = ref_center - (float(value) / 100.0) * ref_width
            self.vmin = center - 0.5 * current_width
            self.vmax = center + 0.5 * current_width
            self.image_artist.set_norm(self._norm())
            self.canvas.draw_idle()
        self.canvas_widget.focus_set()

    def _ask_contrast(self):
        value = simpledialog.askfloat(
            "Contrast",
            "Contrast factor (> 0)\n"
            "1.0 = automatic/reference contrast\n"
            ">1 = stronger, <1 = weaker:",
            initialvalue=self._contrast_factor(),
            minvalue=0.001,
            parent=self.root,
        )
        if value is not None:
            ref_center, ref_width = self._reference_window()
            current_center = 0.5 * (self.vmin + self.vmax)
            width = ref_width / float(value)
            # Preserve the current brightness/level when changing contrast.
            self.vmin = current_center - 0.5 * width
            self.vmax = current_center + 0.5 * width
            self.image_artist.set_norm(self._norm())
            self.canvas.draw_idle()
        self.canvas_widget.focus_set()

    def _popup_display_menu(self, event):
        # Keep current values visible directly in the menu.
        self.display_menu.entryconfigure(
            0, label=f"Scale...    {self.scale:.3g}x"
        )
        self.display_menu.entryconfigure(
            2, label=f"Brightness...    {self._brightness_percent():.1f}%"
        )
        self.display_menu.entryconfigure(
            3, label=f"Contrast...    {self._contrast_factor():.3g}x"
        )
        gui_event = getattr(event, "guiEvent", None)
        x_root = getattr(gui_event, "x_root", None)
        y_root = getattr(gui_event, "y_root", None)
        if x_root is None or y_root is None:
            x_root = self.root.winfo_pointerx()
            y_root = self.root.winfo_pointery()
        try:
            self.display_menu.tk_popup(int(x_root), int(y_root))
        finally:
            self.display_menu.grab_release()

    def _zoom_at_cursor(self, event):
        if (
            event.inaxes is not self.ax
            or event.xdata is None
            or event.ydata is None
        ):
            return

        x, y = float(event.xdata), float(event.ydata)
        old_scale = self.scale
        new_scale = old_scale * (1.2 if float(event.step) > 0 else 1.0 / 1.2)
        new_scale = max(new_scale, 0.01)

        # Preserve the data point below the cursor after changing numeric scale.
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        rx = (x - x0) / (x1 - x0)
        ry = (y - y0) / (y1 - y0)

        bbox = self.ax.bbox
        aspect = max(float(self.volume.plane_aspect(self.axis)), 1.0e-12)
        visible_w = max(float(bbox.width), 1.0) / new_scale
        visible_h = max(float(bbox.height), 1.0) / (new_scale * aspect)

        new_x0 = x - rx * visible_w
        new_y0 = y - ry * visible_h
        self.scale = new_scale
        self.ax.set_xlim(new_x0, new_x0 + visible_w)
        self.ax.set_ylim(new_y0, new_y0 + visible_h)
        self._update_title()
        self._update_status(None)
        self.canvas.draw_idle()

    def _on_button_press(self, event):
        self.canvas_widget.focus_set()
        if event.inaxes is not self.ax:
            return

        if event.button == MouseButton.MIDDLE:
            # Middle mouse opens the compact display-settings menu.
            self._popup_display_menu(event)
            return

        self.drag_start_px = (float(event.x), float(event.y))
        if event.button == MouseButton.LEFT:
            self.drag_mode = "pan"
            self.drag_start_xlim = self.ax.get_xlim()
            self.drag_start_ylim = self.ax.get_ylim()
        elif event.button == MouseButton.RIGHT:
            # One right-button gesture preserves both old display controls:
            # horizontal component -> contrast; vertical component -> brightness.
            self.drag_mode = "display"
            self.drag_start_vmin, self.drag_start_vmax = self.vmin, self.vmax

    def _on_button_release(self, event):
        self.drag_mode = None
        self.drag_start_xlim = None
        self.drag_start_ylim = None

    def _on_motion(self, event):
        self._update_status(event)
        if self.drag_mode is None or event.x is None or event.y is None:
            return

        sx, sy = self.drag_start_px
        dx = float(event.x) - sx
        dy = float(event.y) - sy

        if self.drag_mode == "pan":
            self._drag_pan(dx, dy)
        elif self.drag_mode == "display":
            self._drag_brightness_contrast(dx, dy)

    def _drag_pan(self, dx_px, dy_px):
        if self.drag_start_xlim is None or self.drag_start_ylim is None:
            return
        bbox = self.ax.bbox
        if bbox.width <= 0 or bbox.height <= 0:
            return

        x0, x1 = self.drag_start_xlim
        y0, y1 = self.drag_start_ylim
        dx_data = dx_px / bbox.width * (x1 - x0)
        dy_data = dy_px / bbox.height * (y1 - y0)

        self.ax.set_xlim(x0 - dx_data, x1 - dx_data)
        self.ax.set_ylim(y0 - dy_data, y1 - dy_data)
        self.canvas.draw_idle()

    def _drag_brightness_contrast(self, dx_px, dy_px):
        bbox = self.ax.bbox
        if bbox.width <= 0 or bbox.height <= 0:
            return

        lo0, hi0 = self.drag_start_vmin, self.drag_start_vmax
        center0 = 0.5 * (lo0 + hi0)
        width0 = max(hi0 - lo0, np.finfo(float).eps)

        # Right -> higher contrast (narrower window).
        width = max(
            width0 * math.exp(-dx_px * 0.01),
            np.finfo(float).eps * 100,
        )

        # Up -> brighter: shift density window down.  Use the original width so
        # brightness sensitivity is stable even while contrast is changing.
        shift = -(dy_px / bbox.height) * width0 * 2.0
        center = center0 + shift

        self.vmin = center - 0.5 * width
        self.vmax = center + 0.5 * width
        self.image_artist.set_norm(self._norm())
        self.canvas.draw_idle()

    def close(self):
        if self._resize_job is not None:
            try:
                self.root.after_cancel(self._resize_job)
            except Exception:
                pass
            self._resize_job = None
        try:
            self.volume.close()
        finally:
            self.root.destroy()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="EMAN1-v2-style 3-D MRC slice viewer without Qt."
    )
    p.add_argument(
        "mrc",
        nargs="?",
        help="3-D MRC volume. If omitted, a Tk file chooser is shown.",
    )
    p.add_argument(
        "--load-mode",
        choices=("full", "cache"),
        default="full",
        help="full=load entire volume into RAM; cache=mmap + small slice LRU cache",
    )
    p.add_argument(
        "--cache-slices",
        type=int,
        default=7,
        help="Number of visited slices retained in cache mode (default: 7)",
    )
    p.add_argument(
        "--plane",
        choices=("X", "Y", "Z", "x", "y", "z"),
        default="Z",
        help="Initial slicing axis: Z=XY, Y=XZ, X=YZ",
    )
    p.add_argument(
        "--fast-step",
        type=int,
        choices=(5, 10),
        default=5,
        help="Shift+wheel step: 5 or 10",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    root = tk.Tk()
    root.withdraw()

    path = args.mrc
    if not path:
        path = filedialog.askopenfilename(
            parent=root,
            title="Open 3-D MRC volume",
            filetypes=[
                ("MRC files", "*.mrc *.map *.mrcs"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            root.destroy()
            return 0

    try:
        volume = VolumeSource(
            path,
            load_mode=args.load_mode,
            cache_slices=args.cache_slices,
        )
    except Exception as exc:
        messagebox.showerror("Failed to open MRC", str(exc), parent=root)
        root.destroy()
        return 1

    root.deiconify()
    try:
        V2LiteViewer(
            root,
            volume,
            initial_plane=args.plane.upper(),
            fast_step=args.fast_step,
        )
    except Exception as exc:
        volume.close()
        messagebox.showerror("Viewer error", str(exc), parent=root)
        root.destroy()
        return 1

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
