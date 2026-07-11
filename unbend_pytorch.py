#!/usr/bin/env python3
"""
unbend_torch.py

Standalone PyTorch reimplementation of the core motion-correction and
patch-based distortion-correction path in unbend.cpp.

This version follows the supplied cubicspline/bicubicspline/MovieFrameSpline
and Image::Distortion implementations, including explicit ghost-control
boundaries, dose-coordinate temporal splines, continuous CC-map refinement,
and cisTEM-style forward scan-line distortion resampling.

This low-memory build keeps the algorithm unchanged while streaming the largest
FFT/correlation/shift operations in small frame batches.  Patch alignment and
R1 patch_pix generation use the same source-stack order as unbend.cpp: raw
frames are full-frame Fourier-shifted first and then Fourier-resampled to the
output size before patch clipping.  The final raw-domain correction likewise
applies the global Fourier shift before local Image::Distortion-style warping.
The spline R1 residual-refinement path follows unbend.cpp more strictly: it
regenerates the zero-padded patch_pix stack, applies the fitted R0 spline
shifts to that patch stack, generates 32x32 leave-one-out CC maps with the
original coarse-search B-factor, converts them to interpolating bicubic
coefficients, and optimizes the R1 controls on those continuous CC surfaces.

Dependencies:
    numpy
    torch
    mrcfile
    tifffile          # only required for .tif/.tiff input or TIFF gain/dark
    imagecodecs       # required by tifffile for compressed TIFF variants

Supported input:
    2-D MRC image, 3-D MRC movie stack (Z, Y, X), or TIFF movie stack
    (.tif/.tiff). Multi-page TIFF stacks are read frame-by-frame.

Not supported without extra decoders:
    EER movie decoding and DM4 gain/dark references.

The implementation keeps the original algorithmic structure:
  1. dark/gain correction, optional outlier replacement and anisotropic
     magnification correction;
  2. Fourier binning and iterative full-frame correlation alignment;
  3. patch alignment;
  4. linear, quadratic, or tensor cubic B-spline motion-field fitting;
  5. nonlinear frame resampling;
  6. cisTEM-compatible exposure weighting and optional noise-power restore.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F


Tensor = torch.Tensor
EPS = 1.0e-12


def log(message: str) -> None:
    print(message, flush=True)


def odd_at_least(value: int, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


def round_nearest(value: float) -> int:
    return int(math.floor(value + 0.5))


def available_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(requested)


def torch_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def release_memory(device: torch.device) -> None:
    """Return cached CUDA blocks to the allocator after large tensors are deleted."""
    if device.type == "cuda":
        torch.cuda.empty_cache()


def is_tiff_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".tif", ".tiff"}


def _import_tifffile():
    try:
        import tifffile  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "TIFF input requires tifffile. Install TIFF support with:\n"
            "    pip install tifffile imagecodecs\n"
            "imagecodecs is needed for many compressed TIFF movies."
        ) from exc
    return tifffile


def _decode_tiff_array(read_callable, path: str | Path) -> np.ndarray:
    try:
        return np.asarray(read_callable())
    except Exception as exc:
        raise RuntimeError(
            f"Could not decode TIFF data from {path}. If the TIFF is compressed, install imagecodecs:\n"
            "    pip install imagecodecs\n"
            f"Original decoder error: {exc}"
        ) from exc


def read_reference_image(path: Optional[str], expected_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not path:
        return None
    ref_path = Path(path)
    if is_tiff_path(ref_path):
        tifffile = _import_tifffile()
        with tifffile.TiffFile(str(ref_path)) as tif:
            if len(tif.pages) > 1 and len(tif.pages[0].shape) == 2:
                arr = _decode_tiff_array(lambda: tif.pages[0].asarray(), ref_path)
            else:
                arr = _decode_tiff_array(lambda: tif.asarray(), ref_path)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
        if arr.ndim != 2 or tuple(arr.shape) != expected_shape:
            raise ValueError(f"Reference {path} has shape {arr.shape}; expected {expected_shape}")
        return np.asarray(arr, dtype=np.float32)

    with mrcfile.open(str(ref_path), permissive=True) as mrc:
        arr = np.asarray(mrc.data)
        if arr.ndim == 3:
            if arr.shape[0] != 1:
                raise ValueError(f"Reference {path} must contain one image, got shape {arr.shape}")
            arr = arr[0]
        if arr.ndim != 2 or tuple(arr.shape) != expected_shape:
            raise ValueError(f"Reference {path} has shape {arr.shape}; expected {expected_shape}")
        return np.asarray(arr, dtype=np.float32)


def infer_mrc_pixel_size(mrc: mrcfile.mrcfile.MrcFile) -> Optional[float]:
    try:
        vx = float(mrc.voxel_size.x)
        vy = float(mrc.voxel_size.y)
        if vx > 0 and vy > 0:
            return 0.5 * (vx + vy)
    except Exception:
        pass
    return None


def edge_mean(image: Tensor) -> Tensor:
    if image.ndim != 2:
        raise ValueError("edge_mean expects a 2-D tensor")
    if image.shape[0] < 2 or image.shape[1] < 2:
        return image.mean()
    return torch.cat((image[0], image[-1], image[1:-1, 0], image[1:-1, -1])).mean()


def replace_outliers_with_local_mean(image: Tensor, sigma_threshold: float = 12.0) -> Tensor:
    """Approximate cisTEM ReplaceOutliersWithMean(12)."""
    mean = image.mean()
    sigma = image.std(unbiased=False).clamp_min(1.0e-6)
    bad = torch.abs(image - mean) > sigma_threshold * sigma
    if not bool(bad.any()):
        return image
    local = F.avg_pool2d(image[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    return torch.where(bad, local, image)


def make_magnification_grid(
    height: int,
    width: int,
    angle_degrees: float,
    major_scale: float,
    minor_scale: float,
    device: torch.device,
) -> Tensor:
    """Create a backward sampling grid for anisotropic magnification correction."""
    if major_scale <= 0 or minor_scale <= 0:
        raise ValueError("Magnification scales must be positive")
    theta = math.radians(angle_degrees)
    c, s = math.cos(theta), math.sin(theta)
    rotation = torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=device)
    scale = torch.diag(torch.tensor([major_scale, minor_scale], dtype=torch.float32, device=device))
    matrix = rotation @ scale @ rotation.T

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    points = torch.stack((xx - cx, yy - cy), dim=-1)
    source = points @ matrix.T
    sx = source[..., 0] + cx
    sy = source[..., 1] + cy
    gx = 2.0 * sx / max(width - 1, 1) - 1.0
    gy = 2.0 * sy / max(height - 1, 1) - 1.0
    return torch.stack((gx, gy), dim=-1)[None]


def apply_grid_with_mean_fill(image: Tensor, grid: Tensor) -> Tensor:
    fill = edge_mean(image)
    centered = image - fill
    warped = F.grid_sample(
        centered[None, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    return warped + fill


class MovieSource:
    """MRC or TIFF movie with on-demand preprocessing.

    MRC files are memory-mapped through mrcfile.  Multi-page TIFF files are kept
    open and decoded page-by-page through tifffile, which delegates compressed
    data to imagecodecs when required.  Some single-page shaped TIFF stacks can
    only be decoded as a whole by tifffile; those are supported via an internal
    cache, but multi-page TIFF is preferred for large compressed movies.
    """

    def __init__(
        self,
        filename: str,
        device: torch.device,
        gain_filename: Optional[str] = None,
        dark_filename: Optional[str] = None,
        replace_outliers: bool = True,
        magnification_angle: float = 0.0,
        magnification_major: float = 1.0,
        magnification_minor: float = 1.0,
    ) -> None:
        ext = Path(filename).suffix.lower()
        if ext in {".eer", ".dm4"}:
            raise ValueError(f"{ext} input is not supported. Convert the movie to MRC or TIFF first.")

        self.filename = filename
        self.device = device
        self.kind = "tiff" if is_tiff_path(filename) else "mrc"
        self._mrc = None
        self._tiff = None
        self._tiff_mode = ""
        self._tiff_series = None
        self._tiff_stack_cache: Optional[np.ndarray] = None
        self._tiff_frame_axis = 0
        self._tiff_y_axis = 1
        self._tiff_x_axis = 2
        self.header_pixel_size: Optional[float] = None

        if self.kind == "mrc":
            self._mrc = mrcfile.mmap(filename, mode="r", permissive=True)
            data = self._mrc.data
            if data.ndim == 2:
                self._shape = (1, int(data.shape[0]), int(data.shape[1]))
            elif data.ndim == 3:
                self._shape = tuple(int(v) for v in data.shape)
            else:
                raise ValueError(f"Input MRC must be 2-D or 3-D, got shape {data.shape}")
            self.n_frames, self.height, self.width = self._shape
            self.header_pixel_size = infer_mrc_pixel_size(self._mrc)
        else:
            tifffile = _import_tifffile()
            self._tiff = tifffile.TiffFile(filename)
            if len(self._tiff.pages) == 0:
                raise ValueError(f"TIFF file {filename} contains no pages")
            first_page = self._tiff.pages[0]
            first_shape = tuple(int(v) for v in first_page.shape)
            if len(self._tiff.pages) > 1 and len(first_shape) == 2:
                self._tiff_mode = "pages"
                self.n_frames = int(len(self._tiff.pages))
                self.height, self.width = first_shape
            else:
                self._tiff_series = self._tiff.series[0]
                shape = tuple(int(v) for v in self._tiff_series.shape)
                axes = getattr(self._tiff_series, "axes", "") or ""
                if len(shape) == 2:
                    self._tiff_mode = "single2d"
                    self.n_frames, self.height, self.width = 1, shape[0], shape[1]
                elif len(shape) == 3 and "Y" in axes and "X" in axes:
                    y_axis = axes.index("Y")
                    x_axis = axes.index("X")
                    frame_axes = [i for i in range(3) if i not in (y_axis, x_axis)]
                    if len(frame_axes) != 1 or axes[frame_axes[0]] == "S":
                        raise ValueError(
                            f"Unsupported TIFF series axes {axes!r} with shape {shape}. "
                            "Expected a movie stack with one non-spatial frame axis and Y/X axes."
                        )
                    self._tiff_mode = "series3d"
                    self._tiff_frame_axis = frame_axes[0]
                    self._tiff_y_axis = y_axis
                    self._tiff_x_axis = x_axis
                    self.n_frames = shape[self._tiff_frame_axis]
                    self.height = shape[self._tiff_y_axis]
                    self.width = shape[self._tiff_x_axis]
                    log(
                        "  note: TIFF is stored as a single shaped series; tifffile may decode "
                        "the whole stack once and cache it for frame access"
                    )
                elif len(shape) == 3 and shape[-2] > 1 and shape[-1] > 1 and shape[0] > 1:
                    # Fallback for simple shaped stacks without useful axes metadata.
                    self._tiff_mode = "series3d"
                    self._tiff_frame_axis = 0
                    self._tiff_y_axis = 1
                    self._tiff_x_axis = 2
                    self.n_frames, self.height, self.width = shape
                    log(
                        "  note: TIFF has no explicit frame/Y/X axes; interpreting shape "
                        f"{shape} as (frames, height, width)"
                    )
                else:
                    raise ValueError(
                        f"Unsupported TIFF layout in {filename}: series shape {shape}, "
                        f"axes {axes!r}, first page shape {first_shape}"
                    )
            self._shape = (int(self.n_frames), int(self.height), int(self.width))

        expected = (self.height, self.width)
        gain_np = read_reference_image(gain_filename, expected)
        dark_np = read_reference_image(dark_filename, expected)
        self.gain = None if gain_np is None else torch.as_tensor(gain_np.copy(), device=self.device)
        self.dark = None if dark_np is None else torch.as_tensor(dark_np.copy(), device=self.device)
        self.replace_outliers = replace_outliers
        self.mag_angle = magnification_angle
        self.mag_major = magnification_major
        self.mag_minor = magnification_minor
        self.mag_grid: Optional[Tensor] = None
        if (
            abs(self.mag_angle) > 1.0e-8
            or abs(self.mag_major - 1.0) > 1.0e-8
            or abs(self.mag_minor - 1.0) > 1.0e-8
        ):
            self.mag_grid = make_magnification_grid(
                self.height,
                self.width,
                self.mag_angle,
                self.mag_major,
                self.mag_minor,
                self.device,
            )

    def close(self) -> None:
        if self._mrc is not None:
            self._mrc.close()
        if self._tiff is not None:
            self._tiff.close()

    def __enter__(self) -> "MovieSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _tiff_frame(self, index: int) -> np.ndarray:
        if self._tiff is None:
            raise RuntimeError("TIFF file is not open")
        if self._tiff_mode == "pages":
            return _decode_tiff_array(lambda: self._tiff.pages[index].asarray(), self.filename)
        if self._tiff_mode == "single2d":
            if index != 0:
                raise IndexError(index)
            return _decode_tiff_array(lambda: self._tiff.asarray(), self.filename)
        if self._tiff_mode == "series3d":
            if self._tiff_stack_cache is None:
                assert self._tiff_series is not None
                arr = _decode_tiff_array(lambda: self._tiff_series.asarray(), self.filename)
                axes_order = (self._tiff_frame_axis, self._tiff_y_axis, self._tiff_x_axis)
                if axes_order != (0, 1, 2):
                    arr = np.moveaxis(arr, axes_order, (0, 1, 2))
                self._tiff_stack_cache = np.asarray(arr)
            return self._tiff_stack_cache[index]
        raise RuntimeError(f"Unknown TIFF mode {self._tiff_mode!r}")

    def frame(self, index: int, remove_mean: bool = True) -> Tensor:
        if not 0 <= index < self.n_frames:
            raise IndexError(index)
        if self.kind == "mrc":
            assert self._mrc is not None
            raw = self._mrc.data if self._mrc.data.ndim == 2 else self._mrc.data[index]
        else:
            raw = self._tiff_frame(index)
        arr = np.array(raw, dtype=np.float32, copy=True, order="C")
        if arr.ndim != 2:
            raise ValueError(f"Frame {index} from {self.filename} is not 2-D after decoding; shape {arr.shape}")
        image = torch.as_tensor(arr, device=self.device)
        if self.dark is not None:
            image = image - self.dark
        if self.gain is not None:
            image = image * self.gain
        if self.replace_outliers:
            image = replace_outliers_with_local_mean(image)
        if self.mag_grid is not None:
            image = apply_grid_with_mean_fill(image, self.mag_grid)
        if remove_mean:
            image = image - image.mean()
        return image.contiguous()


def centered_fourier_resample(images: Tensor, out_height: int, out_width: int) -> Tensor:
    """Fourier crop/pad while preserving real-space pixel amplitudes."""
    single = images.ndim == 2
    if single:
        images = images[None]
    if images.ndim != 3:
        raise ValueError("centered_fourier_resample expects [H,W] or [N,H,W]")
    n, in_height, in_width = images.shape
    if (in_height, in_width) == (out_height, out_width):
        # Preserve the original no-op behavior: return the same tensor/view.
        return images[0] if single else images

    spectrum = torch.fft.fftshift(torch.fft.fft2(images), dim=(-2, -1))
    output = torch.zeros(
        (n, out_height, out_width), dtype=spectrum.dtype, device=images.device
    )

    copy_h = min(in_height, out_height)
    copy_w = min(in_width, out_width)
    in_y0 = (in_height - copy_h) // 2
    in_x0 = (in_width - copy_w) // 2
    out_y0 = (out_height - copy_h) // 2
    out_x0 = (out_width - copy_w) // 2
    output[:, out_y0 : out_y0 + copy_h, out_x0 : out_x0 + copy_w] = spectrum[
        :, in_y0 : in_y0 + copy_h, in_x0 : in_x0 + copy_w
    ]
    result = torch.fft.ifft2(torch.fft.ifftshift(output, dim=(-2, -1))).real
    result *= float(out_height * out_width) / float(in_height * in_width)
    return result[0] if single else result


def centered_fourier_resample_stack_chunked(
    images: Tensor,
    out_height: int,
    out_width: int,
    batch_size: int = 1,
    clone_if_same: bool = False,
) -> Tensor:
    """Low-memory stack wrapper around centered_fourier_resample.

    The numerical operation for each frame is unchanged; only execution order is
    streamed so a full complex FFT stack is never resident at once.
    """
    if images.ndim != 3:
        raise ValueError("centered_fourier_resample_stack_chunked expects [N,H,W]")
    n, in_height, in_width = images.shape
    if (in_height, in_width) == (out_height, out_width):
        return images.clone() if clone_if_same else images
    batch_size = max(1, int(batch_size))
    out = torch.empty((n, out_height, out_width), dtype=torch.float32, device=images.device)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        out[start:end] = centered_fourier_resample(images[start:end], out_height, out_width)
    return out


def build_output_stack(
    movie: MovieSource,
    out_height: int,
    out_width: int,
    progress_every: int = 10,
) -> Tensor:
    stack = torch.empty(
        (movie.n_frames, out_height, out_width),
        dtype=torch.float32,
        device=movie.device,
    )
    for i in range(movie.n_frames):
        frame = movie.frame(i, remove_mean=True)
        stack[i] = centered_fourier_resample(frame, out_height, out_width)
    #    if progress_every > 0 and ((i + 1) % progress_every == 0 or i + 1 == movie.n_frames):
    #        log(f"  preprocessed {i + 1}/{movie.n_frames} frames")
    return stack




def build_patch_source_stack_from_raw(
    movie: MovieSource,
    out_height: int,
    out_width: int,
    global_x_output_px: Tensor,
    global_y_output_px: Tensor,
    progress_every: int = 10,
) -> Tensor:
    """Build the patch-source stack using unbend.cpp's raw-stack order.

    Original unbend keeps a raw/super-resolution Fourier stack, applies the
    full-frame PhaseShift to that raw stack, resizes it to the output/image_stack
    dimensions, and only then clips patches.  This function mirrors that order
    in a streaming way:

        raw preprocessed frame -> Fourier global shift at raw scale
        -> Fourier resample to output size -> DC removal.

    The returned stack is in real space and output pixels, ready for
    patch_trimming/extract_patch.
    """
    stack = torch.empty(
        (movie.n_frames, out_height, out_width),
        dtype=torch.float32,
        device=movie.device,
    )
    scale_x = movie.width / out_width
    scale_y = movie.height / out_height
    for i in range(movie.n_frames):
        raw = movie.frame(i, remove_mean=True)
        shifted = phase_shift_stack(
            raw,
            torch.tensor([float(global_x_output_px[i].item()) * scale_x], dtype=torch.float32, device=movie.device),
            torch.tensor([float(global_y_output_px[i].item()) * scale_y], dtype=torch.float32, device=movie.device),
            batch_size=1,
        )
        out = centered_fourier_resample(shifted, out_height, out_width)
        stack[i] = out - out.mean()
    #    if progress_every > 0 and ((i + 1) % progress_every == 0 or i + 1 == movie.n_frames):
    #        log(f"  built patch-source frame {i + 1}/{movie.n_frames}")
    return stack.contiguous()


def rfft_frequency_grids(height: int, width: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32)[:, None]
    fx = torch.fft.rfftfreq(width, device=device, dtype=torch.float32)[None, :]
    return fy, fx


def _phase_shift_stack_lowmem(stack: Tensor, shifts_x: Tensor, shifts_y: Tensor) -> Tensor:
    """Shift a small frame batch using the same Fourier phase formula as before."""
    if stack.ndim != 3:
        raise ValueError("_phase_shift_stack_lowmem expects [N,H,W]")
    n, height, width = stack.shape
    shifts_x = shifts_x.to(device=stack.device, dtype=torch.float32).reshape(n)
    shifts_y = shifts_y.to(device=stack.device, dtype=torch.float32).reshape(n)
    fy, fx = rfft_frequency_grids(height, width, stack.device)
    phase_arg = -2.0 * math.pi * (
        shifts_x[:, None, None] * fx[None] + shifts_y[:, None, None] * fy[None]
    )
    phase = torch.polar(torch.ones_like(phase_arg), phase_arg)
    return torch.fft.irfft2(torch.fft.rfft2(stack) * phase, s=(height, width))


def phase_shift_stack(
    stack: Tensor,
    shifts_x: Tensor,
    shifts_y: Tensor,
    batch_size: int = 1,
) -> Tensor:
    """Shift image content by (+x,+y) pixels using the Fourier shift theorem.

    The old implementation formed a full [N,H,W/2+1] phase tensor. This
    version uses the same phase formula, but only for the current batch.
    """
    single = stack.ndim == 2
    if single:
        stack = stack[None]
        shifts_x = shifts_x.reshape(1)
        shifts_y = shifts_y.reshape(1)
    if stack.ndim != 3:
        raise ValueError("phase_shift_stack expects [H,W] or [N,H,W]")
    n = stack.shape[0]
    batch_size = max(1, int(batch_size))
    out = torch.empty_like(stack)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        out[start:end] = _phase_shift_stack_lowmem(
            stack[start:end], shifts_x[start:end], shifts_y[start:end]
        )
    return out[0] if single else out


def phase_shift_stack_inplace(
    stack: Tensor,
    shifts_x: Tensor,
    shifts_y: Tensor,
    batch_size: int = 1,
) -> Tensor:
    """In-place batched Fourier shift for independent frames.

    Each frame depends only on itself, so overwriting completed chunks is
    algorithmically identical to creating a second full stack and assigning it.
    """
    if stack.ndim != 3:
        raise ValueError("phase_shift_stack_inplace expects [N,H,W]")
    n = stack.shape[0]
    batch_size = max(1, int(batch_size))
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        shifted = _phase_shift_stack_lowmem(
            stack[start:end], shifts_x[start:end], shifts_y[start:end]
        )
        stack[start:end].copy_(shifted)
        del shifted
    return stack


def bfactor_filter_rfft(height: int, width: int, unitless_bfactor: float, device: torch.device) -> Tensor:
    fy, fx = rfft_frequency_grids(height, width, device)
    radius_sq = fx.square() + fy.square()
    return torch.exp(-0.25 * float(unitless_bfactor) * radius_sq)


def central_cross_mask_rfft(
    height: int,
    width: int,
    vertical_half_width: int,
    horizontal_half_width: int,
    device: torch.device,
) -> Tensor:
    mask = torch.ones((height, width // 2 + 1), dtype=torch.float32, device=device)
    if vertical_half_width > 0:
        mask[:, : vertical_half_width + 1] = 0.0
    if horizontal_half_width > 0:
        rows = torch.arange(height, device=device)
        signed_rows = torch.where(rows <= height // 2, rows, rows - height)
        mask[torch.abs(signed_rows) <= horizontal_half_width, :] = 0.0
    mask[0, 0] = 0.0
    return mask


def correlation_search_indices(
    height: int,
    width: int,
    inner_radius: float,
    outer_radius: float,
    device: torch.device,
) -> Tensor:
    """Flat indices matching the old search mask, without materializing HxW bools."""
    if outer_radius < 0:
        raise ValueError("outer_radius must be non-negative")
    maxr = int(math.ceil(float(outer_radius)))
    min_signed_y = -(height - height // 2 - 1)
    max_signed_y = height // 2
    min_signed_x = -(width - width // 2 - 1)
    max_signed_x = width // 2
    y0, y1 = max(-maxr, min_signed_y), min(maxr, max_signed_y)
    x0, x1 = max(-maxr, min_signed_x), min(maxr, max_signed_x)
    ys = torch.arange(y0, y1 + 1, dtype=torch.int64, device=device)
    xs = torch.arange(x0, x1 + 1, dtype=torch.int64, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    radius_sq = yy.to(torch.float32).square() + xx.to(torch.float32).square()
    keep = radius_sq <= float(outer_radius) ** 2
    if inner_radius > 0:
        keep &= radius_sq >= float(inner_radius) ** 2
    dy = yy[keep]
    dx = xx[keep]
    if dy.numel() == 0:
        raise RuntimeError("Empty correlation search region; check inner/outer shift radii")
    py = torch.where(dy >= 0, dy, dy + height)
    px = torch.where(dx >= 0, dx, dx + width)
    return py * width + px


def estimate_correlation_shifts(
    references: Tensor,
    targets: Tensor,
    unitless_bfactor: float,
    inner_radius: float,
    outer_radius: float,
    mask_central_cross: bool,
    vertical_half_width: int,
    horizontal_half_width: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return shifts to apply to targets so that they align to references."""
    n, height, width = targets.shape
    ref_fft = torch.fft.rfft2(references)
    target_fft = torch.fft.rfft2(targets)
    ref_fft.mul_(bfactor_filter_rfft(height, width, unitless_bfactor, targets.device))
    if mask_central_cross:
        ref_fft.mul_(central_cross_mask_rfft(
            height,
            width,
            vertical_half_width,
            horizontal_half_width,
            targets.device,
        ))
    correlation = torch.fft.irfft2(ref_fft * torch.conj(target_fft), s=(height, width))
    search_index = correlation_search_indices(height, width, inner_radius, outer_radius, targets.device)
    # Gather only legal search pixels instead of allocating a full masked copy.
    search_values = correlation.reshape(n, -1).index_select(1, search_index)
    flat_index = search_index[search_values.argmax(dim=1)]
    py = torch.div(flat_index, width, rounding_mode="floor")
    px = flat_index % width
    batch = torch.arange(n, device=targets.device)

    left = correlation[batch, py, (px - 1) % width]
    center = correlation[batch, py, px]
    right = correlation[batch, py, (px + 1) % width]
    up = correlation[batch, (py - 1) % height, px]
    down = correlation[batch, (py + 1) % height, px]

    denom_x = left - 2.0 * center + right
    denom_y = up - 2.0 * center + down
    sub_x = torch.where(
        torch.abs(denom_x) > 1.0e-12,
        0.5 * (left - right) / denom_x,
        torch.zeros_like(center),
    ).clamp(-1.0, 1.0)
    sub_y = torch.where(
        torch.abs(denom_y) > 1.0e-12,
        0.5 * (up - down) / denom_y,
        torch.zeros_like(center),
    ).clamp(-1.0, 1.0)

    signed_x = torch.where(px <= width // 2, px, px - width).to(torch.float32)
    signed_y = torch.where(py <= height // 2, py, py - height).to(torch.float32)
    return signed_x + sub_x, signed_y + sub_y, center

def running_window_bounds(n: int, index: int, window: int) -> Tuple[int, int]:
    if window <= 1:
        return index, index
    window = odd_at_least(window, 3)
    half = (window - 1) // 2
    start = index - half
    end = index + half
    if start < 0:
        end -= start
        start = 0
    if end >= n:
        start -= end - (n - 1)
        end = n - 1
    return max(start, 0), min(end, n - 1)


def running_sum_for_indices(stack: Tensor, window: int, indices: Tensor) -> Tensor:
    """Construct running sums only for requested frames.

    This matches the original cisTEM loop order for each running average and
    avoids a full-stack prefix/cumsum allocation.
    """
    if indices.ndim != 1:
        indices = indices.reshape(-1)
    if window <= 1:
        return stack.index_select(0, indices.to(torch.long))
    n, height, width = stack.shape
    out = torch.empty((indices.numel(), height, width), dtype=stack.dtype, device=stack.device)
    for row, idx_t in enumerate(indices.to(torch.long)):
        idx = int(idx_t.item())
        start, end = running_window_bounds(n, idx, window)
        acc = torch.zeros((height, width), dtype=stack.dtype, device=stack.device)
        for frame in range(start, end + 1):
            acc.add_(stack[frame])
        out[row] = acc
    return out


def running_sum_stack(stack: Tensor, window: int) -> Tensor:
    indices = torch.arange(stack.shape[0], device=stack.device)
    return running_sum_for_indices(stack, window, indices)

def normalize_batch_size(batch_size: int, n_frames: int) -> int:
    return max(1, min(int(batch_size), int(n_frames)))


def running_sum_chunk(stack: Tensor, window: int, start: int, end: int) -> Tensor:
    indices = torch.arange(start, end, dtype=torch.long, device=stack.device)
    return running_sum_for_indices(stack, window, indices)


def running_sum_indices(stack: Tensor, window: int, indices: Tensor) -> Tensor:
    return running_sum_for_indices(stack, window, indices.to(torch.long))



def polynomial_smooth(values: Tensor, degree: int = 4) -> Tensor:
    n = values.numel()
    if n <= 2:
        return values.clone()
    degree = min(degree, n - 1)
    t = torch.linspace(-1.0, 1.0, n, dtype=torch.float64, device=values.device)
    design = torch.stack([t ** p for p in range(degree + 1)], dim=1)
    coeff = torch.linalg.lstsq(design, values.to(torch.float64)[:, None]).solution[:, 0]
    return (design @ coeff).to(values.dtype)


def savitzky_golay_linear(values: Tensor, window: int) -> Tensor:
    """Local degree-1 least-squares smoother with edge-aware windows."""
    n = values.numel()
    if window >= n or n < 3:
        return values.clone()
    window = odd_at_least(window, 3)
    half = window // 2
    out = torch.empty_like(values)
    indices = torch.arange(n, dtype=torch.float64, device=values.device)
    y64 = values.to(torch.float64)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        if end - start < window:
            if start == 0:
                end = min(n, window)
            else:
                start = max(0, n - window)
        x = indices[start:end] - float(i)
        design = torch.stack((torch.ones_like(x), x), dim=1)
        coeff = torch.linalg.lstsq(design, y64[start:end, None]).solution[:, 0]
        out[i] = coeff[0].to(values.dtype)
    return out


def robust_outlier_mask(x: Tensor, y: Tensor, smooth_x: Tensor, smooth_y: Tensor) -> Tensor:
    residual = torch.sqrt((x - smooth_x).square() + (y - smooth_y).square())
    median = residual.median()
    mad = torch.abs(residual - median).median().clamp_min(1.0e-6)
    robust_sigma = 1.4826 * mad
    return residual > median + 4.5 * robust_sigma


@dataclass
class AlignmentOptions:
    max_iterations: int
    unitless_bfactor: float
    inner_radius: float
    outer_radius: float
    convergence_threshold: float
    running_average: int
    savitzky_golay_window: int
    use_smoothed_shifts: bool
    mask_central_cross: bool
    vertical_mask_size: int
    horizontal_mask_size: int
    verbose: bool = True
    batch_size: int = 1


def _estimate_alignment_shifts_for_indices(
    stack: Tensor,
    reference_sum: Tensor,
    indices: Tensor,
    options: AlignmentOptions,
    unitless_bfactor: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    target = running_sum_for_indices(stack, options.running_average, indices)
    references = reference_sum - target
    sx, sy, _ = estimate_correlation_shifts(
        references,
        target,
        options.unitless_bfactor if unitless_bfactor is None else unitless_bfactor,
        options.inner_radius,
        options.outer_radius,
        options.mask_central_cross,
        options.vertical_mask_size,
        options.horizontal_mask_size,
    )
    del target, references
    return sx, sy


def iterative_align(
    stack: Tensor,
    options: AlignmentOptions,
    initial_x: Optional[Tensor] = None,
    initial_y: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    n = stack.shape[0]
    total_x = torch.zeros(n, dtype=torch.float32, device=stack.device) if initial_x is None else initial_x.clone()
    total_y = torch.zeros(n, dtype=torch.float32, device=stack.device) if initial_y is None else initial_y.clone()
    middle = n // 2
    batch_size = normalize_batch_size(options.batch_size, n)

    for iteration in range(1, options.max_iterations + 1):
        reference_sum = stack.sum(dim=0, keepdim=True)
        current_x = torch.empty(n, dtype=torch.float32, device=stack.device)
        current_y = torch.empty(n, dtype=torch.float32, device=stack.device)

        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            target_chunk = running_sum_chunk(stack, options.running_average, start, end)
            ref_chunk = reference_sum - target_chunk
            sx, sy, _ = estimate_correlation_shifts(
                ref_chunk,
                target_chunk,
                options.unitless_bfactor,
                options.inner_radius,
                options.outer_radius,
                options.mask_central_cross,
                options.vertical_mask_size,
                options.horizontal_mask_size,
            )
            current_x[start:end] = sx
            current_y[start:end] = sy
            del target_chunk, ref_chunk, sx, sy
        del reference_sum

        absolute_x = total_x + current_x
        absolute_y = total_y + current_y
        if options.inner_radius > 0:
            smooth_x = polynomial_smooth(absolute_x, 4)
            smooth_y = polynomial_smooth(absolute_y, 4)
            current_x = smooth_x - total_x
            current_y = smooth_y - total_y
        elif options.use_smoothed_shifts:
            smooth_x = savitzky_golay_linear(absolute_x, options.savitzky_golay_window)
            smooth_y = savitzky_golay_linear(absolute_y, options.savitzky_golay_window)
            current_x = smooth_x - total_x
            current_y = smooth_y - total_y
        elif n >= 5:
            smooth_x = savitzky_golay_linear(absolute_x, min(5, odd_at_least(n - 1 if n % 2 == 0 else n)))
            smooth_y = savitzky_golay_linear(absolute_y, min(5, odd_at_least(n - 1 if n % 2 == 0 else n)))
            outliers = robust_outlier_mask(absolute_x, absolute_y, smooth_x, smooth_y)
            if bool(outliers.any()):
                idx = torch.nonzero(outliers, as_tuple=False)[:, 0]
                reference_sum = stack.sum(dim=0, keepdim=True)
                target_chunk = running_sum_indices(stack, options.running_average, idx)
                ref_chunk = reference_sum - target_chunk
                rx, ry, _ = estimate_correlation_shifts(
                    ref_chunk,
                    target_chunk,
                    options.unitless_bfactor * 2.0,
                    options.inner_radius,
                    options.outer_radius,
                    options.mask_central_cross,
                    options.vertical_mask_size,
                    options.horizontal_mask_size,
                )
                current_x[idx] = rx
                current_y[idx] = ry
                del reference_sum, target_chunk, ref_chunk, rx, ry

        current_x = current_x - current_x[middle]
        current_y = current_y - current_y[middle]
        magnitude = torch.sqrt(current_x.square() + current_y.square())
        max_shift = float(magnitude.max().item())
        phase_shift_stack_inplace(stack, current_x, current_y, batch_size=batch_size)
        total_x += current_x
        total_y += current_y

        if options.verbose:
            log(f"    iteration {iteration:2d}: max incremental shift = {max_shift:.4f} px")
        if max_shift <= options.convergence_threshold:
            break
    return stack, total_x, total_y

def choose_prebin(output_pixel_size: float) -> int:
    return max(1, round_nearest(5.0 / output_pixel_size))


def global_align(
    output_stack: Tensor,
    output_pixel_size: float,
    minimum_shift_angstrom: float,
    maximum_shift_angstrom: float,
    termination_angstrom: float,
    bfactor_angstrom_sq: float,
    max_iterations: int,
    running_average: int,
    sg_window: int,
    smooth_global: bool,
    mask_central_cross: bool,
    vertical_mask_size: int,
    horizontal_mask_size: int,
    batch_size: int = 1,
) -> Tuple[Tensor, Tensor, Tensor]:
    n, out_h, out_w = output_stack.shape
    batch_size = normalize_batch_size(batch_size, n)
    prebin = choose_prebin(output_pixel_size)
    pre_h = max(16, round_nearest(out_h / prebin))
    pre_w = max(16, round_nearest(out_w / prebin))
    actual_prebin = 0.5 * (out_h / pre_h + out_w / pre_w)
    align_pixel = output_pixel_size * actual_prebin
    log(
        f"Global alignment: output {out_w}x{out_h}, pre-alignment {pre_w}x{pre_h}, "
        f"alignment pixel {align_pixel:.4f} A"
    )
    same_scale_alignment = (pre_h, pre_w) == (out_h, out_w)
    if same_scale_alignment:
        # Align output_stack itself in-place.  The unaligned output stack is not
        # used again, so this avoids a second full-size aligned copy.
        working = output_stack
    else:
        working = centered_fourier_resample_stack_chunked(
            output_stack, pre_h, pre_w, batch_size=batch_size
        )
    minimum_px = max(1.01, minimum_shift_angstrom / align_pixel)
    maximum_px = maximum_shift_angstrom / align_pixel
    termination_px = max(1.0 if actual_prebin > 1.0 else 0.0, termination_angstrom / align_pixel)
    unitless_b = bfactor_angstrom_sq / (align_pixel * align_pixel)

    initial = AlignmentOptions(
        max_iterations=1,
        unitless_bfactor=unitless_b,
        inner_radius=minimum_px,
        outer_radius=maximum_px,
        convergence_threshold=termination_px,
        running_average=running_average,
        savitzky_golay_window=sg_window,
        use_smoothed_shifts=True,
        mask_central_cross=mask_central_cross,
        vertical_mask_size=vertical_mask_size,
        horizontal_mask_size=horizontal_mask_size,
        batch_size=batch_size,
    )
    log("  initial full-frame search")
    working, sx, sy = iterative_align(working, initial)

    main = AlignmentOptions(
        max_iterations=max_iterations,
        unitless_bfactor=unitless_b,
        inner_radius=0.0,
        outer_radius=maximum_px,
        convergence_threshold=termination_px,
        running_average=running_average,
        savitzky_golay_window=sg_window,
        use_smoothed_shifts=smooth_global,
        mask_central_cross=mask_central_cross,
        vertical_mask_size=vertical_mask_size,
        horizontal_mask_size=horizontal_mask_size,
        batch_size=batch_size,
    )
    log("  main full-frame refinement")
    working, sx, sy = iterative_align(working, main, sx, sy)

    scale_x = out_w / pre_w
    scale_y = out_h / pre_h
    sx_out = sx * scale_x
    sy_out = sy * scale_y

    if same_scale_alignment:
        aligned_output = working
    else:
        del working
        release_memory(output_stack.device)
        aligned_output = phase_shift_stack_inplace(
            output_stack, sx_out, sy_out, batch_size=batch_size
        )

    if actual_prebin > 1.0001:
        final_opts = AlignmentOptions(
            max_iterations=max_iterations,
            unitless_bfactor=bfactor_angstrom_sq / (output_pixel_size * output_pixel_size),
            inner_radius=0.0,
            outer_radius=maximum_shift_angstrom / output_pixel_size,
            convergence_threshold=1.0,
            running_average=running_average,
            savitzky_golay_window=5,
            use_smoothed_shifts=smooth_global,
            mask_central_cross=mask_central_cross,
            vertical_mask_size=vertical_mask_size,
            horizontal_mask_size=horizontal_mask_size,
            batch_size=batch_size,
        )
        log("  final output-scale refinement")
        aligned_output, sx_out, sy_out = iterative_align(
            aligned_output, final_opts, sx_out, sy_out
        )
    return aligned_output, sx_out, sy_out


def auto_patch_geometry(
    height: int,
    width: int,
    output_pixel_size: float,
    model: str,
    requested_x: int,
    requested_y: int,
    requested_size: int,
) -> Tuple[int, int, int, np.ndarray]:
    """Reproduce the patch geometry constructed in unbend.cpp."""
    if requested_size > 0:
        patch_size = int(requested_size)
    elif output_pixel_size < 0.5:
        patch_size = 1024
    elif output_pixel_size > 2.0:
        patch_size = max(16, int(512.0 / output_pixel_size))
        if patch_size % 16:
            patch_size = 16 * (patch_size // 16 + 1)
    else:
        patch_size = 512

    if requested_x > 0 and requested_y > 0:
        nx, ny = int(requested_x), int(requested_y)
        patch_size = max(patch_size, width // nx, height // ny)
        if model == "spline" and (nx < 4 or ny < 4):
            raise ValueError("Spline distortion requires at least 4 x 4 patches")
    else:
        nx = int(math.ceil(width / float(patch_size)))
        ny = int(math.ceil(height / float(patch_size)))
        nx = max(2, nx)
        ny = max(2, ny)
        if model == "spline":
            nx = max(4, nx)
            ny = max(4, ny)

    step_x = round_nearest(width / float(nx) / 2.0)
    step_y = round_nearest(height / float(ny) / 2.0)
    x_centers = np.array([i * step_x * 2 + step_x for i in range(nx)], dtype=np.float64)
    y_centers = np.empty(ny, dtype=np.float64)
    for i in range(ny):
        y_centers[ny - i - 1] = height - i * step_y * 2 - step_y
    centers = np.array([(x, y) for y in y_centers for x in x_centers], dtype=np.float64)
    return nx, ny, int(patch_size), centers


def extract_patch(
    stack: Tensor,
    center_x: float,
    center_y: float,
    size: int,
    mean_padding: bool,
) -> Tensor:
    """Clip a square patch, using per-frame mean padding as in cisTEM ClipInto."""
    if stack.ndim != 3:
        raise ValueError("extract_patch expects [frames,height,width]")
    n, height, width = stack.shape
    # cisTEM receives an integer center offset relative to the image center.
    center_ix = round_nearest(center_x)
    center_iy = round_nearest(center_y)
    x0 = center_ix - size // 2
    y0 = center_iy - size // 2
    x1 = x0 + size
    y1 = y0 + size
    x0c, x1c = max(0, x0), min(width, x1)
    y0c, y1c = max(0, y0), min(height, y1)

    if mean_padding:
        fill = stack.mean(dim=(-2, -1), keepdim=True)
        patch = fill.expand(n, size, size).clone()
    else:
        patch = torch.zeros((n, size, size), dtype=stack.dtype, device=stack.device)
    ox0 = x0c - x0
    oy0 = y0c - y0
    patch[:, oy0 : oy0 + (y1c - y0c), ox0 : ox0 + (x1c - x0c)] = stack[:, y0c:y1c, x0c:x1c]
    return patch.contiguous()


def align_patches(
    stack: Tensor,
    centers: np.ndarray,
    patch_size: int,
    pixel_size: float,
    bfactor_angstrom_sq: float,
    maximum_shift_angstrom: float,
    termination_angstrom: float,
    max_iterations: int,
    running_average: int,
    mask_central_cross: bool,
    vertical_mask_size: int,
    horizontal_mask_size: int,
    batch_size: int = 1,
) -> Tuple[Tensor, Tensor]:
    n_frames = stack.shape[0]
    n_patches = centers.shape[0]
    all_x = torch.empty((n_frames, n_patches), dtype=torch.float32, device=stack.device)
    all_y = torch.empty_like(all_x)
    options = AlignmentOptions(
        max_iterations=max_iterations,
        unitless_bfactor=bfactor_angstrom_sq / (pixel_size * pixel_size),
        inner_radius=0.0,
        outer_radius=maximum_shift_angstrom / pixel_size,
        convergence_threshold=termination_angstrom / pixel_size,
        running_average=running_average,
        savitzky_golay_window=3,
        use_smoothed_shifts=True,
        mask_central_cross=mask_central_cross,
        vertical_mask_size=vertical_mask_size,
        horizontal_mask_size=horizontal_mask_size,
        verbose=False,
        batch_size=batch_size,
    )

    for p, (cx, cy) in enumerate(centers):
        patch = extract_patch(stack, float(cx), float(cy), patch_size, mean_padding=True)
        patch = patch - patch.mean(dim=(-2, -1), keepdim=True)
        _, sx, sy = iterative_align(patch, options)
        all_x[:, p] = sx
        all_y[:, p] = sy
    #    if (p + 1) % max(1, min(8, n_patches)) == 0 or p + 1 == n_patches:
    #        log(f"  aligned patches {p + 1}/{n_patches}")
    return all_x, all_y


def _upper_iqr_outliers(values: Tensor) -> Tensor:
    values64 = values.to(torch.float64)
    q1 = torch.quantile(values64, 0.25, interpolation="linear")
    q3 = torch.quantile(values64, 0.75, interpolation="linear")
    return values64 > q3 + 1.5 * (q3 - q1)


def detect_bad_patch_trajectories(shifts_x: Tensor, shifts_y: Tensor) -> Tensor:
    """Port of utilities.h FixOutliers trajectory-level IQR test."""
    if shifts_x.shape[0] < 2:
        return torch.zeros(shifts_x.shape[1], dtype=torch.bool, device=shifts_x.device)
    std_x = torch.diff(shifts_x.to(torch.float64), dim=0).std(dim=0, unbiased=False)
    std_y = torch.diff(shifts_y.to(torch.float64), dim=0).std(dim=0, unbiased=False)
    return (_upper_iqr_outliers(std_x) | _upper_iqr_outliers(std_y)).to(shifts_x.device)


def replace_bad_patch_trajectories(
    shifts_x: Tensor,
    shifts_y: Tensor,
    centers: np.ndarray,
    bad: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Replace each bad patch by the nearest good patch, matching FixOutliers."""
    if not bool(bad.any()):
        return shifts_x, shifts_y
    good_idx = torch.nonzero(~bad, as_tuple=False).flatten()
    bad_idx = torch.nonzero(bad, as_tuple=False).flatten()
    if good_idx.numel() == 0:
        log("WARNING: every patch was classified as an outlier; no replacement applied")
        return shifts_x, shifts_y
    center_t = torch.as_tensor(centers, dtype=torch.float64, device=shifts_x.device)
    dist = torch.cdist(center_t[bad_idx], center_t[good_idx])
    nearest = good_idx[dist.argmin(dim=1)]
    fixed_x = shifts_x.clone()
    fixed_y = shifts_y.clone()
    fixed_x[:, bad_idx] = shifts_x[:, nearest]
    fixed_y[:, bad_idx] = shifts_y[:, nearest]
    return fixed_x, fixed_y


def ordinary_lstsq(design: Tensor, values: Tensor) -> Tensor:
    """Unregularized least squares with column scaling for numerical stability."""
    a = design.to(torch.float64)
    b = values.to(torch.float64)
    scales = torch.linalg.vector_norm(a, dim=0).clamp_min(1.0e-15)
    scaled = a / scales
    solution = torch.linalg.lstsq(scaled, b).solution / scales
    return solution.to(torch.float32)


class MotionModel:
    def evaluate_points(self, x: Tensor, y: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def evaluate_grid(
        self,
        height: int,
        width: int,
        frame_index: int,
        coordinate_scale_x: float = 1.0,
        coordinate_scale_y: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


@dataclass
class PolynomialMotionModel(MotionModel):
    model_type: str
    coeff_x: Tensor
    coeff_y: Tensor

    def design(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        t = t.to(torch.float64)
        t1, t2, t3 = t, t.square(), t.pow(3)
        if self.model_type == "linear":
            return torch.stack(
                (t1, t2, t3, x * t1, x * t2, x * t3, y * t1, y * t2, y * t3),
                dim=-1,
            )
        return torch.stack(
            (
                t1, t2, t3,
                x * t1, x * t2, x * t3,
                x.square() * t1, x.square() * t2, x.square() * t3,
                y * t1, y * t2, y * t3,
                y.square() * t1, y.square() * t2, y.square() * t3,
                x * y * t1, x * y * t2, x * y * t3,
            ),
            dim=-1,
        )

    def evaluate_points(self, x: Tensor, y: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        d = self.design(x, y, t)
        cx = self.coeff_x.to(device=d.device, dtype=d.dtype)
        cy = self.coeff_y.to(device=d.device, dtype=d.dtype)
        return (d @ cx).to(torch.float32), (d @ cy).to(torch.float32)

    def evaluate_grid(
        self,
        height: int,
        width: int,
        frame_index: int,
        coordinate_scale_x: float = 1.0,
        coordinate_scale_y: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        device = device or self.coeff_x.device
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device) / coordinate_scale_y,
            torch.arange(width, dtype=torch.float32, device=device) / coordinate_scale_x,
            indexing="ij",
        )
        t = float(frame_index)
        t2 = t * t
        t3 = t2 * t

        def evaluate(coeff: Tensor) -> Tensor:
            c = coeff.to(device=device, dtype=torch.float32)
            base = c[0] * t + c[1] * t2 + c[2] * t3
            xterm = c[3] * t + c[4] * t2 + c[5] * t3
            if self.model_type == "linear":
                yterm = c[6] * t + c[7] * t2 + c[8] * t3
                return base + xx * xterm + yy * yterm
            x2term = c[6] * t + c[7] * t2 + c[8] * t3
            yterm = c[9] * t + c[10] * t2 + c[11] * t3
            y2term = c[12] * t + c[13] * t2 + c[14] * t3
            xyterm = c[15] * t + c[16] * t2 + c[17] * t3
            return base + xx * xterm + xx.square() * x2term + yy * yterm + yy.square() * y2term + xx * yy * xyterm

        return evaluate(self.coeff_x), evaluate(self.coeff_y)


def _cubic_weights(frac: Tensor) -> Tensor:
    f2 = frac.square()
    f3 = f2 * frac
    return torch.stack(
        (
            (1.0 - frac).pow(3) / 6.0,
            (3.0 * f3 - 6.0 * f2 + 4.0) / 6.0,
            (-3.0 * f3 + 3.0 * f2 + 3.0 * frac + 1.0) / 6.0,
            f3 / 6.0,
        ),
        dim=-1,
    )


def _extended_cubic_basis(coords: Tensor, n_internal: int, spacing: float) -> Tensor:
    """Basis on the n+2 explicit ghost coefficients used by the C++ splines."""
    if n_internal < 2:
        raise ValueError("At least two internal spline controls are required")
    if spacing <= 0:
        raise ValueError("Spline knot spacing must be positive")
    dtype = coords.dtype if coords.dtype in (torch.float32, torch.float64) else torch.float32
    u = coords.to(dtype) / float(spacing)
    u = u.clamp(0.0, float(n_internal - 1))
    cell = torch.floor(u).to(torch.long)
    endpoint = cell >= n_internal - 1
    cell = torch.where(endpoint, torch.full_like(cell, n_internal - 2), cell)
    frac = torch.where(endpoint, torch.ones_like(u), u - cell.to(u.dtype))
    weights = _cubic_weights(frac)
    indices = cell[:, None] + torch.arange(4, device=coords.device)[None, :]
    basis = torch.zeros((coords.numel(), n_internal + 2), dtype=dtype, device=coords.device)
    basis.scatter_add_(1, indices, weights)
    return basis


def _one_dimensional_ghost_map(n_internal: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    mapping = torch.zeros((n_internal + 2, n_internal), dtype=dtype, device=device)
    mapping[0, 0] = 2.0
    mapping[0, 1] = -1.0
    mapping[1 : n_internal + 1] = torch.eye(n_internal, dtype=dtype, device=device)
    mapping[-1, -1] = 2.0
    mapping[-1, -2] = -1.0
    return mapping


def _internal_cubic_basis(coords: Tensor, n_internal: int, spacing: float) -> Tensor:
    basis = _extended_cubic_basis(coords, n_internal, spacing)
    return basis @ _one_dimensional_ghost_map(n_internal, coords.device, basis.dtype)


def _spatial_ghost_map(ny: int, nx: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Exact UpdateSpline2dControlPoints linear map, including diagonal corners."""
    out = torch.zeros(((ny + 2) * (nx + 2), ny * nx), dtype=dtype, device=device)

    def go(i: int, j: int) -> int:
        return i * (nx + 2) + j

    def ii(i: int, j: int) -> int:
        return i * nx + j

    for i in range(ny):
        for j in range(nx):
            out[go(i + 1, j + 1), ii(i, j)] = 1.0
    for i in range(ny):
        out[go(i + 1, 0), ii(i, 0)] = 2.0
        out[go(i + 1, 0), ii(i, 1)] = -1.0
        out[go(i + 1, nx + 1), ii(i, nx - 1)] = 2.0
        out[go(i + 1, nx + 1), ii(i, nx - 2)] = -1.0
    for j in range(nx):
        out[go(0, j + 1), ii(0, j)] = 2.0
        out[go(0, j + 1), ii(1, j)] = -1.0
        out[go(ny + 1, j + 1), ii(ny - 1, j)] = 2.0
        out[go(ny + 1, j + 1), ii(ny - 2, j)] = -1.0
    out[go(0, 0), ii(0, 0)] = 2.0
    out[go(0, 0), ii(1, 1)] = -1.0
    out[go(ny + 1, nx + 1), ii(ny - 1, nx - 1)] = 2.0
    out[go(ny + 1, nx + 1), ii(ny - 2, nx - 2)] = -1.0
    out[go(0, nx + 1), ii(0, nx - 1)] = 2.0
    out[go(0, nx + 1), ii(1, nx - 2)] = -1.0
    out[go(ny + 1, 0), ii(ny - 1, 0)] = 2.0
    out[go(ny + 1, 0), ii(ny - 2, 1)] = -1.0
    return out


def _spatial_internal_basis(
    x: Tensor,
    y: Tensor,
    nx: int,
    ny: int,
    knot_dx: float,
    knot_dy: float,
) -> Tensor:
    bx = _extended_cubic_basis(x, nx, knot_dx)
    by = _extended_cubic_basis(y, ny, knot_dy)
    ghost_basis = torch.einsum("ni,nj->nij", by, bx).reshape(x.numel(), -1)
    return ghost_basis @ _spatial_ghost_map(ny, nx, x.device, ghost_basis.dtype)


def _extend_spatial_controls(control: Tensor) -> Tensor:
    """Apply bicubicspline::UpdateSpline2dControlPoints to [ny,nx]."""
    ny, nx = control.shape
    q = torch.empty((ny + 2, nx + 2), dtype=control.dtype, device=control.device)
    q[1:-1, 1:-1] = control
    q[1:-1, 0] = 2.0 * control[:, 0] - control[:, 1]
    q[1:-1, -1] = 2.0 * control[:, -1] - control[:, -2]
    q[0, 1:-1] = 2.0 * control[0] - control[1]
    q[-1, 1:-1] = 2.0 * control[-1] - control[-2]
    q[0, 0] = 2.0 * control[0, 0] - control[1, 1]
    q[-1, -1] = 2.0 * control[-1, -1] - control[-2, -2]
    q[0, -1] = 2.0 * control[0, -1] - control[1, -2]
    q[-1, 0] = 2.0 * control[-1, 0] - control[-2, 1]
    return q


@dataclass
class SplineMotionModel(MotionModel):
    control_x: Tensor
    control_y: Tensor
    knot_dx: float
    knot_dy: float
    knot_dose: float
    exposure_per_frame: float
    width: float
    height: float
    n_frames: int

    @property
    def nz(self) -> int:
        return int(self.control_x.shape[0])

    @property
    def ny(self) -> int:
        return int(self.control_x.shape[1])

    @property
    def nx(self) -> int:
        return int(self.control_x.shape[2])

    def _time_coordinates(self, frame_index: Tensor) -> Tensor:
        dtype = frame_index.dtype if frame_index.dtype in (torch.float32, torch.float64) else torch.float32
        return (frame_index.to(dtype) + 1.0) * float(self.exposure_per_frame)

    def design_matrix(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        bt = _internal_cubic_basis(self._time_coordinates(t), self.nz, self.knot_dose)
        bs = _spatial_internal_basis(x.to(torch.float64), y.to(torch.float64), self.nx, self.ny, self.knot_dx, self.knot_dy)
        return torch.einsum("nz,np->nzp", bt, bs).reshape(x.numel(), -1)

    def evaluate_points(self, x: Tensor, y: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        design = self.design_matrix(x, y, t)
        cx = self.control_x.reshape(-1).to(device=design.device, dtype=design.dtype)
        cy = self.control_y.reshape(-1).to(device=design.device, dtype=design.dtype)
        return (design @ cx).to(torch.float32), (design @ cy).to(torch.float32)

    def evaluate_grid(
        self,
        height: int,
        width: int,
        frame_index: int,
        coordinate_scale_x: float = 1.0,
        coordinate_scale_y: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        device = device or self.control_x.device
        time = torch.tensor([float(frame_index)], dtype=torch.float32, device=device)
        bt = _internal_cubic_basis(self._time_coordinates(time), self.nz, self.knot_dose)[0]
        cx = self.control_x.to(device=device, dtype=torch.float32)
        cy = self.control_y.to(device=device, dtype=torch.float32)
        internal_x = torch.einsum("z,zyx->yx", bt, cx)
        internal_y = torch.einsum("z,zyx->yx", bt, cy)
        qx = _extend_spatial_controls(internal_x)
        qy = _extend_spatial_controls(internal_y)
        xcoords = torch.arange(width, dtype=torch.float32, device=device) / coordinate_scale_x
        ycoords = torch.arange(height, dtype=torch.float32, device=device) / coordinate_scale_y
        bx = _extended_cubic_basis(xcoords, self.nx, self.knot_dx)
        by = _extended_cubic_basis(ycoords, self.ny, self.knot_dy)
        field_x = by @ qx @ bx.T
        field_y = by @ qy @ bx.T
        return field_x, field_y


@dataclass
class SumMotionModel(MotionModel):
    first: MotionModel
    second: MotionModel

    def evaluate_points(self, x: Tensor, y: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        ax, ay = self.first.evaluate_points(x, y, t)
        bx, by = self.second.evaluate_points(x, y, t)
        return ax + bx, ay + by

    def evaluate_grid(
        self,
        height: int,
        width: int,
        frame_index: int,
        coordinate_scale_x: float = 1.0,
        coordinate_scale_y: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        ax, ay = self.first.evaluate_grid(height, width, frame_index, coordinate_scale_x, coordinate_scale_y, device)
        bx, by = self.second.evaluate_grid(height, width, frame_index, coordinate_scale_x, coordinate_scale_y, device)
        return ax + bx, ay + by


def fit_polynomial_motion(
    model_type: str,
    centers: np.ndarray,
    shifts_x: Tensor,
    shifts_y: Tensor,
    bad_patches: Tensor,
) -> PolynomialMotionModel:
    n_frames, n_patches = shifts_x.shape
    keep = ~bad_patches
    if not bool(keep.any()):
        keep = torch.ones_like(keep)
    cx = torch.as_tensor(centers[:, 0], dtype=torch.float64, device=shifts_x.device)[keep]
    cy = torch.as_tensor(centers[:, 1], dtype=torch.float64, device=shifts_x.device)[keep]
    x = cx[None].expand(n_frames, -1).reshape(-1)
    y = cy[None].expand(n_frames, -1).reshape(-1)
    t = torch.arange(n_frames, dtype=torch.float64, device=shifts_x.device)[:, None].expand(n_frames, cx.numel()).reshape(-1)
    model = PolynomialMotionModel(model_type, torch.empty(0, device=shifts_x.device), torch.empty(0, device=shifts_x.device))
    design = model.design(x, y, t)
    model.coeff_x = ordinary_lstsq(design, shifts_x[:, keep].reshape(-1))
    model.coeff_y = ordinary_lstsq(design, shifts_y[:, keep].reshape(-1))
    return model


def choose_spline_geometry(
    nx_patches: int,
    ny_patches: int,
    n_frames: int,
    exposure_per_frame: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, float, float, float, float]:
    nx = nx_patches if nx_patches < 6 else round_nearest(nx_patches * 2.0 / 3.0)
    ny = ny_patches if ny_patches < 6 else round_nearest(ny_patches * 2.0 / 3.0)
    nx = max(2, nx)
    ny = max(2, ny)
    effective_exposure = float(exposure_per_frame)
    if effective_exposure <= 0:
        effective_exposure = 1.0
        log("WARNING: spline time coordinate uses 1.0 frame-unit because exposure/frame is zero")
    total_dose = effective_exposure * n_frames
    sample_dose = 4.0 if total_dose >= 12.0 else total_dose / 4.0
    sample_dose = max(sample_dose, effective_exposure * 1.0e-6)
    nz = int(math.ceil(total_dose / sample_dose) + 1)
    if nz > n_frames:
        nz = n_frames
        sample_dose = total_dose / max(nz - 1, 1)  # fix the stale-knot-spacing bug in C++
    nz = max(2, nz)
    knot_dx = float(math.ceil(width / float(nx - 1)))
    knot_dy = float(math.ceil(height / float(ny - 1)))
    return nx, ny, nz, knot_dx, knot_dy, float(sample_dose), effective_exposure


def fit_spline_motion(
    centers: np.ndarray,
    shifts_x: Tensor,
    shifts_y: Tensor,
    width: int,
    height: int,
    nx_control: int,
    ny_control: int,
    nz_control: int,
    knot_dx: float,
    knot_dy: float,
    knot_dose: float,
    exposure_per_frame: float,
) -> SplineMotionModel:
    n_frames, n_patches = shifts_x.shape
    cx = torch.as_tensor(centers[:, 0], dtype=torch.float64, device=shifts_x.device)
    cy = torch.as_tensor(centers[:, 1], dtype=torch.float64, device=shifts_x.device)
    x = cx[None].expand(n_frames, n_patches).reshape(-1)
    y = cy[None].expand(n_frames, n_patches).reshape(-1)
    t = torch.arange(n_frames, dtype=torch.float64, device=shifts_x.device)[:, None].expand(n_frames, n_patches).reshape(-1)
    prototype = SplineMotionModel(
        torch.zeros((nz_control, ny_control, nx_control), device=shifts_x.device),
        torch.zeros((nz_control, ny_control, nx_control), device=shifts_x.device),
        knot_dx, knot_dy, knot_dose, exposure_per_frame, float(width), float(height), n_frames,
    )
    design = prototype.design_matrix(x, y, t)
    coeff_x = ordinary_lstsq(design, shifts_x.reshape(-1))
    coeff_y = ordinary_lstsq(design, shifts_y.reshape(-1))
    prototype.control_x = coeff_x.reshape(nz_control, ny_control, nx_control)
    prototype.control_y = coeff_y.reshape(nz_control, ny_control, nx_control)
    return prototype


def _build_cc_phi(m: int, n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    total = (m + 2) * (n + 2)
    phi = torch.zeros((total, total), dtype=dtype, device=device)
    for j in range(m):
        for i in range(n):
            row = i + j * n
            for dj, wy in enumerate((1.0, 4.0, 1.0)):
                for di, wx in enumerate((1.0, 4.0, 1.0)):
                    col = i + di + (j + dj) * (n + 2)
                    phi[row, col] = wy * wx
    # The original loops are square-map specific; CC maps are fixed square maps.
    for i in range(m):
        row = n * m + i
        base = (i + 1) * (n + 2)
        phi[row, base : base + 3] = torch.tensor((1.0, -2.0, 1.0), dtype=dtype, device=device)
    for i in range(m):
        row = n * m + m + i
        base = (i + 1) * (n + 2) + n - 1
        phi[row, base : base + 3] = torch.tensor((1.0, -2.0, 1.0), dtype=dtype, device=device)
    for i in range(n):
        row = n * m + 2 * m + i
        phi[row, 1 + i] = 1.0
        phi[row, 1 + i + n + 2] = -2.0
        phi[row, 1 + i + 2 * (n + 2)] = 1.0
    for i in range(n):
        row = n * m + 2 * m + n + i
        phi[row, total - 1 - (1 + i)] = 1.0
        phi[row, total - 1 - (1 + i + n + 2)] = -2.0
        phi[row, total - 1 - (1 + i + 2 * (n + 2))] = 1.0
    r = n * m + 2 * m + 2 * n
    phi[r, 0] = 1.0; phi[r, n + 3] = -2.0; phi[r, 2 * (n + 2) + 2] = 1.0
    phi[r + 1, n + 1] = 1.0; phi[r + 1, 2 * (n + 2) - 2] = -2.0; phi[r + 1, 3 * (n + 2) - 3] = 1.0
    phi[r + 2, total - (n + 2)] = 1.0; phi[r + 2, total - 2 * (n + 2) + 1] = -2.0; phi[r + 2, total - 3 * (n + 2) + 2] = 1.0
    phi[r + 3, total - 2 * (n + 2) - 3] = 1.0; phi[r + 3, total - (n + 2) - 2] = -2.0; phi[r + 3, total - 1] = 1.0
    return phi


def cc_maps_to_bicubic_coefficients(cc_maps: Tensor) -> Tensor:
    """Port bicubicspline::CalcPhi/CalcQz for a batch of square CC maps.

    The C++ code uses double-precision dlib matrices.  The maps are only
    32 x 32 in the unbend R1 path, so using float64 here is inexpensive and
    avoids small coefficient differences in the continuous CC objective.
    """
    count, m, n = cc_maps.shape
    if m != n:
        raise ValueError("The supplied bicubicspline implementation assumes square CC maps")
    solve_dtype = torch.float64
    log(f"  converting {count} CC maps to bicubic coefficients")
    t0 = time.perf_counter()
    phi = _build_cc_phi(m, n, cc_maps.device, solve_dtype)
    total = (m + 2) * (n + 2)
    rhs = torch.zeros((total, count), dtype=solve_dtype, device=cc_maps.device)
    # Generate_CoeffSpline stores tmpimg.real_values in row-major order:
    # z_on_knot(ii * dim + jj) = tmpimg(ii, jj).
    rhs[: m * n] = cc_maps.to(solve_dtype).reshape(count, m * n).T
    coeff = torch.linalg.solve(phi, rhs * 36.0)
    log(f"    solved CC coefficients in {time.perf_counter()-t0:.2f} s")
    return coeff.T.reshape(count, m + 2, n + 2).contiguous()

def sample_cc_bicubic(coeff: Tensor, x: Tensor, y: Tensor) -> Tensor:
    count, my, nx = coeff.shape
    m, n = my - 2, nx - 2
    ux = x.to(coeff.dtype).clamp(0.0, float(n - 1))
    uy = y.to(coeff.dtype).clamp(0.0, float(m - 1))
    pv = torch.floor(ux).to(torch.long)
    pu = torch.floor(uy).to(torch.long)
    end_x = pv >= n - 1
    end_y = pu >= m - 1
    pv = torch.where(end_x, torch.full_like(pv, n - 2), pv)
    pu = torch.where(end_y, torch.full_like(pu, m - 2), pu)
    vx = torch.where(end_x, torch.ones_like(ux), ux - pv.to(ux.dtype))
    vy = torch.where(end_y, torch.ones_like(uy), uy - pu.to(uy.dtype))
    wx = _cubic_weights(vx)
    wy = _cubic_weights(vy)
    ar = torch.arange(4, device=coeff.device)
    rows = pu[:, None] + ar[None]
    cols = pv[:, None] + ar[None]
    batch = torch.arange(count, device=coeff.device)
    values = coeff[batch[:, None, None], rows[:, :, None], cols[:, None, :]]
    return torch.einsum("ni,nij,nj->n", wy, values, wx)


def generate_residual_cc_coefficients(
    aligned_stack: Tensor,
    centers: np.ndarray,
    patch_size: int,
    initial_model: SplineMotionModel,
    unitless_bfactor: float,
    cc_size: int = 32,
    batch_size: int = 1,
) -> Tensor:
    """Generate unbend R1 CC-map splines from the R0-shifted patch_pix stack.

    This mirrors the C++ spline path:
      patch_trimming_basedon_locations(..., mean_padding=false, "patch_pix")
      Spline_Shift_Implement(patch_stack)       # apply R0 to each patch/frame
      Generate_CoeffSpline(ccmap_stack, patch_stack, coeffspline_unitless_bfactor)

    Output order is patch-major, i.e. patch * n_frames + frame, matching
    ccmap_stack.spline_stack[patch_ind * image_no + img_ind] in the loss.
    """
    if cc_size % 2 or cc_size < 4:
        raise ValueError("CC map size must be an even integer >= 4")
    n_frames = aligned_stack.shape[0]
    frame_idx = torch.arange(n_frames, dtype=torch.float32, device=aligned_stack.device)
    cc_patch_major = []
    filt = bfactor_filter_rfft(patch_size, patch_size, unitless_bfactor, aligned_stack.device)
    y0 = patch_size // 2 - cc_size // 2
    x0 = patch_size // 2 - cc_size // 2
    for p, (cx, cy) in enumerate(centers):
        # patch_pix uses zero padding, not mean padding.  In C++ the cropped
        # patch is FFT'd and ZeroCentralPixel() is called before R0 PhaseShift.
        patch = extract_patch(aligned_stack, float(cx), float(cy), patch_size, mean_padding=False)
        patch = patch - patch.mean(dim=(-2, -1), keepdim=True)

        px = torch.full_like(frame_idx, float(cx))
        py = torch.full_like(frame_idx, float(cy))
        r0x, r0y = initial_model.evaluate_points(px, py, frame_idx)
        r0_patch = phase_shift_stack(patch, r0x, r0y, batch_size=batch_size)
        del patch

        # ApplyBFactor to every R0-shifted patch first, then form the
        # leave-one-out reference from the filtered frames, exactly as in
        # Generate_CoeffSpline.
        frame_fft = torch.fft.rfft2(r0_patch)
        frame_fft = frame_fft * filt
        reference_fft = frame_fft.sum(dim=0, keepdim=True) - frame_fft
        cc = torch.fft.irfft2(reference_fft * torch.conj(frame_fft), s=(patch_size, patch_size))
        cc = torch.fft.fftshift(cc, dim=(-2, -1))
        cc_patch_major.append(cc[:, y0 : y0 + cc_size, x0 : x0 + cc_size].contiguous())
        del r0_patch, frame_fft, reference_fft, cc

    #    if (p + 1) % max(1, min(8, len(centers))) == 0 or p + 1 == len(centers):
    #        log(f"  generated residual CC maps {p + 1}/{len(centers)}")
    # [patch, frame, y, x] -> patch-major flat, matching C++ stack indexing.
    maps = torch.stack(cc_patch_major, dim=0)
    flat = maps.reshape(len(centers) * n_frames, cc_size, cc_size)
    return cc_maps_to_bicubic_coefficients(flat)


def refine_spline_on_cc_maps(
    prototype: SplineMotionModel,
    centers: np.ndarray,
    cc_coeff: Tensor,
    max_iterations: int,
) -> SplineMotionModel:
    """Optimize R1 controls on patch-major continuous CC-map splines."""
    n_frames = prototype.n_frames
    n_patches = len(centers)
    dtype = torch.float64
    cc_coeff = cc_coeff.to(dtype)
    cx = torch.as_tensor(centers[:, 0], dtype=dtype, device=cc_coeff.device)
    cy = torch.as_tensor(centers[:, 1], dtype=dtype, device=cc_coeff.device)
    # Match the C++ loss loop and stack indexing: patch outer, frame inner.
    x = cx[:, None].expand(n_patches, n_frames).reshape(-1)
    y = cy[:, None].expand(n_patches, n_frames).reshape(-1)
    t = torch.arange(n_frames, dtype=dtype, device=cc_coeff.device)[None, :].expand(n_patches, n_frames).reshape(-1)
    log("  building R1 spline design matrix")
    t0 = time.perf_counter()
    design = prototype.design_matrix(x, y, t).to(dtype)
    log(f"  built R1 design matrix in {time.perf_counter()-t0:.2f} s")
    n_control = design.shape[1]
    control_x = torch.zeros(n_control, dtype=dtype, device=cc_coeff.device, requires_grad=True)
    control_y = torch.zeros_like(control_x, requires_grad=True)
    half = torch.tensor((cc_coeff.shape[-1] - 2) / 2.0, dtype=dtype, device=cc_coeff.device)
    zero = torch.full((design.shape[0],), float(half), dtype=dtype, device=cc_coeff.device)
    with torch.no_grad():
        initial_loss = float((-sample_cc_bicubic(cc_coeff, zero, zero).sum()).item())
    tolerance_change = max(abs(initial_loss) / 1.0e6, 1.0e-12)
    optimizer = torch.optim.LBFGS(
        (control_x, control_y),
        lr=1.0,
        max_iter=max_iterations,
        history_size=min(max(20, 4 * n_control), 1000),
        tolerance_grad=1.0e-7,
        tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe",
    )
    calls = 0

    def closure() -> Tensor:
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        sx = design @ control_x
        sy = design @ control_y
        sampled = sample_cc_bicubic(cc_coeff, sx + half, sy + half)
        loss = -sampled.sum()
        loss.backward()
        calls += 1
        return loss

    log(f"  R1 initial CC loss: {initial_loss:.6g}")
    log(f"  R1 objective-delta stop threshold: {tolerance_change:.6g}")
    if max_iterations > 0:
        optimizer.step(closure)
    with torch.no_grad():
        final_sx = design @ control_x
        final_sy = design @ control_y
        final_loss = float((-sample_cc_bicubic(cc_coeff, final_sx + half, final_sy + half).sum()).item())
    log(f"  R1 final CC loss: {final_loss:.6g} ({calls} closure evaluations)")
    return SplineMotionModel(
        control_x.detach().to(torch.float32).reshape(prototype.nz, prototype.ny, prototype.nx),
        control_y.detach().to(torch.float32).reshape(prototype.nz, prototype.ny, prototype.nx),
        prototype.knot_dx, prototype.knot_dy, prototype.knot_dose,
        prototype.exposure_per_frame, prototype.width, prototype.height, prototype.n_frames,
    )


def fit_patch_motion_model(
    model_type: str,
    aligned_stack: Tensor,
    centers: np.ndarray,
    patch_size: int,
    nx_patches: int,
    ny_patches: int,
    output_pixel_size: float,
    bfactor_angstrom_sq: float,
    maximum_shift_angstrom: float,
    termination_angstrom: float,
    max_iterations: int,
    patch_running_average: int,
    exposure_per_frame: float,
    mask_central_cross: bool,
    vertical_mask_size: int,
    horizontal_mask_size: int,
    spline_residual_refine: bool,
    spline_r1_iterations: int,
    ccmap_unitless_bfactor: Optional[float] = None,
    batch_size: int = 1,
) -> Tuple[MotionModel, Tensor, Tensor, Tensor]:
    log("Patch alignment, round 0")
    shifts_x, shifts_y = align_patches(
        aligned_stack, centers, patch_size, output_pixel_size, bfactor_angstrom_sq,
        maximum_shift_angstrom, termination_angstrom, max_iterations,
        patch_running_average, mask_central_cross, vertical_mask_size, horizontal_mask_size,
        batch_size,
    )
    bad = detect_bad_patch_trajectories(shifts_x, shifts_y)
    bad_indices = torch.nonzero(bad, as_tuple=False).flatten().detach().cpu().tolist()
    log(f"  outlier patches: {bad_indices if bad_indices else 'none'}")
    height, width = aligned_stack.shape[-2:]
    if model_type in {"linear", "quadratic"}:
        model = fit_polynomial_motion(model_type, centers, shifts_x, shifts_y, bad)
        return model, shifts_x, shifts_y, bad

    fixed_x, fixed_y = replace_bad_patch_trajectories(shifts_x, shifts_y, centers, bad)
    nx_ctrl, ny_ctrl, nz_ctrl, knot_dx, knot_dy, knot_dose, effective_exposure = choose_spline_geometry(
        nx_patches, ny_patches, aligned_stack.shape[0], exposure_per_frame, width, height
    )
    log(
        f"Spline controls x/y/time: {nx_ctrl} x {ny_ctrl} x {nz_ctrl}; "
        f"spacing {knot_dx:g}, {knot_dy:g} px, {knot_dose:g} dose"
    )
    initial = fit_spline_motion(
        centers, fixed_x, fixed_y, width, height,
        nx_ctrl, ny_ctrl, nz_ctrl, knot_dx, knot_dy, knot_dose, effective_exposure,
    )
    if not spline_residual_refine:
        return initial, shifts_x, shifts_y, bad

    log("Spline residual refinement on continuous 32 x 32 CC maps")
    if ccmap_unitless_bfactor is None:
        ccmap_unitless_bfactor = bfactor_angstrom_sq / (output_pixel_size * output_pixel_size)
    log(f"  R1 CC-map B-factor uses unitless value {ccmap_unitless_bfactor:.6g}")
    cc_coeff = generate_residual_cc_coefficients(
        aligned_stack, centers, patch_size, initial,
        float(ccmap_unitless_bfactor),
        cc_size=32,
        batch_size=batch_size,
    )
    residual_proto = SplineMotionModel(
        torch.zeros_like(initial.control_x), torch.zeros_like(initial.control_y),
        initial.knot_dx, initial.knot_dy, initial.knot_dose,
        initial.exposure_per_frame, initial.width, initial.height, initial.n_frames,
    )
    residual = refine_spline_on_cc_maps(residual_proto, centers, cc_coeff, spline_r1_iterations)
    total = SumMotionModel(initial, residual)
    # Return the independently aligned patch shifts, matching the C++ files
    # %04i_shift.txt.  Model-predicted R0/R1/GUI shifts are generated later
    # by save_motion_outputs().
    return total, shifts_x, shifts_y, bad


def _backward_inverse_warp(frame: Tensor, shift_x: Tensor, shift_y: Tensor, iterations: int = 4) -> Tensor:
    """Fixed-point inverse deformation; used for --warp-mode inverse/fallback."""
    height, width = frame.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=frame.device),
        torch.arange(width, dtype=torch.float32, device=frame.device),
        indexing="ij",
    )
    source_x = xx - shift_x
    source_y = yy - shift_y
    field = torch.stack((shift_x, shift_y), dim=0)[None]
    for _ in range(max(1, iterations)):
        gx = 2.0 * source_x / max(width - 1, 1) - 1.0
        gy = 2.0 * source_y / max(height - 1, 1) - 1.0
        sampled = F.grid_sample(field, torch.stack((gx, gy), dim=-1)[None], mode="bilinear", padding_mode="border", align_corners=True)[0]
        source_x = xx - sampled[0]
        source_y = yy - sampled[1]
    gx = 2.0 * source_x / max(width - 1, 1) - 1.0
    gy = 2.0 * source_y / max(height - 1, 1) - 1.0
    return apply_grid_with_mean_fill(frame, torch.stack((gx, gy), dim=-1)[None])


def _batched_monotonic_interp(abscissa: Tensor, ordinates: Tensor, target_count: int) -> Tuple[Tensor, Tensor]:
    """Linear interpolation of many monotonic 1-D curves onto integer targets."""
    abscissa = abscissa.contiguous()
    ordinates = ordinates.contiguous()
    batch, length = abscissa.shape
    targets = torch.arange(target_count, dtype=abscissa.dtype, device=abscissa.device)[None].expand(batch, -1).contiguous()
    valid = (targets >= abscissa[:, :1]) & (targets <= abscissa[:, -1:])
    idx = torch.searchsorted(abscissa, targets, right=False)
    hi = idx.clamp(1, length - 1)
    lo = hi - 1
    x0 = torch.gather(abscissa, 1, lo)
    x1 = torch.gather(abscissa, 1, hi)
    z0 = torch.gather(ordinates, 1, lo)
    z1 = torch.gather(ordinates, 1, hi)
    denom = x1 - x0
    alpha = torch.where(torch.abs(denom) > 1.0e-12, (targets - x0) / denom, torch.zeros_like(denom))
    return z0 + alpha * (z1 - z0), valid


def cistem_forward_distort(frame: Tensor, shifted_map_x: Tensor, shifted_map_y: Tensor) -> Tensor:
    """GPU port of Image::Distortion's X-then-Y forward scan-line interpolation."""
    height, width = frame.shape
    if bool(((shifted_map_x[:, 1:] - shifted_map_x[:, :-1]) <= 0).any()):
        log("WARNING: non-monotonic X deformation; falling back to inverse warping")
        yy, xx = torch.meshgrid(torch.arange(height, device=frame.device), torch.arange(width, device=frame.device), indexing="ij")
        return _backward_inverse_warp(frame, shifted_map_x - xx, shifted_map_y - yy)

    temp, valid_x = _batched_monotonic_interp(shifted_map_x, frame, width)
    mapped_y, _ = _batched_monotonic_interp(shifted_map_x, shifted_map_y, width)
    fill = edge_mean(frame)
    temp = torch.where(valid_x, temp, fill)

    valid_cols = valid_x.T  # [width,height]
    yseq = mapped_y.T.contiguous()
    zseq = temp.T.contiguous()
    counts = valid_cols.sum(dim=1)
    usable = counts >= 2
    first = valid_cols.to(torch.int64).argmax(dim=1)
    last = height - 1 - torch.flip(valid_cols, dims=(1,)).to(torch.int64).argmax(dim=1)
    rows = torch.arange(height, device=frame.device)[None].expand(width, -1)
    first_y = yseq.gather(1, first[:, None])
    last_y = yseq.gather(1, last[:, None])
    span = float(max(height, width) * 4 + 1)
    yseq_safe = torch.where(rows < first[:, None], first_y - (first[:, None] - rows).to(yseq.dtype) * span, yseq)
    yseq_safe = torch.where(rows > last[:, None], last_y + (rows - last[:, None]).to(yseq.dtype) * span, yseq_safe)
    inside = (rows >= first[:, None]) & (rows <= last[:, None])
    monotonic_inside = torch.where(inside[:, 1:] & inside[:, :-1], yseq_safe[:, 1:] - yseq_safe[:, :-1], torch.ones_like(yseq_safe[:, 1:]))
    if bool((monotonic_inside[usable] <= 0).any()):
        log("WARNING: non-monotonic Y deformation; falling back to inverse warping")
        yy, xx = torch.meshgrid(torch.arange(height, device=frame.device), torch.arange(width, device=frame.device), indexing="ij")
        return _backward_inverse_warp(frame, shifted_map_x - xx, shifted_map_y - yy)

    out_t, valid_y = _batched_monotonic_interp(yseq_safe, zseq, height)
    targets_y = torch.arange(height, dtype=yseq.dtype, device=frame.device)[None]
    valid_y &= usable[:, None] & (targets_y >= first_y) & (targets_y <= last_y)
    out_t = torch.where(valid_y, out_t, fill)
    return out_t.T.contiguous()


def fourier_shift_image(frame: Tensor, shift_x: float, shift_y: float) -> Tensor:
    """Apply a single-image Fourier PhaseShift."""
    sx = torch.tensor([float(shift_x)], dtype=torch.float32, device=frame.device)
    sy = torch.tensor([float(shift_y)], dtype=torch.float32, device=frame.device)
    return phase_shift_stack(frame, sx, sy, batch_size=1)


def warp_frame_with_motion(
    frame: Tensor,
    global_shift_x_output_px: float,
    global_shift_y_output_px: float,
    local_model: Optional[MotionModel],
    frame_index: int,
    output_width: int,
    output_height: int,
    warp_mode: str,
) -> Tensor:
    """Correct a raw-resolution frame in the same order as unbend.cpp.

    unbend first applies the full-frame shift to the raw/super-resolution frame
    using Fourier PhaseShift.  The local R0/R1 map is then applied with
    Image::Distortion.  Keeping these two steps separate avoids applying the
    global sub-pixel translation with the real-space scan-line interpolator.
    All global shifts are supplied in output pixels and converted to raw pixels
    using the x/y binning factors.
    """
    height, width = frame.shape
    scale_x = width / output_width
    scale_y = height / output_height

    globally_shifted = fourier_shift_image(
        frame,
        global_shift_x_output_px * scale_x,
        global_shift_y_output_px * scale_y,
    )

    if local_model is None:
        return globally_shifted

    lx, ly = local_model.evaluate_grid(
        height,
        width,
        frame_index,
        coordinate_scale_x=scale_x,
        coordinate_scale_y=scale_y,
        device=frame.device,
    )
    sx = lx * scale_x
    sy = ly * scale_y
    if warp_mode == "inverse":
        return _backward_inverse_warp(globally_shifted, sx, sy)
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=frame.device),
        torch.arange(width, dtype=torch.float32, device=frame.device),
        indexing="ij",
    )
    return cistem_forward_distort(globally_shifted, xx + sx, yy + sy)


def warp_output_frame_with_local_motion(
    globally_aligned_frame: Tensor,
    local_model: MotionModel,
    frame_index: int,
    warp_mode: str,
) -> Tensor:
    height, width = globally_aligned_frame.shape
    lx, ly = local_model.evaluate_grid(height, width, frame_index, device=globally_aligned_frame.device)
    if warp_mode == "inverse":
        return _backward_inverse_warp(globally_aligned_frame, lx, ly)
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=globally_aligned_frame.device),
        torch.arange(width, dtype=torch.float32, device=globally_aligned_frame.device),
        indexing="ij",
    )
    return cistem_forward_distort(globally_aligned_frame, xx + lx, yy + ly)

def voltage_scale(voltage_kv: float) -> float:
    if abs(voltage_kv - 300.0) < 1.0:
        return 1.0
    if abs(voltage_kv - 200.0) < 1.0:
        return 0.8
    if abs(voltage_kv - 100.0) < 1.0:
        return 0.532
    raise ValueError("cisTEM dose weighting supports 100, 200, or 300 kV")


def dose_filter_rfft(
    height: int,
    width: int,
    pixel_size: float,
    dose_finish: float,
    voltage_kv: float,
    device: torch.device,
) -> Tensor:
    fy, fx = rfft_frequency_grids(height, width, device)
    spatial_frequency = torch.sqrt(fx.square() + fy.square()) / pixel_size
    scale = voltage_scale(voltage_kv)
    critical = torch.empty_like(spatial_frequency)
    nonzero = spatial_frequency > 0
    critical[nonzero] = (
        0.24499 * spatial_frequency[nonzero].pow(-1.6649) + 2.8141
    ) * scale
    critical[~nonzero] = torch.inf
    filt = torch.exp(-0.5 * dose_finish / critical)
    filt[0, 0] = 1.0
    return filt


def create_mrc_stack_writer(filename: str, shape: Tuple[int, int, int], pixel_size: float):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = mrcfile.new_mmap(
        str(path),
        shape=shape,
        mrc_mode=2,
        overwrite=True,
    )
    writer.voxel_size = pixel_size
    return writer


def write_mrc_image(filename: str, image: np.ndarray, pixel_size: float) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(np.asarray(image, dtype=np.float32))
        mrc.voxel_size = pixel_size
        mrc.update_header_stats()


def final_sum_and_optional_frames(
    movie: MovieSource,
    globally_aligned_output: Optional[Tensor],
    global_x: Tensor,
    global_y: Tensor,
    local_model: Optional[MotionModel],
    output_height: int,
    output_width: int,
    output_pixel_size: float,
    first_frame: int,
    last_frame: int,
    dose_filter_enabled: bool,
    restore_power: bool,
    voltage_kv: float,
    exposure_per_frame: float,
    pre_exposure: float,
    warp_domain: str,
    warp_mode: str,
    aligned_frames_filename: Optional[str],
) -> Tensor:
    writer = None
    if aligned_frames_filename:
        writer = create_mrc_stack_writer(
            aligned_frames_filename,
            (movie.n_frames, output_height, output_width),
            output_pixel_size,
        )

    sum_fft = torch.zeros(
        (output_height, output_width // 2 + 1),
        dtype=torch.complex64,
        device=movie.device,
    )
    sum_squares = torch.zeros(
        (output_height, output_width // 2 + 1),
        dtype=torch.float32,
        device=movie.device,
    )

    try:
        for i in range(movie.n_frames):
            if local_model is None:
                if globally_aligned_output is None:
                    raise RuntimeError("globally_aligned_output is required when no local model is used")
                corrected = globally_aligned_output[i]
            elif warp_domain == "output":
                if globally_aligned_output is None:
                    raise RuntimeError("globally_aligned_output is required for --warp-domain output")
                corrected = warp_output_frame_with_local_motion(
                    globally_aligned_output[i], local_model, i, warp_mode
                )
            else:
                raw = movie.frame(i, remove_mean=True)
                warped_raw = warp_frame_with_motion(
                    raw,
                    float(global_x[i].item()),
                    float(global_y[i].item()),
                    local_model,
                    i,
                    output_width,
                    output_height,
                    warp_mode,
                )
                corrected = centered_fourier_resample(warped_raw, output_height, output_width)
                corrected = corrected - corrected.mean()

            spectrum = torch.fft.rfft2(corrected)
            output_for_writer = corrected
            if dose_filter_enabled and first_frame <= i + 1 <= last_frame:
                finish_dose = pre_exposure + (i + 1) * exposure_per_frame
                filt = dose_filter_rfft(
                    output_height,
                    output_width,
                    output_pixel_size,
                    finish_dose,
                    voltage_kv,
                    movie.device,
                )
                weighted = spectrum * filt
                sum_fft += weighted
                sum_squares += filt.square()
                output_for_writer = torch.fft.irfft2(
                    weighted, s=(output_height, output_width)
                )
            elif first_frame <= i + 1 <= last_frame:
                sum_fft += spectrum

            if writer is not None:
                writer.data[i] = output_for_writer.detach().cpu().numpy().astype(np.float32, copy=False)
        #    if (i + 1) % 10 == 0 or i + 1 == movie.n_frames:
        #        log(f"  corrected/summed {i + 1}/{movie.n_frames} frames")

        if dose_filter_enabled and restore_power:
            sum_fft = torch.where(
                sum_squares > 0,
                sum_fft / torch.sqrt(sum_squares.clamp_min(EPS)),
                sum_fft,
            )
        return torch.fft.irfft2(sum_fft, s=(output_height, output_width)).real
    finally:
        if writer is not None:
            writer.update_header_stats()
            writer.flush()
            writer.close()



def _write_numeric_grid(path: Path, grid: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in grid:
            for value in row:
                f.write(f"{float(value):.8f}\t")
            f.write("\n")


def _evaluate_model_on_patch_centers(model: MotionModel, centers: np.ndarray, n_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return [frame, patch] model predictions in output pixels."""
    device = None
    if isinstance(model, PolynomialMotionModel):
        device = model.coeff_x.device
    elif isinstance(model, SplineMotionModel):
        device = model.control_x.device
    elif isinstance(model, SumMotionModel):
        first = model.first
        if isinstance(first, PolynomialMotionModel):
            device = first.coeff_x.device
        elif isinstance(first, SplineMotionModel):
            device = first.control_x.device
    device = device or torch.device("cpu")
    n_patches = centers.shape[0]
    cx = torch.as_tensor(centers[:, 0], dtype=torch.float32, device=device)
    cy = torch.as_tensor(centers[:, 1], dtype=torch.float32, device=device)
    x = cx[None].expand(n_frames, n_patches).reshape(-1)
    y = cy[None].expand(n_frames, n_patches).reshape(-1)
    t = torch.arange(n_frames, dtype=torch.float32, device=device)[:, None].expand(n_frames, n_patches).reshape(-1)
    with torch.no_grad():
        sx, sy = model.evaluate_points(x, y, t)
    return sx.reshape(n_frames, n_patches).detach().cpu().numpy(), sy.reshape(n_frames, n_patches).detach().cpu().numpy()


def _flatten_spline_control_cistem_order(control: Tensor) -> np.ndarray:
    """C++ Control1d order: for each spatial control y,x, list all z controls."""
    return control.detach().cpu().numpy().transpose(1, 2, 0).reshape(-1)


def _write_control_file(path: Path, control_x: Tensor, control_y: Tensor) -> None:
    joined = np.concatenate((_flatten_spline_control_cistem_order(control_x), _flatten_spline_control_cistem_order(control_y)))
    np.savetxt(path, joined, fmt="%.12g")


def _write_patch_alignment_shift_files(output_dir: Path, patch_x: np.ndarray, patch_y: np.ndarray) -> None:
    """C++ patch-alignment output: %04i_shift.txt, one patch time-series per file."""
    n_frames, n_patches = patch_x.shape
    for pidx in range(n_patches):
        path = output_dir / f"{pidx:04d}_shift.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Patch {pidx} shifts\n")
            for frame in range(n_frames):
                f.write(f"{float(patch_x[frame, pidx]):.8f}\t{float(patch_y[frame, pidx]):.8f}\t\n")


def _write_model_frame_grids(
    output_dir: Path,
    prefix_x: str,
    prefix_y: str,
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    nx_patches: int,
    ny_patches: int,
) -> None:
    """C++ write_shifts/write_linear_shifts format: one patch-grid file per frame."""
    n_frames, n_patches = pred_x.shape
    if n_patches != nx_patches * ny_patches:
        raise ValueError("patch grid dimensions do not match prediction arrays")
    for frame in range(n_frames):
        grid_x = pred_x[frame].reshape(ny_patches, nx_patches)
        grid_y = pred_y[frame].reshape(ny_patches, nx_patches)
        _write_numeric_grid(output_dir / f"{frame:04d}{prefix_x}ccmap.txt", grid_x)
        _write_numeric_grid(output_dir / f"{frame:04d}{prefix_y}ccmap.txt", grid_y)


def _write_patch_shift_gui(
    output_dir: Path,
    centers: np.ndarray,
    total_x: np.ndarray,
    total_y: np.ndarray,
    subtract_frame0: bool,
) -> None:
    """C++ write_shifts_forGUI* format.

    For spline, original write_shifts_forGUI writes R0+R1 minus each patch's
    frame-0 offset.  For linear/quadratic, original write_shifts_forGUI_* writes
    the model value directly.
    """
    n_frames, n_patches = total_x.shape
    path = output_dir / "patch_shift.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Patch number & Frame number\n")
        f.write(f"{n_patches}\t{n_frames}\n")
        for pidx in range(n_patches):
            f.write(f"# Patch {pidx}\n")
            f.write(f"{float(centers[pidx, 0]):.8f}\t{float(centers[pidx, 1]):.8f}\n")
            x0 = float(total_x[0, pidx]) if subtract_frame0 else 0.0
            y0 = float(total_y[0, pidx]) if subtract_frame0 else 0.0
            for frame in range(n_frames):
                f.write(f"{float(total_x[frame, pidx] - x0):.8f}\t{float(total_y[frame, pidx] - y0):.8f}\n")


def save_motion_outputs(
    output_dir: Path,
    input_filename: str,
    output_pixel_size: float,
    global_x: Tensor,
    global_y: Tensor,
    centers: Optional[np.ndarray],
    patch_x: Optional[Tensor],
    patch_y: Optional[Tensor],
    bad_patches: Optional[Tensor],
    model: Optional[MotionModel],
    patch_grid_shape: Optional[Tuple[int, int]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    global_angstrom = torch.stack((global_x, global_y), dim=1).detach().cpu().numpy() * output_pixel_size
    np.savetxt(output_dir / "fullframe_shift.txt", global_angstrom, fmt="%.8f", header=f"X/Y shifts in Angstrom for {input_filename}")

    raw_px = raw_py = None
    if centers is not None and patch_x is not None and patch_y is not None:
        raw_px = patch_x.detach().cpu().numpy()
        raw_py = patch_y.detach().cpu().numpy()
        bad_np = None if bad_patches is None else bad_patches.detach().cpu().numpy().astype(np.uint8)
        _write_patch_alignment_shift_files(output_dir, raw_px, raw_py)
        if bad_np is not None:
            np.savetxt(output_dir / "outliers.txt", np.flatnonzero(bad_np), fmt="%d")
    else:
        bad_np = None

    if centers is None or model is None:
        if raw_px is not None:
            np.savez_compressed(
                output_dir / "patch_motion.npz",
                centers_xy_pixels=centers,
                observed_shifts_x_pixels=raw_px,
                observed_shifts_y_pixels=raw_py,
                bad_patch_mask=bad_np,
                output_pixel_size=np.float32(output_pixel_size),
            )
        return

    n_frames = int(global_x.numel())
    if patch_grid_shape is None:
        xs = np.unique(np.round(centers[:, 0], 6))
        ys = np.unique(np.round(centers[:, 1], 6))
        nx_patches, ny_patches = int(xs.size), int(ys.size)
    else:
        nx_patches, ny_patches = patch_grid_shape
    if nx_patches * ny_patches != centers.shape[0]:
        raise ValueError("patch_grid_shape does not match number of centers")

    model_payload = {"model_class": np.array(type(model).__name__)}

    predicted_total_x = predicted_total_y = None
    predicted_r0_x = predicted_r0_y = None
    predicted_r1_x = predicted_r1_y = None

    if isinstance(model, PolynomialMotionModel):
        predicted_total_x, predicted_total_y = _evaluate_model_on_patch_centers(model, centers, n_frames)
        # Original linear/quadratic model writes only the fitted model with _R0 prefix.
        _write_model_frame_grids(output_dir, "_shiftx_R0", "_shifty_R0", predicted_total_x, predicted_total_y, nx_patches, ny_patches)
        _write_patch_shift_gui(output_dir, centers, predicted_total_x, predicted_total_y, subtract_frame0=False)
        model_payload.update(model_type=np.array(model.model_type), coeff_x=model.coeff_x.detach().cpu().numpy(), coeff_y=model.coeff_y.detach().cpu().numpy())
    elif isinstance(model, SplineMotionModel):
        predicted_r0_x, predicted_r0_y = _evaluate_model_on_patch_centers(model, centers, n_frames)
        predicted_total_x, predicted_total_y = predicted_r0_x, predicted_r0_y
        _write_model_frame_grids(output_dir, "_shiftx_R0", "_shifty_R0", predicted_r0_x, predicted_r0_y, nx_patches, ny_patches)
        _write_patch_shift_gui(output_dir, centers, predicted_total_x, predicted_total_y, subtract_frame0=True)
        _write_control_file(output_dir / "Control_R0.txt", model.control_x, model.control_y)
        model_payload.update(
            control_x=model.control_x.detach().cpu().numpy(),
            control_y=model.control_y.detach().cpu().numpy(),
            spline_geometry=np.array([model.knot_dx, model.knot_dy, model.knot_dose, model.exposure_per_frame, model.width, model.height, model.n_frames]),
        )
    elif isinstance(model, SumMotionModel) and isinstance(model.first, SplineMotionModel) and isinstance(model.second, SplineMotionModel):
        predicted_r0_x, predicted_r0_y = _evaluate_model_on_patch_centers(model.first, centers, n_frames)
        predicted_r1_x, predicted_r1_y = _evaluate_model_on_patch_centers(model.second, centers, n_frames)
        predicted_total_x = predicted_r0_x + predicted_r1_x
        predicted_total_y = predicted_r0_y + predicted_r1_y
        _write_model_frame_grids(output_dir, "_shiftx_R0", "_shifty_R0", predicted_r0_x, predicted_r0_y, nx_patches, ny_patches)
        _write_model_frame_grids(output_dir, "_shiftx_R1", "_shifty_R1", predicted_r1_x, predicted_r1_y, nx_patches, ny_patches)
        _write_patch_shift_gui(output_dir, centers, predicted_total_x, predicted_total_y, subtract_frame0=True)
        _write_control_file(output_dir / "Control_R0.txt", model.first.control_x, model.first.control_y)
        _write_control_file(output_dir / "Control_R1.txt", model.second.control_x, model.second.control_y)
        model_payload.update(
            control_x_round0=model.first.control_x.detach().cpu().numpy(),
            control_y_round0=model.first.control_y.detach().cpu().numpy(),
            control_x_round1=model.second.control_x.detach().cpu().numpy(),
            control_y_round1=model.second.control_y.detach().cpu().numpy(),
            spline_geometry=np.array([model.first.knot_dx, model.first.knot_dy, model.first.knot_dose, model.first.exposure_per_frame, model.first.width, model.first.height, model.first.n_frames]),
        )
    else:
        predicted_total_x, predicted_total_y = _evaluate_model_on_patch_centers(model, centers, n_frames)
        _write_patch_shift_gui(output_dir, centers, predicted_total_x, predicted_total_y, subtract_frame0=False)

    np.savez_compressed(
        output_dir / "patch_motion.npz",
        centers_xy_pixels=centers,
        observed_shifts_x_pixels=raw_px,
        observed_shifts_y_pixels=raw_py,
        model_total_x_pixels=predicted_total_x,
        model_total_y_pixels=predicted_total_y,
        model_r0_x_pixels=predicted_r0_x,
        model_r0_y_pixels=predicted_r0_y,
        model_r1_x_pixels=predicted_r1_x,
        model_r1_y_pixels=predicted_r1_y,
        bad_patch_mask=bad_np,
        output_pixel_size=np.float32(output_pixel_size),
        patch_grid_shape=np.asarray([nx_patches, ny_patches], dtype=np.int32),
    )
    np.savez_compressed(output_dir / "motion_model.npz", **model_payload)

def validate_arguments(args: argparse.Namespace, n_frames: int) -> None:
    if args.pixel_size <= 0:
        raise ValueError("--pixel-size must be positive")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be >= 1")
    if args.align_batch < 1:
        raise ValueError("--align-batch must be >= 1")
    if args.output_binning < 1.0:
        raise ValueError("--output-binning must be >= 1")
    if args.running_average > 1 and args.running_average % 2 == 0:
        raise ValueError("--running-average must be odd")
    if args.patch_running_average > 1 and args.patch_running_average % 2 == 0:
        raise ValueError("--patch-running-average must be odd")
    if args.first_frame < 1:
        raise ValueError("--first-frame is 1-based and must be >= 1")
    if args.last_frame == 0:
        args.last_frame = n_frames
    if args.last_frame > n_frames:
        args.last_frame = n_frames
    if args.first_frame > args.last_frame:
        raise ValueError("--first-frame must not exceed --last-frame")
    if args.dose_filter and args.exposure_per_frame <= 0:
        raise ValueError("--exposure-per-frame must be > 0 when dose filtering is enabled")
    if (args.patch_num_x > 0) != (args.patch_num_y > 0):
        raise ValueError("Specify both --patch-num-x and --patch-num-y, or neither")
    if args.patch_correction and n_frames < 4:
        raise ValueError("Patch distortion correction requires at least four frames")
    if args.spline_r1_iterations < 1:
        raise ValueError("--spline-r1-iterations must be >= 1")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU PyTorch motion correction and patch-based distortion correction for MRC movies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input MRC/MRCS or TIFF movie stack")
    parser.add_argument("output", help="Output aligned/dose-weighted MRC sum")
    parser.add_argument("--pixel-size", type=float, required=True, help="Input physical pixel size in Angstrom")
    parser.add_argument("--output-binning", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--cpu-threads", type=int, default=min(8, os.cpu_count() or 1), help="PyTorch CPU worker threads; limiting this avoids FFT/LAPACK oversubscription")
    parser.add_argument("--align-batch", type=int, default=1, help="Frames per FFT/correlation batch during alignment. Use 1 for lowest memory; larger values are faster but use more memory.")
    parser.add_argument("--output-dir", default=None, help="Directory for shift/model/log files when --write-log-files is enabled")
    parser.add_argument(
        "--write-log-files",
        action="store_true",
        help="Write full-frame, patch, spline-control, patch_shift.txt, and NPZ log/model files. Default is off.",
    )

    parser.add_argument("--minimum-shift", type=float, default=2.0, help="Initial inner search radius in Angstrom")
    parser.add_argument("--maximum-shift", type=float, default=100.0, help="Per-iteration outer search radius in Angstrom")
    parser.add_argument("--termination", type=float, default=None, help="Convergence threshold in Angstrom")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--bfactor", type=float, default=1500.0, help="Alignment low-pass B-factor in A^2")
    parser.add_argument("--running-average", type=int, default=1)
    parser.add_argument("--smooth-global-shifts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask-central-cross", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vertical-mask-size", type=int, default=1)
    parser.add_argument("--horizontal-mask-size", type=int, default=1)

    parser.add_argument("--gain", default=None, help="MRC gain reference; multiplication convention")
    parser.add_argument("--dark", default=None, help="MRC dark reference")
    parser.add_argument("--replace-outliers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mag-distortion-angle", type=float, default=0.0)
    parser.add_argument("--mag-distortion-major", type=float, default=1.0)
    parser.add_argument("--mag-distortion-minor", type=float, default=1.0)

    parser.add_argument("--patch-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distortion-model", choices=("linear", "quadratic", "spline"), default="spline")
    parser.add_argument("--patch-num-x", type=int, default=0)
    parser.add_argument("--patch-num-y", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=0)
    parser.add_argument("--patch-running-average", type=int, default=5)
    parser.add_argument("--patch-max-iterations", type=int, default=None)
    parser.add_argument("--spline-residual-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spline-r1-iterations", type=int, default=50, help="LBFGS iterations for continuous CC-map spline refinement")
    parser.add_argument("--warp-mode", choices=("cistem", "inverse"), default="cistem", help="cistem reproduces Image::Distortion; inverse uses fixed-point grid sampling")
    parser.add_argument(
        "--warp-domain",
        choices=("raw", "output"),
        default="raw",
        help="raw is more faithful; output uses less memory/time",
    )

    parser.add_argument("--dose-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restore-power", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--voltage", type=float, default=300.0, choices=(100.0, 200.0, 300.0))
    parser.add_argument("--exposure-per-frame", type=float, default=1.0)
    parser.add_argument("--pre-exposure", type=float, default=0.0)
    parser.add_argument("--first-frame", type=int, default=1)
    parser.add_argument("--last-frame", type=int, default=0, help="0 means final frame")
    parser.add_argument("--save-aligned-frames", default=None, metavar="OUTPUT_STACK.MRC")
    parser.add_argument("--deterministic", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = available_device(args.device)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    start_time = time.perf_counter()
    output_path = Path(args.output)
    output_dir = Path(args.output_dir) if args.output_dir else output_path.with_suffix("").with_name(output_path.stem + "_unbend")
    if args.write_log_files:
        output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Device: {device}")
    with MovieSource(
        args.input,
        device,
        gain_filename=args.gain,
        dark_filename=args.dark,
        replace_outliers=args.replace_outliers,
        magnification_angle=args.mag_distortion_angle,
        magnification_major=args.mag_distortion_major,
        magnification_minor=args.mag_distortion_minor,
    ) as movie:
        validate_arguments(args, movie.n_frames)
        output_height = round_nearest(movie.height / args.output_binning)
        output_width = round_nearest(movie.width / args.output_binning)
        x_bin = movie.width / output_width
        y_bin = movie.height / output_height
        average_bin = 0.5 * (x_bin + y_bin)
        output_pixel_size = args.pixel_size * average_bin
        # Preserve area-equivalent pixel size after anisotropic magnification correction.
        output_pixel_size /= math.sqrt(args.mag_distortion_major * args.mag_distortion_minor)
        termination = args.termination
        if termination is None:
            termination = 0.5 * output_pixel_size
        sg_window = 5 if args.exposure_per_frame <= 0 else odd_at_least(round_nearest(5.0 / args.exposure_per_frame), 3)

        log(f"Input: {movie.n_frames} frames, {movie.width}x{movie.height}")
        log(f"Output: {output_width}x{output_height}, pixel size {output_pixel_size:.6f} A")

        # unbend.cpp stores coeffspline_unitless_bfactor before the optional
        # final output-scale refinement.  Therefore the R1 CC maps use the
        # coarser pre-alignment pixel size, while patch alignment itself uses
        # output_pixel_size after final refinement.
        prebin_for_coeff = choose_prebin(output_pixel_size)
        pre_h_for_coeff = max(16, round_nearest(output_height / prebin_for_coeff))
        pre_w_for_coeff = max(16, round_nearest(output_width / prebin_for_coeff))
        actual_prebin_for_coeff = 0.5 * (output_height / pre_h_for_coeff + output_width / pre_w_for_coeff)
        coeffspline_unitless_bfactor = args.bfactor / ((output_pixel_size * actual_prebin_for_coeff) ** 2)
        log(f"R1 CC-map coarse unitless B-factor: {coeffspline_unitless_bfactor:.6g}")

        log("Preprocessing and Fourier binning")
        output_stack = build_output_stack(movie, output_height, output_width)

        log("Full-frame motion correction")
        globally_aligned, global_x, global_y = global_align(
            output_stack,
            output_pixel_size,
            args.minimum_shift,
            args.maximum_shift,
            termination,
            args.bfactor,
            args.max_iterations,
            args.running_average,
            sg_window,
            args.smooth_global_shifts,
            args.mask_central_cross,
            args.vertical_mask_size,
            args.horizontal_mask_size,
            batch_size=args.align_batch,
        )

        del output_stack
        release_memory(device)

        local_model: Optional[MotionModel] = None
        centers: Optional[np.ndarray] = None
        patch_x: Optional[Tensor] = None
        patch_y: Optional[Tensor] = None
        bad_patches: Optional[Tensor] = None
        patch_grid_shape: Optional[Tuple[int, int]] = None
        patch_source_stack: Optional[Tensor] = None
        if args.patch_correction:
            nx, ny, patch_size, centers = auto_patch_geometry(
                output_height,
                output_width,
                output_pixel_size,
                args.distortion_model,
                args.patch_num_x,
                args.patch_num_y,
                args.patch_size,
            )
            log(f"Patch geometry: {nx} x {ny}, patch box {patch_size} px")
            patch_grid_shape = (nx, ny)

            # unbend.cpp forms the patch source from raw_image_stack after the
            # full-frame Fourier PhaseShift and then Fourier-resizes it to the
            # output/image_stack dimensions before ClipInto().  Build that stack
            # explicitly instead of clipping from the old output_stack-aligned copy.
            log("Building patch-source stack from raw frames after global Fourier shifts")
            patch_source_stack = build_patch_source_stack_from_raw(
                movie,
                output_height,
                output_width,
                global_x,
                global_y,
            )

            # The old globally_aligned stack is no longer the source used for
            # patch fitting.  For output-domain final warping, use the new
            # patch-source stack because it matches the local model source.
            del globally_aligned
            release_memory(device)

            local_model, patch_x, patch_y, bad_patches = fit_patch_motion_model(
                args.distortion_model,
                patch_source_stack,
                centers,
                patch_size,
                nx,
                ny,
                output_pixel_size,
                args.bfactor,
                args.maximum_shift,
                termination,
                args.patch_max_iterations or args.max_iterations,
                args.patch_running_average,
                args.exposure_per_frame,
                args.mask_central_cross,
                args.vertical_mask_size,
                args.horizontal_mask_size,
                args.spline_residual_refine,
                args.spline_r1_iterations,
                ccmap_unitless_bfactor=coeffspline_unitless_bfactor,
                batch_size=args.align_batch,
            )

        globally_aligned_for_final: Optional[Tensor]
        if local_model is not None:
            if args.warp_domain == "raw":
                globally_aligned_for_final = None
                if patch_source_stack is not None:
                    del patch_source_stack
                release_memory(device)
            else:
                globally_aligned_for_final = patch_source_stack
        else:
            globally_aligned_for_final = globally_aligned

        if args.write_log_files:
            save_motion_outputs(
                output_dir,
                args.input,
                output_pixel_size,
                global_x,
                global_y,
                centers,
                patch_x,
                patch_y,
                bad_patches,
                local_model,
                patch_grid_shape,
            )

        log("Applying final correction and forming sum")
        final_sum = final_sum_and_optional_frames(
            movie,
            globally_aligned_for_final,
            global_x,
            global_y,
            local_model,
            output_height,
            output_width,
            output_pixel_size,
            args.first_frame,
            args.last_frame,
            args.dose_filter,
            args.restore_power,
            args.voltage,
            args.exposure_per_frame,
            args.pre_exposure,
            args.warp_domain,
            args.warp_mode,
            args.save_aligned_frames,
        )
        write_mrc_image(args.output, final_sum.detach().cpu().numpy(), output_pixel_size)

    torch_sync(device)
    elapsed = time.perf_counter() - start_time
    log(f"Completed: {args.output}")
    if args.write_log_files:
        log(f"Shift/model/log files: {output_dir}")
    else:
        log("Shift/model/log files were not written (--write-log-files is off)")
    log(f"Elapsed: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted")
        raise SystemExit(130)
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise
