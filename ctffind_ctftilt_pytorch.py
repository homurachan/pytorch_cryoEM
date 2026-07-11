"""
A wxWidgets-free PyTorch reimplementation of the fast-search path of
CTFFIND 4.1.8, with an optional CTFTILT-style power-spectrum tilt fitter.

Implemented
-----------
* MRC input (one 2-D micrograph per file; 3-D stacks are also accepted)
* Batched PyTorch amplitude-spectrum generation and CTFFIND-like preprocessing
* Batched 1-D mean-defocus grid search
* Batched, derivative-free Powell direction-set refinement implemented in
  PyTorch; SciPy is not used for optimization
* Batched mirror/rotation estimation of the astigmatism angle
* Fixed additional phase shift in the CTF model; phase-search CLI is reserved
* One RELION-style ``.ctf`` diagnostic MRC per micrograph
* Optional tilt fitting from 256-pixel tiles using A^2-B^2 and CTF^2
* Nominal-tilt constrained axis/angle grid search and two-stage refinement
* CTFFIND-style "Thon rings with good fit up to" resolution
* RELION 3.1 ``micrographs_ctf.star`` output written without starfile
* Optional tilted-micrograph CTF refinement using a directly fitted defocus plane
* Per-micrograph colour PNG diagnostics of predicted defocus and tile residuals

Dependencies
------------
    numpy, scipy, torch, mrcfile
    matplotlib (only when tilt PNG output is enabled)

Example
-------
    python ctffind_torch_batched_powell_ctftilt_prior.py "MotionCorr/job003/*.mrc" \
        --pixel-size 1.06 --voltage 300 --cs 2.7 \
        --amplitude-contrast 0.07 --box-size 512 \
        --min-resolution 30 --max-resolution 5 \
        --min-defocus 5000 --max-defocus 50000 \
        --defocus-step 500 --preprocess-batch-size 4 --fit-batch-size 64 \
        --output micrographs_ctf.star --ctf-dir CtfFind/job005
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import savgol_filter


PI = math.pi


@dataclass(frozen=True)
class CtffindConfig:
    """Configuration for the fast CTFFIND-like search."""

    # Microscope / sampling parameters
    pixel_size_A: Optional[float] = None
    acceleration_voltage_kV: float = 300.0
    spherical_aberration_mm: float = 2.7
    amplitude_contrast: float = 0.07

    # Spectrum and fitting range
    box_size: int = 512
    minimum_resolution_A: float = 30.0  # lowest frequency used for fitting
    maximum_resolution_A: float = 5.0   # highest frequency used for fitting

    # Defocus search
    minimum_defocus_A: float = 5_000.0
    maximum_defocus_A: float = 50_000.0
    defocus_search_step_A: float = 500.0
    astigmatism_tolerance_A: float = 200.0  # negative disables restraint

    # Phase-shift interface. Searching is intentionally not implemented yet.
    find_phase_shift: bool = False
    fixed_phase_shift_rad: float = 0.0
    minimum_phase_shift_rad: float = 0.0
    maximum_phase_shift_rad: float = 3.15
    phase_shift_search_step_rad: float = 0.5

    # CTFFIND-style resampling
    resample_if_pixel_too_small: bool = True
    target_pixel_size_after_resampling_A: float = 1.4

    # Astigmatism-angle initial search
    angle_search_half_range_deg: float = 90.0
    angle_search_step_deg: float = 5.0
    angle_rotation_batch_size: int = 8

    # Batched Powell controls. Variables are internally scaled to order unity.
    powell_xtol: float = 1.0e-4
    powell_ftol: float = 1.0e-7
    powell_maxiter_1d: int = 80
    powell_maxiter_2d: int = 30
    powell_line_maxiter: int = 80
    use_powell_defocus_bounds: bool = True

    # Optional tilted-micrograph fitting.  The local tile fit estimates a mean
    # defocus for every tile, robustly fits D(x,y)=D0+gx*x+gy*y, then jointly
    # refines center D1/D2, astigmatism angle, gx and gy against all accepted
    # tile spectra using abs(CTF). Coordinates x/y are in Angstrom, so gx/gy
    # are dimensionless defocus gradients and |g| = tan(tilt_angle).
    fit_tilt: bool = False
    tilt_tile_size: int = 256
    tilt_tile_stride: int = 256
    tilt_min_global_cc: float = 0.10
    tilt_local_search_range_A: float = 12_000.0
    tilt_local_search_step_A: float = 100.0
    tilt_min_tile_cc: float = 0.04
    tilt_min_tiles: int = 6
    tilt_rms_mad_cutoff: float = 3.5
    tilt_plane_mad_cutoff: float = 3.5
    tilt_max_angle_deg: float = 80.0
    tilt_gradient_scale: float = 0.05
    tilt_powell_maxiter: int = 24

    # Nominal-angle prior for tilted micrographs.  A per-micrograph angle file
    # is handled by fit_mrc_files; this value is the optional common fallback.
    tilt_nominal_angle_deg: Optional[float] = None
    tilt_angle_uncertainty_deg: float = 5.0
    tilt_angle_grid_step_deg: float = 2.0
    tilt_axis_grid_step_deg: float = 2.0
    tilt_axis_search_half_range_deg: float = 90.0
    tilt_candidate_batch_size: int = 24
    tilt_prior_scale: float = 1.0
    tilt_hard_range_multiplier: float = 2.0
    tilt_stage1_maxiter: int = 16

    # Runtime.  Raw micrographs are large, while filtered spectra are small;
    # keep the two batch sizes independent so GPU fitting is not limited by
    # the memory footprint of 4K/8K FFT preprocessing.
    preprocess_batch_size: int = 4
    fit_batch_size: int = 64
    optimizer_check_interval: int = 8
    device: str = "auto"  # auto, cpu, cuda, cuda:0, ...

    def validate(self) -> None:
        if self.pixel_size_A is not None and self.pixel_size_A <= 0.0:
            raise ValueError("pixel_size_A must be positive")
        if not (0.0 <= self.amplitude_contrast < 1.0):
            raise ValueError("amplitude_contrast must satisfy 0 <= A < 1")
        if self.box_size < 32 or self.box_size % 2 != 0:
            raise ValueError("box_size must be an even integer >= 32")
        if self.minimum_resolution_A <= self.maximum_resolution_A:
            raise ValueError(
                "minimum_resolution_A must be numerically larger than "
                "maximum_resolution_A (for example 30 A and 5 A)"
            )
        if self.minimum_defocus_A >= self.maximum_defocus_A:
            raise ValueError("minimum_defocus_A must be smaller than maximum_defocus_A")
        if self.defocus_search_step_A <= 0.0:
            raise ValueError("defocus_search_step_A must be positive")
        if self.angle_search_step_deg <= 0.0:
            raise ValueError("angle_search_step_deg must be positive")
        if self.angle_rotation_batch_size < 1:
            raise ValueError("angle_rotation_batch_size must be >= 1")
        if self.preprocess_batch_size < 1:
            raise ValueError("preprocess_batch_size must be >= 1")
        if self.fit_batch_size < 1:
            raise ValueError("fit_batch_size must be >= 1")
        if self.optimizer_check_interval < 1:
            raise ValueError("optimizer_check_interval must be >= 1")
        if self.powell_line_maxiter < 4:
            raise ValueError("powell_line_maxiter must be >= 4")
        if self.tilt_tile_size < 64 or self.tilt_tile_size % 2 != 0:
            raise ValueError("tilt_tile_size must be an even integer >= 64")
        if self.tilt_tile_stride < 1:
            raise ValueError("tilt_tile_stride must be >= 1")
        if self.tilt_local_search_range_A <= 0.0:
            raise ValueError("tilt_local_search_range_A must be positive")
        if self.tilt_local_search_step_A <= 0.0:
            raise ValueError("tilt_local_search_step_A must be positive")
        if self.tilt_min_tiles < 3:
            raise ValueError("tilt_min_tiles must be >= 3")
        if not (0.0 < self.tilt_max_angle_deg < 89.9):
            raise ValueError("tilt_max_angle_deg must be between 0 and 89.9")
        if self.tilt_gradient_scale <= 0.0:
            raise ValueError("tilt_gradient_scale must be positive")
        if self.tilt_nominal_angle_deg is not None and abs(self.tilt_nominal_angle_deg) >= 89.9:
            raise ValueError("tilt_nominal_angle_deg must have magnitude below 89.9 degrees")
        if self.tilt_angle_uncertainty_deg <= 0.0:
            raise ValueError("tilt_angle_uncertainty_deg must be positive")
        if self.tilt_angle_grid_step_deg <= 0.0:
            raise ValueError("tilt_angle_grid_step_deg must be positive")
        if self.tilt_axis_grid_step_deg <= 0.0:
            raise ValueError("tilt_axis_grid_step_deg must be positive")
        if not (0.0 < self.tilt_axis_search_half_range_deg <= 90.0):
            raise ValueError("tilt_axis_search_half_range_deg must be in (0, 90]")
        if self.tilt_candidate_batch_size < 1:
            raise ValueError("tilt_candidate_batch_size must be >= 1")
        if self.tilt_prior_scale < 0.0:
            raise ValueError("tilt_prior_scale must be non-negative")
        if self.tilt_hard_range_multiplier < 1.0:
            raise ValueError("tilt_hard_range_multiplier must be >= 1")
        if self.tilt_stage1_maxiter < 1:
            raise ValueError("tilt_stage1_maxiter must be >= 1")


@dataclass
class CtfFitResult:
    source_file: str
    micrograph_name: str
    ctf_image_name: str
    image_index_1based: int
    pixel_size_input_A: float
    pixel_size_for_fitting_A: float
    defocus1_A: float
    defocus2_A: float
    astigmatism_angle_deg: float
    phase_shift_rad: float
    score: float
    thon_rings_good_fit_resolution_A: float
    ctf_aliasing_resolution_A: float
    coarse_defocus_A: float
    refined_mean_defocus_A: float
    initial_astigmatism_angle_deg: float
    powell_1d_success: bool
    powell_2d_success: bool
    powell_1d_nfev: int
    powell_2d_nfev: int
    powell_1d_message: str
    powell_2d_message: str
    # Optional tilted-micrograph results. NaN/False means not attempted or failed.
    global_thon_rings_good_fit_resolution_A: float = 0.0
    tilt_fitted: bool = False
    tilt_angle_deg: float = float("nan")
    tilt_axis_deg: float = float("nan")
    nominal_tilt_angle_deg: float = float("nan")
    coarse_tilt_angle_deg: float = float("nan")
    coarse_tilt_axis_deg: float = float("nan")
    defocus_gradient_x: float = float("nan")
    defocus_gradient_y: float = float("nan")
    tilt_score: float = float("nan")
    tilt_good_fit_resolution_A: float = 0.0
    tilt_residual_rms_A: float = float("nan")
    tilt_valid_tiles: int = 0
    tilt_total_tiles: int = 0
    tilt_png_name: str = ""
    tilt_message: str = "Not attempted."


@dataclass
class _TiltFitDetails:
    success: bool
    message: str
    center_defocus1_A: float
    center_defocus2_A: float
    astigmatism_angle_rad: float
    gradient_x: float
    gradient_y: float
    tilt_angle_deg: float
    tilt_axis_deg: float
    nominal_tilt_angle_deg: float
    coarse_tilt_angle_deg: float
    coarse_tilt_axis_deg: float
    score: float
    good_fit_resolution_A: float
    residual_rms_A: float
    tile_centers_x_A: np.ndarray
    tile_centers_y_A: np.ndarray
    tile_measured_defocus_A: np.ndarray
    tile_predicted_defocus_A: np.ndarray
    tile_residual_A: np.ndarray
    tile_cc: np.ndarray
    tile_good_fit_resolution_A: np.ndarray
    tile_rms_valid: np.ndarray
    tile_plane_inlier: np.ndarray
    tile_grid_y: np.ndarray
    tile_grid_x: np.ndarray
    image_shape: tuple[int, int]


@dataclass
class _SpectrumFitData:
    spectrum_values: torch.Tensor          # [B, P]
    frequency_squared_Ainv2: torch.Tensor  # [P]
    azimuth_rad: torch.Tensor              # [P]
    image_norm: torch.Tensor               # [B]
    number_of_values: int


@dataclass
class _OneDimensionalCurve:
    values: torch.Tensor                   # [B, R]
    frequencies_Ainv: torch.Tensor         # [R]


@dataclass
class _BatchedOptimizationResult:
    x: torch.Tensor
    fun: torch.Tensor
    success: torch.Tensor
    nfev: torch.Tensor
    nit: int
    messages: list[str]


@dataclass
class _GoodFitStatistics:
    thon_rings_good_fit_resolution_A: float
    ctf_aliasing_resolution_A: float
    spatial_frequency_Ainv: np.ndarray
    rotational_average_astigmatic: np.ndarray
    rotational_average_fit: np.ndarray
    fit_frc: np.ndarray
    fit_frc_sigma: np.ndarray
    prepared_spectrum: torch.Tensor
    chosen_bins: Optional[torch.Tensor]
    last_bin_without_aliasing: int
    last_bin_with_good_fit: int
    minimum_radius_pixels: float
    maximum_radius_pixels: float


def _resolve_device(device_spec: str) -> torch.device:
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({device_spec}), but CUDA is unavailable")
    return device


def _electron_wavelength_A(acceleration_voltage_kV: float) -> float:
    voltage_V = 1000.0 * acceleration_voltage_kV
    return 12.26 / math.sqrt(voltage_V + 0.9784 * voltage_V * voltage_V / 1.0e6)


def _amplitude_contrast_phase(amplitude_contrast: float) -> float:
    # Deliberately follows CTFFIND 4.1.8 ctf.cpp exactly.
    return math.atan(amplitude_contrast / math.sqrt(1.0 - amplitude_contrast))


def _ctf_abs_1d(
    frequencies_Ainv: torch.Tensor,
    defocus_A: torch.Tensor,
    wavelength_A: float,
    spherical_aberration_A: float,
    amplitude_phase_rad: float,
    phase_shift_rad: float,
) -> torch.Tensor:
    """Return abs(CTF) for isotropic defocus.

    defocus_A may be scalar or have any leading dimensions. The frequency axis
    is appended as the final dimension.
    """
    s2 = frequencies_Ainv.square()
    d = defocus_A[..., None]
    phase = (
        PI
        * wavelength_A
        * s2
        * (d - 0.5 * wavelength_A * wavelength_A * s2 * spherical_aberration_A)
        + phase_shift_rad
        + amplitude_phase_rad
    )
    return torch.sin(phase).abs()


def _ctf_abs_2d(
    frequency_squared_Ainv2: torch.Tensor,
    azimuth_rad: torch.Tensor,
    defocus1_A: float,
    defocus2_A: float,
    astigmatism_angle_rad: float,
    wavelength_A: float,
    spherical_aberration_A: float,
    amplitude_phase_rad: float,
    phase_shift_rad: float,
) -> torch.Tensor:
    effective_defocus = 0.5 * (
        defocus1_A
        + defocus2_A
        + torch.cos(2.0 * (azimuth_rad - astigmatism_angle_rad))
        * (defocus1_A - defocus2_A)
    )
    phase = (
        PI
        * wavelength_A
        * frequency_squared_Ainv2
        * (
            effective_defocus
            - 0.5
            * wavelength_A
            * wavelength_A
            * frequency_squared_Ainv2
            * spherical_aberration_A
        )
        + phase_shift_rad
        + amplitude_phase_rad
    )
    return torch.sin(phase).abs()


def _defocus_at_azimuth_A(
    defocus1_A: float,
    defocus2_A: float,
    astigmatism_angle_rad: float,
    azimuth_rad: float,
) -> float:
    return 0.5 * (
        defocus1_A
        + defocus2_A
        + math.cos(2.0 * (azimuth_rad - astigmatism_angle_rad))
        * (defocus1_A - defocus2_A)
    )


def _ctf_phase_2d_full(
    frequency_squared_Ainv2: torch.Tensor,
    azimuth_rad: torch.Tensor,
    defocus1_A: float,
    defocus2_A: float,
    astigmatism_angle_rad: float,
    wavelength_A: float,
    spherical_aberration_A: float,
    amplitude_phase_rad: float,
    phase_shift_rad: float,
) -> torch.Tensor:
    effective_defocus = 0.5 * (
        defocus1_A
        + defocus2_A
        + torch.cos(2.0 * (azimuth_rad - astigmatism_angle_rad))
        * (defocus1_A - defocus2_A)
    )
    return (
        PI
        * wavelength_A
        * frequency_squared_Ainv2
        * (
            effective_defocus
            - 0.5
            * wavelength_A
            * wavelength_A
            * frequency_squared_Ainv2
            * spherical_aberration_A
        )
        + phase_shift_rad
        + amplitude_phase_rad
    )


def _ctf_signed_2d_full(
    frequency_squared_Ainv2: torch.Tensor,
    azimuth_rad: torch.Tensor,
    defocus1_A: float,
    defocus2_A: float,
    astigmatism_angle_rad: float,
    wavelength_A: float,
    spherical_aberration_A: float,
    amplitude_phase_rad: float,
    phase_shift_rad: float,
) -> torch.Tensor:
    phase = _ctf_phase_2d_full(
        frequency_squared_Ainv2,
        azimuth_rad,
        defocus1_A,
        defocus2_A,
        astigmatism_angle_rad,
        wavelength_A,
        spherical_aberration_A,
        amplitude_phase_rad,
        phase_shift_rad,
    )
    return -torch.sin(phase)


def _number_of_extrema_from_phase(phase: torch.Tensor) -> torch.Tensor:
    # Eq. 11 of Rohou & Grigorieff (2015), matching CTF 4.1.8.
    return torch.floor(phase / PI + 0.5).abs().to(torch.int64)


def _squared_frequency_given_phase_Ainv2(
    wanted_phase_rad: float,
    azimuth_rad: float,
    defocus1_A: float,
    defocus2_A: float,
    astigmatism_angle_rad: float,
    wavelength_A: float,
    spherical_aberration_A: float,
    amplitude_phase_rad: float,
    phase_shift_rad: float,
) -> float:
    """CTFFIND's analytic phase-aberration root, in physical A^-2 units."""
    defocus_A = _defocus_at_azimuth_A(
        defocus1_A, defocus2_A, astigmatism_angle_rad, azimuth_rad
    )
    a = -0.5 * PI * wavelength_A**3 * spherical_aberration_A
    b = PI * wavelength_A * defocus_A
    c = phase_shift_rad + amplitude_phase_rad
    if spherical_aberration_A == 0.0:
        if b == 0.0:
            return 0.0
        return max(0.0, (wanted_phase_rad - c) / b)

    determinant = b * b - 4.0 * a * (c - wanted_phase_rad)
    if determinant < 0.0 or a == 0.0:
        return 0.0
    root = math.sqrt(determinant)
    solution_one = (-b + root) / (2.0 * a)
    solution_two = (-b - root) / (2.0 * a)
    if solution_one > 0.0 and solution_two > 0.0:
        return solution_one
    if solution_one > 0.0:
        return solution_one
    if solution_two > 0.0:
        return solution_two
    return 0.0


def _squared_frequency_of_zero_Ainv2(
    which_zero: int,
    azimuth_rad: float,
    defocus1_A: float,
    defocus2_A: float,
    astigmatism_angle_rad: float,
    wavelength_A: float,
    spherical_aberration_A: float,
    amplitude_phase_rad: float,
    phase_shift_rad: float,
) -> float:
    return _squared_frequency_given_phase_Ainv2(
        which_zero * PI,
        azimuth_rad,
        defocus1_A,
        defocus2_A,
        astigmatism_angle_rad,
        wavelength_A,
        spherical_aberration_A,
        amplitude_phase_rad,
        phase_shift_rad,
    )


def _center_pad_to_even_square(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D micrograph, got shape {tuple(image.shape)}")
    height, width = image.shape
    size = max(height, width)
    if size % 2:
        size += 1
    if height == size and width == size:
        return image

    padding_value = image.mean()
    output = torch.full((size, size), padding_value, dtype=image.dtype, device=image.device)
    y0 = size // 2 - height // 2
    x0 = size // 2 - width // 2
    output[y0 : y0 + height, x0 : x0 + width] = image
    return output


def _center_crop_or_pad_2d(
    image: torch.Tensor,
    output_size: int,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Centered crop/pad using CTFFIND's floor(N/2) center convention."""
    if image.ndim != 2:
        raise ValueError("_center_crop_or_pad_2d expects a 2-D tensor")
    in_h, in_w = image.shape
    output = torch.full(
        (output_size, output_size),
        padding_value,
        dtype=image.dtype,
        device=image.device,
    )

    copy_h = min(in_h, output_size)
    copy_w = min(in_w, output_size)
    src_y = in_h // 2 - copy_h // 2
    src_x = in_w // 2 - copy_w // 2
    dst_y = output_size // 2 - copy_h // 2
    dst_x = output_size // 2 - copy_w // 2
    output[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = image[
        src_y : src_y + copy_h, src_x : src_x + copy_w
    ]
    return output


def _fourier_resize_centered_real(image: torch.Tensor, output_size: int) -> torch.Tensor:
    """Fourier crop/pad a centered real 2-D image to a new square size."""
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Fourier resize expects a square 2-D tensor")
    input_size = image.shape[0]
    if input_size == output_size:
        return image.clone()

    origin_image = torch.fft.ifftshift(image)
    fourier = torch.fft.fftshift(torch.fft.fft2(origin_image))

    resized_fourier = torch.zeros(
        (output_size, output_size), dtype=fourier.dtype, device=fourier.device
    )
    copy_size = min(input_size, output_size)
    src0 = input_size // 2 - copy_size // 2
    dst0 = output_size // 2 - copy_size // 2
    resized_fourier[dst0 : dst0 + copy_size, dst0 : dst0 + copy_size] = fourier[
        src0 : src0 + copy_size, src0 : src0 + copy_size
    ]

    resized_origin = torch.fft.ifft2(torch.fft.ifftshift(resized_fourier)).real
    resized = torch.fft.fftshift(resized_origin)

    # Preserve a constant image under the different inverse-FFT normalization.
    resized *= (float(output_size) / float(input_size)) ** 2
    return resized


def _separable_periodic_box_sum(image_4d: torch.Tensor, box_size: int) -> torch.Tensor:
    """Periodic square-window sum using two exact 1-D convolutions."""
    half = box_size // 2
    horizontal_kernel = torch.ones(
        (1, 1, 1, box_size), dtype=image_4d.dtype, device=image_4d.device
    )
    vertical_kernel = torch.ones(
        (1, 1, box_size, 1), dtype=image_4d.dtype, device=image_4d.device
    )
    tmp = F.conv2d(F.pad(image_4d, (half, half, 0, 0), mode="circular"), horizontal_kernel)
    return F.conv2d(F.pad(tmp, (0, 0, half, half), mode="circular"), vertical_kernel)


def _spectrum_box_convolution(
    spectrum: torch.Tensor,
    box_size: int,
    minimum_radius_pixels: float,
) -> torch.Tensor:
    """Reproduce Image::SpectrumBoxConvolution for a 2-D spectrum."""
    if box_size % 2 == 0:
        raise ValueError("Spectrum convolution box size must be odd")
    size = spectrum.shape[0]
    center = size // 2

    y = torch.arange(size, device=spectrum.device)
    x = torch.arange(size, device=spectrum.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # CTFFIND excludes the center row/column +/- 1 from every local box.
    valid_source = (
        (torch.abs(xx - center) > 1) & (torch.abs(yy - center) > 1)
    ).to(spectrum.dtype)

    source = spectrum[None, None] * valid_source[None, None]
    counts_source = valid_source[None, None]
    local_sum = _separable_periodic_box_sum(source, box_size)[0, 0]
    local_count = _separable_periodic_box_sum(counts_source, box_size)[0, 0]
    local_average = local_sum / local_count.clamp_min(1.0)

    radius_squared = (xx - center).square() + (yy - center).square()
    inside = radius_squared <= minimum_radius_pixels * minimum_radius_pixels
    return torch.where(inside, spectrum, local_average)


def _compute_spectrum_mean_sigma(
    spectrum: torch.Tensor,
    minimum_radius_pixels: float,
    maximum_radius_pixels: float,
    cross_half_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = spectrum.shape[0]
    center = size // 2
    y = torch.arange(size, device=spectrum.device)
    x = torch.arange(size, device=spectrum.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    dx = xx - center
    dy = yy - center
    radius_squared = dx.square() + dy.square()
    mask = (
        (radius_squared > minimum_radius_pixels * minimum_radius_pixels)
        & (radius_squared < maximum_radius_pixels * maximum_radius_pixels)
        & (dx.square() > cross_half_width * cross_half_width)
        & (dy.square() > cross_half_width * cross_half_width)
    )
    values = spectrum[mask]
    if values.numel() < 2:
        raise RuntimeError("Too few spectrum pixels for sigma estimation")
    mean = values.mean()
    # EmpiricalDistribution::GetSampleVariance is most closely represented by
    # the unbiased sample variance.
    sigma = values.std(unbiased=True)
    if not torch.isfinite(sigma) or float(sigma) <= 0.0:
        raise RuntimeError("Amplitude spectrum has zero or invalid variance")
    return mean, sigma


def _ctffind_preprocess_micrograph(
    micrograph: torch.Tensor,
    pixel_size_A: float,
    config: CtffindConfig,
) -> tuple[torch.Tensor, float]:
    """Return a filtered amplitude spectrum and its effective pixel size."""
    image = _center_pad_to_even_square(micrograph)

    # CTFFIND calls ForwardFFT(false), then constructs a full, centered
    # amplitude spectrum from the half-complex FFT.
    amplitude = torch.fft.fftshift(torch.fft.fft2(image).abs())
    center = amplitude.shape[0] // 2
    amplitude[center, center] = 0.0

    fitting_pixel_size_A = pixel_size_A
    if (
        config.resample_if_pixel_too_small
        and pixel_size_A < config.target_pixel_size_after_resampling_A
    ):
        temporary_box_size = int(
            round(
                float(config.box_size)
                / pixel_size_A
                * config.target_pixel_size_after_resampling_A
            )
        )
        if temporary_box_size % 2:
            temporary_box_size += 1
        resampled = _fourier_resize_centered_real(amplitude, temporary_box_size)
        spectrum = _center_crop_or_pad_2d(resampled, config.box_size, padding_value=0.0)
        fitting_pixel_size_A = (
            pixel_size_A * float(temporary_box_size) / float(config.box_size)
        )
    else:
        spectrum = _fourier_resize_centered_real(amplitude, config.box_size)

    minimum_radius = (
        float(config.box_size)
        * fitting_pixel_size_A
        / config.minimum_resolution_A
    )
    mean, sigma = _compute_spectrum_mean_sigma(
        spectrum,
        minimum_radius_pixels=minimum_radius,
        maximum_radius_pixels=float(config.box_size),
        cross_half_width=12,
    )

    spectrum = spectrum / sigma
    cross_maximum = mean / sigma + 10.0
    spectrum = spectrum.clone()
    spectrum[config.box_size // 2, :] = torch.minimum(
        spectrum[config.box_size // 2, :], cross_maximum
    )
    spectrum[:, config.box_size // 2] = torch.minimum(
        spectrum[:, config.box_size // 2], cross_maximum
    )

    convolution_box_size = int(
        float(config.box_size)
        * fitting_pixel_size_A
        / config.minimum_resolution_A
        * math.sqrt(2.0)
    )
    if convolution_box_size % 2 == 0:
        convolution_box_size += 1
    convolution_box_size = max(1, convolution_box_size)
    if convolution_box_size >= config.box_size:
        raise RuntimeError(
            f"Background box ({convolution_box_size}) is not smaller than spectrum "
            f"box ({config.box_size}); check pixel size and minimum resolution"
        )

    background = _spectrum_box_convolution(
        spectrum,
        box_size=convolution_box_size,
        minimum_radius_pixels=minimum_radius,
    )
    spectrum = spectrum - background

    # Image::ReturnMaximumValue(3, 3): each dimension must independently be
    # at least 3 pixels from both the center and the image edge.
    size = config.box_size
    coords = torch.arange(size, device=spectrum.device)
    coordinate_is_valid = (
        (coords >= 3)
        & (coords <= size - 4)
        & (torch.abs(coords - size // 2) >= 3)
    )
    valid2d = coordinate_is_valid[:, None] & coordinate_is_valid[None, :]
    threshold = spectrum[valid2d].max()
    spectrum = torch.minimum(spectrum, threshold)

    return spectrum.contiguous(), fitting_pixel_size_A


def _rotational_average_linear(
    spectrum: torch.Tensor,
    fitting_pixel_size_A: float,
) -> _OneDimensionalCurve:
    """CTFFIND-style radial average with linear deposition between bins."""
    size = spectrum.shape[0]
    center = size // 2
    number_of_bins = int(math.ceil(math.sqrt(center * center + center * center)))

    y = torch.arange(size, dtype=spectrum.dtype, device=spectrum.device)
    x = torch.arange(size, dtype=spectrum.dtype, device=spectrum.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius_pixels = torch.sqrt((xx - center).square() + (yy - center).square()).reshape(-1)
    values = spectrum.reshape(-1)

    lower = torch.floor(radius_pixels).to(torch.int64)
    upper = lower + 1
    upper_weight = radius_pixels - lower.to(radius_pixels.dtype)
    lower_weight = 1.0 - upper_weight

    sums = torch.zeros(number_of_bins, dtype=spectrum.dtype, device=spectrum.device)
    counts = torch.zeros_like(sums)

    valid_lower = lower < number_of_bins
    sums.scatter_add_(0, lower[valid_lower], values[valid_lower] * lower_weight[valid_lower])
    counts.scatter_add_(0, lower[valid_lower], lower_weight[valid_lower])

    valid_upper = upper < number_of_bins
    sums.scatter_add_(0, upper[valid_upper], values[valid_upper] * upper_weight[valid_upper])
    counts.scatter_add_(0, upper[valid_upper], upper_weight[valid_upper])

    average = torch.where(counts > 0.0, sums / counts.clamp_min(1.0e-20), torch.zeros_like(sums))
    frequencies = torch.arange(
        number_of_bins, dtype=spectrum.dtype, device=spectrum.device
    ) / (float(size) * fitting_pixel_size_A)
    return _OneDimensionalCurve(values=average, frequencies_Ainv=frequencies)


def _make_2d_fit_data(
    spectrum: torch.Tensor,
    fitting_pixel_size_A: float,
    config: CtffindConfig,
) -> _SpectrumFitData:
    size = spectrum.shape[0]
    center = size // 2

    j = torch.arange(size, dtype=spectrum.dtype, device=spectrum.device)
    i = torch.arange(center, dtype=spectrum.dtype, device=spectrum.device)
    jj, ii = torch.meshgrid(j, i, indexing="ij")

    fx_Ainv = (ii - center) / (float(size) * fitting_pixel_size_A)
    fy_Ainv = (jj - center) / (float(size) * fitting_pixel_size_A)
    frequency_squared = fx_Ainv.square() + fy_Ainv.square()

    lowest = 1.0 / config.minimum_resolution_A
    highest = 1.0 / config.maximum_resolution_A
    central_cross_half_width = 10
    mask = (
        (frequency_squared > lowest * lowest)
        & (frequency_squared < highest * highest)
        & (ii < center - central_cross_half_width)
        & (
            (jj < center - central_cross_half_width)
            | (jj > center + central_cross_half_width)
        )
    )

    values = spectrum[:, :center][mask]
    if values.numel() == 0:
        raise RuntimeError("The 2-D fitting mask contains no pixels")
    azimuth = torch.atan2(fy_Ainv[mask], fx_Ainv[mask])
    freq2 = frequency_squared[mask]
    image_norm = torch.sqrt(torch.sum(values.square()))
    if float(image_norm) <= 0.0:
        raise RuntimeError("The filtered spectrum has zero norm in the fitting annulus")
    return _SpectrumFitData(
        spectrum_values=values,
        frequency_squared_Ainv2=freq2,
        azimuth_rad=azimuth,
        image_norm=image_norm,
        number_of_values=int(values.numel()),
    )


def _mirror_along_y_ctffind(spectrum: torch.Tensor) -> torch.Tensor:
    """Reproduce Image::ApplyMirrorAlongY for an even-sized 2-D image."""
    size = spectrum.shape[0]
    indices = torch.remainder(-torch.arange(size, device=spectrum.device), size)
    mirrored = spectrum.index_select(0, indices).clone()
    mirrored[0, :] = spectrum[0, :].mean()
    return mirrored


def _estimate_astigmatism_angle_deg(
    spectrum: torch.Tensor,
    fitting_pixel_size_A: float,
    config: CtffindConfig,
) -> float:
    """Mirror/rotation search used by CTFFIND's fast path."""
    size = spectrum.shape[0]
    center = size // 2
    mirrored = _mirror_along_y_ctffind(spectrum)

    rotations_deg = np.arange(
        -config.angle_search_half_range_deg,
        config.angle_search_half_range_deg + 0.5 * config.angle_search_step_deg,
        config.angle_search_step_deg,
        dtype=np.float32,
    )

    y = torch.arange(size, dtype=spectrum.dtype, device=spectrum.device)
    x = torch.arange(size, dtype=spectrum.dtype, device=spectrum.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    x_centered = xx - center
    y_centered = yy - center
    radius_squared_reciprocal_pixels = (
        x_centered / float(size)
    ).square() + (y_centered / float(size)).square()
    minimum_radius = fitting_pixel_size_A / config.minimum_resolution_A
    maximum_radius = fitting_pixel_size_A / config.maximum_resolution_A
    annulus = (
        (radius_squared_reciprocal_pixels >= minimum_radius * minimum_radius)
        & (radius_squared_reciprocal_pixels <= maximum_radius * maximum_radius)
    )

    input_image = spectrum[None, None]
    best_cc = -float("inf")
    best_rotation_deg = float(rotations_deg[0])

    for first in range(0, len(rotations_deg), config.angle_rotation_batch_size):
        batch_deg = rotations_deg[first : first + config.angle_rotation_batch_size]
        angles = torch.as_tensor(
            batch_deg * (PI / 180.0), dtype=spectrum.dtype, device=spectrum.device
        )
        cosine = torch.cos(angles)[:, None, None]
        sine = torch.sin(angles)[:, None, None]

        # Same output-to-input coordinate map as the C++ bilinear interpolation.
        source_x = x_centered[None] * cosine - y_centered[None] * sine + center
        source_y = x_centered[None] * sine + y_centered[None] * cosine + center
        valid_bounds = (
            (source_x >= 1.0)
            & (source_x < float(size - 1))
            & (source_y >= 1.0)
            & (source_y < float(size - 1))
        )

        x_norm = 2.0 * source_x / float(size - 1) - 1.0
        y_norm = 2.0 * source_y / float(size - 1) - 1.0
        grid = torch.stack((x_norm, y_norm), dim=-1)
        sampled = F.grid_sample(
            input_image.expand(len(batch_deg), -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0]

        mask = valid_bounds & annulus[None]
        mirror_batch = mirrored[None]
        numerator = torch.sum(torch.where(mask, sampled * mirror_batch, 0.0), dim=(1, 2))
        norm_self = torch.sum(torch.where(mask, sampled.square(), 0.0), dim=(1, 2))
        norm_other = torch.sum(torch.where(mask, mirror_batch.square(), 0.0), dim=(1, 2))
        cc = numerator / torch.sqrt((norm_self * norm_other).clamp_min(1.0e-30))

        local_index = int(torch.argmax(cc).item())
        local_cc = float(cc[local_index].item())
        if local_cc > best_cc:
            best_cc = local_cc
            best_rotation_deg = float(batch_deg[local_index])

    return 0.5 * best_rotation_deg


def _edge_mean_2d(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 2:
        raise ValueError("_edge_mean_2d expects a 2-D tensor")
    if image.shape[0] < 2 or image.shape[1] < 2:
        return image.mean()
    edge_sum = (
        image[0, :].sum()
        + image[-1, :].sum()
        + image[1:-1, 0].sum()
        + image[1:-1, -1].sum()
    )
    number_of_pixels = 2 * image.shape[1] + 2 * max(0, image.shape[0] - 2)
    return edge_sum / float(number_of_pixels)


def _circle_mask_inside_with_ring_average(
    image: torch.Tensor,
    radius_pixels: float,
) -> torch.Tensor:
    """Match Image::CircleMask(radius, invert=true) for a 2-D image."""
    size_y, size_x = image.shape
    center_y = size_y // 2
    center_x = size_x // 2
    y = torch.arange(size_y, device=image.device)
    x = torch.arange(size_x, device=image.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius_squared = (xx - center_x).square() + (yy - center_y).square()
    wanted_squared = float(radius_pixels * radius_pixels)
    ring = torch.abs(radius_squared.to(image.dtype) - wanted_squared) <= 2.0
    ring_value = image[ring].mean() if bool(ring.any()) else image.mean()
    return torch.where(radius_squared <= wanted_squared, ring_value, image)


def _astigmatism_aware_rotational_average(
    spectrum: torch.Tensor,
    fitting_pixel_size_A: float,
    defocus1_A: float,
    defocus2_A: float,
    astigmatism_angle_rad: float,
    wavelength_A: float,
    spherical_aberration_A: float,
    amplitude_phase_rad: float,
    phase_shift_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """Implement CTFFIND's extrema/CTF-value based astigmatic radial average."""
    size = int(spectrum.shape[0])
    center = size // 2
    number_of_bins = int(math.ceil(math.sqrt(center * center + center * center)))

    coords = (
        torch.arange(size, dtype=spectrum.dtype, device=spectrum.device) - center
    ) / (float(size) * fitting_pixel_size_A)
    fy, fx = torch.meshgrid(coords, coords, indexing="ij")
    frequency_squared = fx.square() + fy.square()
    azimuth = torch.atan2(fy, fx)
    phase_image = _ctf_phase_2d_full(
        frequency_squared,
        azimuth,
        defocus1_A,
        defocus2_A,
        astigmatism_angle_rad,
        wavelength_A,
        spherical_aberration_A,
        amplitude_phase_rad,
        phase_shift_rad,
    )
    ctf_image = -torch.sin(phase_image)
    extrema_image = _number_of_extrema_from_phase(phase_image)

    min_axis_distance = 10.0 * PI / 180.0
    profile_azimuth = astigmatism_angle_rad + 0.25 * PI
    angular_distance = math.fmod(profile_azimuth, 0.5 * PI)
    if abs(angular_distance) < min_axis_distance:
        profile_azimuth = min_axis_distance if angular_distance > 0.0 else -min_axis_distance
    if abs(angular_distance) > 0.5 * PI - min_axis_distance:
        profile_azimuth = (
            0.5 * PI - min_axis_distance
            if angular_distance > 0.0
            else -0.5 * PI + min_axis_distance
        )

    spatial_frequency_Ainv_t = torch.arange(
        number_of_bins, dtype=spectrum.dtype, device=spectrum.device
    ) / (float(size) * fitting_pixel_size_A)
    profile_frequency_squared = spatial_frequency_Ainv_t.square()
    profile_azimuth_t = torch.full_like(profile_frequency_squared, profile_azimuth)
    profile_phase = _ctf_phase_2d_full(
        profile_frequency_squared,
        profile_azimuth_t,
        defocus1_A,
        defocus2_A,
        astigmatism_angle_rad,
        wavelength_A,
        spherical_aberration_A,
        amplitude_phase_rad,
        phase_shift_rad,
    )
    profile_ctf = -torch.sin(profile_phase)
    profile_extrema = _number_of_extrema_from_phase(profile_phase)

    flat_extrema = extrema_image.reshape(-1)
    flat_ctf = ctf_image.reshape(-1)
    chosen_bins = torch.full_like(flat_extrema, -1, dtype=torch.int64)
    max_profile_extrema = int(profile_extrema[-1].item())
    above_profile = flat_extrema > max_profile_extrema
    chosen_bins[above_profile] = number_of_bins - 1

    # The C++ routine scans all radial bins, first matching the number of
    # preceding extrema and then choosing the closest signed CTF value. Since
    # both extrema arrays are integer-valued, grouping by extrema is equivalent
    # and much faster than an O(Npixels * Nbins) loop.
    remaining_extrema = torch.unique(flat_extrema[~above_profile])
    for extrema_value_t in remaining_extrema:
        extrema_value = int(extrema_value_t.item())
        pixel_indices = torch.nonzero(
            (flat_extrema == extrema_value) & (~above_profile), as_tuple=False
        ).flatten()
        candidate_bins = torch.nonzero(
            profile_extrema == extrema_value, as_tuple=False
        ).flatten()
        if candidate_bins.numel() == 0:
            nearest_difference = torch.abs(profile_extrema - extrema_value)
            candidate_bins = torch.nonzero(
                nearest_difference == nearest_difference.min(), as_tuple=False
            ).flatten()

        candidate_values = profile_ctf[candidate_bins]
        chunk_size = 65_536
        for first in range(0, int(pixel_indices.numel()), chunk_size):
            current_indices = pixel_indices[first : first + chunk_size]
            differences = torch.abs(
                flat_ctf[current_indices, None] - candidate_values[None, :]
            )
            nearest = torch.argmin(differences, dim=1)
            chosen_bins[current_indices] = candidate_bins[nearest]

    if bool((chosen_bins < 0).any()):
        raise RuntimeError("Could not assign all spectrum pixels to CTF radial bins")

    sums = torch.zeros(number_of_bins, dtype=spectrum.dtype, device=spectrum.device)
    counts = torch.zeros(number_of_bins, dtype=spectrum.dtype, device=spectrum.device)
    sums.scatter_add_(0, chosen_bins, spectrum.reshape(-1))
    counts.scatter_add_(0, chosen_bins, torch.ones_like(spectrum).reshape(-1))
    average = torch.where(counts > 0.0, sums / counts.clamp_min(1.0), torch.zeros_like(sums))
    average_fit = profile_ctf.abs()

    return (
        spatial_frequency_Ainv_t.detach().cpu().numpy().astype(np.float64),
        average.detach().cpu().numpy().astype(np.float64),
        average_fit.detach().cpu().numpy().astype(np.float64),
        profile_extrema.detach().cpu().numpy().astype(np.int64),
        chosen_bins.reshape(size, size),
    )


def _compute_frc_between_1d_spectrum_and_fit(
    average: np.ndarray,
    fit: np.ndarray,
    number_of_extrema_profile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CTFFIND 4.1.8 sliding-window FRC, including its edge convention."""
    number_of_bins = int(len(average))
    if number_of_bins < 3:
        return np.zeros(number_of_bins), np.zeros(number_of_bins)

    minimum_window_half_width = max(1, number_of_bins // 40)
    half_window_width = np.full(
        number_of_bins, minimum_window_half_width, dtype=np.int64
    )
    bin_of_previous_extremum = 0
    for bin_counter in range(1, number_of_bins):
        if number_of_extrema_profile[bin_counter] != number_of_extrema_profile[bin_counter - 1]:
            width = max(
                minimum_window_half_width,
                int(1.5 * float(bin_counter - bin_of_previous_extremum + 1)),
            )
            width = min(width, number_of_bins // 2 - 1)
            half_window_width[bin_of_previous_extremum:bin_counter] = width
            bin_of_previous_extremum = bin_counter
    half_window_width[0] = half_window_width[1]
    tail_width = (
        half_window_width[bin_of_previous_extremum - 1]
        if bin_of_previous_extremum > 0
        else half_window_width[0]
    )
    half_window_width[bin_of_previous_extremum:] = tail_width

    frc = np.zeros(number_of_bins, dtype=np.float64)
    frc_sigma = np.zeros(number_of_bins, dtype=np.float64)
    for bin_counter in range(number_of_bins):
        half_width = int(half_window_width[bin_counter])
        first_bin = bin_counter - half_width
        last_bin = bin_counter + half_width
        if first_bin < 0:
            first_bin = 0
            last_bin = 2 * half_width + 1
        if last_bin >= number_of_bins:
            last_bin = number_of_bins - 1
            first_bin = last_bin - 2 * half_width - 1
        first_bin = max(0, first_bin)
        last_bin = min(number_of_bins - 1, last_bin)

        window_average = average[first_bin : last_bin + 1]
        window_fit = fit[first_bin : last_bin + 1]
        # CTFFIND divides by 2*h+1 even at the two edges, where the inclusive
        # window contains 2*h+2 values. Keep this behavior for numerical match.
        number_in_window = float(2 * half_width + 1)
        spectrum_mean = float(window_average.sum()) / number_in_window
        fit_mean = float(window_fit.sum()) / number_in_window
        spectrum_delta = window_average - spectrum_mean
        fit_delta = window_fit - fit_mean
        cross_product = float(np.sum(spectrum_delta * fit_delta))
        spectrum_sigma = float(np.sum(spectrum_delta * spectrum_delta))
        fit_sigma = float(np.sum(fit_delta * fit_delta))
        if spectrum_sigma > 0.0 and fit_sigma > 0.0:
            frc[bin_counter] = (
                cross_product
                / (math.sqrt(spectrum_sigma / number_in_window)
                   * math.sqrt(fit_sigma / number_in_window))
                / number_in_window
            )
            frc[bin_counter] = min(1.0, max(-1.0, frc[bin_counter]))
        frc_sigma[bin_counter] = 2.0 / math.sqrt(number_in_window)
    return frc, frc_sigma


def _find_good_fit_and_aliasing_bins(
    fit_frc: np.ndarray,
    number_of_extrema_profile: np.ndarray,
    first_zero_frequency_Ainv: float,
    size: int,
    fitting_pixel_size_A: float,
) -> tuple[int, int]:
    number_of_bins = int(len(fit_frc))
    first_bin_to_check = int(first_zero_frequency_Ainv * size * fitting_pixel_size_A)
    first_bin_to_check = min(max(first_bin_to_check, 0), number_of_bins - 1)

    low_threshold = 0.2
    significance_threshold = 0.5
    number_above_low = 0
    number_above_significance = 0
    last_good = -1
    for counter in range(first_bin_to_check, number_of_bins):
        at_last_good = (
            number_above_low > 3 and fit_frc[counter] < low_threshold
        ) or (
            number_above_significance > 3
            and fit_frc[counter] < significance_threshold
        )
        if at_last_good:
            last_good = counter
            break
        if fit_frc[counter] > low_threshold:
            number_above_low += 1
        if fit_frc[counter] > significance_threshold:
            number_above_significance += 1

    if number_above_significance == number_of_bins - first_bin_to_check:
        last_good = number_of_bins - 1
    if last_good < 0 or last_good >= number_of_bins:
        last_good = 0

    last_without_aliasing = 0
    previous_extremum = 0
    for counter in range(1, number_of_bins):
        if number_of_extrema_profile[counter] - number_of_extrema_profile[counter - 1] >= 1:
            if counter - previous_extremum < 4:
                last_without_aliasing = previous_extremum
                break
            previous_extremum = counter
    return last_good, last_without_aliasing


def _smooth_extrema_envelope(
    point_x: list[float],
    point_y: list[float],
    target_x: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    if len(point_x) <= 7:
        return fallback
    values = np.asarray(point_y, dtype=np.float64)
    smoothed = savgol_filter(values, window_length=7, polyorder=2, mode="interp")
    return np.interp(target_x, np.asarray(point_x), smoothed)


def _rescale_spectrum_for_diagnostic(
    spectrum: torch.Tensor,
    spatial_frequency_Ainv: np.ndarray,
    average: np.ndarray,
    average_fit: np.ndarray,
    chosen_bins: torch.Tensor,
    last_bin_without_aliasing: int,
    last_bin_with_good_fit: int,
) -> torch.Tensor:
    """Background-envelope subtraction used to enhance CTFFIND diagnostics."""
    number_of_bins = len(average)
    background = np.zeros(number_of_bins, dtype=np.float64)
    peak = np.zeros(number_of_bins, dtype=np.float64)
    previous_maximum = 0
    previous_minimum = 0
    current_maximum_number = 0
    at_maximum = False
    at_minimum = True
    normalization_bin: Optional[int] = None
    minima_x: list[float] = []
    minima_y: list[float] = []
    maxima_x: list[float] = []
    maxima_y: list[float] = []

    for bin_counter in range(1, number_of_bins - 1):
        maximum_at_previous = at_maximum
        minimum_at_previous = at_minimum
        at_minimum = (
            average_fit[bin_counter] <= average_fit[bin_counter - 1]
            and average_fit[bin_counter] <= average_fit[bin_counter + 1]
        )
        at_maximum = (
            average_fit[bin_counter] >= average_fit[bin_counter - 1]
            and average_fit[bin_counter] >= average_fit[bin_counter + 1]
        )
        if at_maximum and at_minimum:
            at_minimum = minimum_at_previous
            at_maximum = maximum_at_previous

        if at_minimum and bin_counter > previous_minimum:
            indices = np.arange(previous_minimum + 1, bin_counter + 1)
            background[indices] = (
                average[previous_minimum]
                * (bin_counter - indices)
                / float(bin_counter - previous_minimum)
                + average[bin_counter]
                * (indices - previous_minimum)
                / float(bin_counter - previous_minimum)
            )
            previous_minimum = bin_counter
            minima_x.append(float(spatial_frequency_Ainv[bin_counter]))
            minima_y.append(float(average[bin_counter]))

        if at_maximum and bin_counter > previous_maximum:
            if (not maximum_at_previous) and average_fit[bin_counter] > 0.7:
                current_maximum_number += 1
            indices = np.arange(previous_maximum + 1, bin_counter + 1)
            peak[indices] = (
                average[previous_maximum]
                * (bin_counter - indices)
                / float(bin_counter - previous_maximum)
                + average[bin_counter]
                * (indices - previous_maximum)
                / float(bin_counter - previous_maximum)
            )
            if current_maximum_number == 2:
                normalization_bin = bin_counter
            previous_maximum = bin_counter
            maxima_x.append(float(spatial_frequency_Ainv[bin_counter]))
            maxima_y.append(float(average[bin_counter]))

    background = _smooth_extrema_envelope(
        minima_x, minima_y, spatial_frequency_Ainv, background
    )
    peak = _smooth_extrema_envelope(maxima_x, maxima_y, spatial_frequency_Ainv, peak)
    if normalization_bin is None:
        differences = peak - background
        normalization_bin = int(np.argmax(differences))

    if last_bin_without_aliasing != 0:
        last_bin_to_rescale = min(last_bin_with_good_fit, last_bin_without_aliasing)
    else:
        last_bin_to_rescale = last_bin_with_good_fit
    last_bin_to_rescale = min(max(last_bin_to_rescale, 0), number_of_bins - 1)

    if peak[normalization_bin] - background[normalization_bin] <= 0.0:
        return spectrum
    background_t = torch.as_tensor(
        background, dtype=spectrum.dtype, device=spectrum.device
    )
    lookup_bins = torch.minimum(
        chosen_bins,
        torch.tensor(last_bin_to_rescale, device=chosen_bins.device),
    )
    return spectrum - background_t[lookup_bins]


def _compute_good_fit_statistics(
    filtered_spectrum: torch.Tensor,
    fitting_pixel_size_A: float,
    config: CtffindConfig,
    defocus1_A: float,
    defocus2_A: float,
    astigmatism_angle_rad: float,
    wavelength_A: float,
    spherical_aberration_A: float,
    amplitude_phase_rad: float,
    phase_shift_rad: float,
    *,
    keep_diagnostic_support: bool,
) -> _GoodFitStatistics:
    """Compute CTFFIND's good-fit and aliasing statistics.

    This intentionally stops before the display-only Savitzky-Golay envelope
    subtraction, contrast clipping, and theoretical-quadrant overlay.  Thus
    ``--no-diagnostic-output`` no longer pays the cost of rendering a map.
    """
    spectrum = filtered_spectrum.clone()
    size = int(spectrum.shape[0])
    center = size // 2
    spectrum = spectrum - _edge_mean_2d(spectrum)

    zero_1_sq = _squared_frequency_of_zero_Ainv2(
        1, 0.0, defocus1_A, defocus2_A, astigmatism_angle_rad,
        wavelength_A, spherical_aberration_A, amplitude_phase_rad, phase_shift_rad
    )
    zero_2_sq = _squared_frequency_of_zero_Ainv2(
        2, 0.0, defocus1_A, defocus2_A, astigmatism_angle_rad,
        wavelength_A, spherical_aberration_A, amplitude_phase_rad, phase_shift_rad
    )
    zero_3_sq = _squared_frequency_of_zero_Ainv2(
        3, 0.0, defocus1_A, defocus2_A, astigmatism_angle_rad,
        wavelength_A, spherical_aberration_A, amplitude_phase_rad, phase_shift_rad
    )
    minimum_radius = math.sqrt(max(0.0, zero_2_sq)) * size * fitting_pixel_size_A
    maximum_frequency = max(
        1.0 / config.maximum_resolution_A,
        math.sqrt(max(0.0, zero_3_sq)),
    )
    maximum_radius = maximum_frequency * size * fitting_pixel_size_A

    average, sigma = _compute_spectrum_mean_sigma(
        spectrum, minimum_radius, maximum_radius, cross_half_width=2
    )
    spectrum = _circle_mask_inside_with_ring_average(spectrum, 5.0)
    spectrum = spectrum.clone()
    spectrum[center, :] = torch.minimum(spectrum[center, :], average)
    spectrum[:, center] = torch.minimum(spectrum[:, center], average)
    spectrum = torch.clamp(
        spectrum, min=average - 4.0 * sigma, max=average + 4.0 * sigma
    )
    average, sigma = _compute_spectrum_mean_sigma(
        spectrum, minimum_radius, maximum_radius, cross_half_width=2
    )
    spectrum = (spectrum - average) / sigma + average

    (
        spatial_frequency_Ainv,
        rotational_average_astigmatic,
        rotational_average_fit,
        number_of_extrema_profile,
        chosen_bins,
    ) = _astigmatism_aware_rotational_average(
        spectrum,
        fitting_pixel_size_A,
        defocus1_A,
        defocus2_A,
        astigmatism_angle_rad,
        wavelength_A,
        spherical_aberration_A,
        amplitude_phase_rad,
        phase_shift_rad,
    )
    fit_frc, fit_frc_sigma = _compute_frc_between_1d_spectrum_and_fit(
        rotational_average_astigmatic,
        rotational_average_fit,
        number_of_extrema_profile,
    )
    last_good, last_without_aliasing = _find_good_fit_and_aliasing_bins(
        fit_frc,
        number_of_extrema_profile,
        math.sqrt(max(0.0, zero_1_sq)),
        size,
        fitting_pixel_size_A,
    )

    if last_good == 0 or spatial_frequency_Ainv[last_good] <= 0.0:
        good_fit_resolution_A = 0.0
    else:
        good_fit_resolution_A = 1.0 / float(spatial_frequency_Ainv[last_good])
    if (
        last_without_aliasing == 0
        or spatial_frequency_Ainv[last_without_aliasing] <= 0.0
    ):
        aliasing_resolution_A = 0.0
    else:
        aliasing_resolution_A = 1.0 / float(
            spatial_frequency_Ainv[last_without_aliasing]
        )

    return _GoodFitStatistics(
        thon_rings_good_fit_resolution_A=float(good_fit_resolution_A),
        ctf_aliasing_resolution_A=float(aliasing_resolution_A),
        spatial_frequency_Ainv=spatial_frequency_Ainv,
        rotational_average_astigmatic=rotational_average_astigmatic,
        rotational_average_fit=rotational_average_fit,
        fit_frc=fit_frc,
        fit_frc_sigma=fit_frc_sigma,
        prepared_spectrum=spectrum if keep_diagnostic_support else torch.empty(
            0, dtype=spectrum.dtype, device=spectrum.device
        ),
        chosen_bins=chosen_bins if keep_diagnostic_support else None,
        last_bin_without_aliasing=int(last_without_aliasing),
        last_bin_with_good_fit=int(last_good),
        minimum_radius_pixels=float(minimum_radius),
        maximum_radius_pixels=float(maximum_radius),
    )


def _render_diagnostic_map(
    statistics: _GoodFitStatistics,
    fitting_pixel_size_A: float,
    config: CtffindConfig,
    defocus1_A: float,
    defocus2_A: float,
    astigmatism_angle_rad: float,
    wavelength_A: float,
    spherical_aberration_A: float,
    amplitude_phase_rad: float,
    phase_shift_rad: float,
) -> torch.Tensor:
    """Render the display-only RELION/CTFFIND diagnostic map."""
    if statistics.chosen_bins is None or statistics.prepared_spectrum.numel() == 0:
        raise ValueError("Diagnostic support was not retained for this fit")

    spectrum = statistics.prepared_spectrum.clone()
    size = int(spectrum.shape[0])
    center = size // 2
    spectrum = _rescale_spectrum_for_diagnostic(
        spectrum,
        statistics.spatial_frequency_Ainv,
        statistics.rotational_average_astigmatic,
        statistics.rotational_average_fit,
        statistics.chosen_bins,
        statistics.last_bin_without_aliasing,
        statistics.last_bin_with_good_fit,
    )
    average, sigma = _compute_spectrum_mean_sigma(
        spectrum,
        statistics.minimum_radius_pixels,
        statistics.maximum_radius_pixels,
        cross_half_width=2,
    )
    spectrum = torch.clamp(spectrum, min=average - sigma, max=average + 2.0 * sigma)

    coords = (
        torch.arange(size, dtype=spectrum.dtype, device=spectrum.device) - center
    ) / (float(size) * fitting_pixel_size_A)
    fy, fx = torch.meshgrid(coords, coords, indexing="ij")
    frequency_squared = fx.square() + fy.square()
    azimuth = torch.atan2(fy, fx)
    signed_ctf = _ctf_signed_2d_full(
        frequency_squared,
        azimuth,
        defocus1_A,
        defocus2_A,
        astigmatism_angle_rad,
        wavelength_A,
        spherical_aberration_A,
        amplitude_phase_rad,
        phase_shift_rad,
    )
    lowest_frequency = 1.0 / config.minimum_resolution_A
    highest_frequency = 1.0 / config.maximum_resolution_A
    y = torch.arange(size, device=spectrum.device)[:, None]
    x = torch.arange(size, device=spectrum.device)[None, :]
    fitting_annulus = (
        (frequency_squared > lowest_frequency * lowest_frequency)
        & (frequency_squared <= highest_frequency * highest_frequency)
    )
    theoretical_quadrant = fitting_annulus & (y < center) & (x < center)
    spectrum = torch.where(theoretical_quadrant, signed_ctf.abs(), spectrum)
    spectrum = torch.where(
        frequency_squared <= lowest_frequency * lowest_frequency,
        torch.zeros_like(spectrum),
        spectrum,
    )
    return spectrum.contiguous()


def _center_pad_to_even_square_batch(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 3:
        raise ValueError(f"Expected [B,H,W] micrographs, got {tuple(images.shape)}")
    batch, height, width = images.shape
    size = max(height, width)
    if size % 2:
        size += 1
    if height == size and width == size:
        return images
    means = images.mean(dim=(1, 2), keepdim=True)
    output = means.expand(batch, size, size).clone()
    y0 = size // 2 - height // 2
    x0 = size // 2 - width // 2
    output[:, y0:y0 + height, x0:x0 + width] = images
    return output


def _center_crop_or_pad_batch(
    images: torch.Tensor,
    output_size: int,
    padding_value: float = 0.0,
) -> torch.Tensor:
    if images.ndim != 3:
        raise ValueError("_center_crop_or_pad_batch expects [B,H,W]")
    batch, in_h, in_w = images.shape
    output = torch.full(
        (batch, output_size, output_size),
        padding_value,
        dtype=images.dtype,
        device=images.device,
    )
    copy_h = min(in_h, output_size)
    copy_w = min(in_w, output_size)
    src_y = in_h // 2 - copy_h // 2
    src_x = in_w // 2 - copy_w // 2
    dst_y = output_size // 2 - copy_h // 2
    dst_x = output_size // 2 - copy_w // 2
    output[:, dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = images[
        :, src_y:src_y + copy_h, src_x:src_x + copy_w
    ]
    return output


def _fourier_resize_centered_real_batch(
    images: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    if images.ndim != 3 or images.shape[-1] != images.shape[-2]:
        raise ValueError("Fourier resize expects [B,N,N]")
    input_size = images.shape[-1]
    if input_size == output_size:
        return images.clone()
    origin = torch.fft.ifftshift(images, dim=(-2, -1))
    fourier = torch.fft.fftshift(torch.fft.fft2(origin), dim=(-2, -1))
    resized_fourier = torch.zeros(
        (images.shape[0], output_size, output_size),
        dtype=fourier.dtype,
        device=fourier.device,
    )
    copy_size = min(input_size, output_size)
    src0 = input_size // 2 - copy_size // 2
    dst0 = output_size // 2 - copy_size // 2
    resized_fourier[:, dst0:dst0 + copy_size, dst0:dst0 + copy_size] = fourier[
        :, src0:src0 + copy_size, src0:src0 + copy_size
    ]
    resized_origin = torch.fft.ifft2(
        torch.fft.ifftshift(resized_fourier, dim=(-2, -1))
    ).real
    resized = torch.fft.fftshift(resized_origin, dim=(-2, -1))
    resized *= (float(output_size) / float(input_size)) ** 2
    return resized


def _compute_spectrum_mean_sigma_batch(
    spectra: torch.Tensor,
    minimum_radius_pixels: float,
    maximum_radius_pixels: float,
    cross_half_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = spectra.shape[-1]
    center = size // 2
    y = torch.arange(size, device=spectra.device)
    x = torch.arange(size, device=spectra.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    dx = xx - center
    dy = yy - center
    radius_squared = dx.square() + dy.square()
    mask = (
        (radius_squared > minimum_radius_pixels ** 2)
        & (radius_squared < maximum_radius_pixels ** 2)
        & (dx.square() > cross_half_width ** 2)
        & (dy.square() > cross_half_width ** 2)
    )
    values = spectra[:, mask]
    if values.shape[1] < 2:
        raise RuntimeError("Too few spectrum pixels for sigma estimation")
    mean = values.mean(dim=1)
    sigma = values.std(dim=1, unbiased=True)
    if not torch.all(torch.isfinite(sigma) & (sigma > 0.0)):
        raise RuntimeError("At least one amplitude spectrum has invalid variance")
    return mean, sigma


def _spectrum_box_convolution_batch(
    spectra: torch.Tensor,
    box_size: int,
    minimum_radius_pixels: float,
) -> torch.Tensor:
    if box_size % 2 == 0:
        raise ValueError("Spectrum convolution box size must be odd")
    size = spectra.shape[-1]
    center = size // 2
    y = torch.arange(size, device=spectra.device)
    x = torch.arange(size, device=spectra.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    valid_source = (
        (torch.abs(xx - center) > 1) & (torch.abs(yy - center) > 1)
    ).to(spectra.dtype)
    source = spectra[:, None] * valid_source[None, None]
    counts_source = valid_source[None, None].expand(spectra.shape[0], -1, -1, -1)
    local_sum = _separable_periodic_box_sum(source, box_size)[:, 0]
    local_count = _separable_periodic_box_sum(counts_source, box_size)[:, 0]
    local_average = local_sum / local_count.clamp_min(1.0)
    radius_squared = (xx - center).square() + (yy - center).square()
    inside = radius_squared <= minimum_radius_pixels ** 2
    return torch.where(inside[None], spectra, local_average)


def _ctffind_preprocess_batch(
    micrographs: torch.Tensor,
    pixel_size_A: float,
    config: CtffindConfig,
) -> tuple[torch.Tensor, float]:
    images = _center_pad_to_even_square_batch(micrographs)
    amplitudes = torch.fft.fftshift(
        torch.fft.fft2(images).abs(), dim=(-2, -1)
    )
    center = amplitudes.shape[-1] // 2
    amplitudes[:, center, center] = 0.0

    fitting_pixel_size_A = pixel_size_A
    if (
        config.resample_if_pixel_too_small
        and pixel_size_A < config.target_pixel_size_after_resampling_A
    ):
        temporary_box_size = int(round(
            float(config.box_size) / pixel_size_A
            * config.target_pixel_size_after_resampling_A
        ))
        if temporary_box_size % 2:
            temporary_box_size += 1
        resampled = _fourier_resize_centered_real_batch(
            amplitudes, temporary_box_size
        )
        spectra = _center_crop_or_pad_batch(
            resampled, config.box_size, padding_value=0.0
        )
        fitting_pixel_size_A = (
            pixel_size_A * float(temporary_box_size) / float(config.box_size)
        )
    else:
        spectra = _fourier_resize_centered_real_batch(
            amplitudes, config.box_size
        )

    minimum_radius = (
        float(config.box_size) * fitting_pixel_size_A
        / config.minimum_resolution_A
    )
    mean, sigma = _compute_spectrum_mean_sigma_batch(
        spectra,
        minimum_radius_pixels=minimum_radius,
        maximum_radius_pixels=float(config.box_size),
        cross_half_width=12,
    )
    spectra = spectra / sigma[:, None, None]
    cross_maximum = mean / sigma + 10.0
    spectra = spectra.clone()
    c = config.box_size // 2
    spectra[:, c, :] = torch.minimum(spectra[:, c, :], cross_maximum[:, None])
    spectra[:, :, c] = torch.minimum(spectra[:, :, c], cross_maximum[:, None])

    convolution_box_size = int(
        float(config.box_size) * fitting_pixel_size_A
        / config.minimum_resolution_A * math.sqrt(2.0)
    )
    if convolution_box_size % 2 == 0:
        convolution_box_size += 1
    convolution_box_size = max(1, convolution_box_size)
    if convolution_box_size >= config.box_size:
        raise RuntimeError(
            f"Background box ({convolution_box_size}) is not smaller than "
            f"spectrum box ({config.box_size})"
        )
    background = _spectrum_box_convolution_batch(
        spectra, convolution_box_size, minimum_radius
    )
    spectra = spectra - background

    size = config.box_size
    coords = torch.arange(size, device=spectra.device)
    coordinate_is_valid = (
        (coords >= 3) & (coords <= size - 4)
        & (torch.abs(coords - size // 2) >= 3)
    )
    valid2d = coordinate_is_valid[:, None] & coordinate_is_valid[None, :]
    threshold = spectra[:, valid2d].amax(dim=1)
    spectra = torch.minimum(spectra, threshold[:, None, None])
    return spectra.contiguous(), fitting_pixel_size_A


def _ctftilt_preprocess_batch(
    tiles: torch.Tensor,
    pixel_size_A: float,
    config: CtffindConfig,
) -> tuple[torch.Tensor, float]:
    """Prepare local tile power spectra for tilt fitting.

    This follows the important CTFTILT convention rather than the global
    CTFFIND amplitude-spectrum convention:

        experimental = |FFT|^2 - background(|FFT|)^2
        theoretical  = CTF^2

    The slowly varying background is estimated on the amplitude spectrum
    using a 2*N/10+1 square window, matching the original CTFTILT choice.
    Global CTFFIND fitting is deliberately left unchanged.
    """
    images = _center_pad_to_even_square_batch(tiles)
    amplitudes = torch.fft.fftshift(
        torch.fft.fft2(images).abs(), dim=(-2, -1)
    )
    center = amplitudes.shape[-1] // 2
    amplitudes[:, center, center] = 0.0

    fitting_pixel_size_A = pixel_size_A
    if (
        config.resample_if_pixel_too_small
        and pixel_size_A < config.target_pixel_size_after_resampling_A
    ):
        temporary_box_size = int(round(
            float(config.box_size) / pixel_size_A
            * config.target_pixel_size_after_resampling_A
        ))
        if temporary_box_size % 2:
            temporary_box_size += 1
        amplitudes = _fourier_resize_centered_real_batch(
            amplitudes, temporary_box_size
        )
        amplitudes = _center_crop_or_pad_batch(
            amplitudes, config.box_size, padding_value=0.0
        )
        fitting_pixel_size_A = (
            pixel_size_A * float(temporary_box_size) / float(config.box_size)
        )
    elif amplitudes.shape[-1] != config.box_size:
        amplitudes = _fourier_resize_centered_real_batch(
            amplitudes, config.box_size
        )

    # Original CTFTILT: NW=N/10 and window width=2*NW+1.
    half_background_width = max(1, config.box_size // 10)
    background_box_size = 2 * half_background_width + 1
    if background_box_size >= config.box_size:
        background_box_size = config.box_size - 1
        if background_box_size % 2 == 0:
            background_box_size -= 1

    background_amplitude = _spectrum_box_convolution_batch(
        amplitudes,
        box_size=background_box_size,
        minimum_radius_pixels=0.0,
    )

    # Do not use (A-B)^2. CTFTILT explicitly computes A^2-B^2 after
    # smoothing sqrt(power), because this preserves signed background
    # residuals instead of turning every residual positive.
    filtered_power = amplitudes.square() - background_amplitude.square()

    minimum_radius = (
        float(config.box_size) * fitting_pixel_size_A
        / config.minimum_resolution_A
    )
    maximum_radius = (
        float(config.box_size) * fitting_pixel_size_A
        / config.maximum_resolution_A
    )
    _, sigma = _compute_spectrum_mean_sigma_batch(
        filtered_power,
        minimum_radius_pixels=minimum_radius,
        maximum_radius_pixels=maximum_radius,
        cross_half_width=2,
    )
    filtered_power = filtered_power / sigma[:, None, None].clamp_min(1.0e-20)
    return filtered_power.contiguous(), fitting_pixel_size_A


def _rotational_average_linear_batch(
    spectra: torch.Tensor,
    fitting_pixel_size_A: float,
) -> _OneDimensionalCurve:
    batch, size, _ = spectra.shape
    center = size // 2
    number_of_bins = int(math.ceil(math.sqrt(center * center + center * center)))
    y = torch.arange(size, dtype=spectra.dtype, device=spectra.device)
    x = torch.arange(size, dtype=spectra.dtype, device=spectra.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt((xx - center).square() + (yy - center).square()).reshape(-1)
    lower = torch.floor(radius).to(torch.int64)
    upper = lower + 1
    upper_weight = radius - lower.to(radius.dtype)
    lower_weight = 1.0 - upper_weight
    values = spectra.reshape(batch, -1)
    sums = torch.zeros(
        (batch, number_of_bins), dtype=spectra.dtype, device=spectra.device
    )
    counts = torch.zeros_like(sums)
    valid_lower = lower < number_of_bins
    li = lower[valid_lower][None].expand(batch, -1)
    lw = lower_weight[valid_lower][None]
    sums.scatter_add_(1, li, values[:, valid_lower] * lw)
    counts.scatter_add_(1, li, lw.expand(batch, -1))
    valid_upper = upper < number_of_bins
    ui = upper[valid_upper][None].expand(batch, -1)
    uw = upper_weight[valid_upper][None]
    sums.scatter_add_(1, ui, values[:, valid_upper] * uw)
    counts.scatter_add_(1, ui, uw.expand(batch, -1))
    average = torch.where(
        counts > 0.0, sums / counts.clamp_min(1.0e-20), torch.zeros_like(sums)
    )
    frequencies = torch.arange(
        number_of_bins, dtype=spectra.dtype, device=spectra.device
    ) / (float(size) * fitting_pixel_size_A)
    return _OneDimensionalCurve(average, frequencies)


def _make_2d_fit_data_batch(
    spectra: torch.Tensor,
    fitting_pixel_size_A: float,
    config: CtffindConfig,
) -> _SpectrumFitData:
    batch, size, _ = spectra.shape
    center = size // 2
    j = torch.arange(size, dtype=spectra.dtype, device=spectra.device)
    i = torch.arange(center, dtype=spectra.dtype, device=spectra.device)
    jj, ii = torch.meshgrid(j, i, indexing="ij")
    fx = (ii - center) / (float(size) * fitting_pixel_size_A)
    fy = (jj - center) / (float(size) * fitting_pixel_size_A)
    freq2 = fx.square() + fy.square()
    lowest = 1.0 / config.minimum_resolution_A
    highest = 1.0 / config.maximum_resolution_A
    cross = 10
    mask = (
        (freq2 > lowest * lowest) & (freq2 < highest * highest)
        & (ii < center - cross)
        & ((jj < center - cross) | (jj > center + cross))
    )
    flat_mask = mask.reshape(-1)
    values = spectra[:, :, :center].reshape(batch, -1)[:, flat_mask]
    if values.shape[1] == 0:
        raise RuntimeError("The 2-D fitting mask contains no pixels")
    azimuth = torch.atan2(fy.reshape(-1)[flat_mask], fx.reshape(-1)[flat_mask])
    selected_freq2 = freq2.reshape(-1)[flat_mask]
    image_norm = torch.sqrt(torch.sum(values.square(), dim=1))
    if not torch.all(image_norm > 0.0):
        raise RuntimeError("At least one spectrum has zero fitting-annulus norm")
    return _SpectrumFitData(
        spectrum_values=values,
        frequency_squared_Ainv2=selected_freq2,
        azimuth_rad=azimuth,
        image_norm=image_norm,
        number_of_values=int(values.shape[1]),
    )


def _mirror_along_y_ctffind_batch(spectra: torch.Tensor) -> torch.Tensor:
    size = spectra.shape[-1]
    indices = torch.remainder(-torch.arange(size, device=spectra.device), size)
    mirrored = spectra.index_select(1, indices).clone()
    mirrored[:, 0, :] = spectra[:, 0, :].mean(dim=1)[:, None]
    return mirrored


def _estimate_astigmatism_angle_deg_batch(
    spectra: torch.Tensor,
    fitting_pixel_size_A: float,
    config: CtffindConfig,
) -> torch.Tensor:
    batch, size, _ = spectra.shape
    center = size // 2
    mirrored = _mirror_along_y_ctffind_batch(spectra)
    rotations_deg = np.arange(
        -config.angle_search_half_range_deg,
        config.angle_search_half_range_deg + 0.5 * config.angle_search_step_deg,
        config.angle_search_step_deg,
        dtype=np.float32,
    )
    y = torch.arange(size, dtype=spectra.dtype, device=spectra.device)
    x = torch.arange(size, dtype=spectra.dtype, device=spectra.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xc = xx - center
    yc = yy - center
    r2 = (xc / float(size)).square() + (yc / float(size)).square()
    rmin = fitting_pixel_size_A / config.minimum_resolution_A
    rmax = fitting_pixel_size_A / config.maximum_resolution_A
    annulus = (r2 >= rmin * rmin) & (r2 <= rmax * rmax)
    best_cc = torch.full((batch,), -torch.inf, dtype=spectra.dtype, device=spectra.device)
    best_rotation = torch.full_like(best_cc, float(rotations_deg[0]))
    input_images = spectra[:, None]
    for first in range(0, len(rotations_deg), config.angle_rotation_batch_size):
        chunk = rotations_deg[first:first + config.angle_rotation_batch_size]
        k = len(chunk)
        angles = torch.as_tensor(
            chunk * (PI / 180.0), dtype=spectra.dtype, device=spectra.device
        )
        cosine = torch.cos(angles)[:, None, None]
        sine = torch.sin(angles)[:, None, None]
        source_x = xc[None] * cosine - yc[None] * sine + center
        source_y = xc[None] * sine + yc[None] * cosine + center
        valid = (
            (source_x >= 1.0) & (source_x < float(size - 1))
            & (source_y >= 1.0) & (source_y < float(size - 1))
        )
        grid = torch.stack(
            (2.0 * source_x / float(size - 1) - 1.0,
             2.0 * source_y / float(size - 1) - 1.0),
            dim=-1,
        )
        expanded_images = input_images[:, None].expand(batch, k, 1, size, size)
        expanded_grid = grid[None].expand(batch, k, size, size, 2)
        sampled = F.grid_sample(
            expanded_images.reshape(batch * k, 1, size, size),
            expanded_grid.reshape(batch * k, size, size, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0].reshape(batch, k, size, size)
        mask = valid[None] & annulus[None, None]
        mirror = mirrored[:, None]
        numerator = torch.sum(
            torch.where(mask, sampled * mirror, 0.0), dim=(2, 3)
        )
        norm_self = torch.sum(
            torch.where(mask, sampled.square(), 0.0), dim=(2, 3)
        )
        norm_other = torch.sum(
            torch.where(mask, mirror.square(), 0.0), dim=(2, 3)
        )
        cc = numerator / torch.sqrt((norm_self * norm_other).clamp_min(1.0e-30))
        local_cc, local_idx = torch.max(cc, dim=1)
        improve = local_cc > best_cc
        chunk_tensor = torch.as_tensor(chunk, dtype=spectra.dtype, device=spectra.device)
        best_cc = torch.where(improve, local_cc, best_cc)
        best_rotation = torch.where(improve, chunk_tensor[local_idx], best_rotation)
    return 0.5 * best_rotation


def _batched_minimize_scalar_bounded(
    func: Callable[[torch.Tensor], torch.Tensor],
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    xatol: float,
    maxiter: int,
    f_at_zero: Optional[torch.Tensor] = None,
    enabled: Optional[torch.Tensor] = None,
    check_interval: int = 8,
) -> _BatchedOptimizationResult:
    """Vectorized form of Brent's bounded scalar minimizer.

    Each row is an independent scalar minimization, while every objective
    evaluation is issued as one GPU batch.
    """
    device = lower.device
    dtype = lower.dtype
    if enabled is None:
        enabled = torch.ones_like(lower, dtype=torch.bool)
    width = upper - lower
    valid = enabled & torch.isfinite(lower) & torch.isfinite(upper) & (width > 0.0)
    if not bool(torch.any(valid).item()):
        base_fun = (
            f_at_zero.to(dtype) if f_at_zero is not None
            else torch.zeros_like(lower, dtype=dtype)
        )
        return _BatchedOptimizationResult(
            x=torch.zeros_like(lower, dtype=dtype),
            fun=base_fun,
            success=torch.ones_like(valid),
            nfev=torch.zeros_like(lower, dtype=torch.int64),
            nit=0,
            messages=["No active line search."] * lower.numel(),
        )
    golden_mean = 0.5 * (3.0 - math.sqrt(5.0))
    sqrt_eps = math.sqrt(torch.finfo(dtype).eps)
    a = lower.clone()
    b = upper.clone()
    xf = a + golden_mean * (b - a)
    xf = torch.where(valid, xf, torch.zeros_like(xf))
    nfc = xf.clone()
    fulc = xf.clone()
    rat = torch.zeros_like(xf)
    e = torch.zeros_like(xf)
    fx = func(xf).to(dtype)
    if f_at_zero is not None:
        fx = torch.where(valid, fx, f_at_zero.to(dtype))
    fnfc = fx.clone()
    ffulc = fx.clone()
    nfev = valid.to(torch.int64)
    converged = ~valid
    nit = 0

    for nit in range(1, maxiter + 1):
        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * torch.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        active = valid & (
            torch.abs(xf - xm) > (tol2 - 0.5 * (b - a))
        )
        converged = converged | (valid & ~active)
        # Avoid a CUDA->CPU synchronization on every Brent iteration.  A
        # coarse check leaves at most check_interval-1 masked iterations.
        if nit % check_interval == 0 and not bool(torch.any(active).item()):
            break

        old_e = e.clone()
        old_rat = rat.clone()
        can_parabolic = active & (torch.abs(old_e) > tol1)
        r = (xf - nfc) * (fx - ffulc)
        q0 = (xf - fulc) * (fx - fnfc)
        p = (xf - fulc) * q0 - (xf - nfc) * r
        q = 2.0 * (q0 - r)
        p = torch.where(q > 0.0, -p, p)
        qabs = torch.abs(q)
        accept = (
            can_parabolic
            & (qabs > torch.finfo(dtype).tiny)
            & (torch.abs(p) < torch.abs(0.5 * qabs * old_e))
            & (p > qabs * (a - xf))
            & (p < qabs * (b - xf))
        )
        rat_parabolic = p / qabs.clamp_min(torch.finfo(dtype).tiny)
        x_parabolic = xf + rat_parabolic
        near_edge = ((x_parabolic - a) < tol2) | ((b - x_parabolic) < tol2)
        sign_mid = torch.where(xm - xf >= 0.0, 1.0, -1.0)
        rat_parabolic = torch.where(
            near_edge, tol1 * sign_mid, rat_parabolic
        )
        e_golden = torch.where(xf >= xm, a - xf, b - xf)
        rat_golden = golden_mean * e_golden
        rat = torch.where(accept, rat_parabolic, rat_golden)
        e = torch.where(accept, old_rat, e_golden)
        step_sign = torch.where(rat >= 0.0, 1.0, -1.0)
        candidate = xf + step_sign * torch.maximum(torch.abs(rat), tol1)
        candidate = torch.where(active, candidate, xf)
        fu = func(candidate).to(dtype)
        nfev += active.to(torch.int64)

        better = active & (fu <= fx)
        worse = active & ~better
        old_xf = xf.clone()
        old_fx = fx.clone()
        old_nfc = nfc.clone()
        old_fnfc = fnfc.clone()

        a = torch.where(better & (candidate >= old_xf), old_xf, a)
        b = torch.where(better & (candidate < old_xf), old_xf, b)
        fulc = torch.where(better, old_nfc, fulc)
        ffulc = torch.where(better, old_fnfc, ffulc)
        nfc = torch.where(better, old_xf, nfc)
        fnfc = torch.where(better, old_fx, fnfc)
        xf = torch.where(better, candidate, xf)
        fx = torch.where(better, fu, fx)

        a = torch.where(worse & (candidate < old_xf), candidate, a)
        b = torch.where(worse & (candidate >= old_xf), candidate, b)
        replace_nfc = worse & ((fu <= fnfc) | (nfc == old_xf))
        old_nfc2 = nfc.clone()
        old_fnfc2 = fnfc.clone()
        fulc = torch.where(replace_nfc, old_nfc2, fulc)
        ffulc = torch.where(replace_nfc, old_fnfc2, ffulc)
        nfc = torch.where(replace_nfc, candidate, nfc)
        fnfc = torch.where(replace_nfc, fu, fnfc)
        replace_fulc = (
            worse & ~replace_nfc
            & ((fu <= ffulc) | (fulc == old_xf) | (fulc == nfc))
        )
        fulc = torch.where(replace_fulc, candidate, fulc)
        ffulc = torch.where(replace_fulc, fu, ffulc)
    else:
        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * torch.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        converged = ~valid | (
            torch.abs(xf - xm) <= (tol2 - 0.5 * (b - a))
        )

    messages = [
        "Solution found." if bool(v) else "Maximum scalar iterations reached."
        for v in converged.detach().cpu().tolist()
    ]
    return _BatchedOptimizationResult(
        x=xf,
        fun=fx,
        success=converged,
        nfev=nfev,
        nit=nit,
        messages=messages,
    )



def _batched_bracket_minimum(
    func: Callable[[torch.Tensor], torch.Tensor],
    *,
    enabled: torch.Tensor,
    maxiter: int = 100,
    check_interval: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized equivalent of scipy.optimize.bracket with xa=0, xb=1."""
    dtype = torch.float64
    device = enabled.device
    gold = 1.618034
    tiny = 1.0e-21
    grow_limit = 110.0
    xa = torch.zeros(enabled.shape, dtype=dtype, device=device)
    xb = torch.ones_like(xa)
    fa = func(xa).to(dtype)
    fb = func(xb).to(dtype)
    nfev = enabled.to(torch.int64) * 2
    swap = enabled & (fa < fb)
    xa_old, fa_old = xa.clone(), fa.clone()
    xa = torch.where(swap, xb, xa)
    xb = torch.where(swap, xa_old, xb)
    fa = torch.where(swap, fb, fa)
    fb = torch.where(swap, fa_old, fb)
    xc = xb + gold * (xb - xa)
    fc = func(xc).to(dtype)
    nfev += enabled.to(torch.int64)
    searching = enabled & (fc < fb)

    for bracket_iteration in range(1, maxiter + 1):
        if (
            bracket_iteration % check_interval == 0
            and not bool(torch.any(searching).item())
        ):
            break
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        denom = torch.where(
            torch.abs(val) < tiny,
            torch.full_like(val, 2.0 * tiny),
            2.0 * val,
        )
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + grow_limit * (xc - xb)

        between = searching & ((w - xc) * (xb - w) > 0.0)
        beyond_limit = searching & ~between & ((w - wlim) * (wlim - xc) >= 0.0)
        between_limit = (
            searching & ~between & ~beyond_limit
            & ((w - wlim) * (xc - w) > 0.0)
        )
        fallback = searching & ~between & ~beyond_limit & ~between_limit

        eval_w = torch.where(beyond_limit, wlim, w)
        eval_w = torch.where(fallback, xc + gold * (xc - xb), eval_w)
        fw = func(eval_w).to(dtype)
        nfev += (between | beyond_limit | between_limit | fallback).to(torch.int64)

        done_low = between & (fw < fc)
        done_high = between & ~done_low & (fw > fb)
        xa = torch.where(done_low, xb, xa)
        fa = torch.where(done_low, fb, fa)
        xb = torch.where(done_low, eval_w, xb)
        fb = torch.where(done_low, fw, fb)
        xc = torch.where(done_high, eval_w, xc)
        fc = torch.where(done_high, fw, fc)
        done = done_low | done_high

        # Between-case without a completed bracket: evaluate the golden extension.
        between_continue = between & ~done
        w2 = xc + gold * (xc - xb)
        fw2 = func(torch.where(between_continue, w2, eval_w)).to(dtype)
        nfev += between_continue.to(torch.int64)
        eval_w = torch.where(between_continue, w2, eval_w)
        fw = torch.where(between_continue, fw2, fw)

        # In the between-limit case, a successful interpolation is followed by
        # one more golden extension before the standard shift.
        extend_limit = between_limit & (fw < fc)
        xb_pre = torch.where(extend_limit, xc, xb)
        fb_pre = torch.where(extend_limit, fc, fb)
        xc_pre = torch.where(extend_limit, eval_w, xc)
        fc_pre = torch.where(extend_limit, fw, fc)
        w3 = xc_pre + gold * (xc_pre - xb_pre)
        fw3 = func(torch.where(extend_limit, w3, eval_w)).to(dtype)
        nfev += extend_limit.to(torch.int64)
        xb = xb_pre
        fb = fb_pre
        xc = xc_pre
        fc = fc_pre
        eval_w = torch.where(extend_limit, w3, eval_w)
        fw = torch.where(extend_limit, fw3, fw)

        shift = searching & ~done
        xa_new = torch.where(shift, xb, xa)
        fa_new = torch.where(shift, fb, fa)
        xb_new = torch.where(shift, xc, xb)
        fb_new = torch.where(shift, fc, fb)
        xc_new = torch.where(shift, eval_w, xc)
        fc_new = torch.where(shift, fw, fc)
        xa, fa, xb, fb, xc, fc = xa_new, fa_new, xb_new, fb_new, xc_new, fc_new
        searching = searching & ~done & (fc < fb)

    return xa, xb, xc, fa, fb, fc, nfev


def _batched_minimize_scalar_unbounded(
    func: Callable[[torch.Tensor], torch.Tensor],
    *,
    enabled: torch.Tensor,
    xtol: float,
    maxiter: int,
    f_at_zero: Optional[torch.Tensor] = None,
    check_interval: int = 8,
) -> _BatchedOptimizationResult:
    """Vectorized Brent minimization after SciPy-style automatic bracketing."""
    if not bool(torch.any(enabled).item()):
        base_fun = (
            f_at_zero.to(torch.float64) if f_at_zero is not None
            else torch.zeros(enabled.shape, dtype=torch.float64, device=enabled.device)
        )
        return _BatchedOptimizationResult(
            x=torch.zeros(enabled.shape, dtype=torch.float64, device=enabled.device),
            fun=base_fun,
            success=torch.ones_like(enabled),
            nfev=torch.zeros(enabled.shape, dtype=torch.int64, device=enabled.device),
            nit=0,
            messages=["No active line search."] * enabled.numel(),
        )
    xa, xb, xc, fa, fb, fc, nfev = _batched_bracket_minimum(
        func, enabled=enabled, maxiter=maxiter,
        check_interval=check_interval,
    )
    dtype = xa.dtype
    a = torch.minimum(xa, xc)
    b = torch.maximum(xa, xc)
    x = xb.clone()
    w = xb.clone()
    v = xb.clone()
    fx = fb.clone()
    fw = fb.clone()
    fv = fb.clone()
    deltax = torch.zeros_like(x)
    rat = torch.zeros_like(x)
    mintol = 1.0e-11
    cg = 0.3819660
    converged = ~enabled
    nit = 0

    for nit in range(1, maxiter + 1):
        tol1 = xtol * torch.abs(x) + mintol
        tol2 = 2.0 * tol1
        xmid = 0.5 * (a + b)
        active = enabled & (
            torch.abs(x - xmid) >= (tol2 - 0.5 * (b - a))
        )
        converged = converged | (enabled & ~active)
        if nit % check_interval == 0 and not bool(torch.any(active).item()):
            break

        old_deltax = deltax.clone()
        old_rat = rat.clone()
        golden = active & (torch.abs(old_deltax) <= tol1)
        golden_delta = torch.where(x >= xmid, a - x, b - x)
        golden_rat = cg * golden_delta

        tmp1 = (x - w) * (fx - fv)
        tmp2 = (x - v) * (fx - fw)
        p = (x - v) * tmp2 - (x - w) * tmp1
        q = 2.0 * (tmp2 - tmp1)
        p = torch.where(q > 0.0, -p, p)
        qabs = torch.abs(q)
        parabolic_possible = active & ~golden & (qabs > torch.finfo(dtype).tiny)
        accept = (
            parabolic_possible
            & (p > qabs * (a - x))
            & (p < qabs * (b - x))
            & (torch.abs(p) < torch.abs(0.5 * qabs * old_deltax))
        )
        parabolic_rat = p / qabs.clamp_min(torch.finfo(dtype).tiny)
        u_parabolic = x + parabolic_rat
        sign_mid = torch.where(xmid - x >= 0.0, 1.0, -1.0)
        parabolic_rat = torch.where(
            ((u_parabolic - a) < tol2) | ((b - u_parabolic) < tol2),
            tol1 * sign_mid,
            parabolic_rat,
        )
        use_golden = active & ~accept
        deltax = torch.where(use_golden, golden_delta, old_rat)
        rat = torch.where(use_golden, golden_rat, parabolic_rat)
        step = torch.where(
            torch.abs(rat) < tol1,
            torch.where(rat >= 0.0, tol1, -tol1),
            rat,
        )
        u = torch.where(active, x + step, x)
        fu = func(u).to(dtype)
        nfev += active.to(torch.int64)

        worse = active & (fu > fx)
        better = active & ~worse
        old_x, old_fx = x.clone(), fx.clone()
        old_w, old_fw = w.clone(), fw.clone()

        a = torch.where(worse & (u < old_x), u, a)
        b = torch.where(worse & (u >= old_x), u, b)
        replace_w = worse & ((fu <= fw) | (w == old_x))
        v = torch.where(replace_w, old_w, v)
        fv = torch.where(replace_w, old_fw, fv)
        w = torch.where(replace_w, u, w)
        fw = torch.where(replace_w, fu, fw)
        replace_v = (
            worse & ~replace_w
            & ((fu <= fv) | (v == old_x) | (v == w))
        )
        v = torch.where(replace_v, u, v)
        fv = torch.where(replace_v, fu, fv)

        a = torch.where(better & (u >= old_x), old_x, a)
        b = torch.where(better & (u < old_x), old_x, b)
        v = torch.where(better, old_w, v)
        fv = torch.where(better, old_fw, fv)
        w = torch.where(better, old_x, w)
        fw = torch.where(better, old_fx, fw)
        x = torch.where(better, u, x)
        fx = torch.where(better, fu, fx)
    else:
        tol1 = xtol * torch.abs(x) + mintol
        tol2 = 2.0 * tol1
        xmid = 0.5 * (a + b)
        converged = ~enabled | (
            torch.abs(x - xmid) < (tol2 - 0.5 * (b - a))
        )

    if f_at_zero is not None:
        fx = torch.where(enabled, fx, f_at_zero.to(dtype))
        x = torch.where(enabled, x, torch.zeros_like(x))
    messages = [
        "Solution found." if bool(q) else "Maximum scalar iterations reached."
        for q in converged.detach().cpu().tolist()
    ]
    return _BatchedOptimizationResult(
        x=x, fun=fx, success=converged, nfev=nfev, nit=nit, messages=messages
    )
def _batched_minimize_scalar_local_bracket(
    func: Callable[[torch.Tensor], torch.Tensor],
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    xatol: float,
    maxiter: int,
    f_at_zero: Optional[torch.Tensor] = None,
    enabled: Optional[torch.Tensor] = None,
    check_interval: int = 8,
    initial_step: float | torch.Tensor = 1.0,
) -> _BatchedOptimizationResult:
    """
    Batched local line minimization centered at alpha = 0.

    Unlike bounded Brent, [lower, upper] is NOT treated as one globally
    unimodal search interval. The search starts at alpha=0, probes a local
    step, automatically brackets the first nearby minimum, and then applies
    Brent minimization.

    lower and upper are used only as hard safety limits.

    In the current CTFFIND parameterization:
      - initial_step=1 corresponds to about 100 * pixel_size Angstrom
        for a defocus coordinate;
      - for a Powell-updated direction, alpha=1 means one full direction
        vector.
    """
    dtype = lower.dtype
    device = lower.device

    if enabled is None:
        enabled = torch.ones_like(lower, dtype=torch.bool)
    else:
        enabled = enabled.to(device=device, dtype=torch.bool)

    if isinstance(initial_step, torch.Tensor):
        step = initial_step.to(device=device, dtype=dtype)
        if step.ndim == 0:
            step = step.expand_as(lower)
        elif step.shape != lower.shape:
            step = torch.broadcast_to(step, lower.shape).clone()
    else:
        step = torch.full_like(lower, float(initial_step))

    # The sign of the initial step is not important because the automatic
    # bracket routine can reverse direction. Keep only a positive magnitude.
    step = torch.abs(step)

    zero = torch.zeros_like(lower)

    if f_at_zero is None:
        f0 = func(zero).to(dtype)
    else:
        f0 = f_at_zero.to(device=device, dtype=dtype)

    eps = torch.finfo(dtype).eps

    # alpha=0 must lie within the safety interval, because the line search is
    # centered on the current Powell point.
    feasible = (
        enabled
        & torch.isfinite(step)
        & (step > eps)
        & (upper > lower)
        & (lower <= 0.0)
        & (upper >= 0.0)
    )

    def normalized_objective(z: torch.Tensor) -> torch.Tensor:
        """
        z is the dimensionless Brent variable. The physical Powell step is:

            alpha = z * initial_step
        """
        alpha = z.to(dtype) * step

        # Evaluate at the nearest valid safety-boundary point. Values outside
        # the safety interval then receive a large penalty, so the bracket
        # cannot expand indefinitely.
        alpha_clipped = torch.minimum(
            torch.maximum(alpha, lower),
            upper,
        )

        values = func(alpha_clipped).to(dtype)

        outside = (alpha < lower) | (alpha > upper)

        # Keep the penalty finite. Infinite values can create inf-inf during
        # parabolic interpolation.
        penalty_scale = 1.0e6 * (1.0 + torch.abs(f0))
        boundary_distance = torch.abs(alpha - alpha_clipped)

        penalty = (
            f0
            + penalty_scale
            + 1.0e3 * boundary_distance
        )

        values = torch.where(
            outside & feasible,
            penalty,
            values,
        )

        # Disabled or invalid rows remain unchanged.
        return torch.where(feasible, values, f0)

    # The existing routine begins with z=0 and z=1. If +1 is worse than zero,
    # its bracket code reverses direction and probes the negative side.
    raw = _batched_minimize_scalar_unbounded(
        normalized_objective,
        enabled=feasible,
        xtol=xatol,
        maxiter=maxiter,
        f_at_zero=f0,
        check_interval=check_interval,
    )

    alpha = raw.x.to(dtype) * step
    alpha = torch.minimum(
        torch.maximum(alpha, lower),
        upper,
    )
    alpha = torch.where(feasible, alpha, zero)

    # Recompute at the final clipped alpha. This avoids returning the artificial
    # boundary penalty if roundoff placed the Brent result just outside a limit.
    final_fun = func(alpha).to(dtype)
    final_fun = torch.where(feasible, final_fun, f0)

    final_success = raw.success | ~feasible
    final_nfev = raw.nfev + feasible.to(torch.int64)

    messages = []
    raw_messages = raw.messages
    feasible_cpu = feasible.detach().cpu().tolist()

    for i, is_feasible in enumerate(feasible_cpu):
        if is_feasible:
            messages.append(raw_messages[i])
        else:
            messages.append("No active or feasible local line search.")

    return _BatchedOptimizationResult(
        x=alpha,
        fun=final_fun,
        success=final_success,
        nfev=final_nfev,
        nit=raw.nit,
        messages=messages,
    )
def _line_bounds(
    x: torch.Tensor,
    direction: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1.0e-14
    positive = direction > eps
    negative = direction < -eps
    lo_dim = torch.where(
        positive, (lower - x) / direction,
        torch.where(negative, (upper - x) / direction,
                    torch.full_like(direction, -torch.inf)),
    )
    hi_dim = torch.where(
        positive, (upper - x) / direction,
        torch.where(negative, (lower - x) / direction,
                    torch.full_like(direction, torch.inf)),
    )
    alpha_lower = lo_dim.amax(dim=1)
    alpha_upper = hi_dim.amin(dim=1)
    nonzero = torch.any(torch.abs(direction) > eps, dim=1)
    feasible = nonzero & (alpha_upper > alpha_lower)
    return alpha_lower, alpha_upper, feasible


def _batched_powell(
    objective: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    xtol: float,
    ftol: float,
    maxiter: int,
    line_maxiter: int,
    check_interval: int = 8,
    callback: Optional[Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], None]] = None,
) -> _BatchedOptimizationResult:
    """Independent modified-Powell optimizers evaluated as GPU batches."""
    x = x0.clone()
    batch, ndim = x.shape
    eye = torch.eye(ndim, dtype=x.dtype, device=x.device)
    directions = eye[None].expand(batch, -1, -1).clone()
    f = objective(x).to(x.dtype)
    nfev = torch.ones(batch, dtype=torch.int64, device=x.device)
    active = torch.ones(batch, dtype=torch.bool, device=x.device)
    success = torch.zeros_like(active)
    nit = 0

    def line_search(
        current_x: torch.Tensor,
        current_f: torch.Tensor,
        direction: torch.Tensor,
        enabled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_lo, alpha_hi, feasible = _line_bounds(
            current_x, direction, lower, upper
        )
        finite = (
            enabled & feasible
            & torch.isfinite(alpha_lo) & torch.isfinite(alpha_hi)
        )
        unbounded = (
            enabled & feasible
            & torch.isneginf(alpha_lo) & torch.isposinf(alpha_hi)
        )
        # Current CTFFIND bounds produce either fully bounded lines or fully
        # unbounded angle lines.  Unexpected one-sided rows are left unmoved
        # rather than forcing a synchronization merely to raise an exception.
        unsupported_one_sided = enabled & feasible & ~(finite | unbounded)
        feasible = feasible & ~unsupported_one_sided

        def alpha_objective(alpha: torch.Tensor) -> torch.Tensor:
            return objective(current_x + alpha[:, None] * direction)

    #    bounded_result = _batched_minimize_scalar_bounded(
    #        alpha_objective,
    #        alpha_lo,
    #        alpha_hi,
    #        xatol=xtol,
    #        maxiter=line_maxiter,
    #        f_at_zero=current_f,
    #        enabled=finite,
    #        check_interval=check_interval,
    #    )
        bounded_result = _batched_minimize_scalar_local_bracket(
            alpha_objective,
            alpha_lo,
            alpha_hi,
            xatol=xtol,
            maxiter=line_maxiter,
            f_at_zero=current_f,
            enabled=finite,
            check_interval=check_interval,
            initial_step=1.0,
        )
        unbounded_result = _batched_minimize_scalar_unbounded(
            alpha_objective,
            enabled=unbounded,
            xtol=xtol * 100.0,
            maxiter=line_maxiter,
            f_at_zero=current_f,
            check_interval=check_interval,
        )
        alpha = torch.where(unbounded, unbounded_result.x, bounded_result.x)
        line_fun = torch.where(
            unbounded, unbounded_result.fun, bounded_result.fun
        )
        line_success = torch.where(
            unbounded, unbounded_result.success, bounded_result.success
        )
        line_nfev = bounded_result.nfev + unbounded_result.nfev
        moved = enabled & feasible
        step = alpha[:, None] * direction
        candidate_x = current_x + step
        # A bounded scalar search assumes approximate unimodality. CTF scores
        # are strongly multi-modal, so never replace a valid point by a worse
        # line-search result. Direction-set updates are retained, including
        # mixed defocus/angle (and later phase-shift) directions.
        accept = moved & (line_fun <= current_f)
        new_x = torch.where(accept[:, None], candidate_x, current_x)
        new_f = torch.where(accept, line_fun, current_f)
        actual_step = new_x - current_x
        return new_x, new_f, actual_step, line_nfev, line_success

    outer_check_interval = max(2, min(check_interval, 4))
    for nit in range(1, maxiter + 1):
        if (
            nit % outer_check_interval == 0
            and not bool(torch.any(active).item())
        ):
            break
        x_start = x.clone()
        f_start = f.clone()
        biggest_decrease = torch.zeros_like(f)
        biggest_index = torch.zeros(batch, dtype=torch.int64, device=x.device)

        for j in range(ndim):
            f_before = f.clone()
            x, f, _, line_nfev, _ = line_search(
                x, f, directions[:, j, :], active
            )
            nfev += line_nfev
            decrease = f_before - f
            replace = active & (decrease > biggest_decrease)
            biggest_decrease = torch.where(replace, decrease, biggest_decrease)
            biggest_index = torch.where(
                replace, torch.full_like(biggest_index, j), biggest_index
            )

        if callback is not None:
            callback(nit, x.clone(), f.clone(), directions.clone())
        improvement = f_start - f
        threshold = ftol * (torch.abs(f_start) + torch.abs(f)) + 1.0e-20
        displacement = x - x_start
        converged = active & (2.0 * improvement <= threshold)
        success |= converged
        active &= ~converged
        if (
            nit % outer_check_interval == 0
            and not bool(torch.any(active).item())
        ):
            break

        alpha_lo, alpha_hi, feasible = _line_bounds(x, displacement, lower, upper)
        extrap_alpha = torch.minimum(
            torch.ones_like(alpha_hi), alpha_hi
        )
        extrap_alpha = torch.maximum(extrap_alpha, alpha_lo)
        x_extrap = torch.clamp(
            x + extrap_alpha[:, None] * displacement, lower, upper
        )
        f_extrap = objective(x_extrap).to(x.dtype)
        nfev += active.to(torch.int64)
        fx = f_start
        condition1 = active & feasible & (fx > f_extrap)
        t = 2.0 * (fx + f_extrap - 2.0 * f)
        temp = fx - f - biggest_decrease
        t = t * temp.square() - biggest_decrease * (fx - f_extrap).square()
        replace_rows = condition1 & (t < 0.0)
        x_before_extra = x.clone()
        x, f, extra_step, line_nfev, _ = line_search(
            x, f, displacement, replace_rows
        )
        nfev += line_nfev
        nonzero_step = replace_rows & torch.any(
            torch.abs(extra_step) > 1.0e-14, dim=1
        )
        target_index = biggest_index[:, None, None].expand(-1, 1, ndim)
        old_target = directions.gather(1, target_index)
        replacement = torch.where(
            nonzero_step[:, None, None],
            directions[:, -1:, :],
            old_target,
        )
        directions.scatter_(1, target_index, replacement)
        directions[:, -1, :] = torch.where(
            nonzero_step[:, None], extra_step, directions[:, -1, :]
        )

    messages = [
        "Solution found." if bool(v) else "Maximum Powell iterations reached."
        for v in success.detach().cpu().tolist()
    ]
    return _BatchedOptimizationResult(
        x=x,
        fun=f,
        success=success,
        nfev=nfev,
        nit=nit,
        messages=messages,
    )



def _extract_detrended_tiles(
    micrograph: np.ndarray,
    tile_size: int,
    stride: int,
    pixel_size_A: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract a centered regular tile grid and remove a best-fit real-space plane."""
    image = np.asarray(micrograph, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Tilt fitting expects one 2-D micrograph")
    height, width = image.shape
    if height < tile_size or width < tile_size:
        raise ValueError(
            f"Micrograph {width}x{height} is smaller than tilt tile size {tile_size}"
        )

    ny = 1 + (height - tile_size) // stride
    nx = 1 + (width - tile_size) // stride
    covered_h = tile_size + (ny - 1) * stride
    covered_w = tile_size + (nx - 1) * stride
    y_offset = (height - covered_h) // 2
    x_offset = (width - covered_w) // 2

    arrays: list[np.ndarray] = []
    centers_x: list[float] = []
    centers_y: list[float] = []
    grid_y: list[int] = []
    grid_x: list[int] = []
    for iy in range(ny):
        y0 = y_offset + iy * stride
        for ix in range(nx):
            x0 = x_offset + ix * stride
            arrays.append(np.array(image[y0:y0 + tile_size, x0:x0 + tile_size], copy=True))
            centers_x.append((x0 + 0.5 * tile_size - 0.5 * width) * pixel_size_A)
            centers_y.append((y0 + 0.5 * tile_size - 0.5 * height) * pixel_size_A)
            grid_y.append(iy)
            grid_x.append(ix)

    tiles = torch.as_tensor(np.stack(arrays), dtype=dtype, device=device)
    coord = torch.linspace(-1.0, 1.0, tile_size, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    mean = tiles.mean(dim=(1, 2), keepdim=True)
    centered = tiles - mean
    denom_x = torch.sum(xx.square()).clamp_min(1.0e-20)
    denom_y = torch.sum(yy.square()).clamp_min(1.0e-20)
    slope_x = torch.sum(centered * xx[None], dim=(1, 2), keepdim=True) / denom_x
    slope_y = torch.sum(centered * yy[None], dim=(1, 2), keepdim=True) / denom_y
    detrended = centered - slope_x * xx[None] - slope_y * yy[None]
    rms = torch.sqrt(torch.mean(detrended.square(), dim=(1, 2))).detach().cpu().numpy()
    return (
        detrended.contiguous(),
        np.asarray(centers_x, dtype=np.float64),
        np.asarray(centers_y, dtype=np.float64),
        np.asarray(grid_y, dtype=np.int64),
        np.asarray(grid_x, dtype=np.int64),
        np.asarray(rms, dtype=np.float64),
    )


def _robust_mad_mask(values: np.ndarray, cutoff: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return finite
    working = values[finite]
    median = float(np.median(working))
    mad = float(np.median(np.abs(working - median)))
    sigma = 1.4826 * mad
    if not math.isfinite(sigma) or sigma <= 1.0e-12:
        return finite
    return finite & (np.abs(values - median) <= cutoff * sigma)


def _robust_defocus_plane(
    x_A: np.ndarray,
    y_A: np.ndarray,
    defocus_A: np.ndarray,
    tile_cc: np.ndarray,
    eligible: np.ndarray,
    cutoff: float,
    minimum_tiles: int,
    minimum_scale_A: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust weighted fit of D = beta0 + gx*x + gy*y."""
    x_A = np.asarray(x_A, dtype=np.float64)
    y_A = np.asarray(y_A, dtype=np.float64)
    defocus_A = np.asarray(defocus_A, dtype=np.float64)
    tile_cc = np.asarray(tile_cc, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    design = np.column_stack((np.ones_like(x_A), x_A, y_A))
    inlier = eligible & np.isfinite(defocus_A) & np.isfinite(tile_cc)
    if int(np.sum(inlier)) < minimum_tiles:
        # Keep the strongest eligible tiles rather than failing merely because
        # a conservative tile-CC threshold rejected too many.
        candidates = np.flatnonzero(eligible & np.isfinite(defocus_A))
        if candidates.size < minimum_tiles:
            raise RuntimeError(
                f"Only {candidates.size} usable tiles; need at least {minimum_tiles}"
            )
        order = candidates[np.argsort(tile_cc[candidates])[::-1]]
        inlier[order[:minimum_tiles]] = True

    beta = np.zeros(3, dtype=np.float64)
    for _ in range(6):
        idx = np.flatnonzero(inlier)
        if idx.size < minimum_tiles:
            break
        weights = np.clip(tile_cc[idx], 0.01, None) ** 2
        lhs = design[idx] * np.sqrt(weights)[:, None]
        rhs = defocus_A[idx] * np.sqrt(weights)
        beta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        residual = defocus_A - design @ beta
        center = float(np.median(residual[idx]))
        mad = float(np.median(np.abs(residual[idx] - center)))
        scale = max(1.4826 * mad, minimum_scale_A)
        new_inlier = eligible & np.isfinite(residual) & (
            np.abs(residual - center) <= cutoff * scale
        )
        if int(np.sum(new_inlier)) < minimum_tiles:
            break
        if np.array_equal(new_inlier, inlier):
            inlier = new_inlier
            break
        inlier = new_inlier

    idx = np.flatnonzero(inlier)
    if idx.size < minimum_tiles:
        raise RuntimeError(
            f"Robust plane retained only {idx.size} tiles; need {minimum_tiles}"
        )
    weights = np.clip(tile_cc[idx], 0.01, None) ** 2
    lhs = design[idx] * np.sqrt(weights)[:, None]
    rhs = defocus_A[idx] * np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    prediction = design @ beta
    residual = defocus_A - prediction
    return beta, inlier, residual


def _tilt_angle_axis_from_gradient(gx: float, gy: float) -> tuple[float, float]:
    magnitude = math.hypot(gx, gy)
    tilt_angle = math.degrees(math.atan(magnitude))
    if magnitude <= 1.0e-15:
        return 0.0, 0.0
    # CTFTILT convention: gradient normal n=(sin(axis), -cos(axis)).
    axis = math.degrees(math.atan2(gx, -gy)) % 180.0
    return tilt_angle, axis


def _tile_good_fit_from_curve(
    curve_values: np.ndarray,
    frequencies_Ainv: np.ndarray,
    defocus_A: float,
    wavelength_A: float,
    spherical_aberration_A: float,
    amplitude_phase_rad: float,
    phase_shift_rad: float,
    spectrum_size: int,
    fitting_pixel_size_A: float,
) -> float:
    frequencies = np.asarray(frequencies_Ainv, dtype=np.float64)
    s2 = frequencies * frequencies
    phase = (
        PI * wavelength_A * s2
        * (defocus_A - 0.5 * wavelength_A * wavelength_A * s2 * spherical_aberration_A)
        + phase_shift_rad + amplitude_phase_rad
    )
    fit = np.sin(phase) ** 2
    extrema = np.abs(np.floor(phase / PI + 0.5)).astype(np.int64)
    frc, _ = _compute_frc_between_1d_spectrum_and_fit(
        np.asarray(curve_values, dtype=np.float64), fit, extrema
    )
    first_zero_sq = _squared_frequency_given_phase_Ainv2(
        PI, 0.0, defocus_A, defocus_A, 0.0,
        wavelength_A, spherical_aberration_A, amplitude_phase_rad, phase_shift_rad,
    )
    last_good, _ = _find_good_fit_and_aliasing_bins(
        frc, extrema, math.sqrt(max(0.0, first_zero_sq)),
        spectrum_size, fitting_pixel_size_A,
    )
    if last_good <= 0 or last_good >= frequencies.size or frequencies[last_good] <= 0.0:
        return 0.0
    return float(1.0 / frequencies[last_good])


def _write_tilt_png(path: Path, details: _TiltFitDetails) -> None:
    """Write a two-panel colour diagnostic without changing the MRC CTF output."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "Tilt PNG output requires matplotlib (pip install matplotlib)"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = details.image_shape
    x_min = -0.5 * width
    x_max = 0.5 * width
    y_min = -0.5 * height
    y_max = 0.5 * height
    # Convert tile coordinates from Angstrom to image-pixel-like display units
    # only for the extent; colour values retain physical units.
    xA = details.tile_centers_x_A
    yA = details.tile_centers_y_A
    if xA.size > 1:
        px_A = np.median(np.diff(np.unique(np.sort(xA))))
    else:
        px_A = 1.0
    if not np.isfinite(px_A) or px_A == 0.0:
        px_A = 1.0
    xx = np.linspace(float(np.min(xA)), float(np.max(xA)), 240)
    yy = np.linspace(float(np.min(yA)), float(np.max(yA)), 240)
    gy_grid, gx_grid = np.meshgrid(yy, xx, indexing="ij")
    center_mean = 0.5 * (details.center_defocus1_A + details.center_defocus2_A)
    predicted = center_mean + details.gradient_x * gx_grid + details.gradient_y * gy_grid

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    im0 = axes[0].imshow(
        predicted / 10_000.0,
        origin="upper",
        extent=[float(np.min(xA)), float(np.max(xA)), float(np.max(yA)), float(np.min(yA))],
        aspect="auto",
        interpolation="bilinear",
    )
    axes[0].scatter(xA[details.tile_plane_inlier], yA[details.tile_plane_inlier],
                    s=12, facecolors="none", edgecolors="black", linewidths=0.4)
    axes[0].set_title("Predicted mean defocus plane")
    axes[0].set_xlabel("x (Å)")
    axes[0].set_ylabel("y (Å, image downward)")
    fig.colorbar(im0, ax=axes[0], label="Defocus (µm)")

    valid_res = details.tile_plane_inlier & np.isfinite(details.tile_residual_A)
    rejected = ~details.tile_plane_inlier
    residual_nm = details.tile_residual_A / 10.0
    if np.any(valid_res):
        limit = float(np.nanpercentile(np.abs(residual_nm[valid_res]), 95.0))
        limit = max(limit, 1.0)
    else:
        limit = 1.0
    scatter = axes[1].scatter(
        xA[valid_res], yA[valid_res], c=residual_nm[valid_res], s=70,
        marker="s", vmin=-limit, vmax=limit, cmap="coolwarm", edgecolors="black",
        linewidths=0.35,
    )
    if np.any(rejected):
        axes[1].scatter(xA[rejected], yA[rejected], c="0.75", s=45,
                        marker="x", linewidths=0.8, label="rejected")
    axes[1].set_title("Local defocus minus fitted plane")
    axes[1].set_xlabel("x (Å)")
    axes[1].set_ylabel("y (Å, image downward)")
    axes[1].invert_yaxis()
    if np.any(valid_res):
        fig.colorbar(scatter, ax=axes[1], label="Residual (nm)")
    if np.any(rejected):
        axes[1].legend(loc="best", fontsize=8)

    text = (
        f"tilt={details.tilt_angle_deg:.2f}°  nominal={details.nominal_tilt_angle_deg:.2f}°  "
        f"axis={details.tilt_axis_deg:.2f}°\n"
        f"coarse tilt/axis={details.coarse_tilt_angle_deg:.2f}°/"
        f"{details.coarse_tilt_axis_deg:.2f}°  gx={details.gradient_x:.5f}  gy={details.gradient_y:.5f}\n"
        f"center dfU/V={details.center_defocus1_A:.0f}/{details.center_defocus2_A:.0f} Å\n"
        f"tile power/CTF^2 CC={details.score:.4f}  residual RMS={details.residual_rms_A/10.0:.1f} nm\n"
        f"tile median good-fit={details.good_fit_resolution_A:.2f} Å  "
        f"tiles={int(np.sum(details.tile_plane_inlier))}/{details.tile_plane_inlier.size}"
    )
    fig.suptitle(text, fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_tilt_tsv(path: Path, results: Sequence[CtfFitResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    columns = [
        "micrograph_name", "source_file", "image_index", "tilt_fitted",
        "center_defocus_u_A", "center_defocus_v_A", "defocus_angle_deg",
        "gradient_x", "gradient_y", "tilt_angle_deg", "tilt_axis_deg",
        "nominal_tilt_angle_deg", "coarse_tilt_angle_deg", "coarse_tilt_axis_deg",
        "tilt_score", "global_cc", "global_good_fit_A", "tilt_good_fit_A",
        "tile_residual_rms_A", "valid_tiles", "total_tiles", "png", "message",
    ]
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(columns) + "\n")
        for r in results:
            row = [
                r.micrograph_name, r.source_file, str(r.image_index_1based),
                "1" if r.tilt_fitted else "0",
                f"{r.defocus1_A:.6f}", f"{r.defocus2_A:.6f}",
                f"{r.astigmatism_angle_deg:.6f}",
                f"{r.defocus_gradient_x:.9g}", f"{r.defocus_gradient_y:.9g}",
                f"{r.tilt_angle_deg:.6f}", f"{r.tilt_axis_deg:.6f}",
                f"{r.nominal_tilt_angle_deg:.6f}",
                f"{r.coarse_tilt_angle_deg:.6f}", f"{r.coarse_tilt_axis_deg:.6f}",
                f"{r.tilt_score:.8g}", f"{r.score:.8g}",
                f"{r.global_thon_rings_good_fit_resolution_A:.6f}",
                f"{r.tilt_good_fit_resolution_A:.6f}",
                f"{r.tilt_residual_rms_A:.6f}", str(r.tilt_valid_tiles),
                str(r.tilt_total_tiles), r.tilt_png_name,
                r.tilt_message.replace("\t", " ").replace("\n", " "),
            ]
            handle.write("\t".join(row) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Embedded signed CTFTILT backend used only by TorchCtffindPowell.fit_tilt_micrograph.
# It is kept in a private namespace so none of the normal CTFFIND functions or
# constants are replaced when --fit-tilt is not used.
# ---------------------------------------------------------------------------
_EMBEDDED_SIGNED_CTFTILT_SOURCE = '#!/usr/bin/env python3\n"""\nPyTorch port of ctftilt v1.7 (9-Jun-2014).\n\nThe numerical flow follows the supplied Fortran source as closely as practical:\n  * the original tile selection, edge-plane subtraction and background filter;\n  * the original tilt-axis and tilt-angle searches;\n  * the original CTF formula, uncentred correlation and astigmatism penalty;\n  * a direct Python translation of the bundled VA04A Powell minimizer;\n  * MRC I/O through mrcfile and FFTs through torch.fft.\n\nOnly NumPy, PyTorch and mrcfile are required. SciPy is not required.\n\nNotes on literal source compatibility\n-------------------------------------\nThe supplied source contains two suspicious statements. By default this program\npreserves them literally because changing them changes ctftilt behaviour:\n\n1. In FIND_TAXIS_S, A2 uses the value of J left after a completed DO loop,\n   rather than JJ.  In standard Fortran this is upper_bound + 1.\n2. In BOXIMG2, D2 is BOX**2 rather than (BOX/2)**2, so the documented circular\n   mask normally includes the entire square.\n\nPass --fix-source-quirks to use JJ in (1) and a radius of BOX/2 in (2).\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport math\nimport os\nimport sys\nfrom dataclasses import asdict, dataclass\nfrom pathlib import Path\nfrom typing import Callable, Iterable, Optional, Sequence, Tuple\n\nimport mrcfile\nimport numpy as np\nimport torch\nimport torch.nn.functional as F\n\n\nPI = 3.1415926535898\nTWOPI = 6.2831853071796\n\n\n@dataclass\nclass CtfTiltConfig:\n    input_mrc: str\n    output_mrc: str\n    cs_mm: float\n    voltage_kv: float\n    amp_contrast: float\n    magnification: float\n    detector_step_um: float\n    pixel_average: int\n    box: int\n    res_min_a: float\n    res_max_a: float\n    df_min_a: float\n    df_max_a: float\n    df_step_a: float\n    dast_a: float\n    expected_tilt_deg: float\n    tilt_uncertainty_deg: float\n    device: str = "auto"\n    dtype: str = "float32"\n    candidate_batch: int = 256\n    tile_batch: int = 64\n    nr: int = 5\n    fix_source_quirks: bool = False\n    deterministic: bool = False\n    quiet_objective: bool = False\n    result_json: Optional[str] = None\n    fast_gpu: bool = False\n\n\n@dataclass\nclass CtfTiltResult:\n    defocus1_a: float\n    defocus2_a: float\n    astig_angle_deg: float\n    tilt_axis_deg: float\n    tilt_angle_deg: float\n    final_cc: float\n    pixel_size_a: float\n    tiles_total: int\n    tiles_used: int\n    rms_min: float\n    rms_max: float\n\n\ndef nint(x: float) -> int:\n    """Fortran NINT for ordinary finite values: nearest integer, halves away from zero."""\n    if x >= 0.0:\n        return int(math.floor(x + 0.5))\n    return int(math.ceil(x - 0.5))\n\n\ndef fortran_int(x: float) -> int:\n    """Fortran INT truncates toward zero."""\n    return int(x)\n\n\ndef resolve_device(name: str) -> torch.device:\n    if name == "auto":\n        return torch.device("cuda" if torch.cuda.is_available() else "cpu")\n    dev = torch.device(name)\n    if dev.type == "cuda" and not torch.cuda.is_available():\n        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")\n    return dev\n\n\ndef resolve_dtype(name: str) -> torch.dtype:\n    if name == "float32":\n        return torch.float32\n    if name == "float64":\n        return torch.float64\n    raise ValueError(f"Unsupported dtype: {name}")\n\n\ndef read_mrc_2d(path: str) -> np.ndarray:\n    with mrcfile.open(path, mode="r", permissive=True) as mrc:\n        data = np.asarray(mrc.data)\n    if data.ndim == 3:\n        if data.shape[0] != 1:\n            raise ValueError(\n                f"ctftilt expects one 2-D image; received MRC shape {data.shape}. "\n                "Use a single section or an MRC with nz=1."\n            )\n        data = data[0]\n    if data.ndim != 2:\n        raise ValueError(f"Expected a 2-D MRC image, got shape {data.shape}")\n    # Copy also normalizes byte order and makes float16/integer modes safe for torch.\n    return np.array(data, dtype=np.float32, order="C", copy=True)\n\n\ndef write_mrc_2d(path: str, image: torch.Tensor, voxel_size: float) -> None:\n    arr = image.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()\n    out = Path(path)\n    out.parent.mkdir(parents=True, exist_ok=True)\n    with mrcfile.new(out, overwrite=True) as mrc:\n        mrc.set_data(arr)\n        # The original source passes 1/STEPR to IOPEN for the diagnostic image.\n        # mrcfile interprets this field as a voxel size; preserve the numeric value.\n        mrc.voxel_size = float(voxel_size)\n        mrc.update_header_stats()\n\n\ndef pixel_average_image(image: torch.Tensor, pave: int) -> torch.Tensor:\n    if pave <= 1:\n        return image\n    h, w = image.shape\n    h2 = (h // pave) * pave\n    w2 = (w // pave) * pave\n    if h2 == 0 or w2 == 0:\n        raise ValueError("Pixel averaging factor is larger than the image")\n    # Match the source\'s two-stage accumulation closely: vertical sum first,\n    # then horizontal sum, followed by division by pave**2.\n    x = image[:h2, :w2]\n    x = x.reshape(h2 // pave, pave, w2)\n    x = x.sum(dim=1)\n    x = x.reshape(h2 // pave, w2 // pave, pave)\n    x = x.sum(dim=2) / float(pave * pave)\n    return x\n\n\ndef boximg(image: torch.Tensor, ix0: int, iy0: int, box: int) -> Tuple[torch.Tensor, float, float]:\n    """Literal BOXIMG using zero-based upper-left coordinates."""\n    tile = image[iy0 : iy0 + box, ix0 : ix0 + box].clone()\n    if tile.shape != (box, box):\n        return torch.zeros((box, box), device=image.device, dtype=image.dtype), 0.0, 0.0\n    mean_t = tile.mean()\n    m1 = tile[:, 0].mean()\n    m2 = tile[:, -1].mean()\n    m3 = tile[0, :].mean()\n    m4 = tile[-1, :].mean()\n    rms_t = torch.sqrt(torch.mean((tile - mean_t) ** 2))\n    xx = torch.arange(box, device=image.device, dtype=image.dtype)\n    yy = torch.arange(box, device=image.device, dtype=image.dtype)\n    ramp_x = m1 + (m2 - m1) * xx / float(box - 1)\n    ramp_y = m3 + (m4 - m3) * yy / float(box - 1)\n    tile = tile - ramp_x.unsqueeze(0) - ramp_y.unsqueeze(1) + mean_t\n    return tile, float(mean_t.item()), float(rms_t.item())\n\n\ndef boximg2(\n    image: torch.Tensor,\n    ix1: int,\n    iy1: int,\n    box: int,\n    fix_source_quirks: bool,\n) -> Tuple[torch.Tensor, float, float]:\n    """Literal BOXIMG2. ix1/iy1 are one-based as in the Fortran routine."""\n    ix0 = ix1 - 1\n    iy0 = iy1 - 1\n    h, w = image.shape\n    if ix0 < 0 or iy0 < 0 or ix0 + box > w or iy0 + box > h:\n        return torch.zeros((box, box), device=image.device, dtype=image.dtype), 0.0, 0.0\n\n    tile = image[iy0 : iy0 + box, ix0 : ix0 + box].clone()\n    yy = torch.arange(box, device=image.device, dtype=torch.int64)\n    xx = torch.arange(box, device=image.device, dtype=torch.int64)\n    ygrid, xgrid = torch.meshgrid(yy, xx, indexing="ij")\n    rad2 = (ygrid - box // 2) ** 2 + (xgrid - box // 2) ** 2\n    d2 = (box // 2) ** 2 if fix_source_quirks else box**2\n    inside = rad2 <= d2\n\n    n_inside = int(inside.sum().item())\n    if n_inside == 0:\n        return torch.zeros_like(tile), 0.0, 0.0\n    mean_t = tile[inside].sum() / float(n_inside)\n    m1 = tile[:, 0].mean()\n    m2 = tile[:, -1].mean()\n    m3 = tile[0, :].mean()\n    m4 = tile[-1, :].mean()\n\n    xf = torch.arange(box, device=image.device, dtype=image.dtype)\n    yf = torch.arange(box, device=image.device, dtype=image.dtype)\n    ramp_x = m1 + (m2 - m1) * xf / float(box - 1)\n    ramp_y = m3 + (m4 - m3) * yf / float(box - 1)\n    tile = tile - ramp_x.unsqueeze(0) - ramp_y.unsqueeze(1) + mean_t\n    tile = torch.where(inside, tile, torch.zeros((), device=image.device, dtype=image.dtype))\n    rms_t = torch.sqrt(torch.mean(tile**2))\n    return tile, float(mean_t.item()), float(rms_t.item())\n\n\ndef histogram_thresholds(values: Sequence[float], nbin: int = 100) -> Tuple[np.ndarray, float, float, float, float]:\n    data = np.asarray(values, dtype=np.float32)\n    if data.size == 0:\n        raise ValueError("No tiles are available for RMS histogram")\n    vmin = float(np.min(data))\n    vmax = float(np.max(data))\n    if vmin == vmax:\n        vmax += 1.0\n    bins = np.zeros(nbin, dtype=np.float32)\n    indices = np.floor((data - vmin) / (vmax - vmin) * (nbin - 1) + 0.5).astype(np.int64)\n    indices = np.clip(indices, 0, nbin - 1)\n    np.add.at(bins, indices, 1.0)\n\n    peak = int(np.argmax(bins))  # zero-based; Fortran J = peak + 1\n    cmax = float(bins[peak])\n    rms_min = vmin\n    rms_max = vmax\n    if peak > 0:\n        for i0 in range(0, peak):\n            if bins[i0] >= cmax / 20.0:\n                i1 = i0 + 1\n                rms_min = i1 * (vmax - vmin) / float(nbin - 1) + vmin\n                break\n    if peak < nbin - 1:\n        for i0 in range(nbin - 1, peak, -1):\n            if bins[i0] >= cmax / 20.0:\n                i1 = i0 + 1\n                rms_max = i1 * (vmax - vmin) / float(nbin - 1) + vmin\n                break\n    return bins, vmin, vmax, rms_min, rms_max\n\n\ndef msmooth(power: torch.Tensor, nw: int) -> torch.Tensor:\n    """Literal MSMOOTH on a half-complex power layout [Y, Xhalf]."""\n    if power.ndim != 2:\n        raise ValueError("msmooth expects a 2-D tensor")\n    ny, nx = power.shape\n    if nw < 0:\n        raise ValueError("Negative smoothing radius")\n\n    yi = torch.arange(1, ny + 1, device=power.device, dtype=torch.int64).view(ny, 1).expand(ny, nx)\n    xi = torch.arange(1, nx + 1, device=power.device, dtype=torch.int64).view(1, nx).expand(ny, nx)\n    acc = torch.zeros_like(power)\n    cnt = torch.zeros((ny, nx), device=power.device, dtype=torch.int32)\n\n    for dk in range(-nw, nw + 1):\n        for dl in range(-nw, nw + 1):\n            ix = xi + dk\n            iy = yi + dl\n\n            ix = torch.where(ix > nx, ix - 2 * nx, ix)\n            reflect_x = ix < 1\n            ix = torch.where(reflect_x, 1 - ix, ix)\n            iy = torch.where(reflect_x, 1 - iy, iy)\n\n            iy = torch.where(iy > ny, iy - ny, iy)\n            iy = torch.where(iy <= -ny, iy + ny, iy)\n            iy = torch.where(iy < 1, 1 - iy, iy)\n\n            valid = (ix > 1) & (iy > 1) & (ix <= nx) & (iy <= ny)\n            gx = torch.clamp(ix - 1, 0, nx - 1)\n            gy = torch.clamp(iy - 1, 0, ny - 1)\n            sample = power[gy, gx]\n            acc = acc + torch.where(valid, sample, torch.zeros((), device=power.device, dtype=power.dtype))\n            cnt = cnt + valid.to(torch.int32)\n\n    background = torch.where(cnt > 0, acc / cnt.to(power.dtype), power)\n    background = background.clone()\n    background[0, 0] = power[0, 0]\n    return power**2 - background**2\n\n\n\n\n_MSMOOTH_FAST_CACHE = {}\n_BOXIMG2_FAST_CACHE = {}\n\n\ndef _window_sum_2d(x: torch.Tensor, kernel: int) -> torch.Tensor:\n    """Sliding square-window sums for tensors shaped [..., Y, X]."""\n    padded = F.pad(x, (1, 0, 1, 0))\n    integral = padded.cumsum(dim=-2).cumsum(dim=-1)\n    return (\n        integral[..., kernel:, kernel:]\n        - integral[..., :-kernel, kernel:]\n        - integral[..., kernel:, :-kernel]\n        + integral[..., :-kernel, :-kernel]\n    )\n\n\ndef _msmooth_fast_mapping(\n    ny: int,\n    nx: int,\n    nw: int,\n    device: torch.device,\n    dtype: torch.dtype,\n) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:\n    key = (str(device), str(dtype), int(ny), int(nx), int(nw))\n    cached = _MSMOOTH_FAST_CACHE.get(key)\n    if cached is not None:\n        return cached\n\n    xraw = torch.arange(1 - nw, nx + nw + 1, device=device, dtype=torch.int64)\n    yraw = torch.arange(1 - nw, ny + nw + 1, device=device, dtype=torch.int64)\n    yr, xr = torch.meshgrid(yraw, xraw, indexing="ij")\n\n    ix = torch.where(xr > nx, xr - 2 * nx, xr)\n    reflect_x = ix < 1\n    ix = torch.where(reflect_x, 1 - ix, ix)\n    iy = torch.where(reflect_x, 1 - yr, yr)\n    iy = torch.where(iy > ny, iy - ny, iy)\n    iy = torch.where(iy <= -ny, iy + ny, iy)\n    iy = torch.where(iy < 1, 1 - iy, iy)\n\n    valid = (ix > 1) & (iy > 1) & (ix <= nx) & (iy <= ny)\n    gx = torch.clamp(ix - 1, 0, nx - 1)\n    gy = torch.clamp(iy - 1, 0, ny - 1)\n    gather_index = (gy * nx + gx).reshape(-1)\n    valid_f = valid.to(dtype)\n    kernel = 2 * nw + 1\n    counts = _window_sum_2d(valid_f, kernel)\n    cached = (gather_index, valid_f, counts)\n    _MSMOOTH_FAST_CACHE[key] = cached\n    return cached\n\n\ndef msmooth_fast(power: torch.Tensor, nw: int) -> torch.Tensor:\n    """Batched O(N) MSMOOTH path for GPU use.\n\n    It implements the same half-complex coordinate mapping as MSMOOTH, but uses\n    integral-image window sums.  The different floating-point summation order\n    can change the last few float32 bits; the literal msmooth() remains\n    available when bit-level reproduction is more important than speed.\n    """\n    if power.ndim not in (2, 3):\n        raise ValueError("msmooth_fast expects [Y,X] or [B,Y,X]")\n    squeeze = power.ndim == 2\n    x = power.unsqueeze(0) if squeeze else power\n    batch, ny, nx = x.shape\n    gather_index, valid_f, counts = _msmooth_fast_mapping(\n        ny, nx, nw, x.device, x.dtype\n    )\n    ext_h = ny + 2 * nw\n    ext_w = nx + 2 * nw\n    flat = x.reshape(batch, ny * nx)\n    ext = flat[:, gather_index].reshape(batch, ext_h, ext_w)\n    ext = ext * valid_f.unsqueeze(0)\n    kernel = 2 * nw + 1\n    sums = _window_sum_2d(ext, kernel)\n    safe_counts = torch.clamp(counts, min=1.0)\n    background = torch.where(\n        counts.unsqueeze(0) > 0,\n        sums / safe_counts.unsqueeze(0),\n        x,\n    )\n    background = background.clone()\n    background[:, 0, 0] = x[:, 0, 0]\n    result = x * x - background * background\n    return result[0] if squeeze else result\n\n\ndef regular_boximg_batch(\n    image: torch.Tensor,\n    box: int,\n) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:\n    """Batched BOXIMG for the non-overlapping regular tile grid."""\n    h, w = image.shape\n    nx = w // box\n    ny = h // box\n    trimmed = image[: ny * box, : nx * box]\n    raw = (\n        trimmed.reshape(ny, box, nx, box)\n        .permute(0, 2, 1, 3)\n        .reshape(nx * ny, box, box)\n    )\n    mean_t = raw.mean(dim=(1, 2))\n    rms = torch.sqrt(torch.mean((raw - mean_t[:, None, None]) ** 2, dim=(1, 2)))\n    m1 = raw[:, :, 0].mean(dim=1)\n    m2 = raw[:, :, -1].mean(dim=1)\n    m3 = raw[:, 0, :].mean(dim=1)\n    m4 = raw[:, -1, :].mean(dim=1)\n    frac = torch.arange(box, device=image.device, dtype=image.dtype) / float(box - 1)\n    ramp_x = m1[:, None] + (m2 - m1)[:, None] * frac[None, :]\n    ramp_y = m3[:, None] + (m4 - m3)[:, None] * frac[None, :]\n    corrected = raw - ramp_x[:, None, :] - ramp_y[:, :, None] + mean_t[:, None, None]\n\n    ix = torch.arange(nx, device=image.device, dtype=torch.int64).repeat(ny) * box\n    iy = torch.arange(ny, device=image.device, dtype=torch.int64).repeat_interleave(nx) * box\n    centers = torch.stack((ix, iy), dim=1)\n    return corrected, rms, centers, nx, ny\n\n\ndef _boximg2_fast_constants(\n    box: int,\n    fix_source_quirks: bool,\n    device: torch.device,\n    dtype: torch.dtype,\n) -> Tuple[torch.Tensor, torch.Tensor, int]:\n    key = (str(device), str(dtype), int(box), bool(fix_source_quirks))\n    cached = _BOXIMG2_FAST_CACHE.get(key)\n    if cached is not None:\n        return cached\n    yy = torch.arange(box, device=device, dtype=torch.int64)\n    xx = torch.arange(box, device=device, dtype=torch.int64)\n    yg, xg = torch.meshgrid(yy, xx, indexing="ij")\n    rad2 = (yg - box // 2) ** 2 + (xg - box // 2) ** 2\n    d2 = (box // 2) ** 2 if fix_source_quirks else box ** 2\n    inside = rad2 <= d2\n    frac = torch.arange(box, device=device, dtype=dtype) / float(box - 1)\n    n_inside = int(inside.sum().item())\n    cached = (inside, frac, n_inside)\n    _BOXIMG2_FAST_CACHE[key] = cached\n    return cached\n\n\ndef boximg2_batch(\n    raw: torch.Tensor,\n    box: int,\n    fix_source_quirks: bool,\n) -> Tuple[torch.Tensor, torch.Tensor]:\n    """Batched numerical body of BOXIMG2; raw is [B,box,box]."""\n    inside, frac, n_inside = _boximg2_fast_constants(\n        box, fix_source_quirks, raw.device, raw.dtype\n    )\n    inside_f = inside.to(raw.dtype)\n    mean_t = (raw * inside_f.unsqueeze(0)).sum(dim=(1, 2)) / float(n_inside)\n    m1 = raw[:, :, 0].mean(dim=1)\n    m2 = raw[:, :, -1].mean(dim=1)\n    m3 = raw[:, 0, :].mean(dim=1)\n    m4 = raw[:, -1, :].mean(dim=1)\n    ramp_x = m1[:, None] + (m2 - m1)[:, None] * frac[None, :]\n    ramp_y = m3[:, None] + (m4 - m3)[:, None] * frac[None, :]\n    tile = raw - ramp_x[:, None, :] - ramp_y[:, :, None] + mean_t[:, None, None]\n    tile = tile * inside_f.unsqueeze(0)\n    rms = torch.sqrt(torch.mean(tile * tile, dim=(1, 2)))\n    return tile, rms\n\n\ndef find_taxis_single_fast(\n    image: torch.Tensor,\n    box: int,\n    kx: int,\n    rms_min: float,\n    rms_max: float,\n    nr: int,\n    angle_deg: int,\n    fix_source_quirks: bool,\n    store: bool,\n) -> Tuple[float, Optional[torch.Tensor]]:\n    """GPU-batched FIND_TAXIS_S with one host synchronization per angle."""\n    h, w = image.shape\n    nx = min(w // box, h // box)\n    cx = w // 2\n    cy = h // 2\n    irl2 = (box // 2) ** 2\n    alpha32 = np.float32(np.float32(angle_deg) / np.float32(180.0) * np.float32(PI))\n    stale_j = (box // 2) * box + 1\n\n    coords = []\n    groups = []\n    for group, jj in enumerate(range(-nr, nr + 1)):\n        if fix_source_quirks:\n            a2_32 = np.float32(\n                alpha32 + np.float32(jj) * np.float32(60.0) / np.float32(180.0) * np.float32(PI)\n            )\n        else:\n            a2_32 = np.float32(\n                alpha32 + np.float32(stale_j) * np.float32(60.0) / np.float32(180.0) * np.float32(PI)\n            )\n        cos_alpha = np.float32(np.cos(alpha32))\n        sin_alpha = np.float32(np.sin(alpha32))\n        cos_a2 = np.float32(np.cos(a2_32))\n        sin_a2 = np.float32(np.sin(a2_32))\n        upper = nx - abs(jj)\n        for j1 in range(1, upper + 1):\n            ixf = np.float32(\n                np.float32(cx)\n                + cos_alpha * np.float32((j1 - nx // 2) * box)\n                + np.float32(abs(jj)) * cos_a2 * np.float32(box)\n                - np.float32(box // 2)\n            )\n            iyf = np.float32(\n                np.float32(cy)\n                + sin_alpha * np.float32((j1 - nx // 2) * box)\n                + np.float32(abs(jj)) * sin_a2 * np.float32(box)\n                - np.float32(box // 2)\n            )\n            ix0 = fortran_int(float(ixf)) - 1\n            iy0 = fortran_int(float(iyf)) - 1\n            if ix0 < 0 or iy0 < 0 or ix0 + box > w or iy0 + box > h:\n                continue\n            coords.append((ix0, iy0))\n            groups.append(group)\n\n    if not coords:\n        return float("inf"), None\n    raw = torch.stack([image[y:y + box, x:x + box] for x, y in coords], dim=0)\n    tiles, rms = boximg2_batch(raw, box, fix_source_quirks)\n    good = (rms < float(rms_max)) & (rms > float(rms_min))\n    spectra = torch.fft.rfft2(tiles)[:, :, :kx]\n    safe_rms = torch.clamp(rms, min=torch.finfo(image.dtype).tiny)\n    p = (spectra.real * spectra.real + spectra.imag * spectra.imag) / (safe_rms[:, None, None] ** 2)\n    group_t = torch.tensor(groups, device=image.device, dtype=torch.int64)\n\n    yfreq = torch.arange(box, device=image.device, dtype=torch.int64).view(box, 1)\n    xfreq = torch.arange(kx, device=image.device, dtype=torch.int64).view(1, kx)\n    variance_mask = (yfreq > 4) & (xfreq > 4) & (yfreq ** 2 + xfreq ** 2 < irl2)\n\n    out = torch.zeros((box, kx), device=image.device, dtype=image.dtype)\n    var_sum = torch.zeros((), device=image.device, dtype=image.dtype)\n    cnt3 = torch.zeros((), device=image.device, dtype=image.dtype)\n    cnt4 = torch.zeros((), device=image.device, dtype=image.dtype)\n    for group in range(2 * nr + 1):\n        weight = ((group_t == group) & good).to(image.dtype)\n        cnt = weight.sum()\n        sum1 = torch.sum(p * weight[:, None, None], dim=0)\n        sum2 = torch.sum((p * p) * weight[:, None, None], dim=0)\n        out = out + sum1\n        cnt3 = cnt3 + cnt\n        denom = torch.clamp(cnt, min=1.0)\n        mean1 = sum1 / denom\n        var = sum2 / denom - mean1 * mean1\n        var_mean = var[variance_mask].mean()\n        has_var = cnt > 1.0\n        var_sum = var_sum + torch.where(has_var, var_mean, torch.zeros_like(var_mean))\n        cnt4 = cnt4 + has_var.to(image.dtype)\n\n    stats = torch.stack((cnt3, cnt4, var_sum)).detach().cpu().numpy()\n    cnt3_f, cnt4_f, var_sum_f = map(float, stats)\n    value = var_sum_f / cnt4_f if cnt4_f > 0.0 else float("inf")\n    if store:\n        if cnt3_f <= 0.0:\n            raise RuntimeError("No valid spectra were found while storing the tilt-axis power spectrum")\n        return value, torch.sqrt(out / cnt3)\n    return value, None\n\n\ndef fft_amplitude_half(tile: torch.Tensor, kx: int, scale: float = 1.0) -> torch.Tensor:\n    spec = torch.fft.rfft2(tile)\n    if kx > spec.shape[-1]:\n        raise ValueError(f"Requested kx={kx}, but rfft has only {spec.shape[-1]} columns")\n    return torch.abs(spec[:, :kx]) * float(scale)\n\n\ndef fft_power_half(tile: torch.Tensor, kx: int, rms: float) -> torch.Tensor:\n    spec = torch.fft.rfft2(tile)\n    return (spec[:, :kx].real**2 + spec[:, :kx].imag**2) / float(rms * rms)\n\n\ndef find_taxis_single(\n    image: torch.Tensor,\n    box: int,\n    kx: int,\n    rms_min: float,\n    rms_max: float,\n    nr: int,\n    angle_deg: int,\n    fix_source_quirks: bool,\n    store: bool,\n) -> Tuple[float, Optional[torch.Tensor]]:\n    h, w = image.shape\n    nx = min(w // box, h // box)\n    cx = w // 2\n    cy = h // 2\n    irl2 = (box // 2) ** 2\n    # FIND_TAXIS_S uses default REAL arithmetic.  Coordinate truncation can\n    # change by one pixel if these expressions are evaluated in float64.\n    alpha32 = np.float32(np.float32(angle_deg) / np.float32(180.0) * np.float32(PI))\n    alpha = float(alpha32)\n\n    out = torch.zeros((box, kx), device=image.device, dtype=image.dtype)\n    var_sum = 0.0\n    cnt4 = 0\n    cnt3 = 0\n\n    # The preceding Fortran DO J=1,BOX/2*BOX leaves J=upper+1.\n    stale_j = (box // 2) * box + 1\n\n    yfreq = torch.arange(box, device=image.device, dtype=torch.int64).view(box, 1)\n    xfreq = torch.arange(kx, device=image.device, dtype=torch.int64).view(1, kx)\n    variance_mask = (yfreq > 4) & (xfreq > 4) & (yfreq**2 + xfreq**2 < irl2)\n\n    for jj in range(-nr, nr + 1):\n        if fix_source_quirks:\n            a2_32 = np.float32(\n                alpha32\n                + np.float32(jj) * np.float32(60.0) / np.float32(180.0) * np.float32(PI)\n            )\n        else:\n            a2_32 = np.float32(\n                alpha32\n                + np.float32(stale_j) * np.float32(60.0) / np.float32(180.0) * np.float32(PI)\n            )\n        cos_alpha = np.float32(np.cos(alpha32))\n        sin_alpha = np.float32(np.sin(alpha32))\n        cos_a2 = np.float32(np.cos(a2_32))\n        sin_a2 = np.float32(np.sin(a2_32))\n\n        tiles = []\n        rmss = []\n        upper = nx - abs(jj)\n        for j1 in range(1, upper + 1):\n            # All integer divisions match Fortran integer arithmetic.\n            ixf = np.float32(\n                np.float32(cx)\n                + cos_alpha * np.float32((j1 - nx // 2) * box)\n                + np.float32(abs(jj)) * cos_a2 * np.float32(box)\n                - np.float32(box // 2)\n            )\n            iyf = np.float32(\n                np.float32(cy)\n                + sin_alpha * np.float32((j1 - nx // 2) * box)\n                + np.float32(abs(jj)) * sin_a2 * np.float32(box)\n                - np.float32(box // 2)\n            )\n            ix1 = fortran_int(float(ixf))\n            iy1 = fortran_int(float(iyf))\n            tile, _mean, rms = boximg2(image, ix1, iy1, box, fix_source_quirks)\n            if rms < rms_max and rms > rms_min:\n                tiles.append(tile)\n                rmss.append(rms)\n\n        cnt = len(tiles)\n        if cnt == 0:\n            continue\n        batch = torch.stack(tiles, dim=0)\n        spectra = torch.fft.rfft2(batch)[:, :, :kx]\n        rms_t = torch.tensor(rmss, device=image.device, dtype=image.dtype).view(cnt, 1, 1)\n        p = (spectra.real**2 + spectra.imag**2) / (rms_t**2)\n        sum1 = p.sum(dim=0)\n        sum2 = (p**2).sum(dim=0)\n        out += sum1\n        cnt3 += cnt\n\n        if cnt > 1:\n            mean1 = sum1 / float(cnt)\n            var = sum2 / float(cnt) - mean1**2\n            selected = var[variance_mask]\n            if selected.numel() > 0:\n                var_sum += float(selected.mean().item())\n                cnt4 += 1\n\n    value = var_sum / float(cnt4) if cnt4 > 0 else float("inf")\n    if store:\n        if cnt3 <= 0:\n            raise RuntimeError("No valid spectra were found while storing the tilt-axis power spectrum")\n        return value, torch.sqrt(out / float(cnt3))\n    return value, None\n\n\ndef find_taxis(\n    image: torch.Tensor,\n    box: int,\n    kx: int,\n    rms_min: float,\n    rms_max: float,\n    nr: int,\n    fix_source_quirks: bool,\n    fast_gpu: bool = False,\n) -> Tuple[float, torch.Tensor]:\n    print("\\n SEARCHING FOR TILT AXIS...\\n")\n    minv = float("inf")\n    best_angle = 1\n    single = find_taxis_single_fast if fast_gpu else find_taxis_single\n    for angle in range(1, 180, 2):\n        value, _ = single(\n            image, box, kx, rms_min, rms_max, nr, angle, fix_source_quirks, store=False\n        )\n        if value < minv:\n            minv = value\n            best_angle = angle\n            print(f" Angle between tilt axis and X-axis = {float(angle):8.2f}")\n    _value, power = single(\n        image, box, kx, rms_min, rms_max, nr, best_angle, fix_source_quirks, store=True\n    )\n    assert power is not None\n    return float(best_angle) / 180.0 * PI, power\n\n\ndef average_power_entire_image(\n    image: torch.Tensor,\n    box: int,\n    kx: int,\n    rms_min: float,\n    rms_max: float,\n    fast_gpu: bool = False,\n    regular_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]] = None,\n) -> Tuple[torch.Tensor, int]:\n    if fast_gpu:\n        tiles, rms, _centers, _nx, _ny = regular_cache or regular_boximg_batch(image, box)\n        good = (rms < float(rms_max)) & (rms > float(rms_min))\n        cnt = int(good.sum().item())\n        if cnt <= 0:\n            raise RuntimeError("No tiles passed the RMS filter for the initial power spectrum")\n        spec = torch.fft.rfft2(tiles[good])[:, :, :kx]\n        power = torch.sum(spec.real * spec.real + spec.imag * spec.imag, dim=0) / float(box * box)\n        return torch.sqrt(power / float(cnt)), cnt\n\n    h, w = image.shape\n    nx = w // box\n    ny = h // box\n    power = torch.zeros((box, kx), device=image.device, dtype=image.dtype)\n    scale = 1.0 / math.sqrt(float(box * box))\n    cnt = 0\n    for iy in range(ny):\n        for ix in range(nx):\n            tile, _mean, rms = boximg(image, ix * box, iy * box, box)\n            if rms < rms_max and rms > rms_min:\n                amp = fft_amplitude_half(tile, kx, scale)\n                power += amp**2\n                cnt += 1\n    if cnt <= 0:\n        raise RuntimeError("No tiles passed the RMS filter for the initial power spectrum")\n    return torch.sqrt(power / float(cnt)), cnt\n\n\ndef filter_power(\n    power: torch.Tensor,\n    box: int,\n    res_min_pixel: float,\n    fast_gpu: bool = False,\n) -> Tuple[torch.Tensor, float, float, float]:\n    print("\\n FILTERING POWER SPECTRUM...\\n")\n    kx = int(power.shape[1])\n    nw = int(kx * res_min_pixel * math.sqrt(2.0))\n    filtered_k = msmooth_fast(power, nw) if fast_gpu else msmooth(power, nw)\n\n    half = box // 2\n    # The overwhelmingly common path has KXYZ(1)==BOX/2.  In that case the\n    # Fortran resize is an identity copy, so avoid one .item() synchronization\n    # per Fourier pixel.\n    if kx == half:\n        interior = filtered_k[2 : box - 2, 2 : half - 2]\n        dmean_t = interior.sum() / float(box * box) * 2.0\n        dsqr_t = torch.sum(interior * interior) / float(box * box) * 2.0\n        dmax_t = interior.max()\n        stats = torch.stack((dmean_t, dsqr_t, dmax_t)).detach().cpu().numpy()\n        dmean, dsqr, dmax = map(float, stats)\n        drms = math.sqrt(max(0.0, dsqr - dmean * dmean))\n        resized = torch.minimum(filtered_k.clone(), dmax_t)\n        return resized, dmean, drms, dmax\n\n    # Reproduce FILTER\'s in-place resize from KXYZ(1)*Y to BOX/2*Y,\n    # including its sequential flat-memory assignments.\n    src = filtered_k.contiguous().view(-1).clone()\n    work = src.clone()\n    dmean = 0.0\n    dsqr = 0.0\n    dmax = -1.0e30\n    for j1 in range(3, box - 1):  # 3 .. box-2 inclusive\n        for i1 in range(3, half - 1):  # 3 .. half-2 inclusive\n            id0 = (i1 - 1) + half * (j1 - 1)\n            is0 = (i1 - 1) + kx * (j1 - 1)\n            val = float(work[is0].item())\n            work[id0] = val\n            dmean += val\n            dsqr += val * val\n            if val > dmax:\n                dmax = val\n\n    dmean = dmean / float(box * box) * 2.0\n    dsqr = dsqr / float(box * box) * 2.0\n    drms = math.sqrt(max(0.0, dsqr - dmean * dmean))\n    resized = work[: half * box].reshape(box, half).clone()\n    resized = torch.minimum(resized, torch.tensor(dmax, device=power.device, dtype=power.dtype))\n    return resized, dmean, drms, dmax\n\n\nclass CtfEvaluator:\n    def __init__(\n        self,\n        box: int,\n        cs_a: float,\n        wavelength_a: float,\n        wgh1: float,\n        wgh2: float,\n        theta_tr: float,\n        rmin2: float,\n        rmax2: float,\n        hw: float,\n        device: torch.device,\n        dtype: torch.dtype,\n    ) -> None:\n        self.box = int(box)\n        self.cs = float(cs_a)\n        self.wl = float(wavelength_a)\n        self.wgh1 = float(wgh1)\n        self.wgh2 = float(wgh2)\n        self.theta_tr = float(theta_tr)\n        self.rmin2 = float(rmin2)\n        self.rmax2 = float(rmax2)\n        self.hw = float(hw)\n        self.device = device\n        self.dtype = dtype\n\n        mm = torch.arange(box, device=device, dtype=torch.int64)\n        mm = torch.where(mm > box // 2, mm - box, mm).to(dtype)\n        ll = torch.arange(box // 2, device=device, dtype=dtype)\n        mgrid, lgrid = torch.meshgrid(mm, ll, indexing="ij")\n        res2 = (lgrid / float(box)) ** 2 + (mgrid / float(box)) ** 2\n        mask = (res2 <= self.rmax2) & (res2 > self.rmin2)\n        self.mask = mask\n        self.is_count = int(mask.sum().item())\n        if self.is_count == 0:\n            raise ValueError("No Fourier pixels lie inside the requested resolution range")\n\n        l = lgrid[mask]\n        m = mgrid[mask]\n        rad2 = l * l + m * m\n        angspt = torch.atan2(m, l)\n        half_thetatrsq = 0.5 * self.theta_tr * self.theta_tr\n        hangle2 = rad2 * half_thetatrsq\n        self.c1 = (TWOPI / self.wl) * hangle2\n        self.c2 = -self.c1 * self.cs * hangle2\n        self.cos2ang = torch.cos(2.0 * angspt)\n        self.sin2ang = torch.sin(2.0 * angspt)\n        self.origin = rad2 == 0.0\n        self.origin_any = bool(self.origin.any().item())\n        self.res2_selected = res2[mask]\n        self.expv = torch.exp(self.hw * self.res2_selected) if self.hw != 0.0 else None\n\n    def _ctf2(self, df1: torch.Tensor, df2: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:\n        # Inputs are [B]; output is [B,P].\n        dsum = (df1 + df2).unsqueeze(1)\n        ddif = (df1 - df2).unsqueeze(1)\n        a = ang.unsqueeze(1)\n        ccos = self.cos2ang.unsqueeze(0) * torch.cos(2.0 * a) + self.sin2ang.unsqueeze(0) * torch.sin(2.0 * a)\n        df = 0.5 * (dsum + ccos * ddif)\n        chi = self.c1.unsqueeze(0) * df + self.c2.unsqueeze(0)\n        ctf = -self.wgh1 * torch.sin(chi) - self.wgh2 * torch.cos(chi)\n        if self.origin_any:\n            ctf[:, self.origin] = -self.wgh2\n        return ctf * ctf\n\n    def eval_candidates(\n        self,\n        power: torch.Tensor,\n        df1: torch.Tensor,\n        df2: torch.Tensor,\n        ang: torch.Tensor,\n        dast: float,\n        batch_size: int,\n    ) -> Tuple[torch.Tensor, torch.Tensor]:\n        p = power[self.mask].to(device=self.device, dtype=self.dtype)\n        sum2_const = None\n        if self.hw == 0.0:\n            sum2_const = torch.sum(p * p)\n        out_cc = []\n        out_sig2 = []\n        for start in range(0, df1.numel(), batch_size):\n            sl = slice(start, min(start + batch_size, df1.numel()))\n            ctf2 = self._ctf2(df1[sl], df2[sl], ang[sl])\n            if self.hw == 0.0:\n                summ = torch.sum(ctf2 * p.unsqueeze(0), dim=1)\n                sum1 = torch.sum(ctf2 * ctf2, dim=1)\n                sum2 = sum2_const.expand_as(summ)\n            else:\n                assert self.expv is not None\n                weighted_p = p * self.expv\n                summ = torch.sum(ctf2 * weighted_p.unsqueeze(0), dim=1)\n                sum1 = torch.sum(ctf2 * ctf2, dim=1)\n                sum2_scalar = torch.sum((p * self.expv) ** 2)\n                sum2 = sum2_scalar.expand_as(summ)\n            tiny = torch.finfo(self.dtype).tiny\n            safe_sum1 = torch.clamp(sum1, min=tiny)\n            safe_sum = torch.where(torch.abs(summ) > tiny, summ, torch.full_like(summ, tiny))\n            aa = summ / safe_sum1\n            safe_a = torch.where(torch.abs(aa) > tiny, aa, torch.full_like(aa, tiny))\n            sig2 = ((sum2 / safe_a + sum1 * aa) / safe_sum - 2.0) / float(self.is_count)\n            cc = summ / torch.sqrt(torch.clamp(sum1 * sum2, min=tiny))\n            if dast > 0.0:\n                cc = cc - (df1[sl] - df2[sl]) ** 2 / (2.0 * dast * dast * float(self.is_count))\n            out_cc.append(cc)\n            out_sig2.append(sig2)\n        return torch.cat(out_cc), torch.cat(out_sig2)\n\n    def eval_single(\n        self, power: torch.Tensor, df1: float, df2: float, ang: float, dast: float\n    ) -> Tuple[float, float]:\n        t1 = torch.tensor([df1], device=self.device, dtype=self.dtype)\n        t2 = torch.tensor([df2], device=self.device, dtype=self.dtype)\n        ta = torch.tensor([ang], device=self.device, dtype=self.dtype)\n        cc, sig2 = self.eval_candidates(power, t1, t2, ta, dast, 1)\n        return float(cc[0].item()), float(sig2[0].item())\n\n    def eval_paired(\n        self,\n        spectra: torch.Tensor,\n        df1: torch.Tensor,\n        df2: torch.Tensor,\n        ang: float,\n        dast: float,\n        batch_size: int,\n    ) -> Tuple[torch.Tensor, torch.Tensor]:\n        selected = spectra[:, self.mask]\n        out_cc = []\n        out_sig2 = []\n        for start in range(0, spectra.shape[0], batch_size):\n            stop = min(start + batch_size, spectra.shape[0])\n            p = selected[start:stop]\n            n = stop - start\n            a = torch.full((n,), float(ang), device=self.device, dtype=self.dtype)\n            ctf2 = self._ctf2(df1[start:stop], df2[start:stop], a)\n            if self.hw == 0.0:\n                summ = torch.sum(p * ctf2, dim=1)\n                sum1 = torch.sum(ctf2 * ctf2, dim=1)\n                sum2 = torch.sum(p * p, dim=1)\n            else:\n                assert self.expv is not None\n                summ = torch.sum(p * ctf2 * self.expv.unsqueeze(0), dim=1)\n                sum1 = torch.sum(ctf2 * ctf2, dim=1)\n                sum2 = torch.sum((p * self.expv.unsqueeze(0)) ** 2, dim=1)\n            tiny = torch.finfo(self.dtype).tiny\n            safe_sum1 = torch.clamp(sum1, min=tiny)\n            safe_sum = torch.where(torch.abs(summ) > tiny, summ, torch.full_like(summ, tiny))\n            aa = summ / safe_sum1\n            safe_a = torch.where(torch.abs(aa) > tiny, aa, torch.full_like(aa, tiny))\n            sig2 = ((sum2 / safe_a + sum1 * aa) / safe_sum - 2.0) / float(self.is_count)\n            cc = summ / torch.sqrt(torch.clamp(sum1 * sum2, min=tiny))\n            if dast > 0.0:\n                cc = cc - (df1[start:stop] - df2[start:stop]) ** 2 / (\n                    2.0 * dast * dast * float(self.is_count)\n                )\n            out_cc.append(cc)\n            out_sig2.append(sig2)\n        return torch.cat(out_cc), torch.cat(out_sig2)\n\n    def ctf_value_grid(self, df1: float, df2: float, ang: float) -> torch.Tensor:\n        mm = torch.arange(self.box, device=self.device, dtype=torch.int64)\n        mm = torch.where(mm > self.box // 2, mm - self.box, mm).to(self.dtype)\n        ll = torch.arange(self.box // 2, device=self.device, dtype=self.dtype)\n        mgrid, lgrid = torch.meshgrid(mm, ll, indexing="ij")\n        rad = torch.sqrt(lgrid * lgrid + mgrid * mgrid)\n        angle = rad * self.theta_tr\n        angspt = torch.atan2(mgrid, lgrid)\n        c1 = TWOPI * angle * angle / (2.0 * self.wl)\n        c2 = -c1 * self.cs * angle * angle / 2.0\n        ccos = torch.cos(2.0 * (angspt - ang))\n        dfl = 0.5 * (df1 + df2 + ccos * (df1 - df2))\n        chi = c1 * dfl + c2\n        ctf = -self.wgh1 * torch.sin(chi) - self.wgh2 * torch.cos(chi)\n        ctf = torch.where(rad == 0.0, torch.tensor(-self.wgh2, device=self.device, dtype=self.dtype), ctf)\n        return ctf\n\n\ndef search_ctf(\n    evaluator: CtfEvaluator,\n    power: torch.Tensor,\n    df_min: float,\n    df_max: float,\n    fstep: float,\n    dast: float,\n    candidate_batch: int,\n) -> Tuple[float, float, float, float]:\n    print("\\n SEARCHING CTF PARAMETERS...\\n")\n    print("      DFMID1      DFMID2      ANGAST          CC")\n    i1 = fortran_int(df_min / fstep)\n    i2 = fortran_int(df_max / fstep)\n    if i2 < i1:\n        raise ValueError("Invalid defocus grid")\n    ints = list(range(i1, i2 + 1))\n    global_max = -1.0e20\n    best = (df_min, df_max, 0.0)\n\n    # Fortran scans SUMS in flat ID order: J outer, I inner.\n    for k in range(18):\n        df1_list = []\n        df2_list = []\n        ang_list = []\n        for j in ints:\n            for i in ints:\n                df1_list.append(fstep * i)\n                df2_list.append(fstep * j)\n                ang_list.append(5.0 * k / 180.0 * PI)\n        t1 = torch.tensor(df1_list, device=evaluator.device, dtype=evaluator.dtype)\n        t2 = torch.tensor(df2_list, device=evaluator.device, dtype=evaluator.dtype)\n        ta = torch.tensor(ang_list, device=evaluator.device, dtype=evaluator.dtype)\n        sums, _sig = evaluator.eval_candidates(power, t1, t2, ta, dast, candidate_batch)\n        vals = sums.detach().cpu().numpy()\n        for idx, value in enumerate(vals):\n            v = float(value)\n            if v > global_max:\n                global_max = v\n                best = (df1_list[idx], df2_list[idx], ang_list[idx])\n                print(f"{best[0]:12.2f}{best[1]:12.2f}{best[2] / PI * 180.0:12.2f}{v:12.5f}")\n    return best[0], best[1], best[2], global_max\n\n\ndef va04a(\n    x0: Sequence[float],\n    e0: Sequence[float],\n    objective: Callable[[np.ndarray], float],\n    escale: float = 100.0,\n    iprint: int = 0,\n    icon: int = 1,\n    maxit: int = 50,\n) -> Tuple[np.ndarray, float, int]:\n    """Direct state-machine translation of the VA04A routine in the source."""\n    x = np.zeros(len(x0) + 1, dtype=np.float64)\n    e = np.zeros(len(e0) + 1, dtype=np.float64)\n    x[1:] = np.asarray(x0, dtype=np.float64)\n    e[1:] = np.asarray(e0, dtype=np.float64)\n    n = len(x0)\n    w = np.zeros(n * (n + 3) + 1, dtype=np.float64)\n\n    ddmag = 0.1 * escale\n    scer = 0.05 / escale\n    jj = n * n + n\n    jjj = jj + n\n    k = n + 1\n    nfcc = 1\n    ind = 1\n    inn = 1\n    for i in range(1, n + 1):\n        for j in range(1, n + 1):\n            w[k] = 0.0\n            if i == j:\n                w[k] = abs(e[i])\n                w[i] = escale\n            k += 1\n\n    iterc = 1\n    isgrad = 2\n    f = float(objective(x[1:].copy()))\n    fkeep = abs(f) + abs(f)\n    pc = 5\n\n    # Variables are intentionally kept in the outer scope, mirroring FORTRAN labels.\n    _state_steps = 0\n    while True:\n        _state_steps += 1\n        if _state_steps > 2000000:\n            raise RuntimeError(f"VA04A exceeded state-step limit at label {pc}; f={f}; x={x[1:]}")\n        if pc == 5:\n            itone = 1\n            fp = f\n            summ_imp = 0.0\n            ixp = jj\n            for i in range(1, n + 1):\n                ixp += 1\n                w[ixp] = x[i]\n            idirn = n + 1\n            iline = 1\n            pc = 7\n\n        elif pc == 7:\n            dmax = w[iline]\n            dacc = dmax * scer\n            dmag = min(ddmag, 0.1 * dmax)\n            dmag = max(dmag, 20.0 * dacc)\n            ddmax = 10.0 * dmag\n            pc = 71 if itone == 3 else 70\n\n        elif pc == 70:\n            dl = 0.0\n            d = dmag\n            fprev = f\n            is_ = 5\n            fa = f\n            da = dl\n            pc = 8\n\n        elif pc == 71:\n            dl = 1.0\n            ddmax = 5.0\n            fa = fp\n            da = -1.0\n            fb = fhold\n            db = 0.0\n            d = 1.0\n            pc = 10\n\n        elif pc == 8:\n            dd = d - dl\n            dl = d\n            pc = 58\n\n        elif pc == 58:\n            k = idirn\n            for i in range(1, n + 1):\n                x[i] = x[i] + dd * w[k]\n                k += 1\n            f = float(objective(x[1:].copy()))\n            nfcc += 1\n            pc = {1: 10, 2: 11, 3: 12, 4: 13, 5: 14, 6: 96}[is_]\n\n        elif pc == 14:\n            if f < fa:\n                pc = 15\n            elif f == fa:\n                pc = 16\n            else:\n                pc = 24\n\n        elif pc == 16:\n            if abs(d) <= dmax:\n                pc = 17\n            else:\n                print("     VA04A MAXIMUM CHANGE DOES NOT ALTER FUNCTION")\n                pc = 20\n\n        elif pc == 17:\n            d = d + d\n            pc = 8\n\n        elif pc == 15:\n            fb = f\n            db = d\n            pc = 21\n\n        elif pc == 24:\n            fb = fa\n            db = da\n            fa = f\n            da = d\n            pc = 21\n\n        elif pc == 21:\n            pc = 83 if isgrad == 1 else 23\n\n        elif pc == 23:\n            d = db + db - da\n            is_ = 1\n            pc = 8\n\n        elif pc == 83:\n            denom = da - db\n            if denom == 0.0:\n                d = 0.5 * (da + db)\n            else:\n                d = 0.5 * (da + db - (fa - fb) / denom)\n            is_ = 4\n            if (da - d) * (d - db) < 0.0:\n                pc = 25\n            else:\n                pc = 8\n\n        elif pc == 25:\n            is_ = 1\n            if abs(d - db) <= ddmax:\n                pc = 8\n            else:\n                pc = 26\n\n        elif pc == 26:\n            d = db + math.copysign(ddmax, db - da)\n            is_ = 1\n            ddmax = ddmax + ddmax\n            ddmag = ddmag + ddmag\n            if ddmax <= dmax:\n                pc = 8\n            else:\n                pc = 27\n\n        elif pc == 27:\n            ddmax = dmax\n            pc = 8\n\n        elif pc == 13:\n            pc = 28 if f < fa else 23\n\n        elif pc == 28:\n            fc = fb\n            dc = db\n            pc = 29\n\n        elif pc == 29:\n            fb = f\n            db = d\n            pc = 30\n\n        elif pc == 12:\n            if f <= fb:\n                pc = 28\n            else:\n                pc = 31\n\n        elif pc == 31:\n            fa = f\n            da = d\n            pc = 30\n\n        elif pc == 11:\n            pc = 32 if f < fb else 10\n\n        elif pc == 32:\n            fa = fb\n            da = db\n            pc = 29\n\n        elif pc == 10:\n            fc = f\n            dc = d\n            pc = 30\n\n        elif pc == 30:\n            aa = (db - dc) * (fa - fc)\n            bb = (dc - da) * (fb - fc)\n            if (aa + bb) * (da - dc) <= 0.0:\n                pc = 33\n            else:\n                pc = 34\n\n        elif pc == 33:\n            fa = fb\n            da = db\n            fb = fc\n            db = dc\n            pc = 26\n\n        elif pc == 34:\n            denom = aa + bb\n            if denom == 0.0:\n                d = db\n            else:\n                d = 0.5 * (aa * (db + dc) + bb * (da + dc)) / denom\n            di = db\n            fi = fb\n            if fb > fc:\n                di = dc\n                fi = fc\n            pc = 85 if itone == 3 else 86\n\n        elif pc == 85:\n            itone = 2\n            pc = 45\n\n        elif pc == 86:\n            if abs(d - di) <= dacc or abs(d - di) <= 0.03 * abs(d):\n                pc = 41\n            else:\n                pc = 45\n\n        elif pc == 45:\n            if (da - dc) * (dc - d) < 0.0:\n                pc = 47\n            else:\n                pc = 46\n\n        elif pc == 46:\n            fa = fb\n            da = db\n            fb = fc\n            db = dc\n            pc = 25\n\n        elif pc == 47:\n            is_ = 2\n            if (db - d) * (d - dc) < 0.0:\n                pc = 48\n            else:\n                pc = 8\n\n        elif pc == 48:\n            is_ = 3\n            pc = 8\n\n        elif pc == 41:\n            f = fi\n            d = di - dl\n            rad = (dc - db) * (dc - da) * (da - db) / (aa + bb) if (aa + bb) != 0.0 else 0.0\n            dd = math.sqrt(max(0.0, rad))\n            for i in range(1, n + 1):\n                x[i] = x[i] + d * w[idirn]\n                w[idirn] = dd * w[idirn]\n                idirn += 1\n            if dd == 0.0:\n                dd = 1.0e-10\n            w[iline] = w[iline] / dd\n            iline += 1\n            if iprint == 1:\n                print(f"ITERATION {iterc:5d}{nfcc:15d} FUNCTION VALUES          F ={f:21.14E}")\n                print("".join(f"{x[i]:24.14E}" for i in range(1, n + 1)))\n            pc = 51\n\n        elif pc == 51:\n            pc = 55 if itone == 1 else 38\n\n        elif pc == 55:\n            if fprev - f - summ_imp >= 0.0:\n                summ_imp = fprev - f\n                jil = iline\n            if idirn <= jj:\n                pc = 7\n            else:\n                pc = 84\n\n        elif pc == 84:\n            pc = 92 if ind == 1 else 72\n\n        elif pc == 92:\n            fhold = f\n            is_ = 6\n            ixp = jj\n            for i in range(1, n + 1):\n                ixp += 1\n                w[ixp] = x[i] - w[ixp]\n            dd = 1.0\n            pc = 58\n\n        elif pc == 96:\n            pc = 112 if ind == 1 else 87\n\n        elif pc == 112:\n            if fp <= f:\n                pc = 37\n            else:\n                pc = 91\n\n        elif pc == 91:\n            denom = (fp - f) ** 2\n            dtest = 2.0 * (fp + f - 2.0 * fhold) / denom if denom != 0.0 else float("inf")\n            if dtest * (fp - fhold - summ_imp) ** 2 - summ_imp < 0.0:\n                pc = 87\n            else:\n                pc = 37\n\n        elif pc == 87:\n            j = jil * n + 1\n            if j <= jj:\n                pc = 60\n            else:\n                pc = 61\n\n        elif pc == 60:\n            for i in range(j, jj + 1):\n                k = i - n\n                w[k] = w[i]\n            for i in range(jil, n + 1):\n                w[i - 1] = w[i]\n            pc = 61\n\n        elif pc == 61:\n            idirn = idirn - n\n            itone = 3\n            k = idirn\n            ixp = jj\n            aaa = 0.0\n            for i in range(1, n + 1):\n                ixp += 1\n                w[k] = w[ixp]\n                ratio = abs(w[k] / e[i]) if e[i] != 0.0 else float("inf")\n                if aaa < ratio:\n                    aaa = ratio\n                k += 1\n            ddmag = 1.0\n            if aaa == 0.0:\n                aaa = 1.0e-10\n            w[n] = escale / aaa\n            iline = n\n            pc = 7\n\n        elif pc == 37:\n            ixp = jj\n            aaa = 0.0\n            f = fhold\n            for i in range(1, n + 1):\n                ixp += 1\n                x[i] = x[i] - w[ixp]\n                ratio = abs(w[ixp] / e[i]) if e[i] != 0.0 else float("inf")\n                if aaa < ratio:\n                    aaa = ratio\n            pc = 72\n\n        elif pc == 38:\n            aaa = aaa * (1.0 + di)\n            pc = 72 if ind == 1 else 106\n\n        elif pc == 72:\n            if iprint >= 2:\n                print(f"ITERATION {iterc:5d}{nfcc:15d} FUNCTION VALUES          F ={f:21.14E}")\n            pc = 53\n\n        elif pc == 53:\n            pc = 109 if ind == 1 else 88\n\n        elif pc == 109:\n            if aaa <= 0.1:\n                pc = 89\n            else:\n                pc = 76\n\n        elif pc == 89:\n            pc = 20 if icon == 1 else 116\n\n        elif pc == 116:\n            ind = 2\n            pc = 100 if inn == 1 else 101\n\n        elif pc == 100:\n            inn = 2\n            k = jjj\n            for i in range(1, n + 1):\n                k += 1\n                w[k] = x[i]\n                x[i] = x[i] + 10.0 * e[i]\n            fkeep = f\n            f = float(objective(x[1:].copy()))\n            nfcc += 1\n            ddmag = 0.0\n            pc = 108\n\n        elif pc == 76:\n            if f < fp:\n                pc = 35\n            else:\n                print("     VA04A ACCURACY LIMITED BY ERRORS IN F")\n                pc = 20\n\n        elif pc == 88:\n            ind = 1\n            pc = 35\n\n        elif pc == 35:\n            tmp = fp - f\n            ddmag = 0.4 * math.sqrt(tmp) if tmp > 0.0 else 0.0\n            isgrad = 1\n            pc = 108\n\n        elif pc == 108:\n            iterc += 1\n            if iterc <= maxit:\n                pc = 5\n            else:\n                pc = 81\n\n        elif pc == 81:\n            if f <= fkeep:\n                pc = 20\n            else:\n                pc = 110\n\n        elif pc == 110:\n            f = fkeep\n            for i in range(1, n + 1):\n                jjj += 1\n                x[i] = w[jjj]\n            pc = 20\n\n        elif pc == 101:\n            jil = 1\n            fp = fkeep\n            if f < fkeep:\n                pc = 105\n            elif f == fkeep:\n                print("     VA04A ACCURACY LIMITED BY ERRORS IN F")\n                pc = 20\n            else:\n                pc = 104\n\n        elif pc == 104:\n            jil = 2\n            fp = f\n            f = fkeep\n            pc = 105\n\n        elif pc == 105:\n            ixp = jj\n            for i in range(1, n + 1):\n                ixp += 1\n                k = ixp + n\n                if jil == 1:\n                    w[ixp] = w[k]\n                else:\n                    w[ixp] = x[i]\n                    x[i] = w[k]\n            jil = 2\n            pc = 92\n\n        elif pc == 106:\n            pc = 20 if aaa <= 0.1 else 107\n\n        elif pc == 107:\n            inn = 1\n            pc = 35\n\n        elif pc == 20:\n            return x[1:].copy(), float(f), nfcc\n\n        else:\n            raise RuntimeError(f"Internal VA04A state error at label {pc}")\n\n\ndef refine_ctf(\n    evaluator: CtfEvaluator,\n    power: torch.Tensor,\n    df1: float,\n    df2: float,\n    ang: float,\n    dast: float,\n) -> Tuple[float, float, float, float]:\n    print("\\n REFINING CTF PARAMETERS...\\n")\n    print("      DFMID1      DFMID2      ANGAST          CC")\n    x0 = np.array([df1, df2, ang], dtype=np.float64)\n    if x0[0] == x0[1]:\n        x0[0] += 1.0\n\n    def objective(x: np.ndarray) -> float:\n        cc, _sig2 = evaluator.eval_single(power, float(x[0]), float(x[1]), float(x[2]), dast)\n        return -cc\n\n    x, rf, _nf = va04a(x0, [100.0, 100.0, 0.5], objective, escale=100.0, iprint=0, icon=1, maxit=50)\n    df1, df2, ang = map(float, x)\n    print(f"{df1:12.2f}{df2:12.2f}{180.0 * (ang / PI - nint(ang / PI)):12.2f}{-rf:12.5f}  Refined Values at Center")\n    return df1, df2, ang, -rf\n\n\ndef tile_image(\n    image: torch.Tensor,\n    box: int,\n    kx: int,\n    rms_min: float,\n    rms_max: float,\n    fast_gpu: bool = False,\n    tile_batch: int = 64,\n    regular_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]] = None,\n) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:\n    print("\\n TILING IMAGE...\\n")\n    if fast_gpu:\n        tiles, rms, centers_all, nx, ny = regular_cache or regular_boximg_batch(image, box)\n        good = (rms <= float(rms_max)) & (rms >= float(rms_min))\n        cnt = int(good.sum().item())\n        if cnt <= 0:\n            raise RuntimeError("No tiles passed the RMS filter during TILE")\n        good_tiles = tiles[good]\n        good_centers = centers_all[good]\n        nw = kx // 10\n        scale = 1.0 / math.sqrt(float(box * box))\n        spectra_out = []\n        power = torch.zeros((box, kx), device=image.device, dtype=image.dtype)\n        for start in range(0, cnt, tile_batch):\n            batch = good_tiles[start : start + tile_batch]\n            spec = torch.fft.rfft2(batch)[:, :, :kx]\n            amp = torch.abs(spec) * scale\n            power = power + torch.sum(amp * amp, dim=0)\n            spectra_out.append(msmooth_fast(amp, nw)[:, :, : box // 2])\n        average_power = torch.sqrt(power) / float(cnt)\n        spec_tensor = torch.cat(spectra_out, dim=0)\n        print(f" Total tiles and number used = {nx * ny:8d}{cnt:8d}")\n        # The caller does not consume the legacy tiled diagnostic buffer.\n        return image, spec_tensor, good_centers, cnt, nx * ny\n\n    h, w = image.shape\n    nx = w // box\n    ny = h // box\n    tiled = image.clone()\n    average_power = torch.zeros((box, kx), device=image.device, dtype=image.dtype)\n    spectra = []\n    centers = []\n    good_indices = []\n    nw = kx // 10\n    scale = 1.0 / math.sqrt(float(box * box))\n    cnt = 0\n\n    for iy in range(ny):\n        for ix in range(nx):\n            ix0 = ix * box\n            iy0 = iy * box\n            tile, _mean, rms = boximg(image, ix0, iy0, box)\n            flag_y = iy0 + box - 1\n            flag_x = ix0 + box - 1\n            if rms <= rms_max and rms >= rms_min:\n                amp = fft_amplitude_half(tile, kx, scale)\n                average_power += amp**2\n                stripped = msmooth(amp, nw)\n                tiled[iy0 : iy0 + box, ix0 : ix0 + kx] = stripped\n                tiled[flag_y, flag_x] = 1.0\n                spectra.append(stripped[:, : box // 2])\n                centers.append((ix0, iy0))\n                good_indices.append(iy * nx + ix)\n                cnt += 1\n            else:\n                tiled[flag_y, flag_x] = 0.0\n\n    if cnt <= 0:\n        raise RuntimeError("No tiles passed the RMS filter during TILE")\n    average_power = torch.sqrt(average_power) / float(cnt)\n    spec_tensor = torch.stack(spectra, dim=0)\n    center_tensor = torch.tensor(centers, device=image.device, dtype=torch.int64)\n    index_tensor = torch.tensor(good_indices, device=image.device, dtype=torch.int64)\n    print(f" Total tiles and number used = {nx * ny:8d}{cnt:8d}")\n    return tiled, spec_tensor, center_tensor, cnt, nx * ny\n\n\ndef tar(sig2: float, ta: float, tilta: float, tiltr: float) -> float:\n    return -sig2 * (abs(ta) - tilta) ** 2 / (2.0 * tiltr * tiltr)\n\n\ndef eval_tilt(\n    evaluator: CtfEvaluator,\n    spectra: torch.Tensor,\n    centers: torch.Tensor,\n    image_shape: Tuple[int, int],\n    box: int,\n    pixel_size: float,\n    df1: float,\n    df2: float,\n    angast: float,\n    tilt_axis: float,\n    tilt_angle: float,\n    dast: float,\n    tile_batch: int,\n) -> Tuple[float, float]:\n    h, w = image_shape\n    cx = w // 2\n    cy = h // 2\n    n1 = math.sin(tilt_axis)\n    n2 = -math.cos(tilt_axis)\n\n    ta = float(tilt_angle)\n    if abs(abs(ta) - PI / 2.0) < 0.0001:\n        ta = PI / 2.0 - 0.0001 if ta > 0.0 else -PI / 2.0 + 0.0001\n\n    ix0 = centers[:, 0].to(evaluator.dtype)\n    iy0 = centers[:, 1].to(evaluator.dtype)\n    # Source: DX=CX-IX+BOX/2 with IX one-based.\n    dx = float(cx) - (ix0 + 1.0) + float(box // 2)\n    dy = float(cy) - (iy0 + 1.0) + float(box // 2)\n    rr = (n1 * dx + n2 * dy) * float(pixel_size)\n    dchange = rr * math.tan(ta)\n    local1 = dchange + float(df1)\n    local2 = dchange + float(df2)\n    cc, sig2 = evaluator.eval_paired(\n        spectra, local1, local2, float(angast), dast, tile_batch\n    )\n    # The shared SIG2 common-block variable is overwritten by each serial tile;\n    # use the last good tile in row-major order for deterministic source semantics.\n    return float(cc.mean().item()), float(sig2[-1].item())\n\n\ndef signed_tilt_candidates(expected_tilt_rad: float, tilt_range_rad: float) -> list[int]:\n    """Return the source coarse-search angles plus their opposite-sign branch.\n\n    The original FIND_TANGLE searches only around +ABS(TILTA).  That is safe\n    only when FIND_TAXIS chooses the same directed representative of the tilt\n    axis as the downstream defocus-gradient convention.  The axis itself is an\n    undirected line, so the physically equivalent solution may instead be\n    represented by (axis, -tilt).  Preserve the original candidate order, then\n    append the mirrored candidates; strict `>` comparison keeps the original\n    positive branch when the two scores are exactly tied.\n    """\n    itiltr = nint(tilt_range_rad * 180.0 / PI / 5.0) * 5\n    itilta = nint(expected_tilt_rad * 180.0 / PI)\n    source_branch = list(range(itilta - itiltr, itilta + itiltr + 1, 10))\n\n    candidates: list[int] = []\n    seen: set[int] = set()\n    for k in source_branch:\n        if k not in seen:\n            candidates.append(k)\n            seen.add(k)\n    for k in source_branch:\n        mirrored = -k\n        if mirrored not in seen:\n            candidates.append(mirrored)\n            seen.add(mirrored)\n    return candidates\n\n\ndef find_tangle(\n    evaluator: CtfEvaluator,\n    spectra: torch.Tensor,\n    centers: torch.Tensor,\n    image_shape: Tuple[int, int],\n    box: int,\n    pixel_size: float,\n    df1: float,\n    df2: float,\n    angast: float,\n    tilt_axis: float,\n    expected_tilt_rad: float,\n    tilt_range_rad: float,\n    dast: float,\n    tile_batch: int,\n) -> Tuple[float, float]:\n    print("\\n SEARCHING FOR TILT ANGLE (BOTH SIGNS)...\\n")\n    candidates = signed_tilt_candidates(expected_tilt_rad, tilt_range_rad)\n    best_sum = -1.0e30\n    best_k = candidates[0]\n    for k in candidates:\n        ta = float(k) / 180.0 * PI\n        cc, sig2 = eval_tilt(\n            evaluator,\n            spectra,\n            centers,\n            image_shape,\n            box,\n            pixel_size,\n            df1,\n            df2,\n            angast,\n            tilt_axis,\n            ta,\n            dast,\n            tile_batch,\n        )\n        value = cc + tar(sig2, ta, expected_tilt_rad, tilt_range_rad)\n        if value > best_sum:\n            best_sum = value\n            best_k = k\n            print(f"  Tilt angle, CC = {float(k):5.1f}{value:12.5f}")\n    return float(best_k) / 180.0 * PI, best_sum\n\n\ndef refine_tilt(\n    evaluator: CtfEvaluator,\n    spectra: torch.Tensor,\n    centers: torch.Tensor,\n    image_shape: Tuple[int, int],\n    box: int,\n    pixel_size: float,\n    df1: float,\n    df2: float,\n    angast: float,\n    tilt_axis: float,\n    tilt_angle: float,\n    expected_tilt_rad: float,\n    tilt_range_rad: float,\n    dast: float,\n    tile_batch: int,\n    quiet_objective: bool,\n) -> Tuple[float, float, float, float, float, float]:\n    print("\\n REFINING TILT PARAMETERS...\\n")\n    print("      DFMID1      DFMID2      ANGAST     TLTAXIS      TANGLE          CC")\n    x0 = np.array([tilt_axis, tilt_angle, df1, df2, angast], dtype=np.float64)\n    if x0[1] == 0.0:\n        x0[1] = 0.01\n    if x0[2] == x0[3]:\n        x0[2] += 1.0\n\n    def objective(x: np.ndarray) -> float:\n        cc, sig2 = eval_tilt(\n            evaluator,\n            spectra,\n            centers,\n            image_shape,\n            box,\n            pixel_size,\n            float(x[2]),\n            float(x[3]),\n            float(x[4]),\n            float(x[0]),\n            float(x[1]),\n            dast,\n            tile_batch,\n        )\n        rf = -cc - tar(sig2, float(x[1]), expected_tilt_rad, tilt_range_rad)\n        axis_print = float(x[0]) - 2.0 * PI * nint(float(x[0]) / (2.0 * PI))\n        angle_print = float(x[1]) - 2.0 * PI * nint(float(x[1]) / (2.0 * PI))\n        if abs(axis_print) > PI / 2.0:\n            axis_print = axis_print - PI * nint(axis_print / PI)\n            angle_print = -angle_print\n        if not quiet_objective:\n            print(\n                f"{float(x[2]):12.2f}{float(x[3]):12.2f}"\n                f"{180.0 * (float(x[4]) / PI - nint(float(x[4]) / PI)):12.2f}"\n                f"{axis_print / PI * 180.0:12.2f}{angle_print / PI * 180.0:12.2f}{-rf:12.5f}"\n            )\n        return rf\n\n    x, rf, _nf = va04a(\n        x0,\n        [0.1, 0.1, 100.0, 100.0, 0.5],\n        objective,\n        escale=100.0,\n        iprint=0,\n        icon=1,\n        maxit=50,\n    )\n    tilt_axis = float(x[0]) - 2.0 * PI * nint(float(x[0]) / (2.0 * PI))\n    tilt_angle = float(x[1]) - 2.0 * PI * nint(float(x[1]) / (2.0 * PI))\n    df1 = float(x[2])\n    df2 = float(x[3])\n    angast = float(x[4]) - PI * nint(float(x[4]) / PI)\n    if abs(tilt_axis) > PI / 2.0:\n        tilt_axis = tilt_axis - PI * nint(tilt_axis / PI)\n        tilt_angle = -tilt_angle\n    tilt_axis = (tilt_axis + PI) % PI\n    print(\n        f"\\n{df1:12.2f}{df2:12.2f}"\n        f"{180.0 * (angast / PI - nint(angast / PI)):12.2f}"\n        f"{tilt_axis / PI * 180.0:12.2f}{tilt_angle / PI * 180.0:12.2f}{-rf:12.5f}  Final Values"\n    )\n    return df1, df2, angast, tilt_axis, tilt_angle, -rf\n\n\ndef make_diagnostic(\n    evaluator: CtfEvaluator,\n    power: torch.Tensor,\n    drms: float,\n    df1: float,\n    df2: float,\n    angast: float,\n) -> torch.Tensor:\n    box = evaluator.box\n    half = box // 2\n    out = torch.zeros((box, box), device=evaluator.device, dtype=evaluator.dtype)\n\n    for m1 in range(1, box + 1):\n        j1 = m1 + box // 2\n        if j1 > box:\n            j1 -= box\n        right = power[m1 - 1, :] / float(drms) / 2.0 + 0.5 if drms != 0.0 else power[m1 - 1, :] * 0.0 + 0.5\n        right = torch.clamp(right, min=-1.0, max=1.0)\n        out[j1 - 1, half:] = right\n\n    ctf = evaluator.ctf_value_grid(df1, df2, angast)\n    mm = torch.arange(box, device=evaluator.device, dtype=torch.int64)\n    mm = torch.where(mm > box // 2, mm - box, mm).to(evaluator.dtype)\n    ll = torch.arange(half, device=evaluator.device, dtype=evaluator.dtype)\n    mg, lg = torch.meshgrid(mm, ll, indexing="ij")\n    res2_full = (lg / float(box)) ** 2 + (mg / float(box)) ** 2\n    fit_mask = (res2_full <= evaluator.rmax2) & (res2_full >= evaluator.rmin2)\n\n    # Vectorize the inner L loop.  The row mapping remains literal to the\n    # Fortran diagnostic-image code, including the skipped jmirror=box+1 row.\n    for m1 in range(1, box + 1):\n        j1 = m1 + box // 2\n        if j1 > box:\n            j1 -= box\n        jmirror = box - j1 + 2\n        if jmirror <= box:\n            vals = torch.flip(ctf[m1 - 1] ** 2, dims=(0,))\n            mask_row = torch.flip(fit_mask[m1 - 1], dims=(0,))\n            dest = out[jmirror - 1, :half]\n            dest[mask_row] = vals[mask_row]\n    return out\n\n\ndef print_figure(\n    df1: float,\n    df2: float,\n    tilt_axis: float,\n    tilt_angle: float,\n    image_shape: Tuple[int, int],\n    pave: int,\n    pixel_size: float,\n) -> None:\n    h, w = image_shape\n    cx = (w * pave) // 2\n    cy = (h * pave) // 2\n    n1 = math.sin(tilt_axis)\n    n2 = -math.cos(tilt_axis)\n    p_original = pixel_size / float(pave)\n\n    print("\\n\\n     EQUATION FOR CALCULATING DEFOCUS DFL1,DFL2 AT LOCATION NX,NY:\\n")\n    print("          DFL1  = DFMID1 +DF")\n    print("          DFL2  = DFMID2 +DF")\n    print("          DF    = (N1*DX+N2*DY)*PSIZE*TAN(TANGLE)")\n    print("          DX    = CX-NX")\n    print("          DY    = CY-NY")\n    print(f"          CX    = CENTER_X = {cx:12d}")\n    print(f"          CY    = CENTER_Y = {cy:12d}")\n    print(f"          PSIZE = PIXEL SIZE [A] = {p_original:12.4f}")\n    print("          N1,N2 = TILT AXIS NORMAL:")\n    print(f"             N1 =  SIN(TLTAXIS) = {n1:12.6f}")\n    print(f"             N2 = -COS(TLTAXIS) = {n2:12.6f}\\n")\n\n    def local(nx1: int, ny1: int) -> Tuple[float, float]:\n        dx = cx - nx1\n        dy = cy - ny1\n        rr = (n1 * dx + n2 * dy) * p_original\n        d = rr * math.tan(tilt_angle)\n        return df1 + d, df2 + d\n\n    coords = [(1, h * pave), (w * pave, h * pave), (1, 1), (w * pave, 1)]\n    vals = [local(*c) for c in coords]\n    print(f"{vals[0][0]:12.2f},{vals[0][1]:12.2f}    <--(DFMID1,DFMID2)-->   {vals[1][0]:12.2f},{vals[1][1]:12.2f}")\n    print(f"{coords[0][0]:12d},{coords[0][1]:12d}    <------(NX,NY)------>   {coords[1][0]:12d},{coords[1][1]:12d}")\n    print("          +----------------------------------------------------------+")\n    print(f"          |               {df1:12.2f},{df2:12.2f}                  |")\n    print(f"          |               {cx:12d},{cy:12d}                  |")\n    print("          +----------------------------------------------------------+")\n    print(f"{coords[2][0]:12d},{coords[2][1]:12d}    <------(NX,NY)------>   {coords[3][0]:12d},{coords[3][1]:12d}")\n    print(f"{vals[2][0]:12.2f},{vals[2][1]:12.2f}    <--(DFMID1,DFMID2)-->   {vals[3][0]:12.2f},{vals[3][1]:12.2f}")\n\n\ndef validate_config(cfg: CtfTiltConfig) -> None:\n    if cfg.box <= 0 or cfg.box % 2 != 0:\n        raise ValueError("Box size must be a positive even number")\n    if cfg.pixel_average <= 0:\n        raise ValueError("pixel_average must be >= 1")\n    if cfg.df_step_a <= 0.0:\n        raise ValueError("df_step_a must be positive")\n    if not (0.0 <= cfg.amp_contrast <= 1.0):\n        raise ValueError("amp_contrast must lie in [0,1]")\n    if cfg.dast_a < 0.0:\n        cfg.dast_a = 500.0\n        print("\\n Invalid dAst value; reset to 500.0")\n    if cfg.res_min_a < cfg.res_max_a:\n        cfg.res_min_a, cfg.res_max_a = cfg.res_max_a, cfg.res_min_a\n    if cfg.df_max_a < cfg.df_min_a:\n        cfg.df_min_a, cfg.df_max_a = cfg.df_max_a, cfg.df_min_a\n\n\n@torch.inference_mode()\ndef run_ctftilt(cfg: CtfTiltConfig) -> CtfTiltResult:\n    validate_config(cfg)\n    device = resolve_device(cfg.device)\n    dtype = resolve_dtype(cfg.dtype)\n    if cfg.deterministic:\n        torch.use_deterministic_algorithms(True)\n\n    print("\\n CTF TILT DETERMINATION, PyTorch port of V1.7 (9-Jun-2014)")\n    print("  Based on the supplied ctftilt Fortran source.\\n")\n    print(f" Device: {device}; dtype: {cfg.dtype}")\n    print(f" Fast batched path: {\'ENABLED\' if cfg.fast_gpu else \'disabled\'}")\n    if cfg.fix_source_quirks:\n        print(" Source-quirk corrections: ENABLED")\n    else:\n        print(" Source-quirk corrections: disabled (literal source behaviour)")\n\n    image_np = read_mrc_2d(cfg.input_mrc)\n    image = torch.from_numpy(image_np).to(device=device, dtype=dtype)\n    print(f"\\n READING IMAGE...\\n NX, NY= {image.shape[1]:10d}{image.shape[0]:10d}")\n    if cfg.pixel_average > 1:\n        print(f"\\n PIXEL AVERAGING = {cfg.pixel_average:d} x {cfg.pixel_average:d} ...")\n    image = pixel_average_image(image, cfg.pixel_average)\n    h, w = image.shape\n    if w < cfg.box or h < cfg.box:\n        raise ValueError("Image is smaller than one tile after pixel averaging")\n\n    # Original pixel size before and after averaging.\n    pixel_size = cfg.detector_step_um * 1.0e4 / cfg.magnification\n    pixel_size *= cfg.pixel_average\n\n    nx = w // cfg.box\n    ny = h // cfg.box\n    regular_cache = None\n    if cfg.fast_gpu:\n        regular_cache = regular_boximg_batch(image, cfg.box)\n        rms_values = regular_cache[1].detach().cpu().numpy()\n    else:\n        rms_values = []\n        for iy in range(ny):\n            for ix in range(nx):\n                _tile, _mean, rms = boximg(image, ix * cfg.box, iy * cfg.box, cfg.box)\n                rms_values.append(rms)\n    _bins, _vmin, _vmax, rms_min, rms_max = histogram_thresholds(rms_values, 100)\n\n    cs_a = cfg.cs_mm * 1.0e7\n    voltage_v = cfg.voltage_kv * 1000.0\n    wavelength = 12.26 / math.sqrt(voltage_v + 0.9785 * voltage_v * voltage_v / 1.0e6)\n    wgh1 = math.sqrt(1.0 - cfg.amp_contrast * cfg.amp_contrast)\n    wgh2 = cfg.amp_contrast\n    theta_tr = wavelength / (pixel_size * cfg.box)\n\n    res_min_pixel = pixel_size / cfg.res_min_a\n    res_max_pixel = pixel_size / cfg.res_max_a\n    if res_min_pixel < pixel_size / 50.0:\n        res_min_pixel = pixel_size / 50.0\n        print(f"\\n Lower resolution limit reset to {pixel_size / res_min_pixel:.2f} A")\n    if res_min_pixel >= res_max_pixel:\n        raise ValueError("RESMIN >= RESMAX; increase the high-resolution limit")\n    rmin2 = res_min_pixel**2\n    rmax2 = res_max_pixel**2\n\n    kx = cfg.box // 2\n    if kx % 2 != 0:\n        kx += 1\n\n    tilt_axis, axis_power = find_taxis(\n        image, cfg.box, kx, rms_min, rms_max, cfg.nr, cfg.fix_source_quirks,\n        fast_gpu=cfg.fast_gpu\n    )\n\n    if abs(cfg.expected_tilt_deg) <= 20.0:\n        print("\\n CALCULATING AVERAGE POWER SPECTRUM\\n    OF ENTIRE IMAGE FOR INITIAL CTF FIT...\\n")\n        initial_power, _initial_cnt = average_power_entire_image(\n            image, cfg.box, kx, rms_min, rms_max,\n            fast_gpu=cfg.fast_gpu,\n            regular_cache=regular_cache,\n        )\n    else:\n        print("\\n USING AVERAGE POWER SPECTRUM FROM\\n    IMAGE CENTER FOR INITIAL CTF FIT...\\n")\n        initial_power = axis_power\n\n    filtered_power, _dmean, drms, _dmax = filter_power(\n        initial_power, cfg.box, res_min_pixel, fast_gpu=cfg.fast_gpu\n    )\n\n    evaluator = CtfEvaluator(\n        cfg.box,\n        cs_a,\n        wavelength,\n        wgh1,\n        wgh2,\n        theta_tr,\n        rmin2,\n        rmax2,\n        0.0,\n        device,\n        dtype,\n    )\n\n    df1, df2, angast, _grid_cc = search_ctf(\n        evaluator,\n        filtered_power,\n        cfg.df_min_a,\n        cfg.df_max_a,\n        cfg.df_step_a,\n        cfg.dast_a,\n        cfg.candidate_batch,\n    )\n    df1, df2, angast, _center_cc = refine_ctf(\n        evaluator, filtered_power, df1, df2, angast, cfg.dast_a\n    )\n    _cc_center, sig2_center = evaluator.eval_single(\n        filtered_power, df1, df2, angast, cfg.dast_a\n    )\n\n    diagnostic = make_diagnostic(evaluator, filtered_power, drms, df1, df2, angast)\n    write_mrc_2d(cfg.output_mrc, diagnostic, 1.0 / pixel_size if pixel_size != 0.0 else 0.0)\n\n    _tiled, tile_spectra, tile_centers, tiles_used, tiles_total = tile_image(\n        image, cfg.box, kx, rms_min, rms_max,\n        fast_gpu=cfg.fast_gpu,\n        tile_batch=cfg.tile_batch,\n        regular_cache=regular_cache,\n    )\n\n    tilt_range_deg = cfg.tilt_uncertainty_deg\n    if tilt_range_deg < 2.5:\n        tilt_range_deg = 2.4999\n    expected_tilt_rad = abs(cfg.expected_tilt_deg / 180.0 * PI)\n    tilt_range_rad = tilt_range_deg / 180.0 * PI\n\n    tilt_angle, _angle_cc = find_tangle(\n        evaluator,\n        tile_spectra,\n        tile_centers,\n        (h, w),\n        cfg.box,\n        pixel_size,\n        df1,\n        df2,\n        angast,\n        tilt_axis,\n        expected_tilt_rad,\n        tilt_range_rad,\n        cfg.dast_a,\n        cfg.tile_batch,\n    )\n\n    df1, df2, angast, tilt_axis, tilt_angle, final_cc = refine_tilt(\n        evaluator,\n        tile_spectra,\n        tile_centers,\n        (h, w),\n        cfg.box,\n        pixel_size,\n        df1,\n        df2,\n        angast,\n        tilt_axis,\n        tilt_angle,\n        expected_tilt_rad,\n        tilt_range_rad,\n        cfg.dast_a,\n        cfg.tile_batch,\n        cfg.quiet_objective,\n    )\n\n    print_figure(df1, df2, tilt_axis, tilt_angle, (h, w), cfg.pixel_average, pixel_size)\n\n    result = CtfTiltResult(\n        defocus1_a=df1,\n        defocus2_a=df2,\n        astig_angle_deg=180.0 * (angast / PI - nint(angast / PI)),\n        tilt_axis_deg=tilt_axis / PI * 180.0,\n        tilt_angle_deg=tilt_angle / PI * 180.0,\n        final_cc=final_cc,\n        pixel_size_a=pixel_size,\n        tiles_total=tiles_total,\n        tiles_used=tiles_used,\n        rms_min=rms_min,\n        rms_max=rms_max,\n    )\n    if cfg.result_json:\n        out_json = Path(cfg.result_json)\n        out_json.parent.mkdir(parents=True, exist_ok=True)\n        with out_json.open("w", encoding="utf-8") as f:\n            json.dump({"config": asdict(cfg), "result": asdict(result)}, f, indent=2)\n    return result\n\n\ndef prompt_legacy() -> CtfTiltConfig:\n    print(" Input image file name")\n    input_mrc = input().strip()\n    print(input_mrc)\n    print("\\n Output diagnostic file name")\n    output_mrc = input().strip()\n    print(output_mrc)\n    print("\\n CS[mm], HT[kV], AmpCnst, XMAG, DStep[um], PAve")\n    card3 = input().replace(",", " ").split()\n    if len(card3) != 6:\n        raise ValueError("CARD 3 requires 6 values")\n    cs, kv, amp, xmag, dstep = map(float, card3[:5])\n    pave = int(card3[5])\n    print("\\n Positive defocus values for underfocus")\n    print(" Box, ResMin[A], ResMax[A], dFMin[A], dFMax[A], FStep[A], dAst[A], TiltA[deg], TiltR[deg]")\n    card4 = input().replace(",", " ").split()\n    if len(card4) != 9:\n        raise ValueError("CARD 4 requires 9 values")\n    box = int(card4[0])\n    vals = list(map(float, card4[1:]))\n    return CtfTiltConfig(\n        input_mrc=input_mrc,\n        output_mrc=output_mrc,\n        cs_mm=cs,\n        voltage_kv=kv,\n        amp_contrast=amp,\n        magnification=xmag,\n        detector_step_um=dstep,\n        pixel_average=pave,\n        box=box,\n        res_min_a=vals[0],\n        res_max_a=vals[1],\n        df_min_a=vals[2],\n        df_max_a=vals[3],\n        df_step_a=vals[4],\n        dast_a=vals[5],\n        expected_tilt_deg=vals[6],\n        tilt_uncertainty_deg=vals[7],\n    )\n\n\ndef build_parser() -> argparse.ArgumentParser:\n    p = argparse.ArgumentParser(\n        description="PyTorch port of ctftilt v1.7. Run with no arguments for legacy CARD input."\n    )\n    p.add_argument("input_mrc", nargs="?")\n    p.add_argument("output_mrc", nargs="?")\n    p.add_argument("--cs-mm", type=float)\n    p.add_argument("--voltage-kv", type=float)\n    p.add_argument("--amp-contrast", type=float)\n    p.add_argument("--magnification", type=float)\n    p.add_argument("--detector-step-um", type=float)\n    p.add_argument("--pixel-average", type=int, default=1)\n    p.add_argument("--box", type=int)\n    p.add_argument("--res-min-a", type=float)\n    p.add_argument("--res-max-a", type=float)\n    p.add_argument("--df-min-a", type=float)\n    p.add_argument("--df-max-a", type=float)\n    p.add_argument("--df-step-a", type=float)\n    p.add_argument("--dast-a", type=float, default=500.0)\n    p.add_argument("--expected-tilt-deg", type=float, default=0.0)\n    p.add_argument("--tilt-uncertainty-deg", type=float, default=5.0)\n    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")\n    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")\n    p.add_argument("--candidate-batch", type=int, default=256)\n    p.add_argument("--tile-batch", type=int, default=64)\n    p.add_argument("--nr", type=int, default=5)\n    p.add_argument("--fix-source-quirks", action="store_true")\n    p.add_argument("--deterministic", action="store_true")\n    p.add_argument("--quiet-objective", action="store_true")\n    p.add_argument("--result-json")\n    p.add_argument(\n        "--fast-gpu", action="store_true",\n        help="Use batched tile extraction and integral-image MSMOOTH (primarily for CUDA); may change last float32 bits",\n    )\n    return p\n\n\ndef config_from_args(args: argparse.Namespace) -> CtfTiltConfig:\n    required = {\n        "input_mrc": args.input_mrc,\n        "output_mrc": args.output_mrc,\n        "cs_mm": args.cs_mm,\n        "voltage_kv": args.voltage_kv,\n        "amp_contrast": args.amp_contrast,\n        "magnification": args.magnification,\n        "detector_step_um": args.detector_step_um,\n        "box": args.box,\n        "res_min_a": args.res_min_a,\n        "res_max_a": args.res_max_a,\n        "df_min_a": args.df_min_a,\n        "df_max_a": args.df_max_a,\n        "df_step_a": args.df_step_a,\n    }\n    missing = [name for name, value in required.items() if value is None]\n    if missing:\n        raise ValueError("Missing required command-line values: " + ", ".join(missing))\n    return CtfTiltConfig(\n        input_mrc=args.input_mrc,\n        output_mrc=args.output_mrc,\n        cs_mm=args.cs_mm,\n        voltage_kv=args.voltage_kv,\n        amp_contrast=args.amp_contrast,\n        magnification=args.magnification,\n        detector_step_um=args.detector_step_um,\n        pixel_average=args.pixel_average,\n        box=args.box,\n        res_min_a=args.res_min_a,\n        res_max_a=args.res_max_a,\n        df_min_a=args.df_min_a,\n        df_max_a=args.df_max_a,\n        df_step_a=args.df_step_a,\n        dast_a=args.dast_a,\n        expected_tilt_deg=args.expected_tilt_deg,\n        tilt_uncertainty_deg=args.tilt_uncertainty_deg,\n        device=args.device,\n        dtype=args.dtype,\n        candidate_batch=args.candidate_batch,\n        tile_batch=args.tile_batch,\n        nr=args.nr,\n        fix_source_quirks=args.fix_source_quirks,\n        deterministic=args.deterministic,\n        quiet_objective=args.quiet_objective,\n        result_json=args.result_json,\n        fast_gpu=args.fast_gpu,\n    )\n\n\ndef main(argv: Optional[Sequence[str]] = None) -> int:\n    argv = list(sys.argv[1:] if argv is None else argv)\n    try:\n        if len(argv) == 0:\n            cfg = prompt_legacy()\n        else:\n            parser = build_parser()\n            cfg = config_from_args(parser.parse_args(argv))\n        result = run_ctftilt(cfg)\n        print("\\n Result summary:")\n        print(json.dumps(asdict(result), indent=2))\n        return 0\n    except KeyboardInterrupt:\n        print("Interrupted", file=sys.stderr)\n        return 130\n    except Exception as exc:\n        print(f"ERROR: {exc}", file=sys.stderr)\n        return 1\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
_EMBEDDED_SIGNED_CTFTILT_NAMESPACE: Optional[dict[str, object]] = None


def _get_embedded_signed_ctftilt_namespace() -> dict[str, object]:
    global _EMBEDDED_SIGNED_CTFTILT_NAMESPACE
    if _EMBEDDED_SIGNED_CTFTILT_NAMESPACE is None:
        import types

        module_name = "_embedded_ctftilt_signed_backend"
        module = types.ModuleType(module_name)
        module.__file__ = "<embedded_ctftilt_torch_gpu_optimized_signed_tilt.py>"
        # dataclasses inspects sys.modules[cls.__module__] while classes are
        # created, so the private module must be registered before exec().
        sys.modules[module_name] = module
        namespace = module.__dict__
        exec(_EMBEDDED_SIGNED_CTFTILT_SOURCE, namespace)
        _EMBEDDED_SIGNED_CTFTILT_NAMESPACE = namespace
    return _EMBEDDED_SIGNED_CTFTILT_NAMESPACE

class TorchCtffindPowell:
    """Batched CTFFIND estimator with a PyTorch Powell optimizer."""

    def __init__(self, config: CtffindConfig):
        config.validate()
        canonical_phase = math.fmod(config.fixed_phase_shift_rad, PI)
        self.config = replace(config, fixed_phase_shift_rad=canonical_phase)
        self.device = _resolve_device(config.device)
        self.dtype = torch.float32
        self.optimizer_dtype = torch.float64
        self.wavelength_A = _electron_wavelength_A(config.acceleration_voltage_kV)
        self.spherical_aberration_A = config.spherical_aberration_mm * 1.0e7
        self.amplitude_phase_rad = _amplitude_contrast_phase(config.amplitude_contrast)
        if config.find_phase_shift:
            warnings.warn(
                "Phase-shift search is not implemented yet; using the fixed "
                f"phase shift {canonical_phase:.6g} rad.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _score_1d_candidates(
        self,
        curve: _OneDimensionalCurve,
        candidates_A: torch.Tensor,
        *,
        ctf_squared: bool = False,
    ) -> torch.Tensor:
        low = 1.0 / self.config.minimum_resolution_A
        high = 1.0 / self.config.maximum_resolution_A
        mask = (curve.frequencies_Ainv > low) & (curve.frequencies_Ainv < high)
        experimental = curve.values[:, mask]
        frequencies = curve.frequencies_Ainv[mask]
        theoretical = _ctf_abs_1d(
            frequencies,
            candidates_A,
            self.wavelength_A,
            self.spherical_aberration_A,
            self.amplitude_phase_rad,
            self.config.fixed_phase_shift_rad,
        )
        if ctf_squared:
            theoretical = theoretical.square()
        numerator = experimental @ theoretical.transpose(0, 1)
        norm_curve = torch.sum(experimental.square(), dim=1)
        norm_ctf = torch.sum(theoretical.square(), dim=1)
        return numerator / torch.sqrt(
            (norm_curve[:, None] * norm_ctf[None, :]).clamp_min(1.0e-30)
        )

    def _score_1d_per_image(
        self,
        curve: _OneDimensionalCurve,
        defocus_A: torch.Tensor,
        *,
        ctf_squared: bool = False,
    ) -> torch.Tensor:
        low = 1.0 / self.config.minimum_resolution_A
        high = 1.0 / self.config.maximum_resolution_A
        mask = (curve.frequencies_Ainv > low) & (curve.frequencies_Ainv < high)
        experimental = curve.values[:, mask]
        frequencies = curve.frequencies_Ainv[mask]
        theoretical = _ctf_abs_1d(
            frequencies,
            defocus_A.to(self.dtype),
            self.wavelength_A,
            self.spherical_aberration_A,
            self.amplitude_phase_rad,
            self.config.fixed_phase_shift_rad,
        )
        if ctf_squared:
            theoretical = theoretical.square()
        numerator = torch.sum(experimental * theoretical, dim=1)
        norm_curve = torch.sum(experimental.square(), dim=1)
        norm_ctf = torch.sum(theoretical.square(), dim=1)
        return numerator / torch.sqrt(
            (norm_curve * norm_ctf).clamp_min(1.0e-30)
        )

    def _coarse_and_refine_mean_defocus_batch(
        self,
        curve: _OneDimensionalCurve,
        fitting_pixel_size_A: float,
    ) -> tuple[torch.Tensor, torch.Tensor, _BatchedOptimizationResult]:
        cfg = self.config
        candidates_np = np.arange(
            cfg.minimum_defocus_A,
            cfg.maximum_defocus_A + 0.5 * cfg.defocus_search_step_A,
            cfg.defocus_search_step_A,
            dtype=np.float32,
        )
        candidates = torch.as_tensor(
            candidates_np, dtype=self.dtype, device=self.device
        )
        scores = self._score_1d_candidates(curve, candidates)
        coarse_idx = torch.argmax(scores, dim=1)
        coarse = candidates[coarse_idx]
        scale = max(100.0 * fitting_pixel_size_A, 1.0)
        coarse_opt = coarse.to(self.optimizer_dtype)
        lower = (cfg.minimum_defocus_A - coarse_opt) / scale
        upper = (cfg.maximum_defocus_A - coarse_opt) / scale
        if not cfg.use_powell_defocus_bounds:
            span = (cfg.maximum_defocus_A - cfg.minimum_defocus_A) / scale
            lower = torch.full_like(lower, -span)
            upper = torch.full_like(upper, span)

        def objective(u: torch.Tensor) -> torch.Tensor:
            defocus = coarse_opt + u * scale
            return -self._score_1d_per_image(curve, defocus).to(self.optimizer_dtype)

        f0 = objective(torch.zeros_like(lower))
    #    scalar = _batched_minimize_scalar_bounded(
    #        objective,
    #        lower,
    #        upper,
    #        xatol=cfg.powell_xtol,
    #        maxiter=cfg.powell_maxiter_1d,
    #        f_at_zero=f0,
    #        check_interval=cfg.optimizer_check_interval,
    #    )
        scalar = _batched_minimize_scalar_local_bracket(
            objective,
            lower,
            upper,
            xatol=cfg.powell_xtol,
            maxiter=cfg.powell_maxiter_1d,
            f_at_zero=f0,
            check_interval=cfg.optimizer_check_interval,
            initial_step=1.0,
        )
        refined = coarse_opt + scalar.x * scale
        result = _BatchedOptimizationResult(
            x=scalar.x[:, None],
            fun=scalar.fun,
            success=scalar.success,
            nfev=scalar.nfev,
            nit=scalar.nit,
            messages=scalar.messages,
        )
        return coarse, refined, result

    def _score_2d_batch(
        self,
        fit_data: _SpectrumFitData,
        defocus1_A: torch.Tensor,
        defocus2_A: torch.Tensor,
        angle_rad: torch.Tensor,
    ) -> torch.Tensor:
        # Preserve the scalar-operation order of the single-image CTFFIND
        # implementation.  The sums/differences are formed in optimizer
        # precision, then rounded once to float32, just as Python scalar
        # values are converted when combined with a float32 Fourier grid.
        defocus_sum = (defocus1_A + defocus2_A).to(self.dtype)[:, None]
        defocus_difference = (defocus1_A - defocus2_A).to(self.dtype)[:, None]
        angle = angle_rad.to(self.dtype)[:, None]
        effective_defocus = 0.5 * (
            defocus_sum
            + torch.cos(2.0 * (fit_data.azimuth_rad[None] - angle))
            * defocus_difference
        )
        frequency_squared = fit_data.frequency_squared_Ainv2[None]
        phase = (
            PI
            * self.wavelength_A
            * frequency_squared
            * (
                effective_defocus
                - 0.5
                * self.wavelength_A
                * self.wavelength_A
                * frequency_squared
                * self.spherical_aberration_A
            )
            + self.config.fixed_phase_shift_rad
            + self.amplitude_phase_rad
        )
        theoretical = torch.sin(phase).abs()
        cross = torch.sum(fit_data.spectrum_values * theoretical, dim=1)
        norm_ctf = torch.sqrt(torch.sum(theoretical.square(), dim=1))
        score = cross / (fit_data.image_norm * norm_ctf).clamp_min(1.0e-30)
        tolerance = self.config.astigmatism_tolerance_A
        if tolerance > 0.0:
            penalty = (
                0.5 * (defocus1_A - defocus2_A).square()
                / (tolerance * tolerance)
                / float(fit_data.number_of_values)
            ).to(self.dtype)
            score = score - penalty
        return score

    def _refine_2d_batch(
        self,
        fit_data: _SpectrumFitData,
        starting_mean_A: torch.Tensor,
        starting_angle_deg: torch.Tensor,
        fitting_pixel_size_A: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, _BatchedOptimizationResult]:
        cfg = self.config
        batch = starting_mean_A.numel()
        defocus_scale = max(100.0 * fitting_pixel_size_A, 1.0)
        angle_scale = 0.5
        mean_opt = starting_mean_A.to(self.optimizer_dtype)
        angle0 = starting_angle_deg.to(self.optimizer_dtype) * PI / 180.0
        x0 = torch.zeros(
            (batch, 3), dtype=self.optimizer_dtype, device=self.device
        )
        lower = torch.empty_like(x0)
        upper = torch.empty_like(x0)
        lower[:, 0] = (cfg.minimum_defocus_A - mean_opt) / defocus_scale
        lower[:, 1] = lower[:, 0]
        upper[:, 0] = (cfg.maximum_defocus_A - mean_opt) / defocus_scale
        upper[:, 1] = upper[:, 0]
        # Match SciPy/CTFFIND: the astigmatism angle is unbounded during
        # Powell line searches and canonicalized only after optimization.
        lower[:, 2] = -torch.inf
        upper[:, 2] = torch.inf
        if not cfg.use_powell_defocus_bounds:
            span = (cfg.maximum_defocus_A - cfg.minimum_defocus_A) / defocus_scale
            lower[:, :2] = -span
            upper[:, :2] = span

        def decode(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            d1 = mean_opt + x[:, 0] * defocus_scale
            d2 = mean_opt + x[:, 1] * defocus_scale
            angle = angle0 + x[:, 2] * angle_scale
            return d1, d2, angle

        def objective(x: torch.Tensor) -> torch.Tensor:
            d1, d2, angle = decode(x)
            return -self._score_2d_batch(fit_data, d1, d2, angle).to(
                self.optimizer_dtype
            )

        opt = _batched_powell(
            objective,
            x0,
            lower,
            upper,
            xtol=cfg.powell_xtol,
            ftol=cfg.powell_ftol,
            maxiter=cfg.powell_maxiter_2d,
            line_maxiter=cfg.powell_line_maxiter,
            check_interval=cfg.optimizer_check_interval,
        )
        d1, d2, angle = decode(opt.x)
        swap = d1 < d2
        old_d1 = d1.clone()
        d1 = torch.where(swap, d2, d1)
        d2 = torch.where(swap, old_d1, d2)
        angle = torch.where(swap, angle + 0.5 * PI, angle)
        angle = torch.remainder(angle + 0.5 * PI, PI) - 0.5 * PI
        score = self._score_2d_batch(fit_data, d1, d2, angle)
        return d1, d2, angle, score, opt

    def preprocess_batch(
        self,
        micrographs: Sequence[np.ndarray],
        *,
        pixel_size_A: float,
    ) -> tuple[torch.Tensor, float]:
        """Convert a small raw-micrograph batch to filtered GPU spectra."""
        if not micrographs:
            return torch.empty(
                (0, self.config.box_size, self.config.box_size),
                dtype=self.dtype,
                device=self.device,
            ), float(pixel_size_A)
        shape = np.asarray(micrographs[0]).shape
        arrays: list[np.ndarray] = []
        for idx, image in enumerate(micrographs):
            array = np.array(image, dtype=np.float32, copy=True, order="C")
            if array.ndim != 2 or array.shape != shape:
                raise ValueError("All micrographs in a preprocessing batch must share one shape")
            if not np.isfinite(array).all():
                raise ValueError(f"Micrograph {idx} contains NaN or infinity")
            if float(array.max()) == float(array.min()):
                raise ValueError(f"Micrograph {idx} is constant")
            arrays.append(array)
        images = torch.as_tensor(
            np.stack(arrays, axis=0), dtype=self.dtype, device=self.device
        )
        with torch.inference_mode():
            return _ctffind_preprocess_batch(images, pixel_size_A, self.config)

    def fit_spectra_batch(
        self,
        spectra: torch.Tensor,
        *,
        source_files: Sequence[str],
        micrograph_names: Sequence[str],
        ctf_image_names: Sequence[str],
        image_indices_1based: Sequence[int],
        pixel_size_input_A: float,
        fitting_pixel_size_A: float,
        return_filtered_spectra: bool = False,
        return_diagnostic_maps: bool = True,
    ) -> tuple[list[CtfFitResult], Optional[np.ndarray], Optional[np.ndarray]]:
        """Fit a large batch of already-filtered spectra on the GPU."""
        if spectra.ndim != 3:
            raise ValueError("spectra must have shape [B,H,W]")
        batch = int(spectra.shape[0])
        metadata_lengths = {
            len(source_files), len(micrograph_names), len(ctf_image_names),
            len(image_indices_1based),
        }
        if metadata_lengths != {batch}:
            raise ValueError("Spectrum batch and metadata lengths differ")
        spectra = spectra.to(device=self.device, dtype=self.dtype)

        with torch.inference_mode():
            initial_angles = _estimate_astigmatism_angle_deg_batch(
                spectra, fitting_pixel_size_A, self.config
            )
            curves = _rotational_average_linear_batch(
                spectra, fitting_pixel_size_A
            )
            coarse, refined_mean, opt1 = self._coarse_and_refine_mean_defocus_batch(
                curves, fitting_pixel_size_A
            )
            fit_data = _make_2d_fit_data_batch(
                spectra, fitting_pixel_size_A, self.config
            )
            d1, d2, angle, score, opt2 = self._refine_2d_batch(
                fit_data, refined_mean, initial_angles, fitting_pixel_size_A
            )

        # One synchronization and bulk transfer replaces dozens of per-value
        # .item() calls after optimization.
        d1_cpu = d1.detach().cpu().numpy().astype(np.float64, copy=False)
        d2_cpu = d2.detach().cpu().numpy().astype(np.float64, copy=False)
        angle_cpu = angle.detach().cpu().numpy().astype(np.float64, copy=False)
        score_cpu = score.detach().cpu().numpy().astype(np.float64, copy=False)
        coarse_cpu = coarse.detach().cpu().numpy().astype(np.float64, copy=False)
        refined_cpu = refined_mean.detach().cpu().numpy().astype(np.float64, copy=False)
        initial_angle_cpu = initial_angles.detach().cpu().numpy().astype(np.float64, copy=False)
        success1_cpu = opt1.success.detach().cpu().numpy().astype(bool, copy=False)
        success2_cpu = opt2.success.detach().cpu().numpy().astype(bool, copy=False)
        nfev1_cpu = opt1.nfev.detach().cpu().numpy().astype(np.int64, copy=False)
        nfev2_cpu = opt2.nfev.detach().cpu().numpy().astype(np.int64, copy=False)

        statistics: list[_GoodFitStatistics] = []
        diagnostic_tensors: list[torch.Tensor] = []
        with torch.inference_mode():
            for i in range(batch):
                stats = _compute_good_fit_statistics(
                    spectra[i],
                    fitting_pixel_size_A,
                    self.config,
                    float(d1_cpu[i]),
                    float(d2_cpu[i]),
                    float(angle_cpu[i]),
                    self.wavelength_A,
                    self.spherical_aberration_A,
                    self.amplitude_phase_rad,
                    self.config.fixed_phase_shift_rad,
                    keep_diagnostic_support=return_diagnostic_maps,
                )
                statistics.append(stats)
                if return_diagnostic_maps:
                    diagnostic_tensors.append(_render_diagnostic_map(
                        stats,
                        fitting_pixel_size_A,
                        self.config,
                        float(d1_cpu[i]),
                        float(d2_cpu[i]),
                        float(angle_cpu[i]),
                        self.wavelength_A,
                        self.spherical_aberration_A,
                        self.amplitude_phase_rad,
                        self.config.fixed_phase_shift_rad,
                    ))

        results: list[CtfFitResult] = []
        for i in range(batch):
            results.append(CtfFitResult(
                source_file=source_files[i],
                micrograph_name=micrograph_names[i],
                ctf_image_name=ctf_image_names[i],
                image_index_1based=int(image_indices_1based[i]),
                pixel_size_input_A=float(pixel_size_input_A),
                pixel_size_for_fitting_A=float(fitting_pixel_size_A),
                defocus1_A=float(d1_cpu[i]),
                defocus2_A=float(d2_cpu[i]),
                astigmatism_angle_deg=float(angle_cpu[i] * 180.0 / PI),
                phase_shift_rad=float(self.config.fixed_phase_shift_rad),
                score=float(score_cpu[i]),
                thon_rings_good_fit_resolution_A=(
                    statistics[i].thon_rings_good_fit_resolution_A
                ),
                ctf_aliasing_resolution_A=(
                    statistics[i].ctf_aliasing_resolution_A
                ),
                coarse_defocus_A=float(coarse_cpu[i]),
                refined_mean_defocus_A=float(refined_cpu[i]),
                initial_astigmatism_angle_deg=float(initial_angle_cpu[i]),
                powell_1d_success=bool(success1_cpu[i]),
                powell_2d_success=bool(success2_cpu[i]),
                powell_1d_nfev=int(nfev1_cpu[i]),
                powell_2d_nfev=int(nfev2_cpu[i]),
                powell_1d_message=opt1.messages[i],
                powell_2d_message=opt2.messages[i],
            ))

        filtered = None
        if return_filtered_spectra:
            filtered = spectra.detach().cpu().numpy().astype(np.float32, copy=False)
        diagnostic_maps = None
        if return_diagnostic_maps:
            diagnostic_maps = torch.stack(diagnostic_tensors).detach().cpu().numpy().astype(
                np.float32, copy=False
            )
        return results, filtered, diagnostic_maps

    def fit_tilt_micrograph(
        self,
        micrograph: np.ndarray,
        global_result: CtfFitResult,
        nominal_tilt_angle_deg: Optional[float] = None,
    ) -> _TiltFitDetails:
        """Fit tilted defocus parameters using the embedded signed CTFTILT backend.

        This replaces the previous local-plane ``--fit-tilt`` implementation.
        The normal CTFFIND preprocessing/search path is not touched: this method
        is only called after a global CTFFIND result has already been produced
        and only when ``--fit-tilt`` is enabled.
        """
        cfg = self.config
        nominal = (
            cfg.tilt_nominal_angle_deg
            if nominal_tilt_angle_deg is None
            else float(nominal_tilt_angle_deg)
        )
        if nominal is None:
            raise ValueError(
                "Tilt fitting requires --tilt-angle or a matching entry in "
                "--tilt-angle-file"
            )
        if abs(nominal) >= 89.9:
            raise ValueError("Nominal tilt angle magnitude must be below 89.9 degrees")

        image_array = np.asarray(micrograph, dtype=np.float32)
        if image_array.ndim != 2:
            raise ValueError("CTFTILT fitting expects one 2-D micrograph")
        height, width = image_array.shape

        def empty_details(message: str) -> _TiltFitDetails:
            empty_f = np.empty(0, dtype=np.float64)
            empty_b = np.empty(0, dtype=bool)
            empty_i = np.empty(0, dtype=np.int64)
            return _TiltFitDetails(
                success=False,
                message=message,
                center_defocus1_A=global_result.defocus1_A,
                center_defocus2_A=global_result.defocus2_A,
                astigmatism_angle_rad=global_result.astigmatism_angle_deg * PI / 180.0,
                gradient_x=float("nan"),
                gradient_y=float("nan"),
                tilt_angle_deg=float(nominal),
                tilt_axis_deg=float("nan"),
                nominal_tilt_angle_deg=float(nominal),
                coarse_tilt_angle_deg=float("nan"),
                coarse_tilt_axis_deg=float("nan"),
                score=float("nan"),
                good_fit_resolution_A=0.0,
                residual_rms_A=float("nan"),
                tile_centers_x_A=empty_f.copy(),
                tile_centers_y_A=empty_f.copy(),
                tile_measured_defocus_A=empty_f.copy(),
                tile_predicted_defocus_A=empty_f.copy(),
                tile_residual_A=empty_f.copy(),
                tile_cc=empty_f.copy(),
                tile_good_fit_resolution_A=empty_f.copy(),
                tile_rms_valid=empty_b.copy(),
                tile_plane_inlier=empty_b.copy(),
                tile_grid_y=empty_i.copy(),
                tile_grid_x=empty_i.copy(),
                image_shape=(height, width),
            )

        if global_result.score < cfg.tilt_min_global_cc:
            return empty_details(
                f"Skipped: global CC {global_result.score:.4f} < "
                f"{cfg.tilt_min_global_cc:.4f}."
            )
        if height < cfg.tilt_tile_size or width < cfg.tilt_tile_size:
            return empty_details(
                f"Skipped: micrograph {width}x{height} is smaller than "
                f"tilt tile size {cfg.tilt_tile_size}."
            )

        import contextlib
        import io
        import tempfile

        ctftilt_ns = _get_embedded_signed_ctftilt_namespace()
        CtfTiltConfig = ctftilt_ns["CtfTiltConfig"]
        run_ctftilt = ctftilt_ns["run_ctftilt"]

        # CTFTILT derives pixel size as detector_step_um*1e4/magnification.
        # Use a synthetic but exactly equivalent pair so the embedded backend
        # sees the same Angstrom/pixel as CTFFIND used for the input image.
        synthetic_magnification = 10_000.0
        synthetic_detector_step_um = float(global_result.pixel_size_input_A)
        dast = float(cfg.astigmatism_tolerance_A if cfg.astigmatism_tolerance_A > 0.0 else 500.0)

        with tempfile.TemporaryDirectory(prefix="ctffind_ctftilt_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            input_mrc = tmp_dir_path / "input_micrograph.mrc"
            output_mrc = tmp_dir_path / "ctftilt_diagnostic.mrc"
            with mrcfile.new(input_mrc, overwrite=True) as mrc:
                mrc.set_data(np.array(image_array, dtype=np.float32, copy=True, order="C"))
                mrc.voxel_size = float(global_result.pixel_size_input_A)
                mrc.update_header_stats()

            ctftilt_cfg = CtfTiltConfig(
                input_mrc=str(input_mrc),
                output_mrc=str(output_mrc),
                cs_mm=float(cfg.spherical_aberration_mm),
                voltage_kv=float(cfg.acceleration_voltage_kV),
                amp_contrast=float(cfg.amplitude_contrast),
                magnification=synthetic_magnification,
                detector_step_um=synthetic_detector_step_um,
                pixel_average=1,
                box=int(cfg.tilt_tile_size),
                res_min_a=float(cfg.minimum_resolution_A),
                res_max_a=float(cfg.maximum_resolution_A),
                df_min_a=float(cfg.minimum_defocus_A),
                df_max_a=float(cfg.maximum_defocus_A),
                df_step_a=float(cfg.defocus_search_step_A),
                dast_a=dast,
                expected_tilt_deg=float(nominal),
                tilt_uncertainty_deg=float(cfg.tilt_angle_uncertainty_deg),
                device=str(cfg.device),
                dtype="float32",
                candidate_batch=max(1, int(cfg.fit_batch_size)),
                tile_batch=max(1, int(cfg.fit_batch_size)),
                nr=5,
                fix_source_quirks=False,
                deterministic=False,
                quiet_objective=True,
                result_json=None,
                fast_gpu=(self.device.type == "cuda"),
            )

            log_buffer = io.StringIO()
            with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
                ctftilt_result = run_ctftilt(ctftilt_cfg)
            ctftilt_log = log_buffer.getvalue()

        axis_rad = math.radians(float(ctftilt_result.tilt_axis_deg))
        angle_rad = math.radians(float(ctftilt_result.tilt_angle_deg))
        tangent = math.tan(angle_rad)

        # Convert CTFTILT's documented local-defocus equation to the TSV's
        # plane form D(x,y)=D0+gradient_x*x+gradient_y*y, where x is positive
        # to the right and y is positive downward, both in Angstroms from the
        # image center.  CTFTILT uses DX=CX-NX and DY=CY-NY, hence the signs.
        gradient_x = -math.sin(axis_rad) * tangent
        gradient_y =  math.cos(axis_rad) * tangent

        message = (
            "CTFTILT signed-tilt backend; "
            f"tiles={int(ctftilt_result.tiles_used)}/{int(ctftilt_result.tiles_total)}, "
            f"rms_range={float(ctftilt_result.rms_min):.4g}-{float(ctftilt_result.rms_max):.4g}, "
            f"final_cc={float(ctftilt_result.final_cc):.6g}."
        )
        if ctftilt_log:
            # Keep the TSV readable while preserving the most useful status.
            important_lines = [
                line.strip() for line in ctftilt_log.splitlines()
                if line.strip().endswith("Final Values") or "ERROR" in line
            ]
            if important_lines:
                message += " " + " ".join(important_lines[-2:])

        empty_f = np.empty(0, dtype=np.float64)
        empty_b = np.empty(0, dtype=bool)
        empty_i = np.empty(0, dtype=np.int64)
        return _TiltFitDetails(
            success=True,
            message=message,
            center_defocus1_A=float(ctftilt_result.defocus1_a),
            center_defocus2_A=float(ctftilt_result.defocus2_a),
            astigmatism_angle_rad=float(ctftilt_result.astig_angle_deg) * PI / 180.0,
            gradient_x=float(gradient_x),
            gradient_y=float(gradient_y),
            tilt_angle_deg=float(ctftilt_result.tilt_angle_deg),
            tilt_axis_deg=float(ctftilt_result.tilt_axis_deg),
            nominal_tilt_angle_deg=float(nominal),
            coarse_tilt_angle_deg=float(ctftilt_result.tilt_angle_deg),
            coarse_tilt_axis_deg=float(ctftilt_result.tilt_axis_deg),
            score=float(ctftilt_result.final_cc),
            good_fit_resolution_A=0.0,
            residual_rms_A=float("nan"),
            tile_centers_x_A=empty_f.copy(),
            tile_centers_y_A=empty_f.copy(),
            tile_measured_defocus_A=empty_f.copy(),
            tile_predicted_defocus_A=empty_f.copy(),
            tile_residual_A=empty_f.copy(),
            tile_cc=empty_f.copy(),
            tile_good_fit_resolution_A=empty_f.copy(),
            tile_rms_valid=empty_b.copy(),
            tile_plane_inlier=np.ones(int(ctftilt_result.tiles_total), dtype=bool),
            tile_grid_y=empty_i.copy(),
            tile_grid_x=empty_i.copy(),
            image_shape=(height, width),
        )

    def fit_batch(
        self,
        micrographs: Sequence[np.ndarray],
        *,
        source_files: Sequence[str],
        micrograph_names: Sequence[str],
        ctf_image_names: Sequence[str],
        image_indices_1based: Sequence[int],
        pixel_size_A: float,
        return_filtered_spectra: bool = False,
        return_diagnostic_maps: bool = True,
    ) -> tuple[list[CtfFitResult], Optional[np.ndarray], Optional[np.ndarray]]:
        """Compatibility wrapper for callers that do not need two-stage batching."""
        spectra, fitting_pixel_size_A = self.preprocess_batch(
            micrographs, pixel_size_A=pixel_size_A
        )
        return self.fit_spectra_batch(
            spectra,
            source_files=source_files,
            micrograph_names=micrograph_names,
            ctf_image_names=ctf_image_names,
            image_indices_1based=image_indices_1based,
            pixel_size_input_A=pixel_size_A,
            fitting_pixel_size_A=fitting_pixel_size_A,
            return_filtered_spectra=return_filtered_spectra,
            return_diagnostic_maps=return_diagnostic_maps,
        )


def _enforce_ctffind_convention(
    defocus1_A: float,
    defocus2_A: float,
    astigmatism_angle_rad: float,
) -> tuple[float, float, float]:
    if defocus1_A < defocus2_A:
        defocus1_A, defocus2_A = defocus2_A, defocus1_A
        astigmatism_angle_rad += 0.5 * PI
    # Equivalent to CTFFIND's angle -= PI * round(angle / PI), with a stable
    # canonical interval [-PI/2, PI/2).
    astigmatism_angle_rad = (astigmatism_angle_rad + 0.5 * PI) % PI - 0.5 * PI
    return defocus1_A, defocus2_A, astigmatism_angle_rad


def _pixel_size_from_mrc(mrc: mrcfile.mrcfile.MrcFile) -> Optional[float]:
    try:
        value = float(mrc.voxel_size.x)
    except Exception:
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def _expand_input_paths(inputs: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for item in inputs:
        path = Path(item)
        matches: list[str]
        if path.is_dir():
            matches = sorted(
                str(p)
                for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in {".mrc", ".mrcs", ".map"}
            )
        else:
            globbed = sorted(glob.glob(item))
            matches = globbed if globbed else ([item] if path.exists() else [])
        expanded.extend(matches)

    unique: list[str] = []
    seen: set[str] = set()
    for item in expanded:
        absolute = str(Path(item).resolve())
        if absolute not in seen:
            seen.add(absolute)
            unique.append(absolute)
    if not unique:
        raise FileNotFoundError("No MRC input files matched the supplied paths")
    return unique


def _iter_mrc_micrographs(
    path: str,
) -> Iterator[tuple[int, np.ndarray, Optional[float]]]:
    with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
        data = mrc.data
        header_pixel_size = _pixel_size_from_mrc(mrc)
        if data.ndim == 2:
            yield 1, np.asarray(data, dtype=np.float32), header_pixel_size
        elif data.ndim == 3:
            for index in range(data.shape[0]):
                yield index + 1, np.asarray(data[index], dtype=np.float32), header_pixel_size
        else:
            raise ValueError(
                f"{path}: expected a 2-D MRC image or 3-D stack, got shape {data.shape}"
            )


def _count_mrc_micrographs(path: str) -> int:
    with mrcfile.open(path, mode="r", permissive=True, header_only=True) as mrc:
        count = int(mrc.header.nz)
    return max(1, count)




@dataclass
class _MicrographRecord:
    source_file: str
    image_index_1based: int
    image_count: int
    array: np.ndarray
    pixel_size_A: float
    micrograph_name: str
    ctf_path: Path
    ctf_image_name: str


def _relion_path(path: str | Path) -> str:
    absolute = Path(path).resolve()
    try:
        value = os.path.relpath(absolute, Path.cwd())
    except ValueError:
        value = str(absolute)
    return value.replace(os.sep, "/")


def _star_token(value: str) -> str:
    if value and not any(ch.isspace() for ch in value) and not value.startswith(("#", ";")):
        return value
    if '"' not in value:
        return f'"{value}"'
    return "'" + value.replace("'", "''") + "'"


def _diagnostic_path_for_input(
    source_file: str,
    image_index_1based: int,
    image_count: int,
    ctf_dir: Path,
) -> Path:
    stem = Path(source_file).stem
    if image_count > 1:
        stem = f"{stem}_{image_index_1based:06d}"
    return ctf_dir / f"{stem}.ctf"


def _tilt_png_path_for_input(
    source_file: str, image_index_1based: int, image_count: int, png_dir: Path
) -> Path:
    stem = Path(source_file).stem
    if image_count > 1:
        stem = f"{stem}_{image_index_1based:06d}"
    return png_dir / f"{stem}_ctftilt.png"


def _iter_micrograph_records(
    paths: Sequence[str],
    config: CtffindConfig,
    ctf_dir: Path,
) -> Iterator[_MicrographRecord]:
    used_ctf_paths: dict[Path, str] = {}
    for path in paths:
        with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
            data = mrc.data
            header_pixel = _pixel_size_from_mrc(mrc)
            pixel_size = config.pixel_size_A or header_pixel
            if pixel_size is None:
                raise ValueError(
                    f"{path}: pixel size not supplied and absent from MRC header"
                )
            if data.ndim == 2:
                image_count = 1
                indices = [(1, data)]
            elif data.ndim == 3:
                image_count = int(data.shape[0])
                indices = [(i + 1, data[i]) for i in range(image_count)]
            else:
                raise ValueError(f"{path}: expected 2-D/3-D MRC, got {data.shape}")
            for image_index, image in indices:
                ctf_path = _diagnostic_path_for_input(
                    path, image_index, image_count, ctf_dir
                ).resolve()
                previous = used_ctf_paths.get(ctf_path)
                if previous is not None and previous != path:
                    raise RuntimeError(
                        f"Diagnostic filename collision: {ctf_path} for both "
                        f"{previous} and {path}. Use separate --ctf-dir runs or "
                        "rename duplicate micrograph basenames."
                    )
                used_ctf_paths[ctf_path] = path
                source_rel = _relion_path(path)
                if image_count == 1:
                    micrograph_name = source_rel
                else:
                    micrograph_name = f"{image_index:06d}@{source_rel}"
                ctf_rel = _relion_path(ctf_path) + ":mrc"
                yield _MicrographRecord(
                    source_file=str(Path(path).resolve()),
                    image_index_1based=image_index,
                    image_count=image_count,
                    array=np.array(image, dtype=np.float32, copy=True, order="C"),
                    pixel_size_A=float(pixel_size),
                    micrograph_name=micrograph_name,
                    ctf_path=ctf_path,
                    ctf_image_name=ctf_rel,
                )


def _iter_compatible_batches(
    records: Iterable[_MicrographRecord],
    batch_size: int,
) -> Iterator[list[_MicrographRecord]]:
    batch: list[_MicrographRecord] = []
    key = None
    for record in records:
        record_key = (record.array.shape, round(record.pixel_size_A, 8))
        if batch and (record_key != key or len(batch) >= batch_size):
            yield batch
            batch = []
        if not batch:
            key = record_key
        batch.append(record)
    if batch:
        yield batch


def _write_diagnostic_ctf(
    path: Path,
    diagnostic_map: np.ndarray,
    fitting_pixel_size_A: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as output:
        output.set_data(np.asarray(diagnostic_map, dtype=np.float32))
        output.voxel_size = fitting_pixel_size_A


def _write_relion_star(
    path: Path,
    results: Sequence[CtfFitResult],
    config: CtffindConfig,
    *,
    include_ctf_image: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    optics_keys: dict[tuple[float, float, float, float], int] = {}
    optics_rows: list[tuple[int, float]] = []
    row_groups: list[int] = []
    for result in results:
        key = (
            round(result.pixel_size_input_A, 8),
            round(config.acceleration_voltage_kV, 8),
            round(config.spherical_aberration_mm, 8),
            round(config.amplitude_contrast, 8),
        )
        group = optics_keys.get(key)
        if group is None:
            group = len(optics_rows) + 1
            optics_keys[key] = group
            optics_rows.append((group, result.pixel_size_input_A))
        row_groups.append(group)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# version 30001\n\n")
        handle.write("data_optics\n\nloop_ \n")
        optics_columns = [
            "_rlnOpticsGroupName",
            "_rlnOpticsGroup",
            "_rlnMicrographOriginalPixelSize",
            "_rlnVoltage",
            "_rlnSphericalAberration",
            "_rlnAmplitudeContrast",
            "_rlnMicrographPixelSize",
        ]
        for idx, column in enumerate(optics_columns, 1):
            handle.write(f"{column} #{idx} \n")
        for group, pixel in optics_rows:
            handle.write(
                f"opticsGroup{group} {group:d} {pixel:.6f} "
                f"{config.acceleration_voltage_kV:.6f} "
                f"{config.spherical_aberration_mm:.6f} "
                f"{config.amplitude_contrast:.6f} {pixel:.6f}\n"
            )

        handle.write("\n\n# version 30001\n\n")
        handle.write("data_micrographs\n\nloop_ \n")
        columns = ["_rlnMicrographName", "_rlnOpticsGroup"]
        if include_ctf_image:
            columns.append("_rlnCtfImage")
        columns += [
            "_rlnDefocusU",
            "_rlnDefocusV",
            "_rlnCtfAstigmatism",
            "_rlnDefocusAngle",
            "_rlnCtfFigureOfMerit",
            "_rlnCtfMaxResolution",
        ]
        if any(abs(r.phase_shift_rad) > 1.0e-12 for r in results):
            columns.append("_rlnPhaseShift")
        for idx, column in enumerate(columns, 1):
            handle.write(f"{column} #{idx} \n")

        include_phase = columns[-1] == "_rlnPhaseShift"
        for result, group in zip(results, row_groups):
            tokens = [
                _star_token(result.micrograph_name),
                str(group),
            ]
            if include_ctf_image:
                tokens.append(_star_token(result.ctf_image_name))
            tokens.extend([
                f"{result.defocus1_A:.6f}",
                f"{result.defocus2_A:.6f}",
                f"{(result.defocus1_A - result.defocus2_A):.6f}",
                f"{result.astigmatism_angle_deg:.6f}",
                f"{result.score:.6f}",
                f"{result.thon_rings_good_fit_resolution_A:.6f}",
            ])
            if include_phase:
                tokens.append(f"{(result.phase_shift_rad * 180.0 / PI):.6f}")
            handle.write(" ".join(tokens) + "\n")
    os.replace(tmp, path)



def _read_tilt_angle_file(path: str | Path) -> dict[str, float]:
    """Read a simple two-column ``micrograph angle_deg`` text file.

    The first column may be a RELION micrograph name, an absolute/relative
    path, a basename, or a stem. Blank lines and lines starting with ``#``
    are ignored. The final whitespace-separated token is parsed as the angle;
    preceding tokens are joined back together so quoted paths are not needed
    unless they contain trailing numeric words.
    """
    mapping: dict[str, float] = {}
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(
                    f"{input_path}:{line_number}: expected '<micrograph> <angle_deg>'"
                )
            try:
                angle = float(fields[-1])
            except ValueError as exc:
                raise ValueError(
                    f"{input_path}:{line_number}: invalid tilt angle {fields[-1]!r}"
                ) from exc
            if abs(angle) >= 89.9:
                raise ValueError(
                    f"{input_path}:{line_number}: tilt angle magnitude must be < 89.9"
                )
            key = " ".join(fields[:-1]).strip().strip('"').strip("'")
            mapping[key.replace('\\', '/')] = angle
    if not mapping:
        raise ValueError(f"No tilt angles found in {input_path}")
    return mapping


def _lookup_tilt_angle(
    record: _MicrographRecord,
    mapping: dict[str, float],
    fallback: Optional[float],
) -> Optional[float]:
    source = Path(record.source_file)
    source_rel = _relion_path(source)
    candidates = [
        record.micrograph_name,
        record.source_file.replace(os.sep, "/"),
        source_rel,
        source.name,
        source.stem,
    ]
    if record.image_count > 1:
        candidates.extend([
            f"{record.image_index_1based:06d}@{source_rel}",
            f"{source.name}:{record.image_index_1based}",
            f"{source.stem}:{record.image_index_1based}",
            f"{source.stem}_{record.image_index_1based:06d}",
        ])
    for key in candidates:
        normalized = key.replace('\\', '/')
        if normalized in mapping:
            return float(mapping[normalized])
    return fallback

def fit_mrc_files(
    input_paths: Sequence[str],
    config: CtffindConfig,
    output_star: str,
    ctf_output_dir: Optional[str] = None,
    save_filtered_spectra_dir: Optional[str] = None,
    write_diagnostic_maps: bool = True,
    continue_on_error: bool = False,
    tilt_png_output_dir: Optional[str] = None,
    write_tilt_png: bool = True,
    tilt_results_tsv: Optional[str] = None,
    tilt_angle_file: Optional[str] = None,
) -> list[CtfFitResult]:
    """Run a two-stage raw-image/preprocessed-spectrum GPU pipeline."""
    paths = _expand_input_paths(input_paths)
    tilt_angle_mapping = (
        _read_tilt_angle_file(tilt_angle_file) if tilt_angle_file else {}
    )
    estimator = TorchCtffindPowell(config)
    output_path = Path(output_star).resolve()
    ctf_dir = (
        Path(ctf_output_dir).resolve()
        if ctf_output_dir is not None
        else output_path.parent
    )
    ctf_dir.mkdir(parents=True, exist_ok=True)
    tilt_png_dir = (
        Path(tilt_png_output_dir).resolve()
        if tilt_png_output_dir is not None
        else output_path.parent / "ctftilt_png"
    )
    if config.fit_tilt and write_tilt_png:
        tilt_png_dir.mkdir(parents=True, exist_ok=True)
    tilt_tsv_path = (
        Path(tilt_results_tsv).resolve()
        if tilt_results_tsv is not None
        else output_path.with_name(output_path.stem + "_tilt.tsv")
    )
    spectra_dir = None
    if save_filtered_spectra_dir:
        spectra_dir = Path(save_filtered_spectra_dir).resolve()
        spectra_dir.mkdir(parents=True, exist_ok=True)

    total_images = sum(_count_mrc_micrographs(path) for path in paths)
    results: list[CtfFitResult] = []
    processed = 0
    preprocessed = 0
    print(f"Device: {estimator.device}")
    print(f"Input files: {len(paths)}; micrographs/slices: {total_images}")
    print(
        f"Preprocessing batch size: {config.preprocess_batch_size}; "
        f"fitting batch size: {config.fit_batch_size}"
    )
    print(
        f"Optimizer convergence check interval: "
        f"{config.optimizer_check_interval} iterations"
    )

    pending_records: list[_MicrographRecord] = []
    pending_chunks: list[torch.Tensor] = []
    pending_count = 0
    pending_key: Optional[tuple[float, float]] = None

    def pop_pending_spectra(number: int) -> torch.Tensor:
        nonlocal pending_count
        if number < 1 or number > pending_count:
            raise ValueError("Invalid pending-spectrum pop size")
        pieces: list[torch.Tensor] = []
        remaining = number
        while remaining > 0:
            chunk = pending_chunks[0]
            take = min(remaining, int(chunk.shape[0]))
            pieces.append(chunk[:take])
            if take == int(chunk.shape[0]):
                pending_chunks.pop(0)
            else:
                pending_chunks[0] = chunk[take:]
            remaining -= take
        pending_count -= number
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    def write_fit_outputs(
        batch_records: Sequence[_MicrographRecord],
        batch_results: Sequence[CtfFitResult],
        filtered: Optional[np.ndarray],
        diagnostics: Optional[np.ndarray],
    ) -> None:
        nonlocal processed
        for i, (record, result) in enumerate(zip(batch_records, batch_results)):
            if write_diagnostic_maps and diagnostics is not None:
                _write_diagnostic_ctf(
                    record.ctf_path,
                    diagnostics[i],
                    result.pixel_size_for_fitting_A,
                )
            if spectra_dir is not None and filtered is not None:
                spectrum_name = record.ctf_path.stem + "_filtered_spectrum.mrc"
                spectrum_path = spectra_dir / spectrum_name
                with mrcfile.new(spectrum_path, overwrite=True) as output:
                    output.set_data(filtered[i])
                    output.voxel_size = result.pixel_size_for_fitting_A
            if config.fit_tilt:
                result.global_thon_rings_good_fit_resolution_A = (
                    result.thon_rings_good_fit_resolution_A
                )
                try:
                    nominal_angle = _lookup_tilt_angle(
                        record, tilt_angle_mapping, config.tilt_nominal_angle_deg
                    )
                    tilt = estimator.fit_tilt_micrograph(
                        record.array, result, nominal_tilt_angle_deg=nominal_angle
                    )
                    result.tilt_message = tilt.message
                    result.nominal_tilt_angle_deg = tilt.nominal_tilt_angle_deg
                    result.coarse_tilt_angle_deg = tilt.coarse_tilt_angle_deg
                    result.coarse_tilt_axis_deg = tilt.coarse_tilt_axis_deg
                    result.tilt_total_tiles = int(tilt.tile_plane_inlier.size)
                    result.tilt_valid_tiles = int(np.sum(tilt.tile_plane_inlier))
                    if tilt.success:
                        result.tilt_fitted = True
                        result.defocus1_A = tilt.center_defocus1_A
                        result.defocus2_A = tilt.center_defocus2_A
                        result.astigmatism_angle_deg = tilt.astigmatism_angle_rad * 180.0 / PI
                        result.defocus_gradient_x = tilt.gradient_x
                        result.defocus_gradient_y = tilt.gradient_y
                        result.tilt_angle_deg = tilt.tilt_angle_deg
                        result.tilt_axis_deg = tilt.tilt_axis_deg
                        result.tilt_score = tilt.score
                        result.tilt_good_fit_resolution_A = tilt.good_fit_resolution_A
                        result.tilt_residual_rms_A = tilt.residual_rms_A
                        if tilt.good_fit_resolution_A > 0.0:
                            result.thon_rings_good_fit_resolution_A = tilt.good_fit_resolution_A
                        if write_tilt_png and tilt.tile_centers_x_A.size > 0:
                            png_path = _tilt_png_path_for_input(
                                record.source_file, record.image_index_1based,
                                record.image_count, tilt_png_dir,
                            )
                            _write_tilt_png(png_path, tilt)
                            result.tilt_png_name = _relion_path(png_path)
                    else:
                        result.tilt_message = tilt.message
                except Exception as tilt_exc:
                    result.tilt_message = f"Tilt fitting failed: {tilt_exc}"
                    if not continue_on_error:
                        raise
                    print(
                        f"WARNING: {Path(record.source_file).name}: {result.tilt_message}",
                        file=sys.stderr,
                    )
            results.append(result)
            processed += 1
            good = (
                f"{result.thon_rings_good_fit_resolution_A:.2f} A"
                if result.thon_rings_good_fit_resolution_A > 0.0
                else "undetermined"
            )
            tilt_text = (
                f", tilt={result.tilt_angle_deg:.2f} deg, axis={result.tilt_axis_deg:.2f} deg"
                if result.tilt_fitted else ""
            )
            print(
                f"  [{processed}/{total_images}] {Path(result.source_file).name}: "
                f"dfU={result.defocus1_A:.1f}, dfV={result.defocus2_A:.1f}, "
                f"angle={result.astigmatism_angle_deg:.2f}, "
                f"CC={result.score:.5f}, maxres={good}{tilt_text}"
            )
        _write_relion_star(
            output_path,
            results,
            config,
            include_ctf_image=write_diagnostic_maps,
        )
        if config.fit_tilt:
            _write_tilt_tsv(tilt_tsv_path, results)

    def fit_pending(number: int) -> None:
        nonlocal pending_records
        spectra = pop_pending_spectra(number)
        batch_records = pending_records[:number]
        del pending_records[:number]
        if not batch_records:
            return
        input_pixel = batch_records[0].pixel_size_A
        fitting_pixel = float(pending_key[1]) if pending_key is not None else input_pixel
        try:
            batch_results, filtered, diagnostics = estimator.fit_spectra_batch(
                spectra,
                source_files=[r.source_file for r in batch_records],
                micrograph_names=[r.micrograph_name for r in batch_records],
                ctf_image_names=[r.ctf_image_name for r in batch_records],
                image_indices_1based=[r.image_index_1based for r in batch_records],
                pixel_size_input_A=input_pixel,
                fitting_pixel_size_A=fitting_pixel,
                return_filtered_spectra=spectra_dir is not None,
                return_diagnostic_maps=write_diagnostic_maps,
            )
            write_fit_outputs(batch_records, batch_results, filtered, diagnostics)
            return
        except Exception as exc:
            if not continue_on_error or len(batch_records) == 1:
                raise RuntimeError(
                    f"Fitting batch beginning with {batch_records[0].source_file}: {exc}"
                ) from exc
            print(
                f"WARNING: fitting batch failed ({exc}); retrying spectra individually",
                file=sys.stderr,
            )

        for i, record in enumerate(batch_records):
            try:
                rr, ff, dd = estimator.fit_spectra_batch(
                    spectra[i:i + 1],
                    source_files=[record.source_file],
                    micrograph_names=[record.micrograph_name],
                    ctf_image_names=[record.ctf_image_name],
                    image_indices_1based=[record.image_index_1based],
                    pixel_size_input_A=record.pixel_size_A,
                    fitting_pixel_size_A=fitting_pixel,
                    return_filtered_spectra=spectra_dir is not None,
                    return_diagnostic_maps=write_diagnostic_maps,
                )
                write_fit_outputs([record], rr, ff, dd)
            except Exception as single_exc:
                print(f"ERROR: {record.source_file}: {single_exc}", file=sys.stderr)

    def flush_all_pending() -> None:
        while pending_count > 0:
            fit_pending(min(config.fit_batch_size, pending_count))

    def enqueue_preprocessed(
        batch_records: Sequence[_MicrographRecord],
        spectra: torch.Tensor,
        fitting_pixel_size_A: float,
    ) -> None:
        nonlocal pending_count, pending_key
        if not batch_records:
            return
        key = (
            round(batch_records[0].pixel_size_A, 8),
            round(float(fitting_pixel_size_A), 8),
        )
        if pending_count > 0 and key != pending_key:
            flush_all_pending()
        pending_key = key
        pending_records.extend(batch_records)
        pending_chunks.append(spectra)
        pending_count += len(batch_records)
        while pending_count >= config.fit_batch_size:
            fit_pending(config.fit_batch_size)

    records = _iter_micrograph_records(paths, config, ctf_dir)
    for raw_batch in _iter_compatible_batches(records, config.preprocess_batch_size):
        first = preprocessed + 1
        last = preprocessed + len(raw_batch)
        shape = raw_batch[0].array.shape
        print(
            f"Preprocess [{first}-{last}/{total_images}] "
            f"batch={len(raw_batch)}, shape={shape[1]}x{shape[0]}, "
            f"pixel={raw_batch[0].pixel_size_A:.6g} A"
        )
        try:
            spectra, fitting_pixel = estimator.preprocess_batch(
                [r.array for r in raw_batch],
                pixel_size_A=raw_batch[0].pixel_size_A,
            )
            enqueue_preprocessed(raw_batch, spectra, fitting_pixel)
            preprocessed += len(raw_batch)
        except Exception as exc:
            if not continue_on_error or len(raw_batch) == 1:
                raise RuntimeError(
                    f"Preprocessing batch beginning with {raw_batch[0].source_file}: {exc}"
                ) from exc
            print(
                f"WARNING: preprocessing batch failed ({exc}); retrying images individually",
                file=sys.stderr,
            )
            for record in raw_batch:
                try:
                    spectra, fitting_pixel = estimator.preprocess_batch(
                        [record.array], pixel_size_A=record.pixel_size_A
                    )
                    enqueue_preprocessed([record], spectra, fitting_pixel)
                except Exception as single_exc:
                    print(f"ERROR: {record.source_file}: {single_exc}", file=sys.stderr)
                finally:
                    preprocessed += 1

    flush_all_pending()
    print(f"Wrote {len(results)} rows to {_relion_path(output_path)}")
    if write_diagnostic_maps:
        print(f"Wrote one .ctf MRC per micrograph under {_relion_path(ctf_dir)}")
    if config.fit_tilt:
        print(f"Wrote tilt results to {_relion_path(tilt_tsv_path)}")
        if write_tilt_png and any(r.tilt_png_name for r in results):
            print(f"Wrote tilt PNG diagnostics under {_relion_path(tilt_png_dir)}")
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fast CTFFIND-4.1.8-like CTF estimation with batched PyTorch "
            "Powell optimization and RELION STAR/.ctf output."
        )
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="MRC/MRCS files, glob patterns, or directories containing MRC files",
    )
    parser.add_argument("--output", default="micrographs_ctf.star")
    parser.add_argument(
        "--ctf-dir", default=None,
        help="Directory for per-micrograph .ctf MRC files; default: STAR directory",
    )
    parser.add_argument(
        "--pixel-size", type=float, default=None,
        help="Input pixel size in Angstrom; otherwise read each MRC header",
    )
    parser.add_argument("--voltage", type=float, default=300.0)
    parser.add_argument("--cs", type=float, default=2.7)
    parser.add_argument("--amplitude-contrast", type=float, default=0.07)
    parser.add_argument("--box-size", type=int, default=256)
    parser.add_argument("--min-resolution", type=float, default=30.0)
    parser.add_argument("--max-resolution", type=float, default=5.0)
    parser.add_argument("--min-defocus", type=float, default=5000.0)
    parser.add_argument("--max-defocus", type=float, default=50000.0)
    parser.add_argument("--defocus-step", type=float, default=100.0)
    parser.add_argument("--astigmatism-tolerance", type=float, default=300.0)
    parser.add_argument(
        "--find-phase-shift", action="store_true",
        help="Reserved. Uses fixed --phase-shift in this version.",
    )
    parser.add_argument("--phase-shift", type=float, default=0.0)
    parser.add_argument("--min-phase-shift", type=float, default=0.0)
    parser.add_argument("--max-phase-shift", type=float, default=3.15)
    parser.add_argument("--phase-shift-step", type=float, default=0.5)
    parser.add_argument("--no-resample-small-pixel", action="store_true")
    parser.add_argument("--target-fitting-pixel-size", type=float, default=1.4)
    parser.add_argument("--angle-step", type=float, default=5.0)
    parser.add_argument("--rotation-batch-size", type=int, default=8)
    parser.add_argument(
        "--preprocess-batch-size", type=int, default=4,
        help="Raw micrographs processed together during FFT/preprocessing",
    )
    parser.add_argument(
        "--fit-batch-size", type=int, default=64,
        help="Filtered 512x512 spectra fitted together on the GPU",
    )
    parser.add_argument(
        "--optimizer-check-interval", type=int, default=8,
        help=(
            "Check GPU convergence masks on the CPU only every N scalar-search "
            "iterations; larger values reduce synchronization"
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--powell-xtol", type=float, default=1.0e-4)
    parser.add_argument("--powell-ftol", type=float, default=1.0e-7)
    parser.add_argument("--powell-maxiter-1d", type=int, default=80)
    parser.add_argument("--powell-maxiter-2d", type=int, default=30)
    parser.add_argument("--powell-line-maxiter", type=int, default=80)
    parser.add_argument("--no-powell-bounds", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-filtered-spectra", default=None, metavar="DIRECTORY"
    )
    parser.add_argument(
        "--no-diagnostic-output", action="store_true",
        help="Do not write .ctf maps or the _rlnCtfImage STAR column",
    )
    parser.add_argument(
        "--fit-tilt", action="store_true",
        help="Fit a spatial defocus plane from local tiles after the global CTF fit",
    )
    parser.add_argument("--tilt-tile-size", type=int, default=256)
    parser.add_argument("--tilt-tile-stride", type=int, default=256)
    parser.add_argument("--tilt-min-global-cc", type=float, default=0.10)
    parser.add_argument("--tilt-local-range", type=float, default=12000.0,
                        help="Half-range in Angstrom for each tile's local defocus search")
    parser.add_argument("--tilt-local-step", type=float, default=100.0)
    parser.add_argument("--tilt-min-tile-cc", type=float, default=0.04)
    parser.add_argument("--tilt-min-tiles", type=int, default=6)
    parser.add_argument("--tilt-rms-mad-cutoff", type=float, default=3.5)
    parser.add_argument("--tilt-plane-mad-cutoff", type=float, default=3.5)
    parser.add_argument("--tilt-max-angle", type=float, default=80.0)
    parser.add_argument("--tilt-gradient-scale", type=float, default=0.05)
    parser.add_argument("--tilt-powell-maxiter", type=int, default=24)
    parser.add_argument(
        "--tilt-angle", type=float, default=None,
        help="Nominal tilt angle in degrees; may be negative",
    )
    parser.add_argument(
        "--tilt-angle-file", default=None,
        help="Two-column text file: micrograph_name nominal_tilt_angle_deg",
    )
    parser.add_argument("--tilt-angle-uncertainty", type=float, default=5.0)
    parser.add_argument("--tilt-angle-grid-step", type=float, default=2.0)
    parser.add_argument("--tilt-axis-grid-step", type=float, default=2.0)
    parser.add_argument(
        "--tilt-axis-search-half-range", type=float, default=90.0,
        help="Axis grid half-range around the tile-plane estimate; 90 searches 0-180",
    )
    parser.add_argument("--tilt-candidate-batch-size", type=int, default=24)
    parser.add_argument("--tilt-prior-scale", type=float, default=1.0)
    parser.add_argument("--tilt-hard-range-multiplier", type=float, default=2.0)
    parser.add_argument("--tilt-stage1-maxiter", type=int, default=16)
    parser.add_argument("--tilt-png-dir", default=None)
    parser.add_argument("--tilt-results-tsv", default=None)
    parser.add_argument("--no-tilt-png", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    config = CtffindConfig(
        pixel_size_A=args.pixel_size,
        acceleration_voltage_kV=args.voltage,
        spherical_aberration_mm=args.cs,
        amplitude_contrast=args.amplitude_contrast,
        box_size=args.box_size,
        minimum_resolution_A=args.min_resolution,
        maximum_resolution_A=args.max_resolution,
        minimum_defocus_A=args.min_defocus,
        maximum_defocus_A=args.max_defocus,
        defocus_search_step_A=args.defocus_step,
        astigmatism_tolerance_A=args.astigmatism_tolerance,
        find_phase_shift=args.find_phase_shift,
        fixed_phase_shift_rad=args.phase_shift,
        minimum_phase_shift_rad=args.min_phase_shift,
        maximum_phase_shift_rad=args.max_phase_shift,
        phase_shift_search_step_rad=args.phase_shift_step,
        resample_if_pixel_too_small=not args.no_resample_small_pixel,
        target_pixel_size_after_resampling_A=args.target_fitting_pixel_size,
        angle_search_step_deg=args.angle_step,
        angle_rotation_batch_size=args.rotation_batch_size,
        powell_xtol=args.powell_xtol,
        powell_ftol=args.powell_ftol,
        powell_maxiter_1d=args.powell_maxiter_1d,
        powell_maxiter_2d=args.powell_maxiter_2d,
        powell_line_maxiter=args.powell_line_maxiter,
        use_powell_defocus_bounds=not args.no_powell_bounds,
        preprocess_batch_size=(
            args.batch_size
            if args.batch_size is not None
            else args.preprocess_batch_size
        ),
        fit_batch_size=args.fit_batch_size,
        optimizer_check_interval=args.optimizer_check_interval,
        device=args.device,
        fit_tilt=args.fit_tilt,
        tilt_tile_size=args.tilt_tile_size,
        tilt_tile_stride=args.tilt_tile_stride,
        tilt_min_global_cc=args.tilt_min_global_cc,
        tilt_local_search_range_A=args.tilt_local_range,
        tilt_local_search_step_A=args.tilt_local_step,
        tilt_min_tile_cc=args.tilt_min_tile_cc,
        tilt_min_tiles=args.tilt_min_tiles,
        tilt_rms_mad_cutoff=args.tilt_rms_mad_cutoff,
        tilt_plane_mad_cutoff=args.tilt_plane_mad_cutoff,
        tilt_max_angle_deg=args.tilt_max_angle,
        tilt_gradient_scale=args.tilt_gradient_scale,
        tilt_powell_maxiter=args.tilt_powell_maxiter,
        tilt_nominal_angle_deg=args.tilt_angle,
        tilt_angle_uncertainty_deg=args.tilt_angle_uncertainty,
        tilt_angle_grid_step_deg=args.tilt_angle_grid_step,
        tilt_axis_grid_step_deg=args.tilt_axis_grid_step,
        tilt_axis_search_half_range_deg=args.tilt_axis_search_half_range,
        tilt_candidate_batch_size=args.tilt_candidate_batch_size,
        tilt_prior_scale=args.tilt_prior_scale,
        tilt_hard_range_multiplier=args.tilt_hard_range_multiplier,
        tilt_stage1_maxiter=args.tilt_stage1_maxiter,
    )
    config.validate()
    if config.fit_tilt and args.tilt_angle is None and args.tilt_angle_file is None:
        raise SystemExit(
            "--fit-tilt requires either --tilt-angle or --tilt-angle-file"
        )
    fit_mrc_files(
        args.inputs,
        config,
        output_star=args.output,
        ctf_output_dir=args.ctf_dir,
        save_filtered_spectra_dir=args.save_filtered_spectra,
        write_diagnostic_maps=not args.no_diagnostic_output,
        continue_on_error=args.continue_on_error,
        tilt_png_output_dir=args.tilt_png_dir,
        write_tilt_png=not args.no_tilt_png,
        tilt_results_tsv=args.tilt_results_tsv,
        tilt_angle_file=args.tilt_angle_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
