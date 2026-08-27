#!/usr/bin/env python3
"""
CTFFIND5-PyTorch 0.5.4: a wxWidgets-free PyTorch implementation of the
CTFFIND4/CTFFIND5 fitting pipeline for independent 2-D MRC micrographs.

Implemented
-----------
* MRC input (exactly one 2-D micrograph per file; stacks are rejected)
* Batched PyTorch amplitude-spectrum generation and CTFFIND-like preprocessing
* Batched 1-D mean-defocus search and 2-D astigmatic Powell refinement
* CTFFIND-style fit statistics, diagnostic MRCs, and RELION STAR output
* Optional CTFFIND5-style tilt determination and correction:
  - CTFFIND5-compatible fixed 40--10 A local fitting band
  - nominal 5 A/pixel local sampling and 128-pixel 50%-overlap spectra
  - cosine-windowed, variance-normalized tiles with a fixed 55-pixel background
  - one global Pearson score over all local tiles/Fourier pixels
  - CTFFIND-style simplex refinement of axis, angle, and centre mean defocus
  - defocus-scaled, tilt-corrected spectrum at the requested --box-size
  - a final ordinary CTFFIND CTF fit on that corrected spectrum
  - CTFFIND5 non-negative-angle / 0-360-degree-axis output convention
* Optional CTFFIND5 node-based ice-thickness search with coupled 1-D
  thickness/defocus search and local 2-D CTF/thickness refinement
* Built-in spawn-based multi-device scheduler:
  - one persistent process per CUDA device
  - one complete micrograph per worker at a time (never split across GPUs)
  - dynamic queue scheduling for mixed tilt/untilt runtimes
  - deterministic parent-side STAR/TSV/avrot merging

Untilts and tilt-corrected spectra share the same full-2D filtering,
standard CTF fitting, EPA/FRC, and optional thickness pipeline.

Dependencies
------------
    numpy, torch, mrcfile

Example
-------
    python ctffind5_pytorch.py "MotionCorr/job003/*.mrc" \
        --pixel-size 1.06 --voltage 300 --cs 2.7 \
        --amplitude-contrast 0.07 --box-size 512 \
        --min-resolution 30 --max-resolution 5 \
        --min-defocus 5000 --max-defocus 50000 \
        --defocus-step 500 --fit-tilt --estimate-thickness \
        --output micrographs_ctf.star --ctf-dir CtfFind/job005

Multi-GPU example (one micrograph per GPU worker):
    python ctffind5_pytorch.py "MotionCorr/job003/*.mrc" \
        --gpu-ids 0-3 --pixel-size 1.06 --fit-tilt --estimate-thickness \
        --output micrographs_ctf.star --ctf-dir CtfFind/job005
"""

# Upstream notice
# ---------------
# CTFFIND/CTFTILT algorithms and source-ordered compatibility operations are
# adapted from cisTEM (ctffind5_merge), Copyright (c) 2017 Howard Hughes
# Medical Institute, distributed under the Janelia Research Campus Software
# License 1.2.  See LICENSE_CISTEM.txt in the release archive.
# The standard PyTorch CTFFIND path was also derived from the user-provided
# reference implementation at github.com/homurachan/pytorch_cryoEM.

from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as _datetime
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import struct
import sys
import time
import traceback
import warnings
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F


PI = math.pi
VERSION = "0.5.4"


def _add_timing(target: dict[str, float], name: str, seconds: float) -> None:
    """Accumulate one timing value."""
    target[name] = float(target.get(name, 0.0) + float(seconds))


@contextmanager
def _timed_stage(
    enabled: bool,
    device: torch.device | None,
    target: dict[str, float],
    name: str,
) -> Iterator[None]:
    """Measure one stage; CUDA synchronization occurs only under --timing."""
    if not enabled:
        yield
        return
    if device is not None:
        _synchronize_if_cuda(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        if device is not None:
            _synchronize_if_cuda(device)
        _add_timing(target, name, time.perf_counter() - started)


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

    # Optional CTFFIND5-style tilt determination.  This path performs its own
    # local-spectrum CTF initialization and does not use a nominal stage angle,
    # an angle prior, or the old CTFTILT-v1.7 backend.
    fit_tilt: bool = False
    tilt_tile_size: int = 128
    tilt_tile_stride: int = 64
    # None selects native CTFFIND5 local sampling: 10 A / 2 = 5 A/pixel.
    # An explicit value is retained only as an experimental override.
    tilt_target_pixel_size_A: Optional[float] = None
    tilt_axis_step_deg: float = 10.0
    tilt_angle_step_deg: float = 5.0
    tilt_max_angle_deg: float = 80.0
    tilt_candidate_batch_size: int = 32
    tilt_tile_batch_size: int = 48
    tilt_refine_maxiter: int = 90
    tilt_refine_axis_half_range_deg: float = 20.0
    tilt_refine_angle_half_range_deg: float = 10.0
    tilt_refine_defocus_half_range_A: float = 5_000.0
    tilt_min_tiles: int = 3
    tilt_rms_mad_cutoff: float = 0.0
    tilt_diagnostic_defocus_range_A: float = 1_500.0
    tilt_diagnostic_defocus_step_A: float = 100.0

    # Optional CTFFIND5-style sample-thickness (node) fitting.  Lengths are in
    # Angstroms.  The 1-D brute-force stage is always used; the 2-D stage can be
    # disabled for a faster diagnostic estimate.
    estimate_thickness: bool = False
    # CTFFIND5's 1-D node search spans 50--400 nm in 1-nm increments.
    # The public interface remains Angstrom throughout this Python program.
    thickness_min_A: float = 500.0
    thickness_max_A: float = 4_000.0
    thickness_step_A: float = 10.0
    thickness_low_resolution_A: float = 30.0
    thickness_high_resolution_A: float = 3.0
    # The 1-D brute-force stage searches thickness together with mean defocus.
    thickness_defocus_search_range_A: float = 1_000.0
    thickness_defocus_step_A: float = 10.0
    # The 2-D stage is a local joint refinement of thickness, defocus U/V,
    # and astigmatism angle, matching CTFFIND5's four-parameter node fit.
    thickness_2d_refine: bool = True
    thickness_refine_maxiter: int = 40
    thickness_use_rounded_square: bool = False
    thickness_downweight_nodes: bool = False
    thickness_candidate_batch_size: int = 1024
    # Independent four-parameter thickness refinements evaluated together.
    # This is deliberately separate from --fit-batch-size because the 2-D
    # thickness support is much larger than the ordinary CTF support.
    thickness_refine_batch_size: int = 32

    # Runtime.  Raw micrographs are large, while filtered spectra are small;
    # keep the two batch sizes independent so GPU fitting is not limited by
    # the memory footprint of 4K/8K FFT preprocessing.
    preprocess_batch_size: int = 4
    fit_batch_size: int = 64
    optimizer_check_interval: int = 8
    device: str = "auto"  # auto, cpu, cuda, cuda:0, ...
    debug: bool = False
    timing: bool = False

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
        if self.tilt_target_pixel_size_A is not None:
            if self.tilt_target_pixel_size_A <= 0.0:
                raise ValueError("tilt_target_pixel_size_A must be positive")
            if self.tilt_target_pixel_size_A > 5.0 * (1.0 + 1.0e-7):
                raise ValueError(
                    "tilt_target_pixel_size_A cannot exceed 5 A/pixel in "
                    "CTFFIND5 compatibility mode (the local fit ends at 10 A)"
                )
        if self.tilt_axis_step_deg <= 0.0 or self.tilt_axis_step_deg > 180.0:
            raise ValueError("tilt_axis_step_deg must be in (0, 180]")
        if self.tilt_angle_step_deg <= 0.0:
            raise ValueError("tilt_angle_step_deg must be positive")
        if not (0.0 < self.tilt_max_angle_deg < 89.9):
            raise ValueError("tilt_max_angle_deg must be between 0 and 89.9")
        if self.tilt_candidate_batch_size < 1 or self.tilt_tile_batch_size < 1:
            raise ValueError("tilt candidate/tile batch sizes must be >= 1")
        if self.tilt_refine_maxiter < 1:
            raise ValueError("tilt_refine_maxiter must be >= 1")
        if self.tilt_refine_axis_half_range_deg <= 0.0:
            raise ValueError("tilt_refine_axis_half_range_deg must be positive")
        if self.tilt_refine_angle_half_range_deg <= 0.0:
            raise ValueError("tilt_refine_angle_half_range_deg must be positive")
        if self.tilt_refine_defocus_half_range_A <= 0.0:
            raise ValueError("tilt_refine_defocus_half_range_A must be positive")
        if self.tilt_min_tiles < 3:
            raise ValueError("tilt_min_tiles must be >= 3")
        if self.tilt_rms_mad_cutoff < 0.0:
            raise ValueError("tilt_rms_mad_cutoff must be non-negative")
        if self.tilt_diagnostic_defocus_range_A <= 0.0:
            raise ValueError("tilt_diagnostic_defocus_range_A must be positive")
        if self.tilt_diagnostic_defocus_step_A <= 0.0:
            raise ValueError("tilt_diagnostic_defocus_step_A must be positive")
        if self.thickness_min_A < 0.0:
            raise ValueError("thickness_min_A must be non-negative")
        if self.thickness_max_A <= self.thickness_min_A:
            raise ValueError("thickness_max_A must exceed thickness_min_A")
        if self.thickness_step_A <= 0.0:
            raise ValueError("thickness_step_A must be positive")
        if self.thickness_defocus_search_range_A <= 0.0:
            raise ValueError("thickness_defocus_search_range_A must be positive")
        if self.thickness_defocus_step_A <= 0.0:
            raise ValueError("thickness_defocus_step_A must be positive")
        if self.thickness_refine_maxiter < 1:
            raise ValueError("thickness_refine_maxiter must be >= 1")
        if self.thickness_low_resolution_A <= self.thickness_high_resolution_A:
            raise ValueError(
                "thickness_low_resolution_A must be numerically larger than "
                "thickness_high_resolution_A"
            )
        if self.thickness_candidate_batch_size < 1:
            raise ValueError("thickness_candidate_batch_size must be >= 1")
        if self.thickness_refine_batch_size < 1:
            raise ValueError("thickness_refine_batch_size must be >= 1")
        if self.fit_tilt and self.find_phase_shift:
            raise ValueError(
                "CTFFIND5 tilt determination and phase-shift searching cannot be active together"
            )


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
    global_thon_rings_good_fit_resolution_A: float = 0.0
    tilt_fitted: bool = False
    tilt_angle_deg: float = float("nan")
    tilt_axis_deg: float = float("nan")
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
    ice_thickness_fitted: bool = False
    ice_thickness_A: float = float("nan")
    ice_thickness_score: float = float("nan")
    ice_thickness_message: str = "Not attempted."
    debug: Optional[dict[str, object]] = None
    timings: Optional[dict[str, float]] = None
    avrot_spatial_frequency_Ainv: Optional[np.ndarray] = None
    avrot_rotational_average_no_astig: Optional[np.ndarray] = None
    avrot_rotational_average_astig: Optional[np.ndarray] = None
    avrot_rotational_average_fit: Optional[np.ndarray] = None
    avrot_fit_frc: Optional[np.ndarray] = None
    avrot_fit_frc_sigma: Optional[np.ndarray] = None


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
    coarse_tilt_angle_deg: float
    coarse_tilt_axis_deg: float
    # Local tile power/CTF^2 plane score. The final CTF score is stored in
    # final_ctf_result.score after fitting the corrected spectrum.
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
    final_ctf_result: Optional[CtfFitResult] = None
    filtered_spectrum: Optional[np.ndarray] = None
    diagnostic_map: Optional[np.ndarray] = None
    ice_thickness_fitted: bool = False
    ice_thickness_A: float = float("nan")
    ice_thickness_score: float = float("nan")
    ice_thickness_message: str = "Not attempted."
    # Populated only under --debug; arrays are written as separate MRC files
    # and are never embedded in JSON.
    debug_spectra: Optional[dict[str, np.ndarray]] = None


@dataclass
class _ThicknessFitDetails:
    success: bool
    thickness_A: float
    score: float
    coarse_thickness_A: float
    message: str
    node_seed_A: float = float("nan")
    coarse_defocus_A: float = float("nan")
    defocus1_A: float = float("nan")
    defocus2_A: float = float("nan")
    astigmatism_angle_rad: float = float("nan")
    amplitude_contrast: float = float("nan")
    initial_good_fit_resolution_A: float = 0.0
    final_good_fit_resolution_A: float = 0.0
    initial_last_good_bin: int = 0
    final_last_good_bin: int = 0
    debug: Optional[dict[str, object]] = None
    final_epa: Optional["_EPAStatistics"] = None
    timings: Optional[dict[str, float]] = None


@dataclass
class _ThicknessPreparation:
    spectrum: torch.Tensor
    fitting_pixel_size_A: float
    initial_epa: "_EPAStatistics"
    node_seed_A: float
    observed_detrended_np: np.ndarray
    polynomial_coefficients: np.ndarray
    coarse_thickness_A: float
    coarse_profile_defocus_A: float
    coarse_score: float
    coarse_defocus1_A: float
    coarse_defocus2_A: float
    initial_astigmatism_angle_rad: float
    one_d_grid_debug: Optional[dict[str, np.ndarray]]
    timings: dict[str, float]



@dataclass
class _CTFFIND5TiltData:
    power_values: torch.Tensor
    frequency_squared_Ainv2: torch.Tensor
    azimuth_rad: torch.Tensor
    centers_x_A: torch.Tensor
    centers_y_A: torch.Tensor
    valid_mask: torch.Tensor
    rms: np.ndarray
    centers_x_A_numpy: np.ndarray
    centers_y_A_numpy: np.ndarray
    grid_y: np.ndarray
    grid_x: np.ndarray
    fitting_pixel_size_A: float



@dataclass
class _FilteredSpectrumBundle:
    """Outputs of CTFFIND5 ComputeFilteredAmplitudeSpectrumFull2D."""

    raw_amplitude: torch.Tensor
    normalized_cross_capped: torch.Tensor
    background: torch.Tensor
    filtered_unmasked: torch.Tensor
    filtered_masked: torch.Tensor
    fitting_pixel_size_A: float
    timings: Optional[dict[str, float]] = None


@dataclass
class _EPAStatistics:
    spatial_frequency_Ainv: np.ndarray
    observed_profile: np.ndarray
    renormalized_profile: np.ndarray
    theoretical_profile: np.ndarray
    number_of_extrema_profile: np.ndarray
    fit_frc: np.ndarray
    fit_frc_sigma: np.ndarray
    first_fit_bin: int
    last_good_bin: int
    good_fit_resolution_A: float
    profile_azimuth_rad: float
    profile_defocus_A: float
    pre_phase_rad: np.ndarray
    pre_values: np.ndarray
    pre_counts: np.ndarray
    post_phase_rad: np.ndarray
    post_values: np.ndarray
    post_counts: np.ndarray


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


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _electron_wavelength_A(acceleration_voltage_kV: float) -> float:
    voltage_V = 1000.0 * acceleration_voltage_kV
    return 12.26 / math.sqrt(voltage_V + 0.9784 * voltage_V * voltage_V / 1.0e6)


def _amplitude_contrast_phase(amplitude_contrast: float) -> float:
    # Deliberately follows CTFFIND 4.1.8 ctf.cpp exactly.
    return math.atan(amplitude_contrast / math.sqrt(1.0 - amplitude_contrast))


def _amplitude_contrast_phase_tensor(amplitude_contrast: torch.Tensor) -> torch.Tensor:
    """Source-faithful tensor amplitude-contrast phase for CTFFIND5 nodes.

    This helper is intentionally confined to the optional thickness path.  The
    ordinary CTFFIND path retains its previously validated numerical behavior.
    """
    value = amplitude_contrast.clamp(0.0, 1.0 - 1.0e-7)
    return torch.atan(
        value / torch.sqrt((1.0 - value.square()).clamp_min(1.0e-12))
    )


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
    """Compatibility wrapper returning the unmasked filtered spectrum."""
    if micrograph.ndim != 2:
        raise ValueError("Expected one 2-D micrograph")
    bundle = _ctffind_preprocess_bundle_batch(
        micrograph[None], pixel_size_A, config
    )
    return bundle.filtered_unmasked[0], bundle.fitting_pixel_size_A


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


def _savgol_7_2_interp(values: np.ndarray) -> np.ndarray:
    """Seven-point quadratic Savitzky-Golay smoothing without SciPy.

    The interior coefficients are exact.  The first/last three values use the
    same local quadratic extrapolation idea as scipy's ``mode="interp"``.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size < 7:
        return array.copy()
    out = np.empty_like(array)
    kernel = np.asarray([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0]) / 21.0
    out[3:-3] = np.convolve(array, kernel, mode="valid")
    x = np.arange(7, dtype=np.float64)
    left_coeff = np.polyfit(x, array[:7], 2)
    out[:3] = np.polyval(left_coeff, x[:3])
    right_coeff = np.polyfit(x, array[-7:], 2)
    out[-3:] = np.polyval(right_coeff, x[-3:])
    return out


def _smooth_extrema_envelope(
    point_x: list[float],
    point_y: list[float],
    target_x: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    if len(point_x) <= 7:
        return fallback
    values = np.asarray(point_y, dtype=np.float64)
    smoothed = _savgol_7_2_interp(values)
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


def _phase_shift_extremum_sq_frequency_Ainv2(
    defocus_A: float,
    wavelength_A: float,
    spherical_aberration_A: float,
) -> float:
    if defocus_A <= 0.0:
        return 0.0
    denominator = wavelength_A * wavelength_A * spherical_aberration_A
    if denominator <= 0.0:
        return 9999.999
    return defocus_A / denominator


def _phase_given_sq_frequency_and_defocus(
    squared_frequency_Ainv2: np.ndarray | float,
    defocus_A: float,
    wavelength_A: float,
    spherical_aberration_A: float,
    amplitude_phase_rad: float,
    phase_shift_rad: float,
) -> np.ndarray:
    s2 = np.asarray(squared_frequency_Ainv2, dtype=np.float64)
    return (
        PI * wavelength_A * s2
        * (defocus_A - 0.5 * wavelength_A * wavelength_A * s2 * spherical_aberration_A)
        + phase_shift_rad + amplitude_phase_rad
    )


def _return_azimuth_for_1d_plots(astigmatism_angle_rad: float) -> float:
    minimum = 10.0 * PI / 180.0
    azimuth = float(astigmatism_angle_rad) + 0.25 * PI
    angular_distance = math.fmod(azimuth, 0.5 * PI)
    if abs(angular_distance) < minimum:
        azimuth = minimum if angular_distance > 0.0 else -minimum
    if abs(angular_distance) > 0.5 * PI - minimum:
        azimuth = (
            0.5 * PI - minimum if angular_distance > 0.0
            else -0.5 * PI + minimum
        )
    return azimuth


def _curve_linear_deposit(
    x: np.ndarray,
    values: np.ndarray,
    x_min: float,
    x_max: float,
    number_of_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    number_of_points = max(2, int(number_of_points))
    if not math.isfinite(x_min) or not math.isfinite(x_max) or x_max <= x_min:
        x_min = float(np.nanmin(x)) if np.size(x) else 0.0
        x_max = float(np.nanmax(x)) if np.size(x) else x_min + 1.0
        if x_max <= x_min:
            x_max = x_min + 1.0
    axis = np.linspace(x_min, x_max, number_of_points, dtype=np.float64)
    position = (np.asarray(x, dtype=np.float64) - x_min) * (
        (number_of_points - 1) / (x_max - x_min)
    )
    lower = np.floor(position).astype(np.int64)
    fraction = position - lower
    sums = np.zeros(number_of_points, dtype=np.float64)
    counts = np.zeros(number_of_points, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    valid0 = np.isfinite(position) & np.isfinite(data) & (lower >= 0) & (lower < number_of_points)
    np.add.at(sums, lower[valid0], data[valid0] * (1.0 - fraction[valid0]))
    np.add.at(counts, lower[valid0], 1.0 - fraction[valid0])
    upper = lower + 1
    valid1 = np.isfinite(position) & np.isfinite(data) & (upper >= 0) & (upper < number_of_points)
    np.add.at(sums, upper[valid1], data[valid1] * fraction[valid1])
    np.add.at(counts, upper[valid1], fraction[valid1])
    average = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0.0)
    # Empty phase bins can occur at the very ends.  Curve::ReturnLinearInterpolation
    # is better approximated by interpolating across populated neighbours than by
    # introducing artificial zero spikes.
    populated = counts > 0.0
    if np.count_nonzero(populated) >= 2:
        average[~populated] = np.interp(axis[~populated], axis[populated], average[populated])
    elif np.count_nonzero(populated) == 1:
        average[~populated] = average[populated][0]
    return axis, average, counts


def _compute_equi_phase_average(
    spectrum: torch.Tensor,
    fitting_pixel_size_A: float,
    defocus1_A: float,
    defocus2_A: float,
    astigmatism_angle_rad: float,
    wavelength_A: float,
    spherical_aberration_A: float,
    amplitude_phase_rad: float,
    phase_shift_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port CTFFIND's pre/post-maximum equi-phase deposition."""
    array = spectrum.detach().cpu().numpy().astype(np.float64, copy=False)
    size = int(array.shape[0])
    center = size // 2
    coords = (np.arange(size, dtype=np.float64) - center) / (
        float(size) * float(fitting_pixel_size_A)
    )
    fy, fx = np.meshgrid(coords, coords, indexing="ij")
    s2 = fx * fx + fy * fy
    azimuth = np.arctan2(fy, fx)
    effective_defocus = 0.5 * (
        defocus1_A + defocus2_A
        + np.cos(2.0 * (azimuth - astigmatism_angle_rad))
        * (defocus1_A - defocus2_A)
    )
    phase = (
        PI * wavelength_A * s2
        * (effective_defocus - 0.5 * wavelength_A * wavelength_A * s2 * spherical_aberration_A)
        + phase_shift_rad + amplitude_phase_rad
    )
    extremum_s2 = np.where(
        effective_defocus <= 0.0,
        0.0,
        effective_defocus / max(wavelength_A * wavelength_A * spherical_aberration_A, 1.0e-30),
    )

    def phase_maximum(defocus: float) -> float:
        sq = _phase_shift_extremum_sq_frequency_Ainv2(
            defocus, wavelength_A, spherical_aberration_A
        )
        return float(_phase_given_sq_frequency_and_defocus(
            sq, defocus, wavelength_A, spherical_aberration_A,
            amplitude_phase_rad, phase_shift_rad
        ))

    maximum_aberration = max(phase_maximum(defocus1_A), phase_maximum(defocus2_A))
    max_s2 = 0.5 / (float(fitting_pixel_size_A) ** 2)
    edge_phase_1 = float(_phase_given_sq_frequency_and_defocus(
        max_s2, defocus1_A, wavelength_A, spherical_aberration_A,
        amplitude_phase_rad, phase_shift_rad
    ))
    edge_phase_2 = float(_phase_given_sq_frequency_and_defocus(
        max_s2, defocus2_A, wavelength_A, spherical_aberration_A,
        amplitude_phase_rad, phase_shift_rad
    ))
    maximum_abs_edge = max(abs(edge_phase_1), abs(edge_phase_2), 1.0e-6)
    minimum_edge = min(edge_phase_1, edge_phase_2)
    maximum_diagonal_radius = math.sqrt(float(center * center + center * center))
    oversampling = 3.0
    number_pre = max(2, int(math.floor(
        maximum_diagonal_radius * oversampling
        * maximum_aberration / maximum_abs_edge + 0.5
    )))
    number_post = max(2, int(math.floor(
        maximum_diagonal_radius * oversampling + 0.5
    )))
    pre_min = float(phase_shift_rad)
    pre_max = max(float(maximum_aberration), pre_min + 1.0e-6)
    post_min = min(
        float(maximum_aberration),
        float(minimum_edge - 0.5 * abs(minimum_edge)),
    )
    post_max = max(float(maximum_aberration), post_min + 1.0e-6)

    pre_mask = s2 <= extremum_s2
    post_mask = ~pre_mask
    pre_axis, pre_values, pre_counts = _curve_linear_deposit(
        phase[pre_mask], array[pre_mask], pre_min, pre_max, number_pre
    )
    post_axis, post_values, post_counts = _curve_linear_deposit(
        phase[post_mask], array[post_mask], post_min, post_max, number_post
    )
    return pre_axis, pre_values, pre_counts, post_axis, post_values, post_counts


def _ctf_profile_number_of_extrema(
    phase: np.ndarray,
    squared_frequency: np.ndarray,
    extremum_squared_frequency: float,
) -> np.ndarray:
    raw = np.floor(np.asarray(phase) / PI + 0.5).astype(np.int64)
    if extremum_squared_frequency <= 0.0:
        return np.abs(raw)
    phase_at_extremum = np.interp(
        extremum_squared_frequency,
        np.asarray(squared_frequency, dtype=np.float64),
        np.asarray(phase, dtype=np.float64),
    )
    count_at_extremum = int(math.floor(phase_at_extremum / PI + 0.5))
    output = raw.copy()
    post = np.asarray(squared_frequency) > extremum_squared_frequency
    output[post] = count_at_extremum + np.abs(raw[post] - count_at_extremum)
    return np.abs(output)


def _rank_sort(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(order.size, dtype=np.int64)
    return ranks


def _renormalize_1d_spectrum_for_frc(
    average: np.ndarray,
    fit: np.ndarray,
    number_of_extrema_profile: np.ndarray,
) -> np.ndarray:
    result = np.asarray(average, dtype=np.float64).copy()
    fit = np.asarray(fit, dtype=np.float64)
    extrema = np.asarray(number_of_extrema_profile)
    n = result.size
    previous = 0
    current = 0
    for bin_counter in range(1, n):
        delta = extrema[bin_counter] - extrema[bin_counter - 1]
        if 0.9 <= delta <= 1.9:
            current = bin_counter - 1
            if previous > 0 and current > previous + 1:
                zero_bin = previous + (current - previous) // 2
                for i in range(max(previous, 1), min(current, n - 1)):
                    if fit[i] < fit[i - 1] and fit[i] < fit[i + 1]:
                        zero_bin = i
                if extrema[bin_counter] < 7:
                    first = result[previous:zero_bin + 1]
                    if first.size > 1:
                        ranks = _rank_sort(first)
                        result[previous:zero_bin + 1] = np.sin(
                            ranks.astype(np.float64) / float(first.size - 1) * PI * 0.5
                        )
                    second = result[zero_bin + 1:current]
                    if second.size:
                        ranks = _rank_sort(second)
                        result[zero_bin + 1:current] = np.sin(
                            (ranks.astype(np.float64) + 1.0)
                            / float(second.size + 1) * PI * 0.5
                        )
                else:
                    segment = result[previous:current]
                    if segment.size:
                        lo = float(np.min(segment))
                        hi = float(np.max(segment))
                        result[previous:current] = (
                            (segment - lo) / (hi - lo)
                            if hi - lo > 1.0e-4 else segment - lo
                        )
            previous = current
    return result


def _compute_frc_ctffind5(
    average: np.ndarray,
    fit: np.ndarray,
    number_of_extrema_profile: np.ndarray,
    first_fit_bin: int,
    *,
    node_mode: bool,
) -> tuple[np.ndarray, np.ndarray]:
    average = np.asarray(average, dtype=np.float64)
    fit = np.asarray(fit, dtype=np.float64)
    extrema = np.asarray(number_of_extrema_profile, dtype=np.float64)
    n = int(average.size)
    frc = np.zeros(n, dtype=np.float64)
    sigma = np.zeros(n, dtype=np.float64)
    if n < 3:
        return frc, sigma
    minimum = max(1, n // 40)
    half = np.full(n, minimum, dtype=np.int64)
    previous = 0
    for counter in range(1, n):
        if extrema[counter] != extrema[counter - 1]:
            width = max(
                minimum,
                int((1.0 + 0.1 * float(extrema[counter]))
                    * float(counter - previous + 1)),
            )
            width = min(width, max(1, n // 2 - 1))
            half[previous:counter] = width
            previous = counter
    half[0] = half[1]
    fill_width = half[max(0, previous - 1)]
    half[previous:] = fill_width
    if node_mode:
        half = np.maximum(1, np.floor(half.astype(np.float64) * 1.5).astype(np.int64))
        half = np.minimum(half, max(1, n // 2 - 1))
    first_fit_bin = int(np.clip(first_fit_bin, 0, n - 1))
    for counter in range(n):
        if counter < first_fit_bin:
            frc[counter] = 1.0
            continue
        h = int(half[counter])
        first = counter - h
        last = counter + h
        if first < first_fit_bin:
            first = first_fit_bin
            last = first + 2 * h + 1
        if last >= n:
            last = n - 1
            first = last - 2 * h - 1
            if node_mode and first < first_fit_bin:
                first = first_fit_bin
        first = max(first_fit_bin, first, 0)
        last = min(n - 1, max(first, last))
        # Preserve CTFFIND's denominator convention even when an edge-shifted
        # inclusive slice contains one extra value.
        nominal_count = float(2 * h + 1)
        a = average[first:last + 1]
        b = fit[first:last + 1]
        am = float(np.sum(a)) / nominal_count
        bm = float(np.sum(b)) / nominal_count
        da = a - am
        db = b - bm
        cross = float(np.sum(da * db))
        va = float(np.sum(da * da))
        vb = float(np.sum(db * db))
        if va > 0.0 and vb > 0.0:
            frc[counter] = cross / math.sqrt(va * vb)
            frc[counter] = float(np.clip(frc[counter], -1.0, 1.0))
        sigma[counter] = 2.0 / math.sqrt(nominal_count)
    return frc, sigma


def _ctffind5_good_fit_cutoff(frc: np.ndarray) -> int:
    n = int(len(frc))
    if n == 0:
        return 0
    first = int(0.1 * n)
    first = min(max(first, 0), n - 1)
    above_low = 0
    above_sig = 0
    above_high = 0
    last = -1
    for counter in range(first, n):
        if ((above_low > 3 and frc[counter] < 0.1)
                or (above_high > 3 and frc[counter] < 0.5)):
            last = counter
            break
        if frc[counter] > 0.1:
            above_low += 1
        if frc[counter] > 0.5:
            above_sig += 1
        if frc[counter] > 0.66:
            above_high += 1
    if above_sig == n - first:
        last = n - 1
    if above_sig == 0:
        last = 1 if n > 1 else 0
    if last < 0:
        last = n - 1
    return int(np.clip(last, 0, n - 1))


def _compute_epa_statistics(
    masked_spectrum: torch.Tensor,
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
    theoretical_thickness_A: float | None = None,
    node_mode: bool = False,
    rounded_square: bool = False,
) -> _EPAStatistics:
    pre_x, pre_y, pre_count, post_x, post_y, post_count = _compute_equi_phase_average(
        masked_spectrum, fitting_pixel_size_A,
        defocus1_A, defocus2_A, astigmatism_angle_rad,
        wavelength_A, spherical_aberration_A,
        amplitude_phase_rad, phase_shift_rad,
    )
    size = int(masked_spectrum.shape[-1])
    center = size // 2
    number_of_bins = int(math.ceil(math.sqrt(center * center + center * center)))
    frequencies = np.arange(number_of_bins, dtype=np.float64) / (
        float(size) * float(fitting_pixel_size_A)
    )
    profile_azimuth = _return_azimuth_for_1d_plots(astigmatism_angle_rad)
    profile_defocus = _defocus_at_azimuth_A(
        defocus1_A, defocus2_A, astigmatism_angle_rad, profile_azimuth
    )
    s2 = frequencies * frequencies
    phase = _phase_given_sq_frequency_and_defocus(
        s2, profile_defocus, wavelength_A, spherical_aberration_A,
        amplitude_phase_rad, phase_shift_rad
    )
    extremum_s2 = _phase_shift_extremum_sq_frequency_Ainv2(
        profile_defocus, wavelength_A, spherical_aberration_A
    )
    observed = np.empty(number_of_bins, dtype=np.float64)
    pre = s2 <= extremum_s2
    observed[pre] = np.interp(phase[pre], pre_x, pre_y)
    observed[~pre] = np.interp(phase[~pre], post_x, post_y)

    if theoretical_thickness_A is None:
        theoretical = np.abs(np.sin(phase))
        extrema = _ctf_profile_number_of_extrema(phase, s2, extremum_s2)
    else:
        argument = PI * wavelength_A * s2 * float(theoretical_thickness_A)
        if rounded_square:
            argument_t = torch.as_tensor(argument, dtype=torch.float64)
            modulation = _rounded_square_torch(argument_t).cpu().numpy()
        else:
            modulation = np.sinc(
                wavelength_A * s2 * float(theoretical_thickness_A)
            )
        theoretical = 0.5 * (1.0 - modulation * np.cos(2.0 * phase))
        # CTFFIND5 rebuilds extrema from maxima of the thickness profile.
        extrema = np.zeros(number_of_bins, dtype=np.int64)
        count = 0
        previous_slope = 0
        for i in range(1, number_of_bins):
            slope = 1 if theoretical[i] > theoretical[i - 1] else -1
            if slope - previous_slope == -2:
                count += 1
            extrema[i] = count
            previous_slope = slope

    renormalized = _renormalize_1d_spectrum_for_frc(
        observed, theoretical, extrema
    )
    lowest_frequency = 1.0 / float(config.minimum_resolution_A)
    first_fit_bin = int(np.searchsorted(frequencies, lowest_frequency, side="left"))
    first_fit_bin = min(max(first_fit_bin, 0), number_of_bins - 1)
    frc, frc_sigma = _compute_frc_ctffind5(
        renormalized, theoretical, extrema, first_fit_bin,
        node_mode=node_mode,
    )
    last_good = _ctffind5_good_fit_cutoff(frc)
    good_resolution = (
        1.0 / frequencies[last_good]
        if last_good > 0 and frequencies[last_good] > 0.0 else 0.0
    )
    return _EPAStatistics(
        spatial_frequency_Ainv=frequencies,
        observed_profile=observed,
        renormalized_profile=renormalized,
        theoretical_profile=theoretical,
        number_of_extrema_profile=extrema,
        fit_frc=frc,
        fit_frc_sigma=frc_sigma,
        first_fit_bin=first_fit_bin,
        last_good_bin=last_good,
        good_fit_resolution_A=float(good_resolution),
        profile_azimuth_rad=float(profile_azimuth),
        profile_defocus_A=float(profile_defocus),
        pre_phase_rad=pre_x,
        pre_values=pre_y,
        pre_counts=pre_count,
        post_phase_rad=post_x,
        post_values=post_y,
        post_counts=post_count,
    )


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


def _ctffind_raw_amplitude_batch(
    micrographs: torch.Tensor,
    pixel_size_A: float,
    config: CtffindConfig,
) -> tuple[torch.Tensor, float]:
    """Build a centered raw amplitude spectrum before CTFFIND filtering."""
    images = _center_pad_to_even_square_batch(micrographs)
    amplitudes = torch.fft.fftshift(
        torch.fft.fft2(images).abs(), dim=(-2, -1)
    )
    center = amplitudes.shape[-1] // 2
    amplitudes[:, center, center] = 0.0

    fitting_pixel_size_A = float(pixel_size_A)
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
    spectra[:, config.box_size // 2, config.box_size // 2] = 0.0
    return spectra.contiguous(), float(fitting_pixel_size_A)


def _inverted_cosine_mask_with_annulus_mean_batch(
    spectra: torch.Tensor,
    wanted_mask_radius_pixels: float,
    wanted_mask_edge_pixels: float,
) -> torch.Tensor:
    """Match Image::CosineMask(..., invert=true) for centered real images."""
    if spectra.ndim != 3:
        raise ValueError("Expected spectra[B,N,N]")
    size = int(spectra.shape[-1])
    center = size // 2
    y = torch.arange(size, dtype=spectra.dtype, device=spectra.device) - center
    x = torch.arange(size, dtype=spectra.dtype, device=spectra.device) - center
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    inner = max(0.0, float(wanted_mask_radius_pixels) - 0.5 * float(wanted_mask_edge_pixels))
    outer = inner + max(float(wanted_mask_edge_pixels), 1.0e-6)
    annulus = (radius >= inner) & (radius <= outer)
    if not bool(annulus.any()):
        return spectra.clone()
    fill = spectra[:, annulus].mean(dim=1)
    output = spectra.clone()
    inside = radius <= inner
    output[:, inside] = fill[:, None]
    transition = (radius > inner) & (radius < outer)
    if bool(transition.any()):
        edge = 0.5 * (
            1.0 + torch.cos(PI * (radius[transition] - inner) / (outer - inner))
        )
        output[:, transition] = (
            spectra[:, transition] * (1.0 - edge)[None]
            + fill[:, None] * edge[None]
        )
    return output


def _compute_filtered_amplitude_spectrum_full_2d_batch(
    raw_amplitudes: torch.Tensor,
    fitting_pixel_size_A: float,
    config: CtffindConfig,
    *,
    apply_cosine_mask: bool = True,
) -> _FilteredSpectrumBundle:
    """Source-order port of ComputeFilteredAmplitudeSpectrumFull2D.

    This function is shared by the whole-micrograph and tilt-corrected raw
    spectrum producers.  It returns both the ordinary background-subtracted
    spectrum and CTFFIND5's low-resolution-masked fitting spectrum.
    """
    if raw_amplitudes.ndim == 2:
        spectra = raw_amplitudes[None]
    elif raw_amplitudes.ndim == 3:
        spectra = raw_amplitudes
    else:
        raise ValueError("raw_amplitudes must be [N,N] or [B,N,N]")
    size = int(spectra.shape[-1])
    if spectra.shape[-2] != size or size != int(config.box_size):
        raise ValueError(
            f"Expected {config.box_size}x{config.box_size} raw spectra, "
            f"got {tuple(spectra.shape[-2:])}"
        )
    raw = spectra.clone()
    center = size // 2
    raw[:, center, center] = 0.0
    minimum_radius = (
        float(size) * float(fitting_pixel_size_A)
        / float(config.minimum_resolution_A)
    )
    mean, sigma = _compute_spectrum_mean_sigma_batch(
        raw,
        minimum_radius_pixels=minimum_radius,
        maximum_radius_pixels=float(size),
        cross_half_width=12,
    )
    normalized = raw / sigma[:, None, None]
    cross_maximum = mean / sigma + 10.0
    normalized = normalized.clone()
    normalized[:, center, :] = torch.minimum(
        normalized[:, center, :], cross_maximum[:, None]
    )
    normalized[:, :, center] = torch.minimum(
        normalized[:, :, center], cross_maximum[:, None]
    )

    convolution_box_size = int(
        float(size) * float(fitting_pixel_size_A)
        / float(config.minimum_resolution_A) * math.sqrt(2.0)
    )
    if convolution_box_size % 2 == 0:
        convolution_box_size += 1
    convolution_box_size = max(1, convolution_box_size)
    if convolution_box_size >= size:
        raise RuntimeError(
            f"Background box ({convolution_box_size}) is not smaller than "
            f"spectrum box ({size})"
        )
    background = _spectrum_box_convolution_batch(
        normalized, convolution_box_size, minimum_radius
    )
    filtered = normalized - background

    # Image::ReturnMaximumValue(3,3) excludes the edges and both center axes.
    coords = torch.arange(size, device=filtered.device)
    valid_coord = (
        (coords >= 3) & (coords <= size - 4)
        & (torch.abs(coords - center) >= 3)
    )
    valid2d = valid_coord[:, None] & valid_coord[None, :]
    threshold = filtered[:, valid2d].amax(dim=1)
    filtered = torch.minimum(filtered, threshold[:, None, None])

    masked = filtered.clone()
    if apply_cosine_mask:
        wanted_radius = (
            float(size) * float(fitting_pixel_size_A)
            / max(float(config.maximum_resolution_A), 8.0)
        )
        wanted_edge = (
            float(size) * float(fitting_pixel_size_A)
            / max(float(config.maximum_resolution_A), 4.0)
        )
        masked = _inverted_cosine_mask_with_annulus_mean_batch(
            masked, wanted_radius, wanted_edge
        )

    return _FilteredSpectrumBundle(
        raw_amplitude=raw.contiguous(),
        normalized_cross_capped=normalized.contiguous(),
        background=background.contiguous(),
        filtered_unmasked=filtered.contiguous(),
        filtered_masked=masked.contiguous(),
        fitting_pixel_size_A=float(fitting_pixel_size_A),
    )


def _ctffind_preprocess_bundle_batch(
    micrographs: torch.Tensor,
    pixel_size_A: float,
    config: CtffindConfig,
) -> _FilteredSpectrumBundle:
    raw, fitting_pixel_size_A = _ctffind_raw_amplitude_batch(
        micrographs, pixel_size_A, config
    )
    return _compute_filtered_amplitude_spectrum_full_2d_batch(
        raw, fitting_pixel_size_A, config, apply_cosine_mask=True
    )


def _ctffind_preprocess_batch(
    micrographs: torch.Tensor,
    pixel_size_A: float,
    config: CtffindConfig,
) -> tuple[torch.Tensor, float]:
    """Compatibility wrapper preserving the validated ordinary CTF fit input."""
    bundle = _ctffind_preprocess_bundle_batch(micrographs, pixel_size_A, config)
    return bundle.filtered_unmasked, bundle.fitting_pixel_size_A


def _ctffind_filter_centered_amplitude_batch(
    amplitudes: torch.Tensor,
    fitting_pixel_size_A: float,
    config: CtffindConfig,
) -> torch.Tensor:
    """Compatibility wrapper around the shared full-2D filter."""
    return _compute_filtered_amplitude_spectrum_full_2d_batch(
        amplitudes, fitting_pixel_size_A, config, apply_cosine_mask=True
    ).filtered_unmasked


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



def _square_grid_tile_metadata(
    image_shape: tuple[int, int],
    tile_size: int,
    stride: int,
    pixel_size_A: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return full-data tiles on CTFFIND5's mean-padded even-square grid.

    The C++ driver centers the rectangular micrograph in an even square whose
    side is the larger input dimension.  Tiles that would include only/partly
    mean padding carry no useful Thon-ring signal, so this virtual-grid version
    omits them without allocating the often very large padded image.

    x is positive to the right and y follows image row order (positive down).
    Together with _ctffind5_gradient_from_axis_angle this reproduces the
    cisTEM/CTFTILT local-defocus plane convention.
    """
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height < tile_size or width < tile_size:
        raise ValueError(
            f"Micrograph {width}x{height} is smaller than tile size {tile_size}"
        )
    square_size = max(height, width)
    if square_size % 2:
        square_size += 1
    pad_y = (square_size - height) // 2
    pad_x = (square_size - width) // 2
    ny = 1 + (square_size - tile_size) // stride
    nx = 1 + (square_size - tile_size) // stride
    covered_h = tile_size + (ny - 1) * stride
    covered_w = tile_size + (nx - 1) * stride
    square_y_offset = (square_size - covered_h) // 2
    square_x_offset = (square_size - covered_w) // 2

    y0_list: list[int] = []
    x0_list: list[int] = []
    centers_x: list[float] = []
    centers_y: list[float] = []
    grid_y: list[int] = []
    grid_x: list[int] = []
    square_center = 0.5 * float(square_size)
    for iy in range(ny):
        sy0 = square_y_offset + iy * stride
        y0 = sy0 - pad_y
        if y0 < 0 or y0 + tile_size > height:
            continue
        for ix in range(nx):
            sx0 = square_x_offset + ix * stride
            x0 = sx0 - pad_x
            if x0 < 0 or x0 + tile_size > width:
                continue
            y0_list.append(y0)
            x0_list.append(x0)
            centers_x.append(
                (sx0 + 0.5 * tile_size - square_center) * pixel_size_A
            )
            centers_y.append(
                (sy0 + 0.5 * tile_size - square_center) * pixel_size_A
            )
            grid_y.append(iy)
            grid_x.append(ix)
    if not y0_list:
        raise RuntimeError("No complete tiles remain on the square micrograph grid")
    return (
        np.asarray(y0_list, dtype=np.int64),
        np.asarray(x0_list, dtype=np.int64),
        np.asarray(centers_x, dtype=np.float64),
        np.asarray(centers_y, dtype=np.float64),
        np.asarray(grid_y, dtype=np.int64),
        np.asarray(grid_x, dtype=np.int64),
        square_size,
    )


def _detrend_tile_batch(tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if tiles.ndim != 3 or tiles.shape[-1] != tiles.shape[-2]:
        raise ValueError("tiles must have shape [B,N,N]")
    tile_size = int(tiles.shape[-1])
    coord = torch.linspace(
        -1.0, 1.0, tile_size, dtype=tiles.dtype, device=tiles.device
    )
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    mean = tiles.mean(dim=(1, 2), keepdim=True)
    centered = tiles - mean
    denom_x = torch.sum(xx.square()).clamp_min(1.0e-20)
    denom_y = torch.sum(yy.square()).clamp_min(1.0e-20)
    slope_x = torch.sum(centered * xx[None], dim=(1, 2), keepdim=True) / denom_x
    slope_y = torch.sum(centered * yy[None], dim=(1, 2), keepdim=True) / denom_y
    detrended = centered - slope_x * xx[None] - slope_y * yy[None]
    rms = torch.sqrt(torch.mean(detrended.square(), dim=(1, 2)))
    return detrended.contiguous(), rms


def _extract_detrended_tiles(
    micrograph: np.ndarray,
    tile_size: int,
    stride: int,
    pixel_size_A: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract complete tiles on CTFFIND5's centered even-square grid."""
    image = np.asarray(micrograph, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Tilt fitting expects one 2-D micrograph")
    y0, x0, centers_x, centers_y, grid_y, grid_x, _ = _square_grid_tile_metadata(
        image.shape, tile_size, stride, pixel_size_A
    )
    arrays = np.stack(
        [image[y:y + tile_size, x:x + tile_size] for y, x in zip(y0, x0)],
        axis=0,
    ).astype(np.float32, copy=False)
    tiles = torch.as_tensor(arrays, dtype=dtype, device=device)
    detrended, rms_t = _detrend_tile_batch(tiles)
    return (
        detrended,
        centers_x,
        centers_y,
        grid_y,
        grid_x,
        rms_t.detach().cpu().numpy().astype(np.float64, copy=False),
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


def _write_png_rgb(path: Path, rgb: np.ndarray) -> None:
    """Write an 8-bit RGB PNG using only the Python standard library."""
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image must have shape HxWx3")
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw, level=6))
    payload += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _write_tilt_png(path: Path, details: _TiltFitDetails) -> None:
    """Write a dependency-free tilt-plane/residual diagnostic PNG."""
    height, width = 420, 920
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    panel_w = 400
    panel_h = 340
    top = 40
    left0 = 35
    left1 = 485
    xA = np.asarray(details.tile_centers_x_A, dtype=np.float64)
    yA = np.asarray(details.tile_centers_y_A, dtype=np.float64)
    if xA.size == 0 or yA.size == 0:
        _write_png_rgb(path, canvas)
        return
    x_min, x_max = float(np.min(xA)), float(np.max(xA))
    y_min, y_max = float(np.min(yA)), float(np.max(yA))
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    yy, xx = np.meshgrid(
        np.linspace(y_min, y_max, panel_h),
        np.linspace(x_min, x_max, panel_w),
        indexing="ij",
    )
    center_mean = 0.5 * (details.center_defocus1_A + details.center_defocus2_A)
    plane = center_mean + details.gradient_x * xx + details.gradient_y * yy
    lo, hi = np.nanpercentile(plane, [2.0, 98.0])
    scale = np.clip((plane - lo) / max(hi - lo, 1.0e-12), 0.0, 1.0)
    heat = np.empty((panel_h, panel_w, 3), dtype=np.uint8)
    heat[..., 0] = np.rint(255.0 * scale).astype(np.uint8)
    heat[..., 1] = np.rint(220.0 * (1.0 - np.abs(2.0 * scale - 1.0))).astype(np.uint8)
    heat[..., 2] = np.rint(255.0 * (1.0 - scale)).astype(np.uint8)
    canvas[top:top + panel_h, left0:left0 + panel_w] = heat
    canvas[top:top + panel_h, left1:left1 + panel_w] = 238

    px = np.rint((xA - x_min) / (x_max - x_min) * (panel_w - 1)).astype(int)
    py = np.rint((yA - y_min) / (y_max - y_min) * (panel_h - 1)).astype(int)
    residual = np.asarray(details.tile_residual_A, dtype=np.float64)
    inlier = np.asarray(details.tile_plane_inlier, dtype=bool)
    finite = np.isfinite(residual) & inlier
    limit = float(np.nanpercentile(np.abs(residual[finite]), 95.0)) if np.any(finite) else 1.0
    limit = max(limit, 1.0)
    for xpix, ypix, value, good in zip(px, py, residual, inlier):
        xpix = int(np.clip(xpix, 0, panel_w - 1))
        ypix = int(np.clip(ypix, 0, panel_h - 1))
        if good and math.isfinite(float(value)):
            z = float(np.clip(value / limit, -1.0, 1.0))
            colour = np.array(
                [255 if z >= 0.0 else int(255 * (1.0 + z)),
                 int(255 * (1.0 - abs(z))),
                 255 if z <= 0.0 else int(255 * (1.0 - z))],
                dtype=np.uint8,
            )
        else:
            colour = np.array([90, 90, 90], dtype=np.uint8)
        for panel_left in (left0, left1):
            y0, y1 = max(top, top + ypix - 3), min(top + panel_h, top + ypix + 4)
            x0, x1 = max(panel_left, panel_left + xpix - 3), min(panel_left + panel_w, panel_left + xpix + 4)
            canvas[y0:y1, x0:x1] = colour
    _write_png_rgb(path, canvas)


def _write_extended_results_tsv(path: Path, results: Sequence[CtfFitResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    columns = [
        "micrograph_name", "source_file", "image_index",
        "defocus_u_A", "defocus_v_A", "defocus_angle_deg", "global_cc",
        "global_good_fit_A", "tilt_fitted", "gradient_x", "gradient_y",
        "tilt_angle_deg", "tilt_axis_deg", "coarse_tilt_angle_deg",
        "coarse_tilt_axis_deg", "tilt_score", "tilt_good_fit_A",
        "tile_residual_rms_A", "valid_tiles", "total_tiles", "tilt_png",
        "ice_thickness_fitted", "ice_thickness_A", "ice_thickness_score",
        "tilt_message", "ice_thickness_message",
    ]
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(columns) + "\n")
        for r in results:
            row = [
                r.micrograph_name,
                r.source_file,
                str(r.image_index_1based),
                f"{r.defocus1_A:.6f}",
                f"{r.defocus2_A:.6f}",
                f"{r.astigmatism_angle_deg:.6f}",
                f"{r.score:.8g}",
                f"{(r.global_thon_rings_good_fit_resolution_A if r.global_thon_rings_good_fit_resolution_A > 0.0 else r.thon_rings_good_fit_resolution_A):.6f}",
                "1" if r.tilt_fitted else "0",
                f"{r.defocus_gradient_x:.9g}",
                f"{r.defocus_gradient_y:.9g}",
                f"{r.tilt_angle_deg:.6f}",
                f"{r.tilt_axis_deg:.6f}",
                f"{r.coarse_tilt_angle_deg:.6f}",
                f"{r.coarse_tilt_axis_deg:.6f}",
                f"{r.tilt_score:.8g}",
                f"{r.tilt_good_fit_resolution_A:.6f}",
                f"{r.tilt_residual_rms_A:.6f}",
                str(r.tilt_valid_tiles),
                str(r.tilt_total_tiles),
                r.tilt_png_name,
                "1" if r.ice_thickness_fitted else "0",
                f"{r.ice_thickness_A:.6f}",
                f"{r.ice_thickness_score:.8g}",
                r.tilt_message.replace("\t", " ").replace("\n", " "),
                r.ice_thickness_message.replace("\t", " ").replace("\n", " "),
            ]
            handle.write("\t".join(row) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# CTFFIND5-style local-spectrum tilt and finite-thickness helpers.
# These functions are called only by explicitly selected feature paths.
# ---------------------------------------------------------------------------
def _ctffind5_frequency_support(
    size: int,
    pixel_size_A: float,
    minimum_resolution_A: float,
    maximum_resolution_A: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the half-plane support used for local CTFFIND5 power scoring."""
    center = size // 2
    jj = torch.arange(size, device=device, dtype=dtype)
    ii = torch.arange(center, device=device, dtype=dtype)
    y, x = torch.meshgrid(jj, ii, indexing="ij")
    fx = (x - center) / (float(size) * pixel_size_A)
    fy = (y - center) / (float(size) * pixel_size_A)
    freq2 = fx.square() + fy.square()
    low = 1.0 / minimum_resolution_A
    high = 1.0 / maximum_resolution_A
    cross = max(2, min(6, size // 24))
    mask = (
        (freq2 > low * low)
        & (freq2 < high * high)
        & (x < center - cross)
        & ((y < center - cross) | (y > center + cross))
    )
    flat = mask.reshape(-1)
    if int(flat.sum().item()) == 0:
        raise RuntimeError("The CTFFIND5 local-spectrum support contains no pixels")
    return flat, freq2.reshape(-1)[flat], torch.atan2(
        fy.reshape(-1)[flat], fx.reshape(-1)[flat]
    )


def _ctffind5_prepare_tilt_data(
    micrograph: np.ndarray,
    pixel_size_A: float,
    config: CtffindConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> _CTFFIND5TiltData:
    """Create background-subtracted local *power* spectra for tilt scoring.

    CTFFIND5 compares local Thon-ring power patterns with CTF^2.  The previous
    port incorrectly formed A^2 - smooth(A)^2 from amplitude spectra; here the
    operation is the source-equivalent P - smooth(P).
    """
    tiles, x_A, y_A, grid_y, grid_x, rms = _extract_detrended_tiles(
        micrograph,
        config.tilt_tile_size,
        config.tilt_tile_stride,
        pixel_size_A,
        device,
        dtype,
    )
    if config.tilt_rms_mad_cutoff > 0.0:
        rms_valid_np = _robust_mad_mask(rms, config.tilt_rms_mad_cutoff)
    else:
        rms_valid_np = np.isfinite(rms) & (rms > 1.0e-12)
    # Reject effectively blank/padded/constant patches explicitly.  Do not force
    # outliers back in merely to reach the minimum count.
    positive_rms = np.isfinite(rms) & (rms > max(1.0e-12, np.nanmedian(rms) * 1.0e-4))
    rms_valid_np &= positive_rms
    if int(np.sum(rms_valid_np)) < config.tilt_min_tiles:
        raise RuntimeError(
            f"Only {int(np.sum(rms_valid_np))} local spectra pass RMS filtering; "
            f"need at least {config.tilt_min_tiles}"
        )

    n = int(config.tilt_tile_size)
    fft = torch.fft.fft2(tiles)
    power = torch.fft.fftshift(
        fft.real.square() + fft.imag.square(), dim=(-2, -1)
    ) / float(n * n)
    del fft, tiles
    center = n // 2
    power[:, center, center] = 0.0
    rms_t = torch.as_tensor(rms, device=device, dtype=dtype).clamp_min(1.0e-20)
    power = power / rms_t[:, None, None].square()

    minimum_radius = float(n) * pixel_size_A / config.minimum_resolution_A
    background_box = int(
        float(n) * pixel_size_A / config.minimum_resolution_A * math.sqrt(2.0)
    )
    background_box = max(3, background_box)
    if background_box % 2 == 0:
        background_box += 1
    if background_box >= n:
        background_box = n - 1 if (n - 1) % 2 else n - 2
    background = _spectrum_box_convolution_batch(
        power, background_box, minimum_radius
    )
    filtered_power = power - background

    flat_support, freq2, azimuth = _ctffind5_frequency_support(
        n,
        pixel_size_A,
        config.minimum_resolution_A,
        config.maximum_resolution_A,
        device,
        dtype,
    )
    values = filtered_power[:, :, :center].reshape(filtered_power.shape[0], -1)
    values = values[:, flat_support]
    values = values - values.mean(dim=1, keepdim=True)
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    finite = (
        torch.isfinite(values).all(dim=1)
        & torch.isfinite(norms[:, 0])
        & (norms[:, 0] > 1.0e-20)
    )
    valid = torch.as_tensor(rms_valid_np, device=device, dtype=torch.bool) & finite
    if int(valid.sum().item()) < config.tilt_min_tiles:
        raise RuntimeError(
            f"Only {int(valid.sum().item())} finite local power spectra remain; "
            f"need at least {config.tilt_min_tiles}"
        )
    values = values / norms.clamp_min(1.0e-20)
    return _CTFFIND5TiltData(
        power_values=values.contiguous(),
        frequency_squared_Ainv2=freq2.contiguous(),
        azimuth_rad=azimuth.contiguous(),
        centers_x_A=torch.as_tensor(x_A, device=device, dtype=dtype),
        centers_y_A=torch.as_tensor(y_A, device=device, dtype=dtype),
        valid_mask=valid,
        rms=rms,
        centers_x_A_numpy=x_A,
        centers_y_A_numpy=y_A,
        grid_y=grid_y,
        grid_x=grid_x,
        fitting_pixel_size_A=float(pixel_size_A),
    )


def _ctffind5_bin_micrograph_for_tilt(
    micrograph: np.ndarray,
    pixel_size_A: float,
    tile_size: int,
    device: torch.device,
    dtype: torch.dtype,
    target_pixel_size_A: float,
) -> tuple[np.ndarray, float]:
    """Fourier-bin a mean-padded square image for CTFFIND5 tilt search.

    The previous port incorrectly hard-coded 5 A/pixel.  CTFFIND5 instead
    reduces the image only as far as allowed by the requested fitting
    resolution.  For a highest fitted spatial period R, the largest safe pixel
    size is R/2.  The caller supplies that target (or a stricter override).

    The crop size is rounded *up* to an even integer so the exact effective
    pixel size never becomes coarser than the requested target.
    """
    image = np.asarray(micrograph, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Tilt fitting expects one 2-D micrograph")
    if not math.isfinite(target_pixel_size_A) or target_pixel_size_A <= 0.0:
        raise ValueError("target_pixel_size_A must be positive and finite")

    square = _center_pad_to_even_square(
        torch.as_tensor(
            np.array(image, copy=True, order="C"),
            device=device,
            dtype=dtype,
        )
    )
    input_size = int(square.shape[0])
    if pixel_size_A >= target_pixel_size_A:
        effective_pixel = float(pixel_size_A)
        binned = square
    else:
        required = (
            float(input_size) * float(pixel_size_A) / float(target_pixel_size_A)
        )
        output_size = max(int(tile_size), int(math.ceil(required)))
        if output_size % 2:
            output_size += 1
        output_size = min(input_size, output_size)
        effective_pixel = (
            float(pixel_size_A) * float(input_size) / float(output_size)
        )
        if output_size == input_size:
            binned = square
        else:
            fourier = torch.fft.fftshift(torch.fft.fft2(square))
            start_index = input_size // 2 - output_size // 2
            cropped = fourier[
                start_index:start_index + output_size,
                start_index:start_index + output_size,
            ].clone()
            del fourier, square
            binned = torch.fft.ifft2(torch.fft.ifftshift(cropped)).real
            # Preserve the real-space mean/amplitude convention after changing
            # the number of Fourier samples.
            binned *= (float(output_size) / float(input_size)) ** 2
            del cropped

    result = binned.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.ascontiguousarray(result), effective_pixel


def _ctffind5_initial_power_spectrum(
    micrograph: np.ndarray,
    pixel_size_A: float,
    config: CtffindConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Whole-image background-subtracted 128x128 power spectrum.

    Match the CTFFIND5 order described for ``CTFTilt::CalculatePowerSpectra``:
    mean-pad to an even square, calculate the power spectrum, subtract the
    smooth background at the native reciprocal-space sampling, and only then
    bin the filtered power spectrum to the 128-pixel local-spectrum grid.
    """
    image = np.asarray(micrograph, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Tilt fitting expects one 2-D micrograph")
    square = _center_pad_to_even_square(
        torch.as_tensor(
            np.array(image, copy=True, order="C"),
            device=device,
            dtype=dtype,
        )
    )
    # CTFFIND5 embeds a rectangular micrograph in a mean-filled even square.
    # Removing the DC level after padding is equivalent for non-origin Fourier
    # pixels and prevents the origin from dominating the box convolution.
    square = square - square.mean()
    fft = torch.fft.fft2(square)
    power = torch.fft.fftshift(fft.real.square() + fft.imag.square())
    del fft, square

    full_size = int(power.shape[-1])
    full_center = full_size // 2
    power[full_center, full_center] = 0.0
    minimum_radius = (
        float(full_size) * float(pixel_size_A)
        / float(config.minimum_resolution_A)
    )
    background_box = int(
        minimum_radius * math.sqrt(2.0)
    )
    background_box = max(3, background_box)
    if background_box % 2 == 0:
        background_box += 1
    if background_box >= full_size:
        background_box = full_size - 1 if (full_size - 1) % 2 else full_size - 2
    background = _spectrum_box_convolution_batch(
        power[None], background_box, minimum_radius
    )[0]
    filtered = power - background
    del power, background

    n = int(config.tilt_tile_size)
    # Area binning keeps the same Nyquist extent while reducing the filtered
    # whole-image power spectrum to the sampling of a local n x n spectrum.
    filtered = F.interpolate(
        filtered[None, None], size=(n, n), mode="area"
    )[0, 0]
    center = n // 2
    filtered[center, center] = 0.0

    # Match CTFFIND's protection against isolated detector/cross spikes without
    # clipping the Thon-ring support itself.
    coords = torch.arange(n, device=device)
    valid_coord = (
        (coords >= 3) & (coords <= n - 4)
        & (torch.abs(coords - center) >= 3)
    )
    valid2d = valid_coord[:, None] & valid_coord[None]
    threshold = filtered[valid2d].amax()
    return torch.minimum(filtered, threshold).contiguous()



def _round_half_away_from_zero(value: float) -> int:
    """C/C++ round()/cisTEM myroundint convention for scalar coordinates."""
    return int(math.floor(value + 0.5)) if value >= 0.0 else int(math.ceil(value - 0.5))


def _closest_even_dimension(value: float, minimum: int = 2) -> int:
    """Choose the nearest positive even size, matching PixelSizeForFitting."""
    rounded = _round_half_away_from_zero(float(value))
    if rounded % 2:
        rounded += 1
    alternate = rounded - 2
    if alternate >= minimum and abs(float(alternate) - value) < abs(float(rounded) - value):
        rounded = alternate
    return max(minimum, rounded)


def _extract_centered_patch_with_padding(
    image: np.ndarray,
    output_size: int,
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    """Approximate Image::ClipInto for a centered 2-D patch with zero padding."""
    height, width = image.shape
    center_x = width // 2 + int(offset_x)
    center_y = height // 2 + int(offset_y)
    start_x = center_x - output_size // 2
    start_y = center_y - output_size // 2
    patch = np.zeros((output_size, output_size), dtype=np.float32)
    src_x0 = max(0, start_x)
    src_y0 = max(0, start_y)
    src_x1 = min(width, start_x + output_size)
    src_y1 = min(height, start_y + output_size)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - start_x
        dst_y0 = src_y0 - start_y
        patch[
            dst_y0:dst_y0 + (src_y1 - src_y0),
            dst_x0:dst_x0 + (src_x1 - src_x0),
        ] = image[src_y0:src_y1, src_x0:src_x1]
    return patch


def _cosine_rectangular_mask_batch(
    patches: torch.Tensor,
    *,
    inner_fraction: float = 0.90,
    edge_fraction: float = 0.10,
) -> torch.Tensor:
    """CTFFIND tilt-correction edge taper, blended toward each patch mean."""
    if patches.ndim != 3 or patches.shape[-1] != patches.shape[-2]:
        raise ValueError('patches must have shape [B,N,N]')
    size = int(patches.shape[-1])
    coord = torch.arange(size, device=patches.device, dtype=patches.dtype)
    center = float(size // 2)
    distance = torch.abs(coord - center)
    inner = inner_fraction * center
    edge = max(edge_fraction * float(size), 1.0)
    outer = inner + edge
    one_d = torch.ones_like(coord)
    transition = (distance > inner) & (distance < outer)
    one_d = torch.where(
        transition,
        0.5 * (1.0 + torch.cos(PI * (distance - inner) / edge)),
        one_d,
    )
    one_d = torch.where(distance >= outer, torch.zeros_like(one_d), one_d)
    window = one_d[:, None] * one_d[None, :]
    fill = patches.mean(dim=(1, 2), keepdim=True)
    return (fill + (patches - fill) * window[None]).contiguous()


def _tilt_axis_to_output_convention(axis_deg: float) -> float:
    """Convert array y-down tilt-axis convention to CTFFIND/RELION y-up."""
    return (180.0 - float(axis_deg)) % 360.0


def _ctffind5_correction_grid(
    size: int,
    magnification: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Output-to-input grids that magnify spectra by m=sqrt(df_local/df_avg)."""
    center = float(size // 2)
    coord = torch.arange(size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    m = magnification.to(device=device, dtype=dtype).reshape(-1, 1, 1)
    source_x = center + (xx[None] - center) / m
    source_y = center + (yy[None] - center) / m
    return torch.stack(
        (
            2.0 * source_x / float(size - 1) - 1.0,
            2.0 * source_y / float(size - 1) - 1.0,
        ),
        dim=-1,
    )



# ---------------------------------------------------------------------------
# CTFTilt 0.4 compatibility frontend
# ---------------------------------------------------------------------------
# This block intentionally preserves the local CTFTilt path that was validated
# in CTFFIND5-PyTorch 0.4.0.  The standard/tilt-corrected spectrum backend below
# remains the 0.5.x ComputeFilteredAmplitudeSpectrumFull2D + EPA/FRC/thickness
# implementation.  Keeping the boundary explicit prevents ordinary CTFFIND
# filtering choices from silently changing the 128-pixel local tilt objective.

_V04_EPS = 1.0e-20


@dataclass(frozen=True)
class _V04TiltConfig:
    acceleration_voltage_kV: float
    spherical_aberration_mm: float
    amplitude_contrast: float
    phase_shift_rad: float
    minimum_defocus_A: float
    maximum_defocus_A: float
    fit_batch_size: int
    optimizer_max_iterations: int
    tilt_box_size: int
    tilt_low_resolution_A: float
    tilt_high_resolution_A: float
    tilt_ctf_high_resolution_A: float
    tilt_axis_step_deg: float
    tilt_angle_step_deg: float
    tilt_max_angle_deg: float
    tilt_background_box_size: int
    tilt_candidate_batch_size: int
    tilt_tile_batch_size: int
    tilt_refine_iterations: int
    tilt_min_tiles: int
    spectrum_batch_size: int
    debug: bool


@dataclass
class _V04TiltFrontendResult:
    rough_spectrum: torch.Tensor
    rough_pixel_size_A: float
    local_spectra: torch.Tensor
    local_pixel_size_A: float
    centers_A: np.ndarray
    data: _CTFFIND5TiltData
    rough_defocus_grid_A: np.ndarray
    rough_defocus_scores: np.ndarray
    rough_best_isotropic_defocus_A: float
    rough_defocus1_A: float
    rough_defocus2_A: float
    rough_astigmatism_angle_deg: float
    rough_ctf_score: float
    coarse_candidates: np.ndarray
    coarse_scores: np.ndarray
    coarse_axis_deg: float
    coarse_angle_deg: float
    coarse_mean_defocus_A: float
    refined_axis_deg: float
    refined_angle_deg: float
    refined_mean_defocus_A: float
    refined_score: float
    center_defocus1_A: float
    center_defocus2_A: float
    gradient_x: float
    gradient_y: float
    local_defocus_A: np.ndarray
    score_gap: float
    debug: dict[str, object]
    timings: Optional[dict[str, float]] = None


def _make_v04_tilt_config(config: CtffindConfig) -> _V04TiltConfig:
    # Native CTFFIND5 local search is fixed at 40--10 A, with nominal 5 A/pix
    # sampling.  An explicit --tilt-search-pixel-size remains an experimental
    # override by changing only the local Nyquist-derived upper resolution.
    local_high_resolution_A = 10.0
    if config.tilt_target_pixel_size_A is not None:
        local_high_resolution_A = 2.0 * float(config.tilt_target_pixel_size_A)
    return _V04TiltConfig(
        acceleration_voltage_kV=float(config.acceleration_voltage_kV),
        spherical_aberration_mm=float(config.spherical_aberration_mm),
        amplitude_contrast=float(config.amplitude_contrast),
        phase_shift_rad=float(config.fixed_phase_shift_rad),
        minimum_defocus_A=float(config.minimum_defocus_A),
        maximum_defocus_A=float(config.maximum_defocus_A),
        fit_batch_size=max(1, int(config.fit_batch_size)),
        optimizer_max_iterations=max(120, int(config.powell_maxiter_1d)),
        tilt_box_size=int(config.tilt_tile_size),
        tilt_low_resolution_A=40.0,
        tilt_high_resolution_A=float(local_high_resolution_A),
        tilt_ctf_high_resolution_A=5.0,
        tilt_axis_step_deg=float(config.tilt_axis_step_deg),
        tilt_angle_step_deg=float(config.tilt_angle_step_deg),
        tilt_max_angle_deg=float(config.tilt_max_angle_deg),
        tilt_background_box_size=55,
        tilt_candidate_batch_size=max(1, int(config.tilt_candidate_batch_size)),
        tilt_tile_batch_size=max(1, int(config.tilt_tile_batch_size)),
        tilt_refine_iterations=max(1, int(config.tilt_refine_maxiter)),
        tilt_min_tiles=max(3, int(config.tilt_min_tiles)),
        spectrum_batch_size=max(1, min(16, int(config.tilt_tile_batch_size))),
        debug=bool(config.debug),
    )
def _v04_electron_wavelength_A(acceleration_voltage_kV: float) -> float:
    voltage_V = float(acceleration_voltage_kV) * 1000.0
    return 12.2639 / math.sqrt(voltage_V + 0.97845e-6 * voltage_V * voltage_V)

def _v04_amplitude_contrast_phase_rad(amplitude_contrast: float) -> float:
    a = float(amplitude_contrast)
    if not 0.0 <= a <= 1.0:
        raise ValueError("Amplitude contrast must be between 0 and 1")
    if abs(a - 1.0) < 1.0e-3:
        return 0.5 * PI
    return math.atan(a / math.sqrt(max(_V04_EPS, 1.0 - a * a)))

def _v04_canonicalize_ctf_parameters(
    defocus1_A: float,
    defocus2_A: float,
    angle_deg: float,
) -> tuple[float, float, float]:
    d1 = float(defocus1_A)
    d2 = float(defocus2_A)
    angle = float(angle_deg)
    if d1 < d2:
        d1, d2 = d2, d1
        angle += 90.0
    angle = ((angle + 90.0) % 180.0) - 90.0
    return d1, d2, angle

def _v04_ctf_phase_tensor(
    frequency_squared_Ainv2: torch.Tensor,
    azimuth_rad: torch.Tensor,
    defocus1_A: torch.Tensor | float,
    defocus2_A: torch.Tensor | float,
    astigmatism_angle_rad: torch.Tensor | float,
    wavelength_A: float,
    spherical_aberration_A: float,
    amplitude_phase_rad: float,
    additional_phase_shift_rad: float,
) -> torch.Tensor:
    d1 = torch.as_tensor(defocus1_A, dtype=frequency_squared_Ainv2.dtype, device=frequency_squared_Ainv2.device)
    d2 = torch.as_tensor(defocus2_A, dtype=frequency_squared_Ainv2.dtype, device=frequency_squared_Ainv2.device)
    angle = torch.as_tensor(astigmatism_angle_rad, dtype=frequency_squared_Ainv2.dtype, device=frequency_squared_Ainv2.device)
    effective_defocus = 0.5 * (
        d1 + d2 + torch.cos(2.0 * (azimuth_rad - angle)) * (d1 - d2)
    )
    phase = PI * wavelength_A * frequency_squared_Ainv2 * (
        effective_defocus
        - 0.5 * wavelength_A * wavelength_A * frequency_squared_Ainv2 * spherical_aberration_A
    )
    return phase + float(additional_phase_shift_rad) + float(amplitude_phase_rad)

def _v04_frequency_grid(
    size: int,
    pixel_size_A: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinates = (
        torch.arange(size, dtype=dtype, device=device) - size // 2
    ) / (float(size) * float(pixel_size_A))
    fy, fx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    frequency_squared = fx.square() + fy.square()
    frequency = torch.sqrt(frequency_squared)
    azimuth = torch.atan2(fy, fx)
    return fx, fy, frequency, frequency_squared, azimuth

def _v04_pearson_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of x[B,N] against y[N] or y[B,N]."""
    if x.ndim != 2:
        raise ValueError("_pearson_batch expects x[B,N]")
    if y.ndim == 1:
        y_mean = y.mean()
        y_centered = y - y_mean
        y_ss = y_centered.square().sum().clamp_min(_V04_EPS)
        x_centered = x - x.mean(dim=1, keepdim=True)
        numerator = (x_centered * y_centered.unsqueeze(0)).sum(dim=1)
        denominator = torch.sqrt(x_centered.square().sum(dim=1).clamp_min(_V04_EPS) * y_ss)
    elif y.ndim == 2:
        x_centered = x - x.mean(dim=1, keepdim=True)
        y_centered = y - y.mean(dim=1, keepdim=True)
        numerator = (x_centered * y_centered).sum(dim=1)
        denominator = torch.sqrt(
            x_centered.square().sum(dim=1).clamp_min(_V04_EPS)
            * y_centered.square().sum(dim=1).clamp_min(_V04_EPS)
        )
    else:
        raise ValueError("_pearson_batch expects y[N] or y[B,N]")
    return numerator / denominator.clamp_min(_V04_EPS)

def _v04_round_half_away_from_zero(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0.0 else int(math.ceil(x - 0.5))

def _v04_make_even(value: int, *, lower: bool = False) -> int:
    if value % 2 == 0:
        return value
    return value - 1 if lower else value + 1

def _v04_is_fft_friendly(value: int) -> bool:
    if value < 2:
        return False
    n = value
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1

def _v04_closest_fft_friendly(value: int, *, upper: bool) -> int:
    value = max(2, int(value))
    if value % 2:
        value += 1 if upper else -1
    step = 2
    n = value
    if upper:
        while not _v04_is_fft_friendly(n):
            n += step
    else:
        while n > 2 and not _v04_is_fft_friendly(n):
            n -= step
    return max(2, n)

def _v04_center_crop_or_pad_batch(
    images: torch.Tensor,
    output_h: int,
    output_w: int | None = None,
    padding_value: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    if images.ndim == 2:
        images = images.unsqueeze(0)
        squeeze = True
    elif images.ndim == 3:
        squeeze = False
    else:
        raise ValueError("Expected [H,W] or [B,H,W]")
    if output_w is None:
        output_w = output_h
    batch, in_h, in_w = images.shape
    if isinstance(padding_value, torch.Tensor):
        pv = padding_value.to(dtype=images.dtype, device=images.device).reshape(batch, 1, 1)
        output = pv.expand(batch, output_h, output_w).clone()
    else:
        output = torch.full(
            (batch, output_h, output_w),
            float(padding_value),
            dtype=images.dtype,
            device=images.device,
        )
    copy_h = min(in_h, output_h)
    copy_w = min(in_w, output_w)
    src_y = in_h // 2 - copy_h // 2
    src_x = in_w // 2 - copy_w // 2
    dst_y = output_h // 2 - copy_h // 2
    dst_x = output_w // 2 - copy_w // 2
    output[:, dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = images[
        :, src_y:src_y + copy_h, src_x:src_x + copy_w
    ]
    return output[0] if squeeze else output

def _v04_fourier_resize_real_batch(images: torch.Tensor, output_h: int, output_w: int) -> torch.Tensor:
    if images.ndim == 2:
        images = images.unsqueeze(0)
        squeeze = True
    elif images.ndim == 3:
        squeeze = False
    else:
        raise ValueError("Fourier resize expects [H,W] or [B,H,W]")
    batch, input_h, input_w = images.shape
    if (input_h, input_w) == (output_h, output_w):
        result = images.clone()
        return result[0] if squeeze else result
    fourier = torch.fft.fftshift(torch.fft.fft2(images), dim=(-2, -1))
    output_fourier = torch.zeros(
        (batch, output_h, output_w), dtype=fourier.dtype, device=images.device
    )
    copy_h = min(input_h, output_h)
    copy_w = min(input_w, output_w)
    src_y = input_h // 2 - copy_h // 2
    src_x = input_w // 2 - copy_w // 2
    dst_y = output_h // 2 - copy_h // 2
    dst_x = output_w // 2 - copy_w // 2
    output_fourier[:, dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = fourier[
        :, src_y:src_y + copy_h, src_x:src_x + copy_w
    ]
    result = torch.fft.ifft2(torch.fft.ifftshift(output_fourier, dim=(-2, -1))).real
    result *= (float(output_h * output_w) / float(input_h * input_w))
    return result[0] if squeeze else result

def _v04_resample_micrograph_to_pixel_size(
    image: torch.Tensor,
    input_pixel_size_A: float,
    output_pixel_size_A: float,
    *,
    fft_friendly: bool = False,
) -> tuple[torch.Tensor, float]:
    """Fourier-resample a 2-D micrograph and report its exact x sampling.

    CTFTilt resizes the rectangular local-search image to the nearest integer
    dimensions rather than forcing factorized FFT sizes.  PyTorch/cuFFT handles
    those dimensions directly, so ``fft_friendly`` is opt-in and is used only
    where the original code explicitly factorizes a dimension.
    """
    if output_pixel_size_A <= input_pixel_size_A * (1.0 + 1.0e-7):
        return image.clone(), float(input_pixel_size_A)
    scale = input_pixel_size_A / output_pixel_size_A
    output_h = _v04_make_even(
        max(2, _v04_round_half_away_from_zero(image.shape[-2] * scale)), lower=True
    )
    output_w = _v04_make_even(
        max(2, _v04_round_half_away_from_zero(image.shape[-1] * scale)), lower=True
    )
    if fft_friendly:
        output_h = _v04_closest_fft_friendly(output_h, upper=False)
        output_w = _v04_closest_fft_friendly(output_w, upper=False)
    resized = _v04_fourier_resize_real_batch(image, output_h, output_w)
    effective_pixel = input_pixel_size_A * image.shape[-1] / float(output_w)
    return resized, float(effective_pixel)

def _v04_cosine_rectangular_window(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    edge_fraction: float = 0.10,
) -> torch.Tensor:
    def one_dimensional(length: int) -> torch.Tensor:
        edge = max(1, _v04_round_half_away_from_zero(length * edge_fraction))
        w = torch.ones(length, dtype=dtype, device=device)
        if edge > 0:
            t = (torch.arange(edge, dtype=dtype, device=device) + 0.5) / float(edge)
            ramp = 0.5 - 0.5 * torch.cos(PI * t)
            w[:edge] = ramp
            w[-edge:] = torch.flip(ramp, dims=(0,))
        return w
    return one_dimensional(height)[:, None] * one_dimensional(width)[None, :]

def _v04_radial_cosine_mask(
    size: int,
    radius_midpoint: float,
    edge_width: float,
    device: torch.device,
    dtype: torch.dtype,
    invert: bool = False,
) -> torch.Tensor:
    y = torch.arange(size, dtype=dtype, device=device) - size // 2
    x = torch.arange(size, dtype=dtype, device=device) - size // 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    inner = max(0.0, radius_midpoint - 0.5 * edge_width)
    outer = radius_midpoint + 0.5 * edge_width
    mask = torch.ones_like(radius)
    mask[radius >= outer] = 0.0
    transition = (radius > inner) & (radius < outer)
    if torch.any(transition):
        phase = (radius[transition] - inner) / max(_V04_EPS, outer - inner)
        mask[transition] = 0.5 + 0.5 * torch.cos(PI * phase)
    if invert:
        mask = 1.0 - mask
    return mask

def _v04_extract_patch(
    image: torch.Tensor,
    center_y: float,
    center_x: float,
    height: int,
    width: int,
    padding_value: float | None = None,
) -> torch.Tensor:
    if image.ndim != 2:
        raise ValueError("_extract_patch expects a 2-D image")
    if padding_value is None:
        padding_value = float(image.mean().item())
    cy = _v04_round_half_away_from_zero(center_y)
    cx = _v04_round_half_away_from_zero(center_x)
    y0 = cy - height // 2
    x0 = cx - width // 2
    y1 = y0 + height
    x1 = x0 + width
    output = torch.full((height, width), padding_value, dtype=image.dtype, device=image.device)
    sy0 = max(0, y0)
    sx0 = max(0, x0)
    sy1 = min(image.shape[0], y1)
    sx1 = min(image.shape[1], x1)
    if sy1 > sy0 and sx1 > sx0:
        dy0 = sy0 - y0
        dx0 = sx0 - x0
        output[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = image[sy0:sy1, sx0:sx1]
    return output

def _v04_amplitude_spectrum_batch(
    patches: torch.Tensor,
    *,
    square_amplitude: bool,
    normalize_variance: bool,
    mask_padding_value: bool,
    window: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if patches.ndim != 3:
        raise ValueError("Expected patches[B,H,W]")
    means = patches.mean(dim=(-2, -1), keepdim=True)
    work = patches - means
    if normalize_variance:
        std = work.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1.0e-6)
        work = work / std
    if window is None:
        window = _v04_cosine_rectangular_window(
            patches.shape[-2], patches.shape[-1], patches.device, patches.dtype
        )
    elif (
        window.ndim != 2
        or tuple(window.shape) != tuple(patches.shape[-2:])
        or window.device != patches.device
        or window.dtype != patches.dtype
    ):
        raise ValueError("Cached CTFTilt window is incompatible with the patch batch")
    work = work * window
    fourier = torch.fft.fft2(work)
    fourier[..., 0, 0] = 0.0
    amplitude = torch.fft.fftshift(fourier.abs(), dim=(-2, -1))
    if square_amplitude:
        amplitude = amplitude.square()
    return amplitude


_V04_SPECTRUM_BOX_CACHE: dict[
    tuple[str, int | None, str, int, int, int, int, float],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _v04_separable_periodic_box_sum_cached(
    image_4d: torch.Tensor,
    horizontal_kernel: torch.Tensor,
    vertical_kernel: torch.Tensor,
    half_width: int,
) -> torch.Tensor:
    tmp = F.conv2d(
        F.pad(image_4d, (half_width, half_width, 0, 0), mode="circular"),
        horizontal_kernel,
    )
    return F.conv2d(
        F.pad(tmp, (0, 0, half_width, half_width), mode="circular"),
        vertical_kernel,
    )


def _v04_spectrum_box_geometry(
    *,
    height: int,
    width: int,
    box_size: int,
    minimum_radius_pixels: float,
    cross_half_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (
        device.type,
        device.index if device.type == "cuda" else None,
        str(dtype),
        int(height),
        int(width),
        int(box_size),
        int(cross_half_width),
        round(float(minimum_radius_pixels), 8),
    )
    cached = _V04_SPECTRUM_BOX_CACHE.get(key)
    if cached is not None:
        return cached
    cy, cx = height // 2, width // 2
    valid = torch.ones((1, 1, height, width), dtype=dtype, device=device)
    valid[:, :, max(0, cy-cross_half_width):min(height, cy+cross_half_width+1), :] = 0.0
    valid[:, :, :, max(0, cx-cross_half_width):min(width, cx+cross_half_width+1)] = 0.0
    horizontal = torch.ones((1, 1, 1, box_size), dtype=dtype, device=device)
    vertical = torch.ones((1, 1, box_size, 1), dtype=dtype, device=device)
    counts = _v04_separable_periodic_box_sum_cached(
        valid, horizontal, vertical, box_size // 2
    ).clamp_min(1.0)
    y = torch.arange(height, dtype=dtype, device=device) - cy
    x = torch.arange(width, dtype=dtype, device=device) - cx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    inside = ((xx.square() + yy.square()) < float(minimum_radius_pixels) ** 2).unsqueeze(0)
    cached = (valid, counts, inside, horizontal, vertical)
    _V04_SPECTRUM_BOX_CACHE[key] = cached
    return cached


def _v04_spectrum_box_convolution(
    spectra: torch.Tensor,
    box_size: int,
    minimum_radius_pixels: float,
    cross_half_width: int = 1,
) -> torch.Tensor:
    """CTFTilt periodic box average using cached separable 1-D sums."""
    if spectra.ndim == 2:
        spectra = spectra.unsqueeze(0)
        squeeze = True
    elif spectra.ndim == 3:
        squeeze = False
    else:
        raise ValueError("Expected spectrum[H,W] or spectra[B,H,W]")
    if box_size < 1:
        return spectra[0].clone() if squeeze else spectra.clone()
    if box_size % 2 == 0:
        box_size += 1
    _, height, width = spectra.shape
    valid, counts, inside, horizontal, vertical = _v04_spectrum_box_geometry(
        height=height,
        width=width,
        box_size=box_size,
        minimum_radius_pixels=minimum_radius_pixels,
        cross_half_width=cross_half_width,
        device=spectra.device,
        dtype=spectra.dtype,
    )
    sums = _v04_separable_periodic_box_sum_cached(
        spectra.unsqueeze(1) * valid,
        horizontal,
        vertical,
        box_size // 2,
    )
    average = (sums / counts)[:, 0]
    average = torch.where(inside, spectra, average)
    return average[0] if squeeze else average


def _v04_gather_patch_batches(
    image: torch.Tensor,
    top_left_y: Sequence[int],
    top_left_x: Sequence[int],
    height: int,
    width: int,
    batch_size: int,
) -> Iterator[torch.Tensor]:
    """Gather all local patches from one shared, optionally padded image."""
    if image.ndim != 2:
        raise ValueError("CTFTilt patch gather expects one 2-D image")
    if len(top_left_y) != len(top_left_x):
        raise ValueError("Patch coordinate arrays must have equal length")
    if len(top_left_y) == 0:
        return
    minimum_y = min(int(v) for v in top_left_y)
    minimum_x = min(int(v) for v in top_left_x)
    maximum_y = max(int(v) + int(height) for v in top_left_y)
    maximum_x = max(int(v) + int(width) for v in top_left_x)
    pad_top = max(0, -minimum_y)
    pad_left = max(0, -minimum_x)
    pad_bottom = max(0, maximum_y - int(image.shape[0]))
    pad_right = max(0, maximum_x - int(image.shape[1]))
    if pad_top or pad_bottom or pad_left or pad_right:
        # The old implementation recomputed image.mean().item() once per tile.
        # Keep one device-side mean and one shared padded image instead.
        padding_value = image.mean()
        padded = padding_value.expand(
            int(image.shape[0]) + pad_top + pad_bottom,
            int(image.shape[1]) + pad_left + pad_right,
        ).clone()
        padded[
            pad_top:pad_top + int(image.shape[0]),
            pad_left:pad_left + int(image.shape[1]),
        ] = image
    else:
        padded = image
    y0 = torch.as_tensor(
        [int(v) + pad_top for v in top_left_y], dtype=torch.long, device=image.device
    )
    x0 = torch.as_tensor(
        [int(v) + pad_left for v in top_left_x], dtype=torch.long, device=image.device
    )
    rows = torch.arange(height, dtype=torch.long, device=image.device)
    columns = torch.arange(width, dtype=torch.long, device=image.device)
    batch_size = max(1, int(batch_size))
    for first in range(0, len(top_left_y), batch_size):
        ys = y0[first:first + batch_size, None] + rows[None, :]
        xs = x0[first:first + batch_size, None] + columns[None, :]
        yield padded[ys[:, :, None], xs[:, None, :]].contiguous()


def _v04_regular_simplex_3d(start: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    if start.shape != (3,) or ranges.shape != (3,):
        raise ValueError("regular simplex requires three parameters")
    r1, r2, r3 = ranges
    return np.asarray(
        [
            start + np.array([r1 * math.sqrt(8.0 / 9.0), 0.0, -r3 / 3.0]),
            start + np.array([-r1 * math.sqrt(2.0 / 9.0), r2 * math.sqrt(2.0 / 3.0), -r3 / 3.0]),
            start + np.array([-r1 * math.sqrt(2.0 / 9.0), -r2 * math.sqrt(2.0 / 3.0), -r3 / 3.0]),
            start + np.array([0.0, 0.0, r3]),
        ],
        dtype=np.float64,
    )

def _v04_nelder_mead_maximize(
    objective: Callable[[np.ndarray], np.ndarray],
    initial_simplex: np.ndarray,
    max_iterations: int,
    x_tolerance: float = 1.0e-4,
    f_tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, float, int]:
    simplex = np.asarray(initial_simplex, dtype=np.float64).copy()
    if simplex.ndim != 2 or simplex.shape[0] != simplex.shape[1] + 1:
        raise ValueError("initial_simplex must have shape [N+1,N]")
    values = np.asarray(objective(simplex), dtype=np.float64)
    if values.shape != (simplex.shape[0],):
        raise ValueError("objective returned an unexpected shape")
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    iteration = 0
    for iteration in range(max_iterations):
        order = np.argsort(-values, kind="stable")
        simplex = simplex[order]
        values = values[order]
        if (
            np.max(np.abs(simplex[1:] - simplex[0])) <= x_tolerance
            and np.max(np.abs(values[1:] - values[0])) <= f_tolerance
        ):
            break
        centroid = simplex[:-1].mean(axis=0)
        worst = simplex[-1]
        reflected = centroid + alpha * (centroid - worst)
        reflected_value = float(objective(reflected[None, :])[0])
        if values[0] > reflected_value >= values[-2]:
            simplex[-1] = reflected
            values[-1] = reflected_value
            continue
        if reflected_value > values[0]:
            expanded = centroid + gamma * (reflected - centroid)
            expanded_value = float(objective(expanded[None, :])[0])
            if expanded_value > reflected_value:
                simplex[-1] = expanded
                values[-1] = expanded_value
            else:
                simplex[-1] = reflected
                values[-1] = reflected_value
            continue
        if reflected_value > values[-1]:
            contracted = centroid + rho * (reflected - centroid)
        else:
            contracted = centroid + rho * (worst - centroid)
        contracted_value = float(objective(contracted[None, :])[0])
        if contracted_value > max(values[-1], reflected_value if reflected_value <= values[-1] else -np.inf):
            simplex[-1] = contracted
            values[-1] = contracted_value
            continue
        simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
        values[1:] = objective(simplex[1:])
    order = np.argsort(-values, kind="stable")
    return simplex[order[0]].copy(), float(values[order[0]]), iteration + 1

def _v04_annulus_flatten(
    spectrum: torch.Tensor,
    pixel_size_A: float,
    low_resolution_A: float,
    high_resolution_A: float | None,
    *,
    central_cross_half_width: int = 0,
    left_half_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten a centered fitting annulus with optional CTFFIND cross mask."""
    size = spectrum.shape[-1]
    _, _, _, f2, azimuth = _v04_frequency_grid(
        size, pixel_size_A, spectrum.device, spectrum.dtype
    )
    mask = f2 > (1.0 / float(low_resolution_A)) ** 2
    if high_resolution_A is not None:
        mask &= f2 <= (1.0 / float(high_resolution_A)) ** 2
    if central_cross_half_width > 0 or left_half_only:
        coordinates = torch.arange(size, device=spectrum.device) - size // 2
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        if central_cross_half_width > 0:
            half = int(central_cross_half_width)
            mask &= (xx.abs() > half) & (yy.abs() > half)
        if left_half_only:
            mask &= xx < -int(central_cross_half_width)
    values = spectrum[mask]
    return values.contiguous(), f2[mask].contiguous(), azimuth[mask].contiguous()

def _v04_ctf_candidate_scores(
    observed: torch.Tensor,
    frequency_squared: torch.Tensor,
    azimuth: torch.Tensor,
    candidates: np.ndarray,
    config: _V04TiltConfig,
    wavelength_A: float,
    theoretical_power: float,
    batch_size: int,
) -> np.ndarray:
    spherical_aberration_A = config.spherical_aberration_mm * 1.0e7
    amp_phase = _v04_amplitude_contrast_phase_rad(config.amplitude_contrast)
    candidates_np = np.asarray(candidates, dtype=np.float32)
    output: list[torch.Tensor] = []
    for start in range(0, candidates_np.shape[0], batch_size):
        batch_np = candidates_np[start:start + batch_size]
        batch = torch.as_tensor(batch_np, dtype=observed.dtype, device=observed.device)
        d1 = batch[:, 0, None]
        d2 = batch[:, 1, None]
        angle = torch.deg2rad(batch[:, 2, None])
        phase = _v04_ctf_phase_tensor(
            frequency_squared[None, :],
            azimuth[None, :],
            d1,
            d2,
            angle,
            wavelength_A,
            spherical_aberration_A,
            amp_phase,
            config.phase_shift_rad,
        )
        theory = torch.sin(phase).abs()
        if theoretical_power != 1.0:
            theory = theory.pow(theoretical_power)
        output.append(_v04_pearson_batch(theory, observed))
    # Avoid one GPU synchronization per candidate chunk.
    return (
        torch.cat(output).detach().cpu().numpy()
        .astype(np.float64, copy=False)
    )

def _v04_ctftilt_power_spectra(
    image: torch.Tensor,
    input_pixel_size_A: float,
    config: _V04TiltConfig,
    timings: Optional[dict[str, float]] = None,
) -> tuple[torch.Tensor, float, torch.Tensor, float, np.ndarray, dict[str, Any]]:
    """Return global rough spectrum and local 128x128 tilt spectra."""
    timing_enabled = timings is not None
    if timing_enabled:
        _synchronize_if_cuda(image.device)
    stage_started = time.perf_counter()
    tilt_box = int(config.tilt_box_size)
    if tilt_box != 128:
        warnings.warn(
            "CTFFIND5 uses a fixed 128-pixel tilt spectrum. A non-128 value is "
            "supported for experimentation but is not compatibility mode.",
            RuntimeWarning,
        )

    # Rough global CTF spectrum. CTFTilt first mean-pads to an even square,
    # Fourier-bins to a factorized dimension at approximately 2.5 A/pixel,
    # and uses a factorized central subregion of about 2000 A.
    global_target_pixel = max(
        input_pixel_size_A, 0.5 * config.tilt_ctf_high_resolution_A
    )
    square_size = _v04_make_even(max(int(image.shape[0]), int(image.shape[1])))
    square = _v04_center_crop_or_pad_batch(
        image, square_size, square_size, padding_value=image.mean()
    )
    if global_target_pixel > input_pixel_size_A * (1.0 + 1.0e-7):
        global_size = _v04_closest_fft_friendly(
            _v04_round_half_away_from_zero(
                square_size * input_pixel_size_A / global_target_pixel
            ),
            upper=True,
        )
        global_image = _v04_fourier_resize_real_batch(square, global_size, global_size)
        global_pixel = input_pixel_size_A * square_size / float(global_size)
    else:
        global_image = square
        global_pixel = float(input_pixel_size_A)
    target_subregion = _v04_closest_fft_friendly(
        _v04_make_even(_v04_round_half_away_from_zero(2000.0 / global_pixel), lower=True),
        upper=False,
    )
    target_subregion = max(tilt_box, target_subregion)
    target_subregion = min(
        target_subregion, global_image.shape[0], global_image.shape[1]
    )
    target_subregion = _v04_make_even(target_subregion, lower=True)
    rough_patch = _v04_extract_patch(
        global_image,
        0.5 * (global_image.shape[0] - 1),
        0.5 * (global_image.shape[1] - 1),
        target_subregion,
        target_subregion,
    )
    rough_patch = rough_patch - rough_patch.mean()
    rough_window = _v04_cosine_rectangular_window(
        target_subregion,
        target_subregion,
        rough_patch.device,
        rough_patch.dtype,
    )
    rough_patch = rough_patch * rough_window
    rough_fft = torch.fft.fft2(rough_patch)
    rough_fft[0, 0] = 0.0
    rough_amplitude = torch.fft.fftshift(rough_fft.abs())
    rough_resampled = _v04_fourier_resize_real_batch(rough_amplitude, tilt_box, tilt_box)
    rough_min_radius = float(tilt_box) * global_pixel / config.tilt_low_resolution_A
    rough_background = _v04_spectrum_box_convolution(
        rough_resampled,
        config.tilt_background_box_size,
        rough_min_radius,
        cross_half_width=1,
    )
    rough_filtered = rough_resampled - rough_background
    rough_filtered = rough_filtered * _v04_radial_cosine_mask(
        tilt_box,
        0.30 * tilt_box,
        0.40 * tilt_box,
        rough_filtered.device,
        rough_filtered.dtype,
    )
    # Release the square full-image FFT path before starting the second,
    # rectangular full-image resampling.  This matters for K3/K2 micrographs
    # on 24-GB GPUs; PyTorch's allocator can immediately reuse these blocks.
    del (
        square,
        global_image,
        rough_patch,
        rough_window,
        rough_fft,
        rough_amplitude,
        rough_resampled,
        rough_background,
    )
    if timing_enabled:
        _synchronize_if_cuda(image.device)
        now = time.perf_counter()
        _add_timing(
            timings, "tilt_frontend_global_power_spectrum_s",
            now - stage_started,
        )
        stage_started = now

    # Local search spectra: Fourier-bin the entire image to approximately
    # tilt_high_resolution/2 = 5 A/pixel, then construct 50%-overlapping tiles.
    local_target_pixel = max(
        input_pixel_size_A, 0.5 * config.tilt_high_resolution_A
    )
    local_image, local_effective_pixel = _v04_resample_micrograph_to_pixel_size(
        image,
        input_pixel_size_A,
        local_target_pixel,
        fft_friendly=False,
    )
    # CTFFIND5 uses the nominal high-resolution-derived sampling in its CTF
    # and geometry calculations, even when integer resize dimensions make the
    # exact physical sampling differ slightly.
    local_pixel = (
        float(local_target_pixel)
        if local_target_pixel > input_pixel_size_A * (1.0 + 1.0e-7)
        else float(input_pixel_size_A)
    )
    n_sections_x = max(1, int(local_image.shape[1] / float(tilt_box)))
    n_sections_y = max(1, int(local_image.shape[0] / float(tilt_box)))
    subsection_x = _v04_make_even(
        max(2, _v04_round_half_away_from_zero(local_image.shape[1] / float(n_sections_x))),
        lower=True,
    )
    subsection_y = _v04_make_even(
        max(2, _v04_round_half_away_from_zero(local_image.shape[0] / float(n_sections_y))),
        lower=True,
    )
    tile_size = min(subsection_x, subsection_y, tilt_box)
    if tile_size % 2:
        tile_size -= 1
    tile_w = tile_h = max(2, tile_size)
    image_center_y = 0.5 * (local_image.shape[0] - 1)
    image_center_x = 0.5 * (local_image.shape[1] - 1)

    centers_A: list[tuple[float, float]] = []
    top_left_y: list[int] = []
    top_left_x: list[int] = []
    for iy in range(-(n_sections_y - 1), n_sections_y):
        y_offset = 0.5 * iy * subsection_y
        cy = _v04_round_half_away_from_zero(image_center_y + y_offset)
        for ix in range(-(n_sections_x - 1), n_sections_x):
            x_offset = 0.5 * ix * subsection_x
            cx = _v04_round_half_away_from_zero(image_center_x + x_offset)
            top_left_y.append(cy - tile_h // 2)
            top_left_x.append(cx - tile_w // 2)
            centers_A.append((x_offset * local_pixel, y_offset * local_pixel))

    local_spectra_chunks: list[torch.Tensor] = []
    min_radius = float(tilt_box) * local_pixel / config.tilt_low_resolution_A
    local_window = _v04_cosine_rectangular_window(
        tile_h, tile_w, local_image.device, local_image.dtype
    )
    for batch in _v04_gather_patch_batches(
        local_image,
        top_left_y,
        top_left_x,
        tile_h,
        tile_w,
        config.spectrum_batch_size,
    ):
        power = _v04_amplitude_spectrum_batch(
            batch,
            square_amplitude=True,
            normalize_variance=True,
            mask_padding_value=False,
            window=local_window,
        )
        if (tile_h, tile_w) != (tilt_box, tilt_box):
            power = _v04_fourier_resize_real_batch(power, tilt_box, tilt_box)
        background = _v04_spectrum_box_convolution(
            power,
            config.tilt_background_box_size,
            min_radius,
            cross_half_width=1,
        )
        local_spectra_chunks.append((power - background).contiguous())
    if not local_spectra_chunks:
        raise RuntimeError("No CTFTilt local spectra were generated")
    local_spectra = torch.cat(local_spectra_chunks, dim=0)
    centers_array = np.asarray(centers_A, dtype=np.float32)
    number_of_tiles = int(local_spectra.shape[0])
    if number_of_tiles < int(config.tilt_min_tiles):
        raise RuntimeError(
            "CTFFIND5 tilt needs at least "
            f"{config.tilt_min_tiles} local spectra after approximately "
            f"{local_pixel:.3f} A/pixel binning; only {number_of_tiles} were "
            "generated. Use a larger micrograph or omit --fit-tilt."
        )
    centered_geometry = centers_array.astype(np.float64) - centers_array.mean(
        axis=0, dtype=np.float64
    )
    geometry_scale = float(np.max(np.abs(centered_geometry)))
    geometry_rank = (
        0
        if geometry_scale <= 0.0
        else int(np.linalg.matrix_rank(centered_geometry / geometry_scale, tol=1.0e-6))
    )
    if geometry_rank < 2:
        raise RuntimeError(
            "CTFFIND5 tilt axis and angle are not identifiable because the local "
            f"spectrum centers span only {geometry_rank} dimension(s). Use a "
            "micrograph large enough to contain tiles in both x and y, or omit "
            "--fit-tilt."
        )
    if timing_enabled:
        _synchronize_if_cuda(image.device)
        now = time.perf_counter()
        _add_timing(
            timings, "tilt_frontend_local_power_spectra_s",
            now - stage_started,
        )

    debug = {
        "global_pixel_size_A": float(global_pixel),
        "global_subregion_size": int(target_subregion),
        "tilt_pixel_size_A": float(local_pixel),
        "tilt_effective_resampling_pixel_size_A": float(local_effective_pixel),
        "n_sections_x": int(n_sections_x),
        "n_sections_y": int(n_sections_y),
        "subsection_dimension_x": int(subsection_x),
        "subsection_dimension_y": int(subsection_y),
        "tile_width": int(tile_w),
        "tile_height": int(tile_h),
        "number_of_tiles": number_of_tiles,
        "tile_geometry_rank": geometry_rank,
        "tile_centers_A": centers_array.tolist(),
    }
    return rough_filtered, global_pixel, local_spectra, local_pixel, centers_array, debug

def _v04_tilt_global_score_candidates(
    observed: torch.Tensor,
    frequency_squared: torch.Tensor,
    azimuth: torch.Tensor,
    candidates: np.ndarray,
    config: _V04TiltConfig,
    wavelength_A: float,
) -> np.ndarray:
    return _v04_ctf_candidate_scores(
        observed,
        frequency_squared,
        azimuth,
        candidates,
        config,
        wavelength_A,
        theoretical_power=4.0,
        batch_size=config.fit_batch_size,
    )

@dataclass
class _V04TiltScoreContext:
    observed: torch.Tensor
    centers: torch.Tensor
    coefficient: torch.Tensor
    base_phase: torch.Tensor
    count: float
    mean_y: torch.Tensor
    var_y: torch.Tensor
    config: _V04TiltConfig


def _v04_make_tilt_score_context(
    local_spectra: torch.Tensor,
    local_pixel_size_A: float,
    tile_centers_A: np.ndarray,
    defocus1_A: float,
    defocus2_A: float,
    astigmatism_angle_deg: float,
    config: _V04TiltConfig,
    wavelength_A: float,
) -> _V04TiltScoreContext:
    """Precompute all candidate-independent local CTFTilt score tensors."""
    device = local_spectra.device
    dtype = local_spectra.dtype
    size = local_spectra.shape[-1]
    _, _, _, frequency_squared, azimuth = _v04_frequency_grid(
        size, local_pixel_size_A, device, dtype
    )
    mask = frequency_squared > (1.0 / config.tilt_low_resolution_A) ** 2
    if config.tilt_high_resolution_A > 2.0 * local_pixel_size_A + 1.0e-6:
        mask &= frequency_squared <= (1.0 / config.tilt_high_resolution_A) ** 2
    f2 = frequency_squared[mask].contiguous()
    az = azimuth[mask].contiguous()
    observed = local_spectra[:, mask].contiguous()
    centers = torch.as_tensor(tile_centers_A, dtype=dtype, device=device)
    spherical_aberration_A = config.spherical_aberration_mm * 1.0e7
    amp_phase = _v04_amplitude_contrast_phase_rad(config.amplitude_contrast)
    half_astig = 0.5 * (defocus1_A - defocus2_A)
    astig_angle = math.radians(astigmatism_angle_deg)
    coefficient = PI * wavelength_A * f2
    base_phase = coefficient * (
        half_astig * torch.cos(2.0 * (az - astig_angle))
    )
    base_phase = base_phase - (
        PI
        * wavelength_A
        * 0.5
        * wavelength_A
        * wavelength_A
        * spherical_aberration_A
        * f2.square()
    )
    base_phase = base_phase + config.phase_shift_rad + amp_phase

    count = float(observed.numel())
    sum_y = observed.sum().to(torch.float64)
    sum_y2 = observed.square().sum().to(torch.float64)
    mean_y = sum_y / count
    var_y = (sum_y2 / count - mean_y.square()).clamp_min(_V04_EPS)
    return _V04TiltScoreContext(
        observed=observed,
        centers=centers,
        coefficient=coefficient,
        base_phase=base_phase,
        count=count,
        mean_y=mean_y,
        var_y=var_y,
        config=config,
    )


def _v04_tilt_local_scores(
    candidates: np.ndarray,
    context: _V04TiltScoreContext,
) -> np.ndarray:
    """Score candidates with one global Pearson coefficient.

    Static frequency support and observed-spectrum moments are cached in
    ``context``.  Candidate chunks stay on the GPU and are copied to the host
    only once, avoiding repeated synchronization during the coarse grid and
    every simplex objective call.
    """
    observed = context.observed
    centers = context.centers
    coefficient = context.coefficient
    base_phase = context.base_phase
    config = context.config
    device = observed.device
    dtype = observed.dtype

    candidates_np = np.asarray(candidates, dtype=np.float32)
    all_scores: list[torch.Tensor] = []
    for candidate_start in range(
        0, candidates_np.shape[0], config.tilt_candidate_batch_size
    ):
        chunk_np = candidates_np[
            candidate_start:candidate_start + config.tilt_candidate_batch_size
        ]
        chunk = torch.as_tensor(chunk_np, dtype=dtype, device=device)
        axis = torch.deg2rad(chunk[:, 0])
        tilt = torch.deg2rad(chunk[:, 1])
        mean_defocus = chunk[:, 2]
        cos_axis = torch.cos(axis)[:, None]
        sin_axis = torch.sin(axis)[:, None]
        x = centers[:, 0][None, :]
        y = centers[:, 1][None, :]
        y_rotated = x * sin_axis + y * cos_axis
        local_mean = (
            mean_defocus[:, None]
            + y_rotated * torch.tan(tilt)[:, None]
        )

        sum_x = torch.zeros(
            chunk.shape[0], dtype=torch.float64, device=device
        )
        sum_x2 = torch.zeros_like(sum_x)
        sum_xy = torch.zeros_like(sum_x)
        for tile_start in range(
            0, observed.shape[0], config.tilt_tile_batch_size
        ):
            tile_stop = min(
                observed.shape[0], tile_start + config.tilt_tile_batch_size
            )
            local = local_mean[:, tile_start:tile_stop]
            phase = (
                coefficient[None, None, :] * local[:, :, None]
                + base_phase[None, None, :]
            )
            theory = torch.sin(phase).square()
            obs = observed[tile_start:tile_stop][None, :, :]
            sum_x += theory.sum(dim=(1, 2)).to(torch.float64)
            sum_x2 += theory.square().sum(dim=(1, 2)).to(torch.float64)
            sum_xy += (theory * obs).sum(dim=(1, 2)).to(torch.float64)
        mean_x = sum_x / context.count
        var_x = (
            sum_x2 / context.count - mean_x.square()
        ).clamp_min(_V04_EPS)
        covariance = sum_xy / context.count - mean_x * context.mean_y
        scores = covariance / torch.sqrt(var_x * context.var_y)
        excess = (chunk[:, 1].abs() - 85.0).clamp_min(0.0) / 5.0
        all_scores.append(scores - excess.to(scores.dtype))
    return (
        torch.cat(all_scores).detach().cpu().numpy()
        .astype(np.float64, copy=False)
    )

def _v04_build_tilt_data(
    local_spectra: torch.Tensor,
    local_pixel_size_A: float,
    centers_A: np.ndarray,
    spectra_debug: dict[str, object],
    config: CtffindConfig,
    compatibility: _V04TiltConfig,
) -> tuple[torch.Tensor, np.ndarray, _CTFFIND5TiltData]:
    """Create 0.5.x diagnostics data without altering the 0.4 search objective."""
    spectra = local_spectra
    centers = np.asarray(centers_A, dtype=np.float32)
    rms_all = torch.sqrt(
        torch.mean(
            (spectra - spectra.mean(dim=(-2, -1), keepdim=True)).square(),
            dim=(-2, -1),
        ).clamp_min(0.0)
    ).detach().cpu().numpy().astype(np.float64, copy=False)
    keep = np.isfinite(rms_all)
    if config.tilt_rms_mad_cutoff > 0.0:
        keep &= _robust_mad_mask(rms_all, float(config.tilt_rms_mad_cutoff))
    if np.count_nonzero(keep) < compatibility.tilt_min_tiles:
        raise RuntimeError(
            f"Only {np.count_nonzero(keep)} finite/local-RMS-compatible CTFTilt "
            f"tiles remain; at least {compatibility.tilt_min_tiles} are required"
        )
    if not np.all(keep):
        index = torch.as_tensor(np.flatnonzero(keep), device=spectra.device, dtype=torch.long)
        spectra = spectra.index_select(0, index)
        centers = centers[keep]
        rms_all = rms_all[keep]

    size = int(spectra.shape[-1])
    _, _, _, freq2, azimuth = _v04_frequency_grid(
        size, float(local_pixel_size_A), spectra.device, spectra.dtype
    )
    support = freq2 > 1.0 / (compatibility.tilt_low_resolution_A ** 2)
    if compatibility.tilt_high_resolution_A > 2.0 * float(local_pixel_size_A) + 1.0e-6:
        support &= freq2 < 1.0 / (compatibility.tilt_high_resolution_A ** 2)
    if int(torch.count_nonzero(support).item()) < 16:
        raise RuntimeError("CTFTilt local fitting annulus contains too few Fourier pixels")

    power_values = spectra[:, support].contiguous()
    frequency_squared = freq2[support].contiguous()
    azimuth_values = azimuth[support].contiguous()
    n_sections_x = int(spectra_debug.get('n_sections_x', 1))
    n_sections_y = int(spectra_debug.get('n_sections_y', 1))
    grid_y_full = np.repeat(
        np.arange(-(n_sections_y - 1), n_sections_y, dtype=np.int64),
        2 * n_sections_x - 1,
    )
    grid_x_full = np.tile(
        np.arange(-(n_sections_x - 1), n_sections_x, dtype=np.int64),
        2 * n_sections_y - 1,
    )
    if grid_y_full.size != len(keep):
        grid_y_full = np.zeros(len(keep), dtype=np.int64)
        grid_x_full = np.arange(len(keep), dtype=np.int64)
    grid_y = grid_y_full[keep]
    grid_x = grid_x_full[keep]
    valid = torch.ones(power_values.shape[0], device=spectra.device, dtype=torch.bool)
    data = _CTFFIND5TiltData(
        power_values=power_values,
        frequency_squared_Ainv2=frequency_squared,
        azimuth_rad=azimuth_values,
        centers_x_A=torch.as_tensor(centers[:, 0], device=spectra.device, dtype=spectra.dtype),
        centers_y_A=torch.as_tensor(centers[:, 1], device=spectra.device, dtype=spectra.dtype),
        valid_mask=valid,
        rms=np.asarray(rms_all, dtype=np.float64),
        centers_x_A_numpy=centers[:, 0].astype(np.float64, copy=False),
        centers_y_A_numpy=centers[:, 1].astype(np.float64, copy=False),
        grid_y=grid_y,
        grid_x=grid_x,
        fitting_pixel_size_A=float(local_pixel_size_A),
    )
    return spectra, centers, data


def _v04_fit_tilt_frontend(
    image: torch.Tensor,
    input_pixel_size_A: float,
    config: CtffindConfig,
) -> _V04TiltFrontendResult:
    """Exact 0.4 local CTFTilt frontend, stopping before spectrum correction."""
    compatibility = _make_v04_tilt_config(config)
    timing_enabled = bool(config.timing)
    timings: dict[str, float] = {}
    if timing_enabled:
        _synchronize_if_cuda(image.device)
    total_started = time.perf_counter()
    stage_started = total_started

    def checkpoint(name: str) -> None:
        nonlocal stage_started
        if not timing_enabled:
            return
        _synchronize_if_cuda(image.device)
        now = time.perf_counter()
        _add_timing(timings, name, now - stage_started)
        stage_started = now
    (
        rough_spectrum,
        rough_pixel,
        local_spectra,
        local_pixel,
        centers_A,
        spectra_debug,
    ) = _v04_ctftilt_power_spectra(
        image, input_pixel_size_A, compatibility,
        timings if timing_enabled else None,
    )
    if timing_enabled:
        _synchronize_if_cuda(image.device)
        stage_started = time.perf_counter()
    local_spectra, centers_A, data = _v04_build_tilt_data(
        local_spectra,
        local_pixel,
        centers_A,
        spectra_debug,
        config,
        compatibility,
    )
    checkpoint("tilt_frontend_build_fit_data_s")

    wavelength = _v04_electron_wavelength_A(compatibility.acceleration_voltage_kV)
    rough_observed, rough_f2, rough_azimuth = _v04_annulus_flatten(
        rough_spectrum,
        rough_pixel,
        compatibility.tilt_low_resolution_A,
        None,
    )
    defocus_values = np.arange(
        compatibility.minimum_defocus_A,
        compatibility.maximum_defocus_A,
        100.0,
        dtype=np.float64,
    )
    if defocus_values.size == 0:
        raise RuntimeError("CTFTilt rough defocus grid is empty")
    rough_candidates = np.column_stack(
        (defocus_values, defocus_values, np.zeros_like(defocus_values))
    )
    rough_scores = _v04_tilt_global_score_candidates(
        rough_observed,
        rough_f2,
        rough_azimuth,
        rough_candidates,
        compatibility,
        wavelength,
    )
    checkpoint("tilt_frontend_rough_defocus_grid_s")
    best_defocus = float(defocus_values[int(np.argmax(rough_scores))])

    def global_objective(parameters: np.ndarray) -> np.ndarray:
        params = np.asarray(parameters, dtype=np.float64).copy()
        penalties = np.zeros(params.shape[0], dtype=np.float64)
        for i in range(params.shape[0]):
            d1, d2, angle = _v04_canonicalize_ctf_parameters(*params[i])
            params[i] = (d1, d2, angle)
            mean = 0.5 * (d1 + d2)
            if mean < compatibility.minimum_defocus_A:
                penalties[i] += (compatibility.minimum_defocus_A - mean) / 1000.0
            if mean > compatibility.maximum_defocus_A:
                penalties[i] += (mean - compatibility.maximum_defocus_A) / 1000.0
        return _v04_tilt_global_score_candidates(
            rough_observed,
            rough_f2,
            rough_azimuth,
            params,
            compatibility,
            wavelength,
        ) - penalties

    global_start = np.asarray([best_defocus, best_defocus, 0.0], dtype=np.float64)
    global_simplex = _v04_regular_simplex_3d(
        global_start, np.asarray([1000.0, 1000.0, 180.0], dtype=np.float64)
    )
    global_refined, global_score, global_iterations = _v04_nelder_mead_maximize(
        global_objective,
        global_simplex,
        max_iterations=compatibility.optimizer_max_iterations,
        x_tolerance=0.02,
        f_tolerance=1.0e-7,
    )
    rough_d1, rough_d2, rough_astig_angle = _v04_canonicalize_ctf_parameters(
        *global_refined
    )
    rough_mean = 0.5 * (rough_d1 + rough_d2)
    checkpoint("tilt_frontend_rough_ctf_refine_s")

    score_context = _v04_make_tilt_score_context(
        local_spectra,
        local_pixel,
        centers_A,
        rough_d1,
        rough_d2,
        rough_astig_angle,
        compatibility,
        wavelength,
    )
    checkpoint("tilt_frontend_score_context_s")

    tilt_angles = np.arange(
        0.0,
        compatibility.tilt_max_angle_deg + 0.5 * compatibility.tilt_angle_step_deg,
        compatibility.tilt_angle_step_deg,
        dtype=np.float64,
    )
    tilt_axes = np.arange(
        0.0,
        360.0,
        compatibility.tilt_axis_step_deg,
        dtype=np.float64,
    )
    coarse_candidates = np.asarray(
        [(axis, angle, rough_mean) for angle in tilt_angles for axis in tilt_axes],
        dtype=np.float64,
    )
    coarse_scores = _v04_tilt_local_scores(
        coarse_candidates, score_context
    )
    checkpoint("tilt_frontend_coarse_search_s")
    best_index = int(np.argmax(coarse_scores))
    coarse_best = coarse_candidates[best_index].copy()

    def local_objective(parameters: np.ndarray) -> np.ndarray:
        params = np.asarray(parameters, dtype=np.float64).copy()
        params[:, 0] = np.mod(params[:, 0], 360.0)
        scores = _v04_tilt_local_scores(params, score_context)
        mean = params[:, 2]
        penalty = np.maximum(0.0, compatibility.minimum_defocus_A - mean) / 1000.0
        penalty += np.maximum(0.0, mean - compatibility.maximum_defocus_A) / 1000.0
        return scores - penalty

    tilt_simplex = _v04_regular_simplex_3d(
        coarse_best,
        np.asarray([20.0, 10.0, 1000.0], dtype=np.float64),
    )
    refined, refined_score, tilt_iterations = _v04_nelder_mead_maximize(
        local_objective,
        tilt_simplex,
        max_iterations=compatibility.tilt_refine_iterations,
        x_tolerance=1.0e-4,
        f_tolerance=1.0e-7,
    )
    checkpoint("tilt_frontend_refine_s")
    axis = float(refined[0] % 360.0)
    angle = float(refined[1])
    refined_mean = float(refined[2])
    if angle < 0.0:
        angle = -angle
        axis = (axis + 180.0) % 360.0
    dmean = refined_mean - rough_mean
    center_d1 = rough_d1 + dmean
    center_d2 = rough_d2 + dmean
    gradient_x = math.sin(math.radians(axis)) * math.tan(math.radians(angle))
    gradient_y = math.cos(math.radians(axis)) * math.tan(math.radians(angle))
    local_defocus = (
        refined_mean
        + gradient_x * centers_A[:, 0]
        + gradient_y * centers_A[:, 1]
    ).astype(np.float64)
    ordered_scores = np.sort(np.asarray(coarse_scores, dtype=np.float64))
    score_gap = (
        float(ordered_scores[-1] - ordered_scores[-2])
        if ordered_scores.size >= 2 else float('nan')
    )
    debug: dict[str, object] = {
        'algorithm': 'CTFFIND5-PyTorch-0.4-compatible-local-frontend',
        'power_spectra': spectra_debug,
        'local_fit_low_resolution_A': float(compatibility.tilt_low_resolution_A),
        'local_fit_high_resolution_A': float(compatibility.tilt_high_resolution_A),
        'local_background_box_size': int(compatibility.tilt_background_box_size),
        'local_objective': 'single_global_Pearson_over_all_tiles_and_pixels',
        'rough_best_isotropic_defocus_A': float(best_defocus),
        'rough_defocus_score': float(np.max(rough_scores)),
        'rough_ctf': {
            'defocus1_A': float(rough_d1),
            'defocus2_A': float(rough_d2),
            'astigmatism_angle_deg': float(rough_astig_angle),
            'score': float(global_score),
            'iterations': int(global_iterations),
        },
        'coarse_best': {
            'axis_internal_deg': float(coarse_best[0]),
            'angle_deg': float(coarse_best[1]),
            'center_mean_defocus_A': float(coarse_best[2]),
            'score': float(coarse_scores[best_index]),
            'score_gap': float(score_gap),
        },
        'refined': {
            'axis_internal_deg': float(axis),
            'angle_deg': float(angle),
            'center_mean_defocus_A': float(refined_mean),
            'score': float(refined_score),
            'iterations': int(tilt_iterations),
            'gradient_x': float(gradient_x),
            'gradient_y': float(gradient_y),
        },
    }
    if config.debug:
        debug['rough_defocus_grid_A'] = defocus_values.tolist()
        debug['rough_defocus_scores'] = np.asarray(rough_scores).tolist()
        debug['coarse_search'] = [
            [float(c[0]), float(c[1]), float(c[2]), float(score)]
            for c, score in zip(coarse_candidates, coarse_scores)
        ]
    if timing_enabled:
        timings['tilt_frontend_total_s'] = float(
            time.perf_counter() - total_started
        )

    return _V04TiltFrontendResult(
        rough_spectrum=rough_spectrum,
        rough_pixel_size_A=float(rough_pixel),
        local_spectra=local_spectra,
        local_pixel_size_A=float(local_pixel),
        centers_A=np.asarray(centers_A, dtype=np.float32),
        data=data,
        rough_defocus_grid_A=defocus_values,
        rough_defocus_scores=np.asarray(rough_scores, dtype=np.float64),
        rough_best_isotropic_defocus_A=float(best_defocus),
        rough_defocus1_A=float(rough_d1),
        rough_defocus2_A=float(rough_d2),
        rough_astigmatism_angle_deg=float(rough_astig_angle),
        rough_ctf_score=float(global_score),
        coarse_candidates=coarse_candidates,
        coarse_scores=np.asarray(coarse_scores, dtype=np.float64),
        coarse_axis_deg=float(coarse_best[0]),
        coarse_angle_deg=float(coarse_best[1]),
        coarse_mean_defocus_A=float(coarse_best[2]),
        refined_axis_deg=float(axis),
        refined_angle_deg=float(angle),
        refined_mean_defocus_A=float(refined_mean),
        refined_score=float(refined_score),
        center_defocus1_A=float(center_d1),
        center_defocus2_A=float(center_d2),
        gradient_x=float(gradient_x),
        gradient_y=float(gradient_y),
        local_defocus_A=local_defocus,
        score_gap=float(score_gap),
        debug=debug,
        timings=timings if timing_enabled else None,
    )


def _ctffind5_gradient_from_axis_angle(
    axis_rad: torch.Tensor,
    angle_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CTFFIND5 non-negative-angle plane convention.

    The tilt axis is reported over 0..360 degrees.  The defocus gradient is
    normal to the axis.  A 180-degree axis change represents the opposite
    signed plane while the reported tilt angle remains non-negative.
    """
    tangent = torch.tan(angle_rad)
    # Internal MRC/NumPy y-down convention used by the validated 0.4 plane.
    # Convert the reported axis only at the output boundary.
    return torch.sin(axis_rad) * tangent, torch.cos(axis_rad) * tangent


def _ctffind5_canonical_tilt(axis_deg: float, angle_deg: float) -> tuple[float, float]:
    if angle_deg < 0.0:
        angle_deg = -angle_deg
        axis_deg += 180.0
    return axis_deg % 360.0, max(0.0, angle_deg)


def _rounded_square_torch(x: torch.Tensor, factor: float = 5.0, exponent: float = 1.0) -> torch.Tensor:
    """cisTEM CTF::rounded_square, with the same default odd factor/exponent."""
    sin_x = torch.sin(x)
    threshold = math.sin(PI / (2.0 * factor))
    rounded = torch.where(
        torch.abs(sin_x) > threshold,
        torch.sign(sin_x),
        torch.sin(factor * x).pow(exponent),
    )
    return torch.where(x < PI / 2.0, torch.ones_like(x), rounded)


def _finite_thickness_power_model(
    frequency_squared_Ainv2: torch.Tensor,
    azimuth_rad: torch.Tensor,
    defocus1_A: torch.Tensor,
    defocus2_A: torch.Tensor,
    astigmatism_angle_rad: torch.Tensor,
    thickness_A: torch.Tensor,
    wavelength_A: float,
    spherical_aberration_A: float,
    amplitude_phase_rad: float | torch.Tensor,
    phase_shift_rad: float,
    use_rounded_square: bool,
) -> torch.Tensor:
    """cisTEM CTF::EvaluatePowerspectrumWithThickness in Angstrom units."""
    freq2 = frequency_squared_Ainv2
    az = azimuth_rad

    def expand_parameter(value: torch.Tensor) -> torch.Tensor:
        expanded = value
        while expanded.ndim < freq2.ndim:
            expanded = expanded.unsqueeze(-1)
        return expanded

    d1 = expand_parameter(defocus1_A)
    d2 = expand_parameter(defocus2_A)
    angle = expand_parameter(astigmatism_angle_rad)
    effective_defocus = 0.5 * (
        d1 + d2 + torch.cos(2.0 * (az - angle)) * (d1 - d2)
    )
    amp_phase = torch.as_tensor(
        amplitude_phase_rad, device=freq2.device, dtype=freq2.dtype
    )
    while amp_phase.ndim < freq2.ndim:
        amp_phase = amp_phase.unsqueeze(-1)
    phase = (
        PI
        * wavelength_A
        * freq2
        * (
            effective_defocus
            - 0.5
            * wavelength_A
            * wavelength_A
            * freq2
            * spherical_aberration_A
        )
        + phase_shift_rad
        + amp_phase
    )
    thickness_expanded = expand_parameter(thickness_A)
    argument = PI * wavelength_A * freq2 * thickness_expanded
    if use_rounded_square:
        modulation = _rounded_square_torch(argument)
    else:
        # torch.sinc(z) = sin(pi*z)/(pi*z), so z=lambda*s^2*t reproduces
        # cisTEM's sinc(pi*lambda*s^2*t).
        modulation = torch.sinc(wavelength_A * freq2 * thickness_expanded)
    return 0.5 * (1.0 - modulation * torch.cos(2.0 * phase))


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
        self._thickness_2d_geometry_cache: dict[
            tuple[int, float, str, int | None, str, float, float],
            dict[str, torch.Tensor | int],
        ] = {}
        if config.find_phase_shift:
            warnings.warn(
                "Phase-shift search is not implemented yet; using the fixed "
                f"phase shift {canonical_phase:.6g} rad.",
                RuntimeWarning,
                stacklevel=2,
            )
        if config.fit_tilt and int(config.tilt_tile_stride) != 64:
            warnings.warn(
                "--tilt-tile-stride is ignored by CTFFIND5 compatibility mode; "
                "the local grid uses source-compatible subsection geometry with "
                "approximately 50% overlap.",
                RuntimeWarning,
                stacklevel=2,
            )
        if (
            config.fit_tilt
            and config.tilt_target_pixel_size_A is not None
            and abs(float(config.tilt_target_pixel_size_A) - 5.0) > 1.0e-7
        ):
            warnings.warn(
                "A non-5-A --tilt-search-pixel-size changes the validated "
                "CTFFIND5 40--10 A local-search frontend and is experimental.",
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
            experimental = experimental - experimental.mean(dim=1, keepdim=True)
            theoretical = theoretical - theoretical.mean(dim=1, keepdim=True)
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
            experimental = experimental - experimental.mean(dim=1, keepdim=True)
            theoretical = theoretical - theoretical.mean(dim=1, keepdim=True)
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
        *,
        ctf_squared: bool = False,
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
        scores = self._score_1d_candidates(curve, candidates, ctf_squared=ctf_squared)
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
            return -self._score_1d_per_image(
                curve, defocus, ctf_squared=ctf_squared
            ).to(self.optimizer_dtype)

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
        *,
        ctf_squared: bool = False,
        apply_astigmatism_penalty: bool = True,
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
        if ctf_squared:
            theoretical = theoretical.square()
            experimental = fit_data.spectrum_values
            experimental = experimental - experimental.mean(dim=1, keepdim=True)
            theoretical = theoretical - theoretical.mean(dim=1, keepdim=True)
            image_norm = torch.sqrt(torch.sum(experimental.square(), dim=1))
        else:
            experimental = fit_data.spectrum_values
            image_norm = fit_data.image_norm
        cross = torch.sum(experimental * theoretical, dim=1)
        norm_ctf = torch.sqrt(torch.sum(theoretical.square(), dim=1))
        score = cross / (image_norm * norm_ctf).clamp_min(1.0e-30)
        tolerance = self.config.astigmatism_tolerance_A
        if apply_astigmatism_penalty and tolerance > 0.0:
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
        *,
        ctf_squared: bool = False,
        apply_astigmatism_penalty: bool = True,
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
            return -self._score_2d_batch(
                fit_data,
                d1,
                d2,
                angle,
                ctf_squared=ctf_squared,
                apply_astigmatism_penalty=apply_astigmatism_penalty,
            ).to(self.optimizer_dtype)

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
        score = self._score_2d_batch(
            fit_data,
            d1,
            d2,
            angle,
            ctf_squared=ctf_squared,
            apply_astigmatism_penalty=apply_astigmatism_penalty,
        )
        return d1, d2, angle, score, opt

    def _fit_ctffind5_initial_ctf(
        self,
        power_spectrum: torch.Tensor,
        fitting_pixel_size_A: float,
    ) -> tuple[float, float, float, float]:
        """CTFFIND5 FindRoughDefocus + FindDefocusAstigmatism on power/CTF^2."""
        if power_spectrum.ndim == 2:
            spectrum = power_spectrum[None]
        elif power_spectrum.ndim == 3 and power_spectrum.shape[0] == 1:
            spectrum = power_spectrum
        else:
            raise ValueError("Expected one 2-D initial power spectrum")
        init_cfg = replace(
            self.config,
            box_size=int(spectrum.shape[-1]),
            fit_tilt=False,
            estimate_thickness=False,
            resample_if_pixel_too_small=False,
        )
        initial_angle = _estimate_astigmatism_angle_deg_batch(
            spectrum, fitting_pixel_size_A, init_cfg
        )
        curve = _rotational_average_linear_batch(spectrum, fitting_pixel_size_A)
        _coarse, refined_mean, _opt1 = self._coarse_and_refine_mean_defocus_batch(
            curve, fitting_pixel_size_A, ctf_squared=True
        )
        fit_data = _make_2d_fit_data_batch(
            spectrum, fitting_pixel_size_A, init_cfg
        )
        d1, d2, astig_angle, score, _opt2 = self._refine_2d_batch(
            fit_data,
            refined_mean,
            initial_angle,
            fitting_pixel_size_A,
            ctf_squared=True,
            apply_astigmatism_penalty=False,
        )
        return (
            float(d1[0].item()),
            float(d2[0].item()),
            float(astig_angle[0].item()),
            float(score[0].item()),
        )

    def _score_ctffind5_tilt_candidates(
        self,
        data: _CTFFIND5TiltData,
        axis_deg: torch.Tensor,
        angle_deg: torch.Tensor,
        mean_defocus_A: torch.Tensor,
        astigmatism_A: float,
        astigmatism_angle_rad: float,
    ) -> torch.Tensor:
        """Average local-spectrum CC for candidate tilt planes.

        Unlike the removed CTFTILT backend, this objective has no nominal-angle
        prior.  Axis and angle are searched jointly, as in CTFFIND5.
        """
        axis_deg = axis_deg.to(device=self.device, dtype=self.dtype).reshape(-1)
        angle_deg = angle_deg.to(device=self.device, dtype=self.dtype).reshape(-1)
        mean_defocus_A = mean_defocus_A.to(
            device=self.device, dtype=self.dtype
        ).reshape(-1)
        if mean_defocus_A.numel() == 1 and axis_deg.numel() > 1:
            mean_defocus_A = mean_defocus_A.expand_as(axis_deg)
        if not (axis_deg.numel() == angle_deg.numel() == mean_defocus_A.numel()):
            raise ValueError("Tilt candidate arrays must have matching lengths")

        valid = data.valid_mask
        observed = data.power_values[valid]
        x_A = data.centers_x_A[valid]
        y_A = data.centers_y_A[valid]
        axis_rad = axis_deg * (PI / 180.0)
        angle_rad = angle_deg * (PI / 180.0)
        gx, gy = _ctffind5_gradient_from_axis_angle(axis_rad, angle_rad)
        local_mean = (
            mean_defocus_A[:, None]
            + gx[:, None] * x_A[None]
            + gy[:, None] * y_A[None]
        )

        freq2 = data.frequency_squared_Ainv2
        azimuth = data.azimuth_rad
        astig_component = 0.5 * float(astigmatism_A) * torch.cos(
            2.0 * (azimuth - float(astigmatism_angle_rad))
        )
        scores = torch.zeros(
            axis_deg.numel(), device=self.device, dtype=self.dtype
        )
        count = 0
        tile_batch = max(1, int(self.config.tilt_tile_batch_size))
        for first in range(0, observed.shape[0], tile_batch):
            obs = observed[first:first + tile_batch]
            local = local_mean[:, first:first + tile_batch]
            effective_defocus = local[:, :, None] + astig_component[None, None]
            phase = (
                PI
                * self.wavelength_A
                * freq2[None, None]
                * (
                    effective_defocus
                    - 0.5
                    * self.wavelength_A
                    * self.wavelength_A
                    * freq2[None, None]
                    * self.spherical_aberration_A
                )
                + self.config.fixed_phase_shift_rad
                + self.amplitude_phase_rad
            )
            theoretical = torch.sin(phase).square()
            theoretical = theoretical - theoretical.mean(dim=2, keepdim=True)
            theoretical = theoretical / torch.linalg.vector_norm(
                theoretical, dim=2, keepdim=True
            ).clamp_min(1.0e-20)
            scores += torch.sum(
                theoretical * obs[None], dim=2
            ).sum(dim=1)
            count += int(obs.shape[0])
        return scores / float(max(1, count))

    def _search_ctffind5_tilt(
        self,
        data: _CTFFIND5TiltData,
        mean_defocus_A: float,
        astigmatism_A: float,
        astigmatism_angle_rad: float,
    ) -> tuple[float, float, float, float]:
        """Native CTFFIND5 coarse convention: axis 0..360, tilt 0..max."""
        cfg = self.config
        axes = np.arange(0.0, 360.0, cfg.tilt_axis_step_deg, dtype=np.float32)
        angles = np.arange(
            0.0,
            cfg.tilt_max_angle_deg + 0.5 * cfg.tilt_angle_step_deg,
            cfg.tilt_angle_step_deg,
            dtype=np.float32,
        )
        candidate_axis: list[float] = []
        candidate_angle: list[float] = []
        for angle in angles:
            if abs(float(angle)) < 1.0e-8:
                candidate_axis.append(0.0)
                candidate_angle.append(0.0)
            else:
                candidate_axis.extend(float(v) for v in axes)
                candidate_angle.extend([float(angle)] * len(axes))
        axis_t = torch.tensor(candidate_axis, device=self.device, dtype=self.dtype)
        angle_t = torch.tensor(candidate_angle, device=self.device, dtype=self.dtype)
        mean_t = torch.full_like(axis_t, float(mean_defocus_A))
        all_scores: list[torch.Tensor] = []
        batch = max(1, int(cfg.tilt_candidate_batch_size))
        with torch.inference_mode():
            for first in range(0, axis_t.numel(), batch):
                all_scores.append(self._score_ctffind5_tilt_candidates(
                    data,
                    axis_t[first:first + batch],
                    angle_t[first:first + batch],
                    mean_t[first:first + batch],
                    astigmatism_A,
                    astigmatism_angle_rad,
                ))
        scores = torch.cat(all_scores)
        order = torch.argsort(scores, descending=True)
        best = int(order[0].item())
        best_axis = float(axis_t[best].item())
        best_angle = float(angle_t[best].item())
        best_score = float(scores[best].item())
        # Report a useful gap rather than the difference to a neighboring grid
        # point in the same broad peak.
        second_score = best_score
        for index in order[1:].tolist():
            axis_i = float(axis_t[index].item())
            angle_i = float(angle_t[index].item())
            axis_distance = abs(((axis_i - best_axis + 180.0) % 360.0) - 180.0)
            angle_distance = abs(angle_i - best_angle)
            if (
                axis_distance >= max(2.0 * cfg.tilt_axis_step_deg, 10.0)
                or angle_distance >= max(2.0 * cfg.tilt_angle_step_deg, 5.0)
            ):
                second_score = float(scores[index].item())
                break
        return best_axis, best_angle, best_score, best_score - second_score

    def _refine_ctffind5_tilt(
        self,
        data: _CTFFIND5TiltData,
        coarse_axis_deg: float,
        coarse_angle_deg: float,
        initial_mean_defocus_A: float,
        astigmatism_A: float,
        astigmatism_angle_rad: float,
    ) -> tuple[float, float, float, float, _BatchedOptimizationResult]:
        cfg = self.config
        axis_scale = max(cfg.tilt_axis_step_deg, 1.0)
        angle_scale = max(cfg.tilt_angle_step_deg, 1.0)
        defocus_scale = max(cfg.defocus_search_step_A, 100.0)
        x0 = torch.zeros((1, 3), device=self.device, dtype=self.optimizer_dtype)
        lower = torch.empty_like(x0)
        upper = torch.empty_like(x0)
        lower[0, 0] = -cfg.tilt_refine_axis_half_range_deg / axis_scale
        upper[0, 0] = cfg.tilt_refine_axis_half_range_deg / axis_scale
        lower_angle = max(
            0.0,
            coarse_angle_deg - cfg.tilt_refine_angle_half_range_deg,
        )
        upper_angle = min(
            cfg.tilt_max_angle_deg,
            coarse_angle_deg + cfg.tilt_refine_angle_half_range_deg,
        )
        lower[0, 1] = (lower_angle - coarse_angle_deg) / angle_scale
        upper[0, 1] = (upper_angle - coarse_angle_deg) / angle_scale
        lower_df = max(
            cfg.minimum_defocus_A,
            initial_mean_defocus_A - cfg.tilt_refine_defocus_half_range_A,
        )
        upper_df = min(
            cfg.maximum_defocus_A,
            initial_mean_defocus_A + cfg.tilt_refine_defocus_half_range_A,
        )
        lower[0, 2] = (lower_df - initial_mean_defocus_A) / defocus_scale
        upper[0, 2] = (upper_df - initial_mean_defocus_A) / defocus_scale

        def decode(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            axis = coarse_axis_deg + x[:, 0] * axis_scale
            angle = coarse_angle_deg + x[:, 1] * angle_scale
            mean_df = initial_mean_defocus_A + x[:, 2] * defocus_scale
            return axis, angle, mean_df

        def objective(x: torch.Tensor) -> torch.Tensor:
            axis, angle, mean_df = decode(x)
            return -self._score_ctffind5_tilt_candidates(
                data, axis, angle, mean_df,
                astigmatism_A, astigmatism_angle_rad,
            ).to(self.optimizer_dtype)

        opt = _batched_powell(
            objective,
            x0,
            lower,
            upper,
            xtol=cfg.powell_xtol,
            ftol=cfg.powell_ftol,
            maxiter=cfg.tilt_refine_maxiter,
            line_maxiter=cfg.powell_line_maxiter,
            check_interval=cfg.optimizer_check_interval,
        )
        axis, angle, mean_df = decode(opt.x)
        axis_value, angle_value = _ctffind5_canonical_tilt(
            float(axis[0].item()), float(angle[0].item())
        )
        score = -float(opt.fun[0].item())
        return axis_value, angle_value, float(mean_df[0].item()), score, opt

    def _calculate_ctffind5_tilt_corrected_spectrum(
        self,
        micrograph: np.ndarray | torch.Tensor,
        pixel_size_A: float,
        axis_deg: float,
        angle_deg: float,
        mean_defocus_A: float,
        timings: Optional[dict[str, float]] = None,
    ) -> tuple[torch.Tensor, float, int]:
        """Return CTFFIND5's raw tilt-corrected amplitude spectrum.

        The section coordinates and arithmetic match 0.5.1, but a regular
        zero-padded ``Tensor.unfold`` view replaces thousands of NumPy patch
        allocations and repeated host-to-device transfers.  Only the current
        section batch is materialized.
        """
        cfg = self.config
        timing_enabled = timings is not None
        local_timings = timings if timings is not None else {}

        with _timed_stage(
            timing_enabled,
            self.device,
            local_timings,
            'tilt_correction_input_and_geometry_s',
        ):
            if isinstance(micrograph, torch.Tensor):
                image = micrograph.to(device=self.device, dtype=self.dtype)
            else:
                array = np.asarray(micrograph, dtype=np.float32)
                if array.ndim != 2 or not np.isfinite(array).all():
                    raise ValueError(
                        'Tilt correction expects one finite 2-D micrograph'
                    )
                image = torch.as_tensor(
                    array, device=self.device, dtype=self.dtype
                )
            if image.ndim != 2:
                raise ValueError(
                    'Tilt correction expects one finite 2-D micrograph'
                )
            height, width = int(image.shape[0]), int(image.shape[1])
            box = int(cfg.box_size)
            if min(height, width) < box:
                raise RuntimeError(
                    f'Micrograph {width}x{height} is smaller than '
                    f'corrected-spectrum box {box}'
                )
            if not math.isfinite(mean_defocus_A) or mean_defocus_A <= 100.0:
                raise ValueError(
                    'Mean defocus must exceed 100 A for tilt correction'
                )

            if (
                cfg.resample_if_pixel_too_small
                and pixel_size_A < cfg.target_pixel_size_after_resampling_A
            ):
                temporary_box = _round_half_away_from_zero(
                    float(box) / float(pixel_size_A)
                    * float(cfg.target_pixel_size_after_resampling_A)
                )
                if temporary_box % 2:
                    temporary_box += 1
                fitting_pixel_size_A = (
                    float(pixel_size_A) * float(temporary_box) / float(box)
                )
                base_resize_dimension = int(temporary_box)
            else:
                fitting_pixel_size_A = float(pixel_size_A)
                base_resize_dimension = box

            n_sec = max(width // box, height // box)
            if n_sec % 2 == 0:
                n_sec += 1
            n_sec = max(1, n_sec)
            subsection_x = width // (n_sec + 1)
            subsection_y = height // (n_sec + 1)
            if subsection_x % 2:
                subsection_x -= 1
            if subsection_y % 2:
                subsection_y -= 1
            subsection_x = max(2, subsection_x)
            subsection_y = max(2, subsection_y)
            step_x = subsection_x // 2
            step_y = subsection_y // 2
            coordinate_extent = n_sec - 1
            grid_count_x = 2 * coordinate_extent + 1
            grid_count_y = 2 * coordinate_extent + 1

            axis = torch.tensor(
                float(axis_deg) * PI / 180.0,
                device=self.device,
                dtype=self.dtype,
            )
            angle = torch.tensor(
                float(angle_deg) * PI / 180.0,
                device=self.device,
                dtype=self.dtype,
            )
            gx, gy = _ctffind5_gradient_from_axis_angle(axis, angle)
            gx_value = float(gx.item())
            gy_value = float(gy.item())

            flat_indices: list[int] = []
            local_defocus: list[float] = []
            for grid_y, iy in enumerate(
                range(-coordinate_extent, coordinate_extent + 1)
            ):
                for grid_x, ix in enumerate(
                    range(-coordinate_extent, coordinate_extent + 1)
                ):
                    x_A = (
                        float(ix) * float(subsection_x) * 0.5
                        * float(pixel_size_A)
                    )
                    y_A = (
                        float(iy) * float(subsection_y) * 0.5
                        * float(pixel_size_A)
                    )
                    local_df = (
                        float(mean_defocus_A)
                        + gx_value * x_A + gy_value * y_A
                    )
                    if math.isfinite(local_df) and local_df > 1.0:
                        flat_indices.append(grid_y * grid_count_x + grid_x)
                        local_defocus.append(local_df)
            if len(flat_indices) < cfg.tilt_min_tiles:
                raise RuntimeError(
                    f'Only {len(flat_indices)} corrected-spectrum sections '
                    'have positive defocus'
                )

            first_offset_x = -coordinate_extent * step_x
            first_offset_y = -coordinate_extent * step_y
            first_start_x = width // 2 + first_offset_x - box // 2
            first_start_y = height // 2 + first_offset_y - box // 2
            last_start_x = (
                first_start_x + (grid_count_x - 1) * step_x
            )
            last_start_y = (
                first_start_y + (grid_count_y - 1) * step_y
            )
            pad_left = max(0, -first_start_x)
            pad_top = max(0, -first_start_y)
            pad_right = max(0, last_start_x + box - width)
            pad_bottom = max(0, last_start_y + box - height)
            padded = F.pad(
                image[None, None],
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='constant',
                value=0.0,
            )[0, 0]
            region_start_x = first_start_x + pad_left
            region_start_y = first_start_y + pad_top
            region_width = (grid_count_x - 1) * step_x + box
            region_height = (grid_count_y - 1) * step_y + box
            region = padded[
                region_start_y:region_start_y + region_height,
                region_start_x:region_start_x + region_width,
            ]
            patch_grid = (
                region.unfold(0, box, step_y)
                .unfold(1, box, step_x)
            )
            if tuple(patch_grid.shape[:2]) != (
                grid_count_y, grid_count_x
            ):
                raise RuntimeError(
                    'Internal corrected-spectrum patch-grid geometry mismatch: '
                    f'{tuple(patch_grid.shape[:2])} vs '
                    f'{(grid_count_y, grid_count_x)}'
                )

            flat_index_t = torch.as_tensor(
                flat_indices, device=self.device, dtype=torch.long
            )
            grid_y_t = torch.div(
                flat_index_t, grid_count_x, rounding_mode='floor'
            )
            grid_x_t = torch.remainder(flat_index_t, grid_count_x)
            local_df_t = torch.as_tensor(
                np.asarray(local_defocus, dtype=np.float32),
                device=self.device,
                dtype=self.dtype,
            )
            stretch = torch.sqrt(
                torch.abs(local_df_t) / float(mean_defocus_A)
            )
            # One host transfer for all integer resize dimensions, instead of
            # one synchronization for every section batch.
            resize_dimensions = [
                _closest_even_dimension(
                    float(base_resize_dimension) * float(value), minimum=2
                )
                for value in stretch.detach().cpu().tolist()
            ]

        sum_amplitude = torch.zeros(
            (box, box), device=self.device, dtype=self.dtype
        )
        counts = torch.zeros_like(sum_amplitude)
        batch_size = max(1, int(cfg.tilt_tile_batch_size))
        used = 0

        with _timed_stage(
            timing_enabled,
            self.device,
            local_timings,
            'tilt_correction_patch_fft_resize_accumulate_s',
        ):
            for first in range(0, len(flat_indices), batch_size):
                stop = min(len(flat_indices), first + batch_size)
                sections = patch_grid[
                    grid_y_t[first:stop], grid_x_t[first:stop]
                ].contiguous()
                sections = _cosine_rectangular_mask_batch(sections)
                fft = torch.fft.fft2(sections)
                fft[:, 0, 0] = 0.0
                amplitude = torch.fft.fftshift(
                    torch.abs(fft), dim=(-2, -1)
                )

                current_dimensions = resize_dimensions[first:stop]
                for dimension in sorted(set(current_dimensions)):
                    indices = [
                        index for index, value in enumerate(current_dimensions)
                        if value == dimension
                    ]
                    index_t = torch.as_tensor(
                        indices, device=self.device, dtype=torch.int64
                    )
                    resized = _fourier_resize_centered_real_batch(
                        amplitude.index_select(0, index_t), int(dimension)
                    )
                    clipped = _center_crop_or_pad_batch(
                        resized, box, padding_value=0.0
                    )
                    sum_amplitude += clipped.sum(dim=0)
                    counts += (
                        (clipped != 0.0).to(self.dtype).sum(dim=0)
                    )
                    used += int(clipped.shape[0])

        with _timed_stage(
            timing_enabled,
            self.device,
            local_timings,
            'tilt_correction_finalize_s',
        ):
            raw = sum_amplitude / counts.clamp_min(1.0)
            raw[counts <= 0.0] = 0.0
            raw[box // 2, box // 2] = 0.0
            if not bool(torch.isfinite(raw).all().item()):
                raise RuntimeError(
                    'Tilt-corrected raw amplitude spectrum contains NaN/Inf'
                )
        return raw.contiguous(), float(fitting_pixel_size_A), int(used)

    def _ctffind5_local_defocus_diagnostics(
        self,
        data: _CTFFIND5TiltData,
        predicted_defocus_A: np.ndarray,
        astigmatism_A: float,
        astigmatism_angle_rad: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        count = len(predicted_defocus_A)
        predicted_all = torch.as_tensor(
            np.asarray(predicted_defocus_A, dtype=np.float32),
            device=self.device,
            dtype=self.dtype,
        )
        measured_all = torch.full(
            (count,), float('nan'), device=self.device, dtype=self.dtype
        )
        cc_all = torch.full_like(measured_all, float('nan'))
        valid_indices = torch.nonzero(
            data.valid_mask, as_tuple=False
        ).flatten()
        offsets = torch.arange(
            -cfg.tilt_diagnostic_defocus_range_A,
            cfg.tilt_diagnostic_defocus_range_A
            + 0.5 * cfg.tilt_diagnostic_defocus_step_A,
            cfg.tilt_diagnostic_defocus_step_A,
            device=self.device,
            dtype=self.dtype,
        )
        freq2 = data.frequency_squared_Ainv2
        azimuth = data.azimuth_rad
        astig_component = 0.5 * float(astigmatism_A) * torch.cos(
            2.0 * (azimuth - float(astigmatism_angle_rad))
        )
        batch = max(1, int(cfg.tilt_tile_batch_size))
        with torch.inference_mode():
            for first in range(0, valid_indices.numel(), batch):
                idx = valid_indices[first:first + batch]
                obs = data.power_values.index_select(0, idx)
                predicted = predicted_all.index_select(0, idx)
                means = predicted[:, None] + offsets[None]
                effective = means[:, :, None] + astig_component[None, None]
                phase = (
                    PI * self.wavelength_A * freq2[None, None]
                    * (
                        effective
                        - 0.5 * self.wavelength_A * self.wavelength_A
                        * freq2[None, None] * self.spherical_aberration_A
                    )
                    + self.config.fixed_phase_shift_rad
                    + self.amplitude_phase_rad
                )
                theory = torch.sin(phase).square()
                theory = theory - theory.mean(dim=2, keepdim=True)
                obs_centered = obs - obs.mean(dim=1, keepdim=True)
                numerator = torch.sum(
                    theory * obs_centered[:, None], dim=2
                )
                denominator = torch.sqrt(
                    torch.sum(theory.square(), dim=2)
                    * torch.sum(obs_centered.square(), dim=1)[:, None]
                ).clamp_min(1.0e-20)
                scores = numerator / denominator
                best_score, best_index = torch.max(scores, dim=1)
                best_mean = means.gather(1, best_index[:, None])[:, 0]
                measured_all.index_copy_(0, idx, best_mean)
                cc_all.index_copy_(0, idx, best_score)
        return (
            measured_all.detach().cpu().numpy().astype(
                np.float64, copy=False
            ),
            cc_all.detach().cpu().numpy().astype(np.float64, copy=False),
        )

    def _prepare_ice_thickness_single(
        self,
        masked_spectrum: torch.Tensor,
        fitting_pixel_size_A: float,
        defocus1_A: float,
        defocus2_A: float,
        astigmatism_angle_rad: float,
        initial_epa: Optional[_EPAStatistics],
    ) -> _ThicknessPreparation:
        """Run EPA detrending and the source-compatible 1-D node grid."""
        cfg = self.config
        timing_enabled = bool(cfg.timing)
        timings: dict[str, float] = {}
        if timing_enabled:
            _synchronize_if_cuda(self.device)
        stage_started = time.perf_counter()

        def checkpoint(name: str) -> None:
            nonlocal stage_started
            if not timing_enabled:
                return
            _synchronize_if_cuda(self.device)
            now = time.perf_counter()
            _add_timing(timings, name, now - stage_started)
            stage_started = now

        if masked_spectrum.ndim != 2 or masked_spectrum.shape[0] != masked_spectrum.shape[1]:
            raise ValueError("Thickness fitting expects one square masked spectrum")
        if not math.isfinite(fitting_pixel_size_A) or fitting_pixel_size_A <= 0.0:
            raise ValueError("fitting_pixel_size_A must be positive and finite")
        spectrum = masked_spectrum.to(device=self.device, dtype=self.dtype)
        if initial_epa is None:
            initial_epa = _compute_epa_statistics(
                spectrum,
                fitting_pixel_size_A,
                cfg,
                float(defocus1_A),
                float(defocus2_A),
                float(astigmatism_angle_rad),
                self.wavelength_A,
                self.spherical_aberration_A,
                self.amplitude_phase_rad,
                cfg.fixed_phase_shift_rad,
                theoretical_thickness_A=None,
                node_mode=False,
                rounded_square=cfg.thickness_use_rounded_square,
            )
        checkpoint("thickness_initial_epa_s")

        initial_good_resolution = float(initial_epa.good_fit_resolution_A)
        if not math.isfinite(initial_good_resolution) or initial_good_resolution <= 0.0:
            raise RuntimeError("EPA/FRC did not produce a finite initial fit cutoff")
        node_seed_A = initial_good_resolution * initial_good_resolution / self.wavelength_A
        node_seed_A = float(np.clip(node_seed_A, cfg.thickness_min_A, cfg.thickness_max_A))

        frequencies_np = np.asarray(initial_epa.spatial_frequency_Ainv, dtype=np.float64)
        observed_raw_np = np.asarray(initial_epa.observed_profile, dtype=np.float64).copy()
        if observed_raw_np.size < 12:
            raise RuntimeError("EPA profile contains too few bins")
        fit_indices = np.arange(1, observed_raw_np.size, dtype=np.int64)
        finite = np.isfinite(observed_raw_np[fit_indices]) & np.isfinite(
            frequencies_np[fit_indices]
        )
        if np.count_nonzero(finite) < 8:
            raise RuntimeError("EPA profile has too few finite bins for polynomial detrending")
        coefficients = np.polyfit(
            frequencies_np[fit_indices][finite],
            observed_raw_np[fit_indices][finite],
            deg=3,
        )
        polynomial = np.polyval(coefficients, frequencies_np)
        observed_detrended_np = observed_raw_np - polynomial + 0.5
        observed_detrended_np[0] = observed_raw_np[0]

        low_frequency = 1.0 / float(cfg.thickness_low_resolution_A)
        high_frequency = min(
            1.0 / float(cfg.thickness_high_resolution_A),
            0.5 / float(fitting_pixel_size_A),
        )
        fit_mask_np = (
            np.isfinite(observed_detrended_np)
            & np.isfinite(frequencies_np)
            & (frequencies_np >= low_frequency)
            & (frequencies_np <= high_frequency)
        )
        if np.count_nonzero(fit_mask_np) < 12:
            raise RuntimeError("The 1-D thickness fitting support contains too few bins")
        freq2_1d = torch.as_tensor(
            frequencies_np[fit_mask_np] ** 2,
            device=self.device,
            dtype=self.dtype,
        )
        observed_1d = torch.as_tensor(
            observed_detrended_np[fit_mask_np],
            device=self.device,
            dtype=self.dtype,
        )
        observed_1d_centered = observed_1d - observed_1d.mean()
        observed_1d_norm = torch.linalg.vector_norm(observed_1d_centered).clamp_min(1.0e-20)
        profile_azimuth = float(initial_epa.profile_azimuth_rad)
        profile_defocus = float(initial_epa.profile_defocus_A)
        def1_offset = float(defocus1_A) - profile_defocus
        def2_offset = float(defocus2_A) - profile_defocus
        azimuth_1d = torch.full(
            (1, freq2_1d.numel()),
            profile_azimuth,
            device=self.device,
            dtype=self.dtype,
        )
        fixed_angle = torch.tensor(
            float(astigmatism_angle_rad), device=self.device, dtype=self.dtype
        )
        checkpoint("thickness_prepare_1d_s")

        def score_1d(
            thickness_values: torch.Tensor,
            profile_df_values: torch.Tensor,
        ) -> torch.Tensor:
            count = int(thickness_values.numel())
            model = _finite_thickness_power_model(
                freq2_1d[None],
                azimuth_1d,
                profile_df_values + def1_offset,
                profile_df_values + def2_offset,
                fixed_angle.expand(count),
                thickness_values,
                self.wavelength_A,
                self.spherical_aberration_A,
                self.amplitude_phase_rad,
                cfg.fixed_phase_shift_rad,
                cfg.thickness_use_rounded_square,
            )
            model_centered = model - model.mean(dim=1, keepdim=True)
            denominator = (
                torch.linalg.vector_norm(model_centered, dim=1).clamp_min(1.0e-20)
                * observed_1d_norm
            )
            return torch.sum(
                model_centered * observed_1d_centered[None], dim=1
            ) / denominator

        thickness_grid = torch.arange(
            float(cfg.thickness_min_A),
            float(cfg.thickness_max_A) + 0.5 * float(cfg.thickness_step_A),
            float(cfg.thickness_step_A),
            device=self.device,
            dtype=self.dtype,
        )
        df_lower = max(
            float(cfg.minimum_defocus_A),
            profile_defocus - float(cfg.thickness_defocus_search_range_A),
        )
        df_upper = min(
            float(cfg.maximum_defocus_A),
            profile_defocus + float(cfg.thickness_defocus_search_range_A),
        )
        profile_df_grid = torch.arange(
            df_lower,
            df_upper + 0.5 * float(cfg.thickness_defocus_step_A),
            float(cfg.thickness_defocus_step_A),
            device=self.device,
            dtype=self.dtype,
        )
        if thickness_grid.numel() == 0 or profile_df_grid.numel() == 0:
            raise RuntimeError("Thickness/defocus brute-force grid is empty")
        tt, dd = torch.meshgrid(thickness_grid, profile_df_grid, indexing="ij")
        pair_t = tt.reshape(-1)
        pair_df = dd.reshape(-1)
        score_chunks: list[torch.Tensor] = []
        candidate_batch = max(1, int(cfg.thickness_candidate_batch_size))
        with torch.inference_mode():
            for first in range(0, pair_t.numel(), candidate_batch):
                score_chunks.append(
                    score_1d(
                        pair_t[first:first + candidate_batch],
                        pair_df[first:first + candidate_batch],
                    )
                )
        scores_1d = torch.cat(score_chunks)
        best_score = torch.max(scores_1d)
        tied = torch.nonzero(
            scores_1d >= best_score - 1.0e-7, as_tuple=False
        ).flatten()
        if tied.numel() > 1:
            cost = torch.abs(pair_t[tied] - node_seed_A) / max(
                float(cfg.thickness_step_A), 1.0
            )
            cost += (
                1.0e-3
                * torch.abs(pair_df[tied] - profile_defocus)
                / max(float(cfg.thickness_defocus_step_A), 1.0)
            )
            best_index = int(tied[int(torch.argmin(cost).item())].item())
        else:
            best_index = int(torch.argmax(scores_1d).item())
        coarse_t = float(pair_t[best_index].item())
        coarse_profile_df = float(pair_df[best_index].item())
        coarse_score = float(scores_1d[best_index].item())
        checkpoint("thickness_1d_grid_search_s")

        grid_debug = None
        if cfg.debug:
            grid_debug = {
                "thickness_A": pair_t.detach().cpu().numpy().astype(np.float64, copy=False),
                "profile_defocus_A": pair_df.detach().cpu().numpy().astype(np.float64, copy=False),
                "score": scores_1d.detach().cpu().numpy().astype(np.float64, copy=False),
            }
        return _ThicknessPreparation(
            spectrum=spectrum,
            fitting_pixel_size_A=float(fitting_pixel_size_A),
            initial_epa=initial_epa,
            node_seed_A=float(node_seed_A),
            observed_detrended_np=observed_detrended_np,
            polynomial_coefficients=np.asarray(coefficients, dtype=np.float64),
            coarse_thickness_A=float(coarse_t),
            coarse_profile_defocus_A=float(coarse_profile_df),
            coarse_score=float(coarse_score),
            coarse_defocus1_A=float(coarse_profile_df + def1_offset),
            coarse_defocus2_A=float(coarse_profile_df + def2_offset),
            initial_astigmatism_angle_rad=float(astigmatism_angle_rad),
            one_d_grid_debug=grid_debug,
            timings=timings,
        )

    def _get_thickness_2d_geometry(
        self,
        size: int,
        fitting_pixel_size_A: float,
    ) -> dict[str, torch.Tensor | int]:
        cfg = self.config
        key = (
            int(size),
            round(float(fitting_pixel_size_A), 10),
            self.device.type,
            self.device.index if self.device.type == "cuda" else None,
            str(self.dtype),
            round(float(cfg.thickness_low_resolution_A), 8),
            round(float(cfg.thickness_high_resolution_A), 8),
        )
        cached = self._thickness_2d_geometry_cache.get(key)
        if cached is not None:
            return cached
        center = int(size) // 2
        y = torch.arange(size, device=self.device, dtype=self.dtype)
        x = torch.arange(center, device=self.device, dtype=self.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        fx = (xx - center) / (float(size) * float(fitting_pixel_size_A))
        fy = (yy - center) / (float(size) * float(fitting_pixel_size_A))
        frequency_squared = fx.square() + fy.square()
        low_frequency = 1.0 / float(cfg.thickness_low_resolution_A)
        high_frequency = min(
            1.0 / float(cfg.thickness_high_resolution_A),
            0.5 / float(fitting_pixel_size_A),
        )
        cross = 10
        support = (
            (frequency_squared > low_frequency * low_frequency)
            & (frequency_squared < high_frequency * high_frequency)
            & (xx < center - cross)
            & ((yy < center - cross) | (yy > center + cross))
        )
        frequency_squared = frequency_squared[support].contiguous()
        azimuth = torch.atan2(fy[support], fx[support])
        defocus_coefficient = (PI * self.wavelength_A * frequency_squared).contiguous()
        phase_base = (
            -0.5
            * PI
            * self.wavelength_A ** 3
            * self.spherical_aberration_A
            * frequency_squared.square()
            + cfg.fixed_phase_shift_rad
            + self.amplitude_phase_rad
        ).contiguous()
        cached = {
            "support": support,
            "center": int(center),
            "cos2azimuth": torch.cos(2.0 * azimuth).contiguous(),
            "sin2azimuth": torch.sin(2.0 * azimuth).contiguous(),
            "defocus_coefficient": defocus_coefficient,
            "phase_base": phase_base,
            "thickness_coefficient": defocus_coefficient,
            "number_of_values": int(frequency_squared.numel()),
        }
        self._thickness_2d_geometry_cache[key] = cached
        return cached

    def _refine_ice_thickness_2d_batch(
        self,
        preparations: Sequence[_ThicknessPreparation],
    ) -> tuple[list[tuple[float, float, float, float, float]], float, float]:
        """Evaluate independent four-parameter Powell searches as one batch."""
        if not preparations:
            return [], 0.0, 0.0
        cfg = self.config
        if cfg.timing:
            _synchronize_if_cuda(self.device)
        preparation_started = time.perf_counter()
        size = int(preparations[0].spectrum.shape[-1])
        fitting_pixel_size_A = float(preparations[0].fitting_pixel_size_A)
        for item in preparations:
            if (
                int(item.spectrum.shape[-1]) != size
                or abs(float(item.fitting_pixel_size_A) - fitting_pixel_size_A) > 1.0e-9
            ):
                raise ValueError("Batched thickness refinement requires compatible spectra")
        geometry = self._get_thickness_2d_geometry(size, fitting_pixel_size_A)
        support = geometry["support"]
        center = int(geometry["center"])
        spectra = torch.stack([item.spectrum for item in preparations], dim=0)
        observed = spectra[:, :, :center][:, support]
        if not bool(torch.all(torch.isfinite(observed)).item()):
            raise RuntimeError("The 2-D thickness spectrum contains non-finite pixels")
        if observed.shape[1] < 32:
            raise RuntimeError("The 2-D thickness fitting support contains too few pixels")
        image_mean = observed.mean(dim=1)
        norm_image = torch.sum(
            (observed - image_mean[:, None]).square(), dim=1
        ).clamp_min(1.0e-20)
        number_of_values = float(observed.shape[1])

        coarse_t = torch.tensor(
            [item.coarse_thickness_A for item in preparations],
            device=self.device,
            dtype=self.optimizer_dtype,
        )
        coarse_d1 = torch.tensor(
            [item.coarse_defocus1_A for item in preparations],
            device=self.device,
            dtype=self.optimizer_dtype,
        )
        coarse_d2 = torch.tensor(
            [item.coarse_defocus2_A for item in preparations],
            device=self.device,
            dtype=self.optimizer_dtype,
        )
        angle0 = torch.tensor(
            [item.initial_astigmatism_angle_rad for item in preparations],
            device=self.device,
            dtype=self.optimizer_dtype,
        )
        t_scale = max(float(cfg.thickness_step_A), 10.0)
        df_scale = max(float(cfg.thickness_defocus_step_A), 100.0)
        angle_scale = 0.05
        batch = len(preparations)
        x0 = torch.zeros((batch, 4), device=self.device, dtype=self.optimizer_dtype)
        lower = torch.empty_like(x0)
        upper = torch.empty_like(x0)
        lower[:, 0] = (float(cfg.thickness_min_A) - coarse_t) / t_scale
        upper[:, 0] = (float(cfg.thickness_max_A) - coarse_t) / t_scale
        local_df = max(float(cfg.thickness_defocus_search_range_A), 500.0)
        lower[:, 1] = (
            torch.clamp(coarse_d1 - local_df, min=float(cfg.minimum_defocus_A))
            - coarse_d1
        ) / df_scale
        upper[:, 1] = (
            torch.clamp(coarse_d1 + local_df, max=float(cfg.maximum_defocus_A))
            - coarse_d1
        ) / df_scale
        lower[:, 2] = (
            torch.clamp(coarse_d2 - local_df, min=float(cfg.minimum_defocus_A))
            - coarse_d2
        ) / df_scale
        upper[:, 2] = (
            torch.clamp(coarse_d2 + local_df, max=float(cfg.maximum_defocus_A))
            - coarse_d2
        ) / df_scale
        lower[:, 3] = -0.5 * PI / angle_scale
        upper[:, 3] = 0.5 * PI / angle_scale

        def decode(
            coordinates: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                coarse_t + coordinates[:, 0] * t_scale,
                coarse_d1 + coordinates[:, 1] * df_scale,
                coarse_d2 + coordinates[:, 2] * df_scale,
                angle0 + coordinates[:, 3] * angle_scale,
            )

        cos2azimuth = geometry["cos2azimuth"]
        sin2azimuth = geometry["sin2azimuth"]
        defocus_coefficient = geometry["defocus_coefficient"]
        phase_base = geometry["phase_base"]
        thickness_coefficient = geometry["thickness_coefficient"]

        def score_fused(
            thickness_values: torch.Tensor,
            d1_values: torch.Tensor,
            d2_values: torch.Tensor,
            angle_values: torch.Tensor,
        ) -> torch.Tensor:
            # Optimizer coordinates remain float64, but CTF generation uses the
            # same once-rounded float32 physical parameters as the validated
            # ordinary CTFFIND scorer.  This also avoids consumer-GPU FP64
            # throughput becoming the dominant cost.
            thickness = thickness_values.to(self.dtype)
            d1 = d1_values.to(self.dtype)
            d2 = d2_values.to(self.dtype)
            angle = angle_values.to(self.dtype)
            mean_defocus = 0.5 * (d1 + d2)
            half_difference = 0.5 * (d1 - d2)
            astigmatic_direction = (
                torch.cos(2.0 * angle)[:, None] * cos2azimuth[None]
                + torch.sin(2.0 * angle)[:, None] * sin2azimuth[None]
            )
            effective_defocus = (
                mean_defocus[:, None]
                + half_difference[:, None] * astigmatic_direction
            )
            phase = (
                defocus_coefficient[None] * effective_defocus
                + phase_base[None]
            )
            argument = thickness_coefficient[None] * thickness[:, None]
            if cfg.thickness_use_rounded_square:
                modulation = _rounded_square_torch(argument)
            else:
                # argument/pi == lambda*s^2*t, matching torch.sinc's convention.
                modulation = torch.sinc(argument / PI)
            model = -0.5 * modulation * torch.cos(2.0 * phase)
            if cfg.thickness_downweight_nodes:
                node_window = _rounded_square_torch(argument)
                model = torch.where(
                    torch.abs(node_window) > 0.99,
                    model,
                    torch.zeros_like(model),
                )
            cross_product = torch.sum(model * observed, dim=1)
            model_sum = torch.sum(model, dim=1)
            model_square_sum = torch.sum(model.square(), dim=1)
            model_variance = (
                model_square_sum - model_sum.square() / number_of_values
            ).clamp_min(1.0e-20)
            return (
                (cross_product - image_mean * model_sum)
                / torch.sqrt(norm_image * model_variance)
            )

        def objective(coordinates: torch.Tensor) -> torch.Tensor:
            thickness, d1, d2, angle = decode(coordinates)
            return -score_fused(thickness, d1, d2, angle).to(self.optimizer_dtype)

        if cfg.timing:
            _synchronize_if_cuda(self.device)
        preparation_elapsed = time.perf_counter() - preparation_started
        started = time.perf_counter()
        optimization = _batched_powell(
            objective,
            x0,
            lower,
            upper,
            xtol=cfg.powell_xtol,
            ftol=cfg.powell_ftol,
            maxiter=cfg.thickness_refine_maxiter,
            line_maxiter=cfg.powell_line_maxiter,
            check_interval=cfg.optimizer_check_interval,
        )
        thickness, d1, d2, angle = decode(optimization.x)
        score = -optimization.fun
        if cfg.timing:
            _synchronize_if_cuda(self.device)
        elapsed = time.perf_counter() - started
        host = torch.stack((thickness, d1, d2, angle, score), dim=1).detach().cpu().numpy()
        return (
            [tuple(float(value) for value in row) for row in host],
            float(preparation_elapsed),
            float(elapsed),
        )

    def _finalize_ice_thickness(
        self,
        preparation: _ThicknessPreparation,
        final_values: tuple[float, float, float, float, float],
        shared_prepare_seconds: float,
        shared_refine_seconds: float,
    ) -> _ThicknessFitDetails:
        cfg = self.config
        final_t, final_d1, final_d2, final_angle, final_score = final_values
        final_d1, final_d2, final_angle = _enforce_ctffind_convention(
            final_d1, final_d2, final_angle
        )
        if cfg.timing:
            _synchronize_if_cuda(self.device)
        started = time.perf_counter()
        final_epa = _compute_epa_statistics(
            preparation.spectrum,
            preparation.fitting_pixel_size_A,
            cfg,
            final_d1,
            final_d2,
            final_angle,
            self.wavelength_A,
            self.spherical_aberration_A,
            self.amplitude_phase_rad,
            cfg.fixed_phase_shift_rad,
            theoretical_thickness_A=final_t,
            node_mode=True,
            rounded_square=cfg.thickness_use_rounded_square,
        )
        if cfg.timing:
            _synchronize_if_cuda(self.device)
        final_epa_seconds = time.perf_counter() - started
        timings = dict(preparation.timings)
        if cfg.timing:
            timings["thickness_prepare_2d_s"] = float(shared_prepare_seconds)
            timings["thickness_2d_refine_s"] = float(shared_refine_seconds)
            timings["thickness_final_epa_s"] = float(final_epa_seconds)
            timings["thickness_total_s"] = float(sum(timings.values()))

        initial_epa = preparation.initial_epa
        debug = None
        if cfg.debug:
            grid = preparation.one_d_grid_debug
            debug = {
                "initial": {
                    "spatial_frequency_Ainv": initial_epa.spatial_frequency_Ainv.tolist(),
                    "observed_epa": initial_epa.observed_profile.tolist(),
                    "renormalized_epa": initial_epa.renormalized_profile.tolist(),
                    "theoretical_profile": initial_epa.theoretical_profile.tolist(),
                    "frc": initial_epa.fit_frc.tolist(),
                    "frc_sigma": initial_epa.fit_frc_sigma.tolist(),
                    "last_good_bin": int(initial_epa.last_good_bin),
                    "good_fit_resolution_A": float(initial_epa.good_fit_resolution_A),
                    "node_seed_A": float(preparation.node_seed_A),
                    "epa_pre_phase_rad": initial_epa.pre_phase_rad.tolist(),
                    "epa_pre_values": initial_epa.pre_values.tolist(),
                    "epa_pre_counts": initial_epa.pre_counts.tolist(),
                    "epa_post_phase_rad": initial_epa.post_phase_rad.tolist(),
                    "epa_post_values": initial_epa.post_values.tolist(),
                    "epa_post_counts": initial_epa.post_counts.tolist(),
                },
                "detrended_epa": preparation.observed_detrended_np.tolist(),
                "polynomial_coefficients": preparation.polynomial_coefficients.tolist(),
                "one_d_grid": {
                    "thickness_A": [] if grid is None else grid["thickness_A"].astype(float).tolist(),
                    "profile_defocus_A": [] if grid is None else grid["profile_defocus_A"].astype(float).tolist(),
                    "score": [] if grid is None else grid["score"].astype(float).tolist(),
                    "best_thickness_A": float(preparation.coarse_thickness_A),
                    "best_profile_defocus_A": float(preparation.coarse_profile_defocus_A),
                },
                "final": {
                    "spatial_frequency_Ainv": final_epa.spatial_frequency_Ainv.tolist(),
                    "observed_epa": final_epa.observed_profile.tolist(),
                    "renormalized_epa": final_epa.renormalized_profile.tolist(),
                    "theoretical_profile": final_epa.theoretical_profile.tolist(),
                    "frc": final_epa.fit_frc.tolist(),
                    "frc_sigma": final_epa.fit_frc_sigma.tolist(),
                    "last_good_bin": int(final_epa.last_good_bin),
                    "good_fit_resolution_A": float(final_epa.good_fit_resolution_A),
                },
            }
        return _ThicknessFitDetails(
            success=True,
            thickness_A=float(final_t),
            score=float(final_score),
            coarse_thickness_A=float(preparation.coarse_thickness_A),
            message=(
                "CTFFIND5 EPA node fit; "
                f"initial cutoff={initial_epa.good_fit_resolution_A:.3g} A "
                f"(seed={preparation.node_seed_A:.1f} A), "
                f"1D={preparation.coarse_thickness_A:.1f} A/"
                f"profile_df={preparation.coarse_profile_defocus_A:.1f} A "
                f"(CC={preparation.coarse_score:.6g}), "
                f"final={final_t:.1f} A, df1/df2={final_d1:.1f}/{final_d2:.1f} A, "
                f"angle={final_angle * 180.0 / PI:.3f} deg, "
                f"post-node cutoff={final_epa.good_fit_resolution_A:.3g} A "
                f"(CC={final_score:.6g})."
            ),
            node_seed_A=float(preparation.node_seed_A),
            coarse_defocus_A=float(preparation.coarse_profile_defocus_A),
            defocus1_A=float(final_d1),
            defocus2_A=float(final_d2),
            astigmatism_angle_rad=float(final_angle),
            amplitude_contrast=float(cfg.amplitude_contrast),
            initial_good_fit_resolution_A=float(initial_epa.good_fit_resolution_A),
            final_good_fit_resolution_A=float(final_epa.good_fit_resolution_A),
            initial_last_good_bin=int(initial_epa.last_good_bin),
            final_last_good_bin=int(final_epa.last_good_bin),
            debug=debug,
            final_epa=final_epa,
            timings=timings if cfg.timing else None,
        )

    def estimate_ice_thickness_batch(
        self,
        masked_spectra: torch.Tensor,
        fitting_pixel_size_A: float,
        defocus1_A: Sequence[float],
        defocus2_A: Sequence[float],
        astigmatism_angle_rad: Sequence[float],
        initial_epas: Sequence[Optional[_EPAStatistics]],
    ) -> list[_ThicknessFitDetails]:
        """Batch the expensive four-parameter thickness Powell stage."""
        if masked_spectra.ndim != 3:
            raise ValueError("masked_spectra must have shape [B,N,N]")
        count = int(masked_spectra.shape[0])
        if not (
            len(defocus1_A)
            == len(defocus2_A)
            == len(astigmatism_angle_rad)
            == len(initial_epas)
            == count
        ):
            raise ValueError("Thickness batch metadata lengths differ")
        preparations: list[Optional[_ThicknessPreparation]] = [None] * count
        outputs: list[Optional[_ThicknessFitDetails]] = [None] * count
        for index in range(count):
            try:
                preparations[index] = self._prepare_ice_thickness_single(
                    masked_spectra[index],
                    fitting_pixel_size_A,
                    float(defocus1_A[index]),
                    float(defocus2_A[index]),
                    float(astigmatism_angle_rad[index]),
                    initial_epas[index],
                )
            except Exception as exc:
                outputs[index] = _ThicknessFitDetails(
                    success=False,
                    thickness_A=float("nan"),
                    score=float("nan"),
                    coarse_thickness_A=float("nan"),
                    message=f"Thickness fit failed: {exc}",
                )

        valid_indices = [
            index for index, item in enumerate(preparations) if item is not None
        ]
        refine_batch_size = max(1, int(self.config.thickness_refine_batch_size))
        for first in range(0, len(valid_indices), refine_batch_size):
            indices = valid_indices[first:first + refine_batch_size]
            prepared = [preparations[index] for index in indices]
            prepared = [item for item in prepared if item is not None]
            if self.config.thickness_2d_refine:
                try:
                    final_values, prepare_elapsed, refine_elapsed = (
                        self._refine_ice_thickness_2d_batch(prepared)
                    )
                except Exception:
                    # Retain the old per-image fault isolation for unusual NaN
                    # supports or device-specific optimizer failures.  A failed
                    # row must not abort the remaining micrographs in the batch.
                    final_values = []
                    prepare_elapsed = 0.0
                    refine_elapsed = 0.0
                    surviving_indices: list[int] = []
                    surviving_prepared: list[_ThicknessPreparation] = []
                    for output_index, item in zip(indices, prepared):
                        try:
                            values, prep_seconds, refine_seconds = (
                                self._refine_ice_thickness_2d_batch([item])
                            )
                        except Exception as exc:
                            outputs[output_index] = _ThicknessFitDetails(
                                success=False,
                                thickness_A=float("nan"),
                                score=float("nan"),
                                coarse_thickness_A=float(item.coarse_thickness_A),
                                message=f"Thickness fit failed: {exc}",
                            )
                            continue
                        final_values.extend(values)
                        prepare_elapsed += prep_seconds
                        refine_elapsed += refine_seconds
                        surviving_indices.append(output_index)
                        surviving_prepared.append(item)
                    indices = surviving_indices
                    prepared = surviving_prepared
                shared_prepare_seconds = prepare_elapsed / max(1, len(prepared))
                shared_refine_seconds = refine_elapsed / max(1, len(prepared))
            else:
                final_values = [
                    (
                        item.coarse_thickness_A,
                        item.coarse_defocus1_A,
                        item.coarse_defocus2_A,
                        item.initial_astigmatism_angle_rad,
                        item.coarse_score,
                    )
                    for item in prepared
                ]
                shared_prepare_seconds = 0.0
                shared_refine_seconds = 0.0
            for index, item, values in zip(indices, prepared, final_values):
                try:
                    outputs[index] = self._finalize_ice_thickness(
                        item, values, shared_prepare_seconds, shared_refine_seconds
                    )
                except Exception as exc:
                    outputs[index] = _ThicknessFitDetails(
                        success=False,
                        thickness_A=float("nan"),
                        score=float("nan"),
                        coarse_thickness_A=float(item.coarse_thickness_A),
                        message=f"Thickness fit failed: {exc}",
                    )
        return [
            item
            if item is not None
            else _ThicknessFitDetails(
                success=False,
                thickness_A=float("nan"),
                score=float("nan"),
                coarse_thickness_A=float("nan"),
                message="Thickness fit failed without a diagnostic message.",
            )
            for item in outputs
        ]

    def estimate_ice_thickness(
        self,
        masked_spectrum: torch.Tensor,
        fitting_pixel_size_A: float,
        defocus1_A: float,
        defocus2_A: float,
        astigmatism_angle_rad: float,
        initial_epa: Optional[_EPAStatistics] = None,
    ) -> _ThicknessFitDetails:
        """Single-image compatibility wrapper around the batched path."""
        return self.estimate_ice_thickness_batch(
            masked_spectrum[None],
            fitting_pixel_size_A,
            [float(defocus1_A)],
            [float(defocus2_A)],
            [float(astigmatism_angle_rad)],
            [initial_epa],
        )[0]


    def preprocess_bundle_batch(
        self,
        micrographs: Sequence[np.ndarray],
        *,
        pixel_size_A: float,
    ) -> _FilteredSpectrumBundle:
        """Build whole-image spectra and apply the shared full-2D filter."""
        timing_enabled = bool(self.config.timing)
        timings: dict[str, float] = {}
        total_started = time.perf_counter()
        if not micrographs:
            empty = torch.empty(
                (0, self.config.box_size, self.config.box_size),
                dtype=self.dtype,
                device=self.device,
            )
            return _FilteredSpectrumBundle(
                raw_amplitude=empty,
                normalized_cross_capped=empty,
                background=empty,
                filtered_unmasked=empty,
                filtered_masked=empty,
                fitting_pixel_size_A=float(pixel_size_A),
                timings=timings if timing_enabled else None,
            )

        with _timed_stage(
            timing_enabled,
            None,
            timings,
            'preprocess_validate_and_stack_batch_s',
        ):
            shape = np.asarray(micrographs[0]).shape
            arrays: list[np.ndarray] = []
            for idx, image in enumerate(micrographs):
                array = np.asarray(image, dtype=np.float32)
                if not array.flags.c_contiguous:
                    array = np.ascontiguousarray(array)
                if array.ndim != 2 or array.shape != shape:
                    raise ValueError(
                        'All micrographs in a preprocessing batch must share '
                        'one shape'
                    )
                if not np.isfinite(array).all():
                    raise ValueError(
                        f'Micrograph {idx} contains NaN or infinity'
                    )
                if float(array.max()) == float(array.min()):
                    raise ValueError(f'Micrograph {idx} is constant')
                arrays.append(array)
            stacked = np.stack(arrays, axis=0)

        with _timed_stage(
            timing_enabled,
            self.device,
            timings,
            'preprocess_host_to_device_batch_s',
        ):
            images = torch.as_tensor(
                stacked, dtype=self.dtype, device=self.device
            )

        with torch.inference_mode():
            with _timed_stage(
                timing_enabled,
                self.device,
                timings,
                'preprocess_whole_micrograph_fft_resize_batch_s',
            ):
                raw, fitting_pixel_size_A = _ctffind_raw_amplitude_batch(
                    images, pixel_size_A, self.config
                )
            with _timed_stage(
                timing_enabled,
                self.device,
                timings,
                'preprocess_full_2d_filter_batch_s',
            ):
                bundle = _compute_filtered_amplitude_spectrum_full_2d_batch(
                    raw,
                    fitting_pixel_size_A,
                    self.config,
                    apply_cosine_mask=True,
                )
        if timing_enabled:
            timings['preprocess_total_batch_s'] = float(
                time.perf_counter() - total_started
            )
            bundle.timings = timings
        return bundle

    def preprocess_batch(
        self,
        micrographs: Sequence[np.ndarray],
        *,
        pixel_size_A: float,
    ) -> tuple[torch.Tensor, float]:
        """Compatibility wrapper returning the validated unmasked fit spectrum."""
        bundle = self.preprocess_bundle_batch(micrographs, pixel_size_A=pixel_size_A)
        return bundle.filtered_unmasked, bundle.fitting_pixel_size_A


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
        masked_spectra: Optional[torch.Tensor] = None,
        return_filtered_spectra: bool = False,
        return_diagnostic_maps: bool = True,
        run_thickness: Optional[bool] = None,
    ) -> tuple[list[CtfFitResult], Optional[np.ndarray], Optional[np.ndarray]]:
        """Fit shared CTFFIND spectra, then EPA/thickness on masked spectra.

        Timing distinguishes truly batched stages from per-micrograph serial
        stages.  Batched GPU costs are reported per result as amortized values;
        EPA, FRC, thickness, and diagnostic rendering are measured for each
        micrograph separately.
        """
        timing_enabled = bool(self.config.timing)
        batch_timings: dict[str, float] = {}
        timing_started = time.perf_counter()

        if spectra.ndim != 3:
            raise ValueError('spectra must have shape [B,H,W]')
        batch = int(spectra.shape[0])
        per_image_timings: list[dict[str, float]] = [dict() for _ in range(batch)]
        metadata_lengths = {
            len(source_files), len(micrograph_names), len(ctf_image_names),
            len(image_indices_1based),
        }
        if metadata_lengths != {batch}:
            raise ValueError('Spectrum batch and metadata lengths differ')

        with _timed_stage(
            timing_enabled,
            self.device,
            batch_timings,
            'standard_input_validation_transfer_batch_s',
        ):
            spectra = spectra.to(device=self.device, dtype=self.dtype)
            if masked_spectra is None:
                masked_spectra = spectra
            masked_spectra = masked_spectra.to(device=self.device, dtype=self.dtype)
            if masked_spectra.shape != spectra.shape:
                raise ValueError('masked_spectra must match spectra shape')
            if run_thickness is None:
                run_thickness = bool(
                    self.config.estimate_thickness and not self.config.fit_tilt
                )

        with torch.inference_mode():
            with _timed_stage(
                timing_enabled,
                self.device,
                batch_timings,
                'standard_astigmatism_angle_search_batch_s',
            ):
                initial_angles = _estimate_astigmatism_angle_deg_batch(
                    spectra, fitting_pixel_size_A, self.config
                )
            with _timed_stage(
                timing_enabled,
                self.device,
                batch_timings,
                'standard_rotational_average_batch_s',
            ):
                curves = _rotational_average_linear_batch(
                    spectra, fitting_pixel_size_A
                )
            with _timed_stage(
                timing_enabled,
                self.device,
                batch_timings,
                'standard_mean_defocus_search_refine_batch_s',
            ):
                coarse, refined_mean, opt1 = (
                    self._coarse_and_refine_mean_defocus_batch(
                        curves, fitting_pixel_size_A
                    )
                )
            with _timed_stage(
                timing_enabled,
                self.device,
                batch_timings,
                'standard_2d_fit_data_batch_s',
            ):
                fit_data = _make_2d_fit_data_batch(
                    spectra, fitting_pixel_size_A, self.config
                )
            with _timed_stage(
                timing_enabled,
                self.device,
                batch_timings,
                'standard_2d_refine_batch_s',
            ):
                d1, d2, angle, score, opt2 = self._refine_2d_batch(
                    fit_data, refined_mean, initial_angles,
                    fitting_pixel_size_A,
                )

        with _timed_stage(
            timing_enabled,
            self.device,
            batch_timings,
            'standard_results_to_host_batch_s',
        ):
            d1_cpu = d1.detach().cpu().numpy().astype(np.float64, copy=False)
            d2_cpu = d2.detach().cpu().numpy().astype(np.float64, copy=False)
            angle_cpu = angle.detach().cpu().numpy().astype(np.float64, copy=False)
            score_cpu = score.detach().cpu().numpy().astype(np.float64, copy=False)
            coarse_cpu = coarse.detach().cpu().numpy().astype(np.float64, copy=False)
            refined_cpu = refined_mean.detach().cpu().numpy().astype(np.float64, copy=False)
            initial_angle_cpu = (
                initial_angles.detach().cpu().numpy().astype(np.float64, copy=False)
            )
            success1_cpu = opt1.success.detach().cpu().numpy().astype(bool, copy=False)
            success2_cpu = opt2.success.detach().cpu().numpy().astype(bool, copy=False)
            nfev1_cpu = opt1.nfev.detach().cpu().numpy().astype(np.int64, copy=False)
            nfev2_cpu = opt2.nfev.detach().cpu().numpy().astype(np.int64, copy=False)
            # One transfer replaces one synchronization/copy per result.
            curves_cpu = (
                curves.values.detach().cpu().numpy().astype(np.float64, copy=False)
            )

        statistics: list[_GoodFitStatistics] = []
        with torch.inference_mode():
            for i in range(batch):
                with _timed_stage(
                    timing_enabled,
                    self.device,
                    per_image_timings[i],
                    'standard_good_fit_statistics_s',
                ):
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
                ctf_aliasing_resolution_A=statistics[i].ctf_aliasing_resolution_A,
                coarse_defocus_A=float(coarse_cpu[i]),
                refined_mean_defocus_A=float(refined_cpu[i]),
                initial_astigmatism_angle_deg=float(initial_angle_cpu[i]),
                powell_1d_success=bool(success1_cpu[i]),
                powell_2d_success=bool(success2_cpu[i]),
                powell_1d_nfev=int(nfev1_cpu[i]),
                powell_2d_nfev=int(nfev2_cpu[i]),
                powell_1d_message=opt1.messages[i],
                powell_2d_message=opt2.messages[i],
                avrot_spatial_frequency_Ainv=(
                    statistics[i].spatial_frequency_Ainv.copy()
                ),
                avrot_rotational_average_no_astig=np.array(
                    curves_cpu[i], dtype=np.float64, copy=True
                ),
                avrot_rotational_average_astig=(
                    statistics[i].rotational_average_astigmatic.copy()
                ),
                avrot_rotational_average_fit=(
                    statistics[i].rotational_average_fit.copy()
                ),
                avrot_fit_frc=statistics[i].fit_frc.copy(),
                avrot_fit_frc_sigma=statistics[i].fit_frc_sigma.copy(),
            ))

        # Diagnostic rendering is intentionally deferred until all optional
        # thickness updates are complete.  Version 0.5.1 rendered an initial
        # map and discarded it after a successful thickness refinement.
        diagnostic_statistics: list[_GoodFitStatistics] = list(statistics)

        if run_thickness:
            initial_epas: list[Optional[_EPAStatistics]] = [None] * batch
            valid_thickness_indices: list[int] = []
            for i, result in enumerate(results):
                try:
                    with _timed_stage(
                        timing_enabled,
                        self.device,
                        per_image_timings[i],
                        "epa_initial_statistics_s",
                    ):
                        initial_epa = _compute_epa_statistics(
                            masked_spectra[i],
                            fitting_pixel_size_A,
                            self.config,
                            result.defocus1_A,
                            result.defocus2_A,
                            result.astigmatism_angle_deg * PI / 180.0,
                            self.wavelength_A,
                            self.spherical_aberration_A,
                            self.amplitude_phase_rad,
                            self.config.fixed_phase_shift_rad,
                            theoretical_thickness_A=None,
                            node_mode=False,
                            rounded_square=self.config.thickness_use_rounded_square,
                        )
                    initial_epas[i] = initial_epa
                    valid_thickness_indices.append(i)
                    result.global_thon_rings_good_fit_resolution_A = (
                        result.thon_rings_good_fit_resolution_A
                    )
                    result.thon_rings_good_fit_resolution_A = (
                        initial_epa.good_fit_resolution_A
                    )
                    result.avrot_spatial_frequency_Ainv = (
                        initial_epa.spatial_frequency_Ainv.copy()
                    )
                    result.avrot_rotational_average_astig = (
                        initial_epa.observed_profile.copy()
                    )
                    result.avrot_rotational_average_fit = (
                        initial_epa.theoretical_profile.copy()
                    )
                    result.avrot_fit_frc = initial_epa.fit_frc.copy()
                    result.avrot_fit_frc_sigma = initial_epa.fit_frc_sigma.copy()
                except Exception as exc:
                    result.ice_thickness_message = f"Thickness fit failed: {exc}"

            if valid_thickness_indices:
                index_tensor = torch.as_tensor(
                    valid_thickness_indices,
                    dtype=torch.long,
                    device=masked_spectra.device,
                )
                valid_spectra = masked_spectra.index_select(0, index_tensor)
                thickness_results = self.estimate_ice_thickness_batch(
                    valid_spectra,
                    fitting_pixel_size_A,
                    [results[i].defocus1_A for i in valid_thickness_indices],
                    [results[i].defocus2_A for i in valid_thickness_indices],
                    [
                        results[i].astigmatism_angle_deg * PI / 180.0
                        for i in valid_thickness_indices
                    ],
                    [initial_epas[i] for i in valid_thickness_indices],
                )
                for i, thickness in zip(valid_thickness_indices, thickness_results):
                    result = results[i]
                    if thickness.timings:
                        for name, seconds in thickness.timings.items():
                            _add_timing(per_image_timings[i], name, float(seconds))
                    result.ice_thickness_fitted = thickness.success
                    result.ice_thickness_A = thickness.thickness_A
                    result.ice_thickness_score = thickness.score
                    result.ice_thickness_message = thickness.message
                    base_debug = result.debug if result.debug is not None else {}
                    base_debug["thickness"] = thickness.debug
                    result.debug = base_debug
                    if not thickness.success:
                        continue

                    result.defocus1_A = thickness.defocus1_A
                    result.defocus2_A = thickness.defocus2_A
                    result.astigmatism_angle_deg = (
                        thickness.astigmatism_angle_rad * 180.0 / PI
                    )
                    result.coarse_defocus_A = thickness.coarse_defocus_A
                    result.refined_mean_defocus_A = 0.5 * (
                        thickness.defocus1_A + thickness.defocus2_A
                    )
                    if thickness.final_good_fit_resolution_A > 0.0:
                        result.thon_rings_good_fit_resolution_A = (
                            thickness.final_good_fit_resolution_A
                        )

                    final_epa_for_output = thickness.final_epa
                    if final_epa_for_output is None:
                        with _timed_stage(
                            timing_enabled,
                            self.device,
                            per_image_timings[i],
                            "epa_final_output_fallback_s",
                        ):
                            final_epa_for_output = _compute_epa_statistics(
                                masked_spectra[i],
                                fitting_pixel_size_A,
                                self.config,
                                result.defocus1_A,
                                result.defocus2_A,
                                result.astigmatism_angle_deg * PI / 180.0,
                                self.wavelength_A,
                                self.spherical_aberration_A,
                                self.amplitude_phase_rad,
                                self.config.fixed_phase_shift_rad,
                                theoretical_thickness_A=result.ice_thickness_A,
                                node_mode=True,
                                rounded_square=self.config.thickness_use_rounded_square,
                            )
                    result.avrot_spatial_frequency_Ainv = (
                        final_epa_for_output.spatial_frequency_Ainv.copy()
                    )
                    result.avrot_rotational_average_astig = (
                        final_epa_for_output.observed_profile.copy()
                    )
                    result.avrot_rotational_average_fit = (
                        final_epa_for_output.theoretical_profile.copy()
                    )
                    result.avrot_fit_frc = final_epa_for_output.fit_frc.copy()
                    result.avrot_fit_frc_sigma = (
                        final_epa_for_output.fit_frc_sigma.copy()
                    )

                    if return_diagnostic_maps:
                        with _timed_stage(
                            timing_enabled,
                            self.device,
                            per_image_timings[i],
                            "diagnostic_final_statistics_s",
                        ):
                            diagnostic_statistics[i] = _compute_good_fit_statistics(
                                spectra[i],
                                fitting_pixel_size_A,
                                self.config,
                                result.defocus1_A,
                                result.defocus2_A,
                                result.astigmatism_angle_deg * PI / 180.0,
                                self.wavelength_A,
                                self.spherical_aberration_A,
                                self.amplitude_phase_rad,
                                self.config.fixed_phase_shift_rad,
                                keep_diagnostic_support=True,
                            )
        diagnostic_tensors: list[torch.Tensor] = []
        if return_diagnostic_maps:
            with torch.inference_mode():
                for i, result in enumerate(results):
                    with _timed_stage(
                        timing_enabled,
                        self.device,
                        per_image_timings[i],
                        'diagnostic_render_s',
                    ):
                        diagnostic_tensors.append(_render_diagnostic_map(
                            diagnostic_statistics[i],
                            fitting_pixel_size_A,
                            self.config,
                            result.defocus1_A,
                            result.defocus2_A,
                            result.astigmatism_angle_deg * PI / 180.0,
                            self.wavelength_A,
                            self.spherical_aberration_A,
                            self.amplitude_phase_rad,
                            self.config.fixed_phase_shift_rad,
                        ))

        filtered = None
        diagnostic_maps = None
        with _timed_stage(
            timing_enabled,
            self.device,
            batch_timings,
            'standard_outputs_to_host_batch_s',
        ):
            if return_filtered_spectra:
                filtered = (
                    spectra.detach().cpu().numpy().astype(np.float32, copy=False)
                )
            if return_diagnostic_maps:
                diagnostic_maps = (
                    torch.stack(diagnostic_tensors)
                    .detach().cpu().numpy().astype(np.float32, copy=False)
                )

        if timing_enabled:
            _synchronize_if_cuda(self.device)
            elapsed = time.perf_counter() - timing_started
            amortized_total = elapsed / max(1, batch)
            for i, result in enumerate(results):
                detailed = dict(per_image_timings[i])
                for name, seconds in batch_timings.items():
                    if name.endswith('_batch_s'):
                        out_name = name[:-8] + '_amortized_s'
                    else:
                        out_name = name + '_amortized_s'
                    detailed[out_name] = float(seconds) / max(1, batch)
                detailed['shared_standard_epa_thickness_pipeline_s'] = float(
                    amortized_total
                )
                result.timings = detailed

        if self.config.debug:
            for i, result in enumerate(results):
                base_debug = result.debug if result.debug is not None else {}
                base_debug['standard_fit'] = {
                    'coarse_defocus_A': float(coarse_cpu[i]),
                    'refined_mean_defocus_A': float(refined_cpu[i]),
                    'defocus1_A': float(result.defocus1_A),
                    'defocus2_A': float(result.defocus2_A),
                    'astigmatism_angle_deg': float(result.astigmatism_angle_deg),
                    'score': float(result.score),
                    'ordinary_good_fit_resolution_A': float(
                        statistics[i].thon_rings_good_fit_resolution_A
                    ),
                    'aliasing_resolution_A': float(
                        statistics[i].ctf_aliasing_resolution_A
                    ),
                }
                result.debug = base_debug
        return results, filtered, diagnostic_maps

    def fit_tilt_micrograph(
        self,
        micrograph: np.ndarray,
        global_result: CtfFitResult,
        return_diagnostic_map: bool = True,
    ) -> _TiltFitDetails:
        """Run the validated 0.4 CTFTilt frontend and the 0.5.x shared backend."""
        cfg = self.config
        timing_enabled = bool(cfg.timing)
        tilt_timings: dict[str, float] = {}
        if timing_enabled:
            _synchronize_if_cuda(self.device)
        tilt_timing_started = time.perf_counter()
        image_array = np.asarray(micrograph, dtype=np.float32)
        if image_array.ndim != 2:
            raise ValueError("CTFFIND5 tilt fitting expects one 2-D micrograph")
        height, width = image_array.shape
        empty_f = np.empty(0, dtype=np.float64)
        empty_b = np.empty(0, dtype=bool)
        empty_i = np.empty(0, dtype=np.int64)

        def empty_details(message: str) -> _TiltFitDetails:
            return _TiltFitDetails(
                success=False,
                message=message,
                center_defocus1_A=global_result.defocus1_A,
                center_defocus2_A=global_result.defocus2_A,
                astigmatism_angle_rad=global_result.astigmatism_angle_deg * PI / 180.0,
                gradient_x=float("nan"),
                gradient_y=float("nan"),
                tilt_angle_deg=float("nan"),
                tilt_axis_deg=float("nan"),
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

        if min(height, width) < cfg.tilt_tile_size:
            return empty_details(
                f"Skipped: micrograph {width}x{height} is smaller than "
                f"{cfg.tilt_tile_size}-pixel CTFFIND5 tile"
            )

        pixel_size_A = float(global_result.pixel_size_input_A)
        try:
            with _timed_stage(
                timing_enabled,
                self.device,
                tilt_timings,
                'tilt_input_host_to_device_s',
            ):
                image_tensor = torch.as_tensor(
                    image_array, device=self.device, dtype=self.dtype
                )
            frontend = _v04_fit_tilt_frontend(
                image_tensor, pixel_size_A, cfg
            )
            if frontend.timings:
                for name, seconds in frontend.timings.items():
                    _add_timing(tilt_timings, name, seconds)
        except Exception as exc:
            return empty_details(f"CTFFIND5 0.4-compatible tilt frontend failed: {exc}")

        data = frontend.data
        axis = float(frontend.refined_axis_deg)
        angle = float(frontend.refined_angle_deg)
        plane_mean_defocus = float(frontend.refined_mean_defocus_A)
        plane_score = float(frontend.refined_score)
        coarse_axis = float(frontend.coarse_axis_deg)
        coarse_angle = float(frontend.coarse_angle_deg)
        initial_d1 = float(frontend.rough_defocus1_A)
        initial_d2 = float(frontend.rough_defocus2_A)
        initial_astig_angle = math.radians(frontend.rough_astigmatism_angle_deg)
        initial_mean = 0.5 * (initial_d1 + initial_d2)
        astigmatism_A = initial_d1 - initial_d2

        try:
            corrected_raw_spectrum, corrected_pixel_size_A, corrected_tiles = (
                self._calculate_ctffind5_tilt_corrected_spectrum(
                    image_tensor,
                    pixel_size_A,
                    axis,
                    angle,
                    plane_mean_defocus,
                    tilt_timings if timing_enabled else None,
                )
            )
            with _timed_stage(
                timing_enabled,
                self.device,
                tilt_timings,
                'tilt_corrected_full_2d_filter_s',
            ):
                corrected_bundle = (
                    _compute_filtered_amplitude_spectrum_full_2d_batch(
                        corrected_raw_spectrum[None],
                        corrected_pixel_size_A,
                        cfg,
                        apply_cosine_mask=True,
                    )
                )
            with _timed_stage(
                timing_enabled,
                self.device,
                tilt_timings,
                'tilt_corrected_standard_refit_wrapper_s',
            ):
                final_results, final_filtered, final_diagnostics = (
                    self.fit_spectra_batch(
                        corrected_bundle.filtered_unmasked,
                        source_files=[global_result.source_file],
                        micrograph_names=[global_result.micrograph_name],
                        ctf_image_names=[global_result.ctf_image_name],
                        image_indices_1based=[global_result.image_index_1based],
                        pixel_size_input_A=pixel_size_A,
                        fitting_pixel_size_A=corrected_pixel_size_A,
                        masked_spectra=corrected_bundle.filtered_masked,
                        return_filtered_spectra=True,
                        return_diagnostic_maps=return_diagnostic_map,
                        run_thickness=bool(cfg.estimate_thickness),
                    )
                )
            final_result = final_results[0]
            filtered_np = (
                final_filtered[0]
                if final_filtered is not None
                else corrected_bundle.filtered_unmasked[0]
                .detach().cpu().numpy().astype(np.float32, copy=False)
            )
            diagnostic_np = (
                final_diagnostics[0] if final_diagnostics is not None else None
            )
            if cfg.debug:
                if final_result.debug is None:
                    final_result.debug = {}
                final_result.debug['tilt_search'] = frontend.debug
                final_result.debug['tilt_corrected_spectrum'] = {
                    'corrected_tiles': int(corrected_tiles),
                    'fitting_pixel_size_A': float(corrected_pixel_size_A),
                    'raw_min': float(torch.amin(corrected_bundle.raw_amplitude).item()),
                    'raw_max': float(torch.amax(corrected_bundle.raw_amplitude).item()),
                    'normalized_min': float(torch.amin(corrected_bundle.normalized_cross_capped).item()),
                    'normalized_max': float(torch.amax(corrected_bundle.normalized_cross_capped).item()),
                    'background_min': float(torch.amin(corrected_bundle.background).item()),
                    'background_max': float(torch.amax(corrected_bundle.background).item()),
                    'filtered_min': float(torch.amin(corrected_bundle.filtered_unmasked).item()),
                    'filtered_max': float(torch.amax(corrected_bundle.filtered_unmasked).item()),
                    'masked_min': float(torch.amin(corrected_bundle.filtered_masked).item()),
                    'masked_max': float(torch.amax(corrected_bundle.filtered_masked).item()),
                }
        except Exception as exc:
            return empty_details(
                f"CTFFIND5 tilt correction/shared CTF-EPA backend failed: {exc}"
            )

        gx = float(frontend.gradient_x)
        gy = float(frontend.gradient_y)
        predicted = np.asarray(frontend.local_defocus_A, dtype=np.float64)
        with _timed_stage(
            timing_enabled,
            self.device,
            tilt_timings,
            'tilt_local_defocus_diagnostics_s',
        ):
            measured, tile_cc = self._ctffind5_local_defocus_diagnostics(
                data, predicted, astigmatism_A, initial_astig_angle
            )
        residual = measured - predicted
        plane_inlier = data.valid_mask.detach().cpu().numpy().astype(bool, copy=False)
        finite_residual = plane_inlier & np.isfinite(residual)
        residual_rms = (
            float(np.sqrt(np.mean(residual[finite_residual] ** 2)))
            if np.any(finite_residual) else float('nan')
        )

        thickness = _ThicknessFitDetails(
            success=bool(final_result.ice_thickness_fitted),
            thickness_A=float(final_result.ice_thickness_A),
            score=float(final_result.ice_thickness_score),
            coarse_thickness_A=float(final_result.ice_thickness_A),
            message=str(final_result.ice_thickness_message),
            defocus1_A=float(final_result.defocus1_A),
            defocus2_A=float(final_result.defocus2_A),
            astigmatism_angle_rad=float(final_result.astigmatism_angle_deg) * PI / 180.0,
        )
        output_axis = _tilt_axis_to_output_convention(axis)
        output_coarse_axis = _tilt_axis_to_output_convention(coarse_axis)
        message = (
            "CTFFIND5 0.4-compatible local tilt search plus 0.5.x corrected-spectrum "
            "full-2D/EPA backend; "
            f"local_pixel={frontend.local_pixel_size_A:.4g} A, "
            f"local_band=40-{_make_v04_tilt_config(cfg).tilt_high_resolution_A:.4g} A, "
            f"rough_df={initial_mean:.1f} A, "
            f"coarse_axis/angle(output)={output_coarse_axis:.3f}/{coarse_angle:.3f} deg, "
            f"score_gap={frontend.score_gap:.6g}; "
            f"refined_axis/angle(output)={output_axis:.3f}/{angle:.3f} deg, "
            f"plane_df={plane_mean_defocus:.1f} A, plane_CC={plane_score:.6g}, "
            f"corrected_tiles={corrected_tiles}, final_CTF_CC={final_result.score:.6g}, "
            f"final_good_fit={final_result.thon_rings_good_fit_resolution_A:.3g} A."
        )
        if timing_enabled:
            _synchronize_if_cuda(self.device)
            combined_timings = dict(final_result.timings or {})
            for name, seconds in tilt_timings.items():
                _add_timing(combined_timings, name, seconds)
            combined_timings['tilt_search_correction_and_refit_total_s'] = float(
                time.perf_counter() - tilt_timing_started
            )
            final_result.timings = combined_timings
        return _TiltFitDetails(
            success=True,
            message=message,
            center_defocus1_A=float(final_result.defocus1_A),
            center_defocus2_A=float(final_result.defocus2_A),
            astigmatism_angle_rad=float(final_result.astigmatism_angle_deg) * PI / 180.0,
            gradient_x=gx,
            gradient_y=gy,
            tilt_angle_deg=angle,
            tilt_axis_deg=output_axis,
            coarse_tilt_angle_deg=coarse_angle,
            coarse_tilt_axis_deg=output_coarse_axis,
            score=plane_score,
            good_fit_resolution_A=float(final_result.thon_rings_good_fit_resolution_A),
            residual_rms_A=residual_rms,
            tile_centers_x_A=data.centers_x_A_numpy.copy(),
            tile_centers_y_A=data.centers_y_A_numpy.copy(),
            tile_measured_defocus_A=measured,
            tile_predicted_defocus_A=predicted,
            tile_residual_A=residual,
            tile_cc=tile_cc,
            tile_good_fit_resolution_A=np.full_like(measured, np.nan),
            tile_rms_valid=data.valid_mask.detach().cpu().numpy().astype(bool, copy=False),
            tile_plane_inlier=plane_inlier,
            tile_grid_y=data.grid_y.copy(),
            tile_grid_x=data.grid_x.copy(),
            image_shape=(height, width),
            final_ctf_result=final_result,
            filtered_spectrum=np.asarray(filtered_np, dtype=np.float32),
            diagnostic_map=(
                None if diagnostic_np is None
                else np.asarray(diagnostic_np, dtype=np.float32)
            ),
            debug_spectra=(
                {
                    "tilt_corrected_raw": corrected_bundle.raw_amplitude[0].detach().cpu().numpy().astype(np.float32, copy=False),
                    "tilt_corrected_normalized_cross_capped": corrected_bundle.normalized_cross_capped[0].detach().cpu().numpy().astype(np.float32, copy=False),
                    "tilt_corrected_background": corrected_bundle.background[0].detach().cpu().numpy().astype(np.float32, copy=False),
                    "tilt_corrected_filtered_unmasked": corrected_bundle.filtered_unmasked[0].detach().cpu().numpy().astype(np.float32, copy=False),
                    "tilt_corrected_filtered_masked": corrected_bundle.filtered_masked[0].detach().cpu().numpy().astype(np.float32, copy=False),
                }
                if cfg.debug else None
            ),
            ice_thickness_fitted=thickness.success,
            ice_thickness_A=thickness.thickness_A,
            ice_thickness_score=thickness.score,
            ice_thickness_message=thickness.message,
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
        """Whole-image raw spectrum producer followed by the shared pipeline."""
        bundle = self.preprocess_bundle_batch(micrographs, pixel_size_A=pixel_size_A)
        return self.fit_spectra_batch(
            bundle.filtered_unmasked,
            source_files=source_files,
            micrograph_names=micrograph_names,
            ctf_image_names=ctf_image_names,
            image_indices_1based=image_indices_1based,
            pixel_size_input_A=pixel_size_A,
            fitting_pixel_size_A=bundle.fitting_pixel_size_A,
            masked_spectra=bundle.filtered_masked,
            return_filtered_spectra=return_filtered_spectra,
            return_diagnostic_maps=return_diagnostic_maps,
            run_thickness=bool(self.config.estimate_thickness and not self.config.fit_tilt),
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
            raise ValueError(
                f"{path}: 3-D MRC stacks are intentionally unsupported; provide one 2-D micrograph per file"
            )
        else:
            raise ValueError(
                f"{path}: expected one 2-D MRC micrograph, got shape {data.shape}"
            )


def _count_mrc_micrographs(path: str) -> int:
    with mrcfile.open(path, mode="r", permissive=True, header_only=True) as mrc:
        count = int(mrc.header.nz)
    if count != 1:
        raise ValueError(
            f"{path}: MRC nz={count}; stacks/movies are intentionally unsupported"
        )
    return 1




@dataclass
class _MicrographRecord:
    source_file: str
    image_index_1based: int
    image_count: int
    array: Optional[np.ndarray]
    pixel_size_A: float
    micrograph_name: str
    ctf_path: Path
    ctf_image_name: str
    timings: Optional[dict[str, float]] = None


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
        read_started = time.perf_counter()
        with mrcfile.mmap(path, mode='r', permissive=True) as mrc:
            data = mrc.data
            header_pixel = _pixel_size_from_mrc(mrc)
            pixel_size = config.pixel_size_A or header_pixel
            if pixel_size is None:
                raise ValueError(
                    f'{path}: pixel size not supplied and absent from MRC header'
                )
            if data.ndim == 2:
                image_count = 1
                image_index = 1
                image_array = np.array(
                    data, dtype=np.float32, copy=True, order='C'
                )
            elif data.ndim == 3:
                raise ValueError(
                    f'{path}: 3-D MRC stacks/movies are intentionally '
                    'unsupported; provide one 2-D micrograph per file'
                )
            else:
                raise ValueError(
                    f'{path}: expected one 2-D MRC, got {data.shape}'
                )
        read_seconds = time.perf_counter() - read_started

        ctf_path = _diagnostic_path_for_input(
            path, image_index, image_count, ctf_dir
        ).resolve()
        previous = used_ctf_paths.get(ctf_path)
        if previous is not None and previous != path:
            raise RuntimeError(
                f'Diagnostic filename collision: {ctf_path} for both '
                f'{previous} and {path}. Use separate --ctf-dir runs or '
                'rename duplicate micrograph basenames.'
            )
        used_ctf_paths[ctf_path] = path
        source_rel = _relion_path(path)
        micrograph_name = source_rel
        ctf_rel = _relion_path(ctf_path) + ':mrc'
        yield _MicrographRecord(
            source_file=str(Path(path).resolve()),
            image_index_1based=image_index,
            image_count=image_count,
            array=image_array,
            pixel_size_A=float(pixel_size),
            micrograph_name=micrograph_name,
            ctf_path=ctf_path,
            ctf_image_name=ctf_rel,
            timings=(
                {'input_mrc_read_and_copy_s': float(read_seconds)}
                if config.timing else None
            ),
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



def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _format_numeric_line(values: Optional[np.ndarray]) -> str:
    if values is None:
        return ""
    return " ".join(f"{float(value):.8g}" for value in np.asarray(values).ravel())


def _write_ctffind_summary(
    path: Path,
    results: Sequence[CtfFitResult],
    config: CtffindConfig,
    input_paths: Sequence[str],
) -> None:
    lines = [
        f"# Output from CTFFind version 5.0.2 (ctffind5_pytorch {VERSION})",
        "# Input files: " + " ".join(input_paths),
        (
            f"# acceleration voltage: {config.acceleration_voltage_kV:.3f} kV ; "
            f"spherical aberration: {config.spherical_aberration_mm:.6f} mm ; "
            f"amplitude contrast: {config.amplitude_contrast:.6f}"
        ),
        (
            f"# box size: {config.box_size} ; min resolution: "
            f"{config.minimum_resolution_A:.4f} A ; max resolution: "
            f"{config.maximum_resolution_A:.4f} A"
        ),
        "# Columns: micrograph_number defocus1_A defocus2_A astigmatism_angle_deg phase_shift_rad score fit_resolution_A tilt_axis_deg tilt_angle_deg thickness_A",
    ]
    for index, result in enumerate(results, 1):
        lines.append(
            f"{index:6d} {result.defocus1_A:13.3f} {result.defocus2_A:13.3f} "
            f"{result.astigmatism_angle_deg:10.4f} {result.phase_shift_rad:11.6f} "
            f"{result.score:11.7f} {result.thon_rings_good_fit_resolution_A:10.4f} "
            f"{result.tilt_axis_deg:10.4f} {result.tilt_angle_deg:10.4f} "
            f"{result.ice_thickness_A:12.3f}"
        )
    _atomic_write_text(path, "\n".join(lines) + "\n")


def _write_avrot(
    path: Path,
    results: Sequence[CtfFitResult],
    config: CtffindConfig,
    input_paths: Sequence[str],
) -> None:
    lines = [
        f"# Output from CTFFind version 5.0.2 (ctffind5_pytorch {VERSION})",
        f"# Number of micrographs: {len(results)} ; input files: {' '.join(input_paths)}",
        (
            f"# voltage: {config.acceleration_voltage_kV:.3f} kV ; Cs: "
            f"{config.spherical_aberration_mm:.6f} mm ; amplitude contrast: "
            f"{config.amplitude_contrast:.6f}"
        ),
        "# Six lines per micrograph: spatial frequency; radial average; EPA/astigmatic average; CTF fit; FRC; 2sigma",
    ]
    for result in results:
        if result.avrot_spatial_frequency_Ainv is None:
            continue
        lines.extend([
            _format_numeric_line(result.avrot_spatial_frequency_Ainv),
            _format_numeric_line(result.avrot_rotational_average_no_astig),
            _format_numeric_line(result.avrot_rotational_average_astig),
            _format_numeric_line(result.avrot_rotational_average_fit),
            _format_numeric_line(result.avrot_fit_frc),
            _format_numeric_line(result.avrot_fit_frc_sigma),
        ])
    _atomic_write_text(path, "\n".join(lines) + "\n")


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_result_debug_json(path: Path, result: CtfFitResult) -> None:
    payload = {
        "program": "ctffind5_pytorch",
        "version": VERSION,
        "micrograph": result.micrograph_name,
        "source_file": result.source_file,
        "result": {
            "defocus1_A": result.defocus1_A,
            "defocus2_A": result.defocus2_A,
            "astigmatism_angle_deg": result.astigmatism_angle_deg,
            "score": result.score,
            "fit_resolution_A": result.thon_rings_good_fit_resolution_A,
            "tilt_axis_deg": result.tilt_axis_deg,
            "tilt_angle_deg": result.tilt_angle_deg,
            "thickness_A": result.ice_thickness_A,
        },
        "debug": result.debug,
        "timings_seconds": result.timings,
    }
    _atomic_write_text(path, json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _write_timing_json(
    path: Path,
    results: Sequence[CtfFitResult],
    *,
    program_timings: Optional[dict[str, float]] = None,
    config: Optional[CtffindConfig] = None,
    resolved_device: Optional[torch.device] = None,
    diagnostic_output: Optional[bool] = None,
    runtime_settings_extra: Optional[dict[str, object]] = None,
    extra_payload: Optional[dict[str, object]] = None,
) -> None:
    stage_values: dict[str, list[float]] = {}
    for result in results:
        for name, value in (result.timings or {}).items():
            if math.isfinite(float(value)):
                stage_values.setdefault(name, []).append(float(value))
    payload: dict[str, object] = {
        'program': 'ctffind5_pytorch',
        'version': VERSION,
        'micrograph_count': len(results),
        'micrographs': [
            {
                'micrograph': result.micrograph_name,
                'timings_seconds': result.timings or {},
            }
            for result in results
        ],
        'stage_summary_seconds': {
            name: {
                'count': len(values),
                'total': float(sum(values)),
                'mean': float(sum(values) / len(values)),
                'minimum': float(min(values)),
                'maximum': float(max(values)),
            }
            for name, values in sorted(stage_values.items())
            if values
        },
        'program_timings_seconds': program_timings or {},
    }
    if config is not None:
        device_text = str(
            resolved_device if resolved_device is not None else config.device
        )
        runtime_settings: dict[str, object] = {
            'requested_device': config.device,
            'resolved_device': device_text,
            'torch_version': torch.__version__,
            'torch_num_threads': int(torch.get_num_threads()),
            'preprocess_batch_size': config.preprocess_batch_size,
            'fit_batch_size': config.fit_batch_size,
            'tilt_candidate_batch_size': config.tilt_candidate_batch_size,
            'tilt_tile_batch_size': config.tilt_tile_batch_size,
            'thickness_candidate_batch_size': (
                config.thickness_candidate_batch_size
            ),
            'thickness_refine_batch_size': config.thickness_refine_batch_size,
            'diagnostic_output': (
                bool(diagnostic_output)
                if diagnostic_output is not None else None
            ),
            'fit_tilt': config.fit_tilt,
            'estimate_thickness': config.estimate_thickness,
            'timing_note': (
                'CUDA stages are synchronized at timing boundaries. '
                'Names ending in _amortized_s are batch wall times divided '
                'by the number of micrographs in that batch; nested total '
                'and component stages must not be summed together.'
            ),
        }
        if resolved_device is not None and resolved_device.type == 'cuda':
            runtime_settings['cuda_device_name'] = torch.cuda.get_device_name(
                resolved_device
            )
        if runtime_settings_extra:
            runtime_settings.update(runtime_settings_extra)
        payload['runtime_settings'] = runtime_settings
    elif runtime_settings_extra:
        payload['runtime_settings'] = dict(runtime_settings_extra)
    if extra_payload:
        payload.update(extra_payload)
    wall = float((program_timings or {}).get('program_total_wall_s', 0.0))
    if wall > 0.0 and results:
        payload['throughput_micrographs_per_second'] = float(
            len(results) / wall
        )
    _atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=True) + '\n'
    )

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
    extended_results_tsv: Optional[str] = None,
    output_ctffind: Optional[str] = None,
    output_avrot: Optional[str] = None,
    debug_output_dir: Optional[str] = None,
    timing_json: Optional[str] = None,
    _estimator: Optional["TorchCtffindPowell"] = None,
    _write_aggregate_outputs: bool = True,
    _print_progress: bool = True,
    _paths_are_expanded: bool = False,
    _skip_header_validation: bool = False,
) -> list[CtfFitResult]:
    """Run the two-stage raw-image/preprocessed-spectrum GPU pipeline."""
    program_started = time.perf_counter()
    program_timings: dict[str, float] = {}
    stage_started = time.perf_counter()
    paths = (
        [str(Path(item).resolve()) for item in input_paths]
        if _paths_are_expanded
        else _expand_input_paths(input_paths)
    )
    if config.timing:
        program_timings['input_path_expansion_s'] = float(
            time.perf_counter() - stage_started
        )
    estimator = _estimator if _estimator is not None else TorchCtffindPowell(config)
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
        else output_path.parent / "ctffind5_tilt_png"
    )
    if config.fit_tilt and write_tilt_png:
        tilt_png_dir.mkdir(parents=True, exist_ok=True)
    extended_tsv_path = (
        Path(extended_results_tsv).resolve()
        if extended_results_tsv is not None
        else output_path.with_name(output_path.stem + ".tsv")
    )
    ctffind_text_path = (
        Path(output_ctffind).resolve()
        if output_ctffind is not None
        else output_path.with_name(output_path.stem + ".txt")
    )
    avrot_path = (
        Path(output_avrot).resolve()
        if output_avrot is not None
        else output_path.with_name(output_path.stem + "_avrot.txt")
    )
    debug_dir = (
        Path(debug_output_dir).resolve()
        if debug_output_dir is not None else output_path.parent / "ctffind5_debug"
    )
    if config.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
    timing_path = (
        Path(timing_json).resolve()
        if timing_json is not None else output_path.with_name(output_path.stem + "_timing.json")
    )
    spectra_dir = None
    if save_filtered_spectra_dir:
        spectra_dir = Path(save_filtered_spectra_dir).resolve()
        spectra_dir.mkdir(parents=True, exist_ok=True)

    stage_started = time.perf_counter()
    total_images = (
        len(paths)
        if _skip_header_validation
        else sum(_count_mrc_micrographs(path) for path in paths)
    )
    if config.timing:
        program_timings['input_header_validation_s'] = float(
            time.perf_counter() - stage_started
        )
    results: list[CtfFitResult] = []
    processed = 0
    preprocessed = 0
    if _print_progress:
        print(f"Device: {estimator.device}")
        print(f"Input files: {len(paths)}; independent micrographs: {total_images}")
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
    pending_masked_chunks: list[torch.Tensor] = []
    pending_count = 0
    pending_key: Optional[tuple[float, float]] = None

    def attach_preprocess_timings(
        batch_records: Sequence[_MicrographRecord],
        timings: Optional[dict[str, float]],
    ) -> None:
        if not config.timing or not timings or not batch_records:
            return
        count = max(1, len(batch_records))
        for record in batch_records:
            target = dict(record.timings or {})
            for name, seconds in timings.items():
                if name.endswith('_batch_s'):
                    output_name = name[:-8] + '_amortized_s'
                else:
                    output_name = name + '_amortized_s'
                _add_timing(target, output_name, float(seconds) / count)
            record.timings = target

    def pop_pending_spectra(number: int) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal pending_count
        if number < 1 or number > pending_count:
            raise ValueError("Invalid pending-spectrum pop size")

        def pop_from(chunks: list[torch.Tensor]) -> torch.Tensor:
            pieces: list[torch.Tensor] = []
            remaining = number
            while remaining > 0:
                chunk = chunks[0]
                take = min(remaining, int(chunk.shape[0]))
                pieces.append(chunk[:take])
                if take == int(chunk.shape[0]):
                    chunks.pop(0)
                else:
                    chunks[0] = chunk[take:]
                remaining -= take
            return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

        spectra = pop_from(pending_chunks)
        masked = pop_from(pending_masked_chunks)
        pending_count -= number
        return spectra, masked

    def write_fit_outputs(
        batch_records: Sequence[_MicrographRecord],
        batch_results: Sequence[CtfFitResult],
        filtered: Optional[np.ndarray],
        diagnostics: Optional[np.ndarray],
    ) -> None:
        nonlocal processed
        for i, (record, result) in enumerate(zip(batch_records, batch_results)):
            selected_diagnostic = (
                diagnostics[i] if diagnostics is not None else None
            )
            selected_filtered = (
                filtered[i] if filtered is not None else None
            )
            selected_fitting_pixel = result.pixel_size_for_fitting_A
            tilt_debug_spectra: Optional[dict[str, np.ndarray]] = None
            record_timings = dict(record.timings or {})
            initial_fit_timings = dict(result.timings or {})
            timings_merged = False
            if config.fit_tilt:
                result.global_thon_rings_good_fit_resolution_A = (
                    result.thon_rings_good_fit_resolution_A
                )
                try:
                    if record.array is None:
                        raise RuntimeError("Raw micrograph was released before tilt fitting")
                    tilt = estimator.fit_tilt_micrograph(
                        record.array,
                        result,
                        return_diagnostic_map=write_diagnostic_maps,
                    )
                    result.tilt_message = tilt.message
                    tilt_debug_spectra = tilt.debug_spectra
                    result.coarse_tilt_angle_deg = tilt.coarse_tilt_angle_deg
                    result.coarse_tilt_axis_deg = tilt.coarse_tilt_axis_deg
                    result.tilt_total_tiles = int(tilt.tile_plane_inlier.size)
                    result.tilt_valid_tiles = int(np.sum(tilt.tile_plane_inlier))
                    if tilt.success:
                        result.tilt_fitted = True
                        result.defocus1_A = tilt.center_defocus1_A
                        result.defocus2_A = tilt.center_defocus2_A
                        result.astigmatism_angle_deg = (
                            tilt.astigmatism_angle_rad * 180.0 / PI
                        )
                        result.defocus_gradient_x = tilt.gradient_x
                        result.defocus_gradient_y = tilt.gradient_y
                        result.tilt_angle_deg = tilt.tilt_angle_deg
                        result.tilt_axis_deg = tilt.tilt_axis_deg
                        result.tilt_score = tilt.score
                        result.tilt_good_fit_resolution_A = tilt.good_fit_resolution_A
                        result.tilt_residual_rms_A = tilt.residual_rms_A
                        if tilt.final_ctf_result is not None:
                            final_ctf = tilt.final_ctf_result
                            result.score = final_ctf.score
                            result.thon_rings_good_fit_resolution_A = (
                                final_ctf.thon_rings_good_fit_resolution_A
                            )
                            result.ctf_aliasing_resolution_A = (
                                final_ctf.ctf_aliasing_resolution_A
                            )
                            result.coarse_defocus_A = final_ctf.coarse_defocus_A
                            result.refined_mean_defocus_A = final_ctf.refined_mean_defocus_A
                            result.initial_astigmatism_angle_deg = (
                                final_ctf.initial_astigmatism_angle_deg
                            )
                            result.powell_1d_success = final_ctf.powell_1d_success
                            result.powell_2d_success = final_ctf.powell_2d_success
                            result.powell_1d_nfev = final_ctf.powell_1d_nfev
                            result.powell_2d_nfev = final_ctf.powell_2d_nfev
                            result.powell_1d_message = final_ctf.powell_1d_message
                            result.powell_2d_message = final_ctf.powell_2d_message
                            result.pixel_size_for_fitting_A = (
                                final_ctf.pixel_size_for_fitting_A
                            )
                            result.avrot_spatial_frequency_Ainv = final_ctf.avrot_spatial_frequency_Ainv
                            result.avrot_rotational_average_no_astig = final_ctf.avrot_rotational_average_no_astig
                            result.avrot_rotational_average_astig = final_ctf.avrot_rotational_average_astig
                            result.avrot_rotational_average_fit = final_ctf.avrot_rotational_average_fit
                            result.avrot_fit_frc = final_ctf.avrot_fit_frc
                            result.avrot_fit_frc_sigma = final_ctf.avrot_fit_frc_sigma
                            result.debug = final_ctf.debug
                            combined_timings = dict(record_timings)
                            for name, seconds in initial_fit_timings.items():
                                _add_timing(
                                    combined_timings,
                                    'initial_' + name,
                                    seconds,
                                )
                            for name, seconds in (final_ctf.timings or {}).items():
                                _add_timing(combined_timings, name, seconds)
                            result.timings = combined_timings
                            timings_merged = True
                        if tilt.filtered_spectrum is not None:
                            selected_filtered = tilt.filtered_spectrum
                        if tilt.diagnostic_map is not None:
                            selected_diagnostic = tilt.diagnostic_map
                        selected_fitting_pixel = result.pixel_size_for_fitting_A
                        if config.estimate_thickness:
                            result.ice_thickness_fitted = tilt.ice_thickness_fitted
                            result.ice_thickness_A = tilt.ice_thickness_A
                            result.ice_thickness_score = tilt.ice_thickness_score
                            result.ice_thickness_message = tilt.ice_thickness_message
                        if write_tilt_png and tilt.tile_centers_x_A.size > 0:
                            png_path = _tilt_png_path_for_input(
                                record.source_file,
                                record.image_index_1based,
                                record.image_count,
                                tilt_png_dir,
                            )
                            png_started = time.perf_counter()
                            _write_tilt_png(png_path, tilt)
                            if config.timing:
                                target = result.timings if result.timings is not None else {}
                                _add_timing(
                                    target, 'output_tilt_png_write_s',
                                    time.perf_counter() - png_started,
                                )
                                result.timings = target
                            result.tilt_png_name = _relion_path(png_path)
                except Exception as tilt_exc:
                    result.tilt_message = f"Tilt fitting failed: {tilt_exc}"
                    if not continue_on_error:
                        raise
                    print(
                        f"WARNING: {Path(record.source_file).name}: "
                        f"{result.tilt_message}",
                        file=sys.stderr,
                    )
            if config.fit_tilt:
                record.array = None
            if not timings_merged:
                combined_timings = dict(record_timings)
                for name, seconds in initial_fit_timings.items():
                    _add_timing(combined_timings, name, seconds)
                result.timings = combined_timings if config.timing else result.timings

            if write_diagnostic_maps and selected_diagnostic is not None:
                output_started = time.perf_counter()
                _write_diagnostic_ctf(
                    record.ctf_path,
                    selected_diagnostic,
                    selected_fitting_pixel,
                )
                if config.timing:
                    target = result.timings if result.timings is not None else {}
                    _add_timing(
                        target, 'output_diagnostic_ctf_write_s',
                        time.perf_counter() - output_started,
                    )
                    result.timings = target
            if spectra_dir is not None and selected_filtered is not None:
                spectrum_name = record.ctf_path.stem + "_filtered_spectrum.mrc"
                spectrum_path = spectra_dir / spectrum_name
                output_started = time.perf_counter()
                with mrcfile.new(spectrum_path, overwrite=True) as output:
                    output.set_data(np.asarray(selected_filtered, dtype=np.float32))
                    output.voxel_size = selected_fitting_pixel
                if config.timing:
                    target = result.timings if result.timings is not None else {}
                    _add_timing(
                        target, 'output_filtered_spectrum_write_s',
                        time.perf_counter() - output_started,
                    )
                    result.timings = target
            results.append(result)
            if config.debug:
                debug_json = debug_dir / f"{record.ctf_path.stem}_debug.json"
                _write_result_debug_json(debug_json, result)
                if selected_filtered is not None:
                    debug_spectrum = debug_dir / f"{record.ctf_path.stem}_filtered_spectrum.mrc"
                    with mrcfile.new(debug_spectrum, overwrite=True) as output:
                        output.set_data(np.asarray(selected_filtered, dtype=np.float32))
                        output.voxel_size = selected_fitting_pixel
                if tilt_debug_spectra:
                    for stage_name, stage_array in tilt_debug_spectra.items():
                        stage_path = debug_dir / f"{record.ctf_path.stem}_{stage_name}.mrc"
                        with mrcfile.new(stage_path, overwrite=True) as output:
                            output.set_data(np.asarray(stage_array, dtype=np.float32))
                            output.voxel_size = selected_fitting_pixel
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
            thickness_text = (
                f", thickness={result.ice_thickness_A / 10.0:.1f} nm"
                if result.ice_thickness_fitted else ""
            )
            if _print_progress:
                print(
                    f"  [{processed}/{total_images}] {Path(result.source_file).name}: "
                    f"dfU={result.defocus1_A:.1f}, dfV={result.defocus2_A:.1f}, "
                    f"angle={result.astigmatism_angle_deg:.2f}, "
                    f"CC={result.score:.5f}, maxres={good}{tilt_text}{thickness_text}"
                )
        if _write_aggregate_outputs:
            checkpoint_started = time.perf_counter()
            _write_relion_star(
                output_path,
                results,
                config,
                include_ctf_image=write_diagnostic_maps,
            )
            _write_extended_results_tsv(extended_tsv_path, results)
            _write_ctffind_summary(ctffind_text_path, results, config, paths)
            _write_avrot(avrot_path, results, config, paths)
            if config.timing:
                _add_timing(
                    program_timings,
                    'output_checkpoint_tables_write_s',
                    time.perf_counter() - checkpoint_started,
                )
                program_timings['program_total_wall_s'] = float(
                    time.perf_counter() - program_started
                )
                _write_timing_json(
                    timing_path, results,
                    program_timings=program_timings,
                    config=config,
                    resolved_device=estimator.device,
                    diagnostic_output=write_diagnostic_maps,
                )

    def fit_pending(number: int) -> None:
        nonlocal pending_records
        spectra, masked_spectra = pop_pending_spectra(number)
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
                masked_spectra=masked_spectra,
                return_filtered_spectra=(spectra_dir is not None or config.debug),
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
                    masked_spectra=masked_spectra[i:i + 1],
                    return_filtered_spectra=(spectra_dir is not None or config.debug),
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
        masked_spectra: torch.Tensor,
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
        if masked_spectra.shape != spectra.shape:
            raise ValueError("Masked and unmasked preprocessed spectra must match")
        pending_records.extend(batch_records)
        pending_chunks.append(spectra)
        pending_masked_chunks.append(masked_spectra)
        pending_count += len(batch_records)
        while pending_count >= config.fit_batch_size:
            fit_pending(config.fit_batch_size)

    records = _iter_micrograph_records(paths, config, ctf_dir)
    for raw_batch in _iter_compatible_batches(records, config.preprocess_batch_size):
        first = preprocessed + 1
        last = preprocessed + len(raw_batch)
        shape = raw_batch[0].array.shape
        if _print_progress:
            print(
                f"Preprocess [{first}-{last}/{total_images}] "
                f"batch={len(raw_batch)}, shape={shape[1]}x{shape[0]}, "
                f"pixel={raw_batch[0].pixel_size_A:.6g} A"
            )
        try:
            bundle = estimator.preprocess_bundle_batch(
                [r.array for r in raw_batch],
                pixel_size_A=raw_batch[0].pixel_size_A,
            )
            attach_preprocess_timings(raw_batch, bundle.timings)
            enqueue_preprocessed(
                raw_batch,
                bundle.filtered_unmasked,
                bundle.filtered_masked,
                bundle.fitting_pixel_size_A,
            )
            if not config.fit_tilt:
                # The pending queue needs only metadata and 512-pixel spectra.
                # Release full-resolution NumPy arrays once each preprocess batch
                # has been transferred and filtered.
                for record in raw_batch:
                    record.array = None
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
                    bundle = estimator.preprocess_bundle_batch(
                        [record.array], pixel_size_A=record.pixel_size_A
                    )
                    attach_preprocess_timings([record], bundle.timings)
                    enqueue_preprocessed(
                        [record],
                        bundle.filtered_unmasked,
                        bundle.filtered_masked,
                        bundle.fitting_pixel_size_A,
                    )
                except Exception as single_exc:
                    print(f"ERROR: {record.source_file}: {single_exc}", file=sys.stderr)
                finally:
                    if not config.fit_tilt:
                        record.array = None
                    preprocessed += 1

    flush_all_pending()
    if _write_aggregate_outputs and config.timing:
        program_timings['program_total_wall_s'] = float(
            time.perf_counter() - program_started
        )
        _write_timing_json(
            timing_path, results,
            program_timings=program_timings,
            config=config,
            resolved_device=estimator.device,
            diagnostic_output=write_diagnostic_maps,
        )
    if _write_aggregate_outputs and _print_progress:
        print(f"Wrote {len(results)} rows to {_relion_path(output_path)}")
        if write_diagnostic_maps:
            print(f"Wrote one .ctf MRC per micrograph under {_relion_path(ctf_dir)}")
        print(f"Wrote TSV results to {_relion_path(extended_tsv_path)}")
        print(f"Wrote CTFFIND text to {_relion_path(ctffind_text_path)}")
        print(f"Wrote avrot curves to {_relion_path(avrot_path)}")
        if config.debug:
            print(f"Wrote debug outputs under {_relion_path(debug_dir)}")
        if config.timing:
            print(f"Wrote timing JSON to {_relion_path(timing_path)}")
        if config.fit_tilt and write_tilt_png and any(r.tilt_png_name for r in results):
            print(f"Wrote tilt PNG diagnostics under {_relion_path(tilt_png_dir)}")
    return results


@dataclass(frozen=True)
class _MultiDeviceRunOptions:
    """Picklable output options shared by persistent device workers."""

    output_star: str
    ctf_output_dir: Optional[str]
    save_filtered_spectra_dir: Optional[str]
    write_diagnostic_maps: bool
    tilt_png_output_dir: Optional[str]
    write_tilt_png: bool
    extended_results_tsv: Optional[str]
    output_ctffind: Optional[str]
    output_avrot: Optional[str]
    debug_output_dir: Optional[str]
    timing_json: Optional[str]


@dataclass(frozen=True)
class _AggregateOutputLayout:
    output_path: Path
    ctf_dir: Path
    tilt_png_dir: Path
    extended_tsv_path: Path
    ctffind_text_path: Path
    avrot_path: Path
    debug_dir: Path
    timing_path: Path
    spectra_dir: Optional[Path]


def _resolve_aggregate_output_layout(
    output_star: str,
    ctf_output_dir: Optional[str],
    save_filtered_spectra_dir: Optional[str],
    tilt_png_output_dir: Optional[str],
    extended_results_tsv: Optional[str],
    output_ctffind: Optional[str],
    output_avrot: Optional[str],
    debug_output_dir: Optional[str],
    timing_json: Optional[str],
) -> _AggregateOutputLayout:
    output_path = Path(output_star).resolve()
    ctf_dir = (
        Path(ctf_output_dir).resolve()
        if ctf_output_dir is not None
        else output_path.parent
    )
    tilt_png_dir = (
        Path(tilt_png_output_dir).resolve()
        if tilt_png_output_dir is not None
        else output_path.parent / "ctffind5_tilt_png"
    )
    extended_tsv_path = (
        Path(extended_results_tsv).resolve()
        if extended_results_tsv is not None
        else output_path.with_name(output_path.stem + ".tsv")
    )
    ctffind_text_path = (
        Path(output_ctffind).resolve()
        if output_ctffind is not None
        else output_path.with_name(output_path.stem + ".txt")
    )
    avrot_path = (
        Path(output_avrot).resolve()
        if output_avrot is not None
        else output_path.with_name(output_path.stem + "_avrot.txt")
    )
    debug_dir = (
        Path(debug_output_dir).resolve()
        if debug_output_dir is not None
        else output_path.parent / "ctffind5_debug"
    )
    timing_path = (
        Path(timing_json).resolve()
        if timing_json is not None
        else output_path.with_name(output_path.stem + "_timing.json")
    )
    spectra_dir = (
        Path(save_filtered_spectra_dir).resolve()
        if save_filtered_spectra_dir is not None
        else None
    )
    return _AggregateOutputLayout(
        output_path=output_path,
        ctf_dir=ctf_dir,
        tilt_png_dir=tilt_png_dir,
        extended_tsv_path=extended_tsv_path,
        ctffind_text_path=ctffind_text_path,
        avrot_path=avrot_path,
        debug_dir=debug_dir,
        timing_path=timing_path,
        spectra_dir=spectra_dir,
    )


def _write_aggregate_outputs(
    layout: _AggregateOutputLayout,
    results: Sequence[CtfFitResult],
    config: CtffindConfig,
    input_paths: Sequence[str],
    *,
    include_ctf_image: bool,
) -> None:
    """Atomically write all aggregate tables in deterministic input order."""
    _write_relion_star(
        layout.output_path,
        results,
        config,
        include_ctf_image=include_ctf_image,
    )
    _write_extended_results_tsv(layout.extended_tsv_path, results)
    _write_ctffind_summary(layout.ctffind_text_path, results, config, input_paths)
    _write_avrot(layout.avrot_path, results, config, input_paths)


def _parse_gpu_ids(value: str) -> list[int]:
    """Parse comma-separated GPU IDs, including compact ranges such as 0-3."""
    ids: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            pieces = token.split("-", 1)
            if len(pieces) != 2:
                raise ValueError(f"Invalid GPU range: {token}")
            first, last = int(pieces[0]), int(pieces[1])
            if first < 0 or last < first:
                raise ValueError(f"Invalid GPU range: {token}")
            ids.extend(range(first, last + 1))
        else:
            gpu_id = int(token)
            if gpu_id < 0:
                raise ValueError("GPU IDs must be non-negative")
            ids.append(gpu_id)
    if not ids:
        raise ValueError("No GPU IDs were supplied")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate GPU IDs are not allowed")
    return ids


def _parse_multi_device_specs(
    devices_value: Optional[str],
    gpu_ids_value: Optional[str],
) -> Optional[list[str]]:
    """Return normalized worker devices, or None for the original single process."""
    if devices_value is None and gpu_ids_value is None:
        return None
    if devices_value is not None:
        raw = [item.strip() for item in devices_value.split(",") if item.strip()]
        if not raw:
            raise ValueError("--devices requires at least one device")
        normalized: list[str] = []
        seen_cuda: set[str] = set()
        for item in raw:
            device = torch.device(item)
            if device.type == "cuda":
                index = 0 if device.index is None else int(device.index)
                spec = f"cuda:{index}"
                if spec in seen_cuda:
                    raise ValueError(f"Duplicate CUDA worker device: {spec}")
                seen_cuda.add(spec)
                normalized.append(spec)
            elif device.type == "cpu":
                # Duplicate CPU workers are useful for scheduler regression tests.
                normalized.append("cpu")
            else:
                raise ValueError(
                    f"Multi-device workers currently support only CUDA or CPU, got {item!r}"
                )
    else:
        assert gpu_ids_value is not None
        if gpu_ids_value.strip().lower() == "all":
            count = int(torch.cuda.device_count())
            if count < 1:
                raise RuntimeError("--gpu-ids all requested, but no CUDA devices are visible")
            normalized = [f"cuda:{index}" for index in range(count)]
        else:
            normalized = [f"cuda:{index}" for index in _parse_gpu_ids(gpu_ids_value)]

    cuda_specs = [spec for spec in normalized if spec.startswith("cuda:")]
    if cuda_specs:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA worker devices were requested, but PyTorch CUDA is unavailable")
        visible = int(torch.cuda.device_count())
        for spec in cuda_specs:
            index = int(spec.split(":", 1)[1])
            if index >= visible:
                raise ValueError(
                    f"Requested {spec}, but only {visible} CUDA device(s) are visible. "
                    "GPU IDs are logical IDs after CUDA_VISIBLE_DEVICES is applied."
                )
    return normalized


def _validate_parallel_output_collisions(
    paths: Sequence[str],
    layout: _AggregateOutputLayout,
    config: CtffindConfig,
    *,
    write_tilt_png: bool,
) -> None:
    """Reject duplicate basenames before workers can race on per-image files."""
    generated: dict[Path, str] = {}
    for source in paths:
        stem = Path(source).stem
        candidates: list[Path] = [layout.ctf_dir / f"{stem}.ctf"]
        if config.fit_tilt and write_tilt_png:
            candidates.append(layout.tilt_png_dir / f"{stem}_ctftilt.png")
        if layout.spectra_dir is not None:
            candidates.append(layout.spectra_dir / f"{stem}_filtered_spectrum.mrc")
        if config.debug:
            candidates.extend(
                [
                    layout.debug_dir / f"{stem}_debug.json",
                    layout.debug_dir / f"{stem}_filtered_spectrum.mrc",
                ]
            )
        for candidate in candidates:
            resolved = candidate.resolve()
            previous = generated.get(resolved)
            if previous is not None and previous != source:
                raise RuntimeError(
                    f"Parallel output filename collision: {resolved} for both "
                    f"{previous} and {source}. Rename duplicate micrograph basenames "
                    "or run them with separate output directories."
                )
            generated[resolved] = source


def _multi_device_worker_loop(
    worker_index: int,
    device_spec: str,
    config: CtffindConfig,
    options: _MultiDeviceRunOptions,
    job_queue: Any,
    result_queue: Any,
    cpu_threads: int,
) -> None:
    """Persistent spawn worker: one fixed device and one micrograph at a time."""
    try:
        os.environ["OMP_NUM_THREADS"] = str(max(1, cpu_threads))
        os.environ["MKL_NUM_THREADS"] = str(max(1, cpu_threads))
        torch.set_num_threads(max(1, cpu_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        device = torch.device(device_spec)
        if device.type == "cuda":
            torch.cuda.set_device(0 if device.index is None else device.index)
        worker_config = replace(config, device=device_spec)
        worker_config.validate()
        estimator = TorchCtffindPowell(worker_config)
        result_queue.put(
            {
                "kind": "ready",
                "worker_index": worker_index,
                "device": device_spec,
                "device_name": (
                    torch.cuda.get_device_name(estimator.device)
                    if estimator.device.type == "cuda"
                    else "CPU"
                ),
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "kind": "fatal",
                "worker_index": worker_index,
                "device": device_spec,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return

    while True:
        job = job_queue.get()
        if job is None:
            break
        job_index, source_file = job
        job_started = time.perf_counter()
        try:
            fitted = fit_mrc_files(
                [source_file],
                worker_config,
                output_star=options.output_star,
                ctf_output_dir=options.ctf_output_dir,
                save_filtered_spectra_dir=options.save_filtered_spectra_dir,
                write_diagnostic_maps=options.write_diagnostic_maps,
                continue_on_error=False,
                tilt_png_output_dir=options.tilt_png_output_dir,
                write_tilt_png=options.write_tilt_png,
                extended_results_tsv=options.extended_results_tsv,
                output_ctffind=options.output_ctffind,
                output_avrot=options.output_avrot,
                debug_output_dir=options.debug_output_dir,
                timing_json=options.timing_json,
                _estimator=estimator,
                _write_aggregate_outputs=False,
                _print_progress=False,
                _paths_are_expanded=True,
                _skip_header_validation=True,
            )
            if len(fitted) != 1:
                raise RuntimeError(
                    f"Worker expected one result for {source_file}, got {len(fitted)}"
                )
            result = fitted[0]
            job_wall = float(time.perf_counter() - job_started)
            if worker_config.timing:
                timing_values = dict(result.timings or {})
                timing_values["multi_gpu_job_wall_s"] = job_wall
                result.timings = timing_values
            result_queue.put(
                {
                    "kind": "result",
                    "worker_index": worker_index,
                    "device": device_spec,
                    "job_index": int(job_index),
                    "source_file": source_file,
                    "job_wall_s": job_wall,
                    "result": dict(result.__dict__),
                }
            )
        except BaseException as exc:
            if estimator.device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            result_queue.put(
                {
                    "kind": "error",
                    "worker_index": worker_index,
                    "device": device_spec,
                    "job_index": int(job_index),
                    "source_file": source_file,
                    "job_wall_s": float(time.perf_counter() - job_started),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    result_queue.put(
        {
            "kind": "stopped",
            "worker_index": worker_index,
            "device": device_spec,
        }
    )


def fit_mrc_files_multi_device(
    input_paths: Sequence[str],
    config: CtffindConfig,
    devices: Sequence[str],
    output_star: str,
    ctf_output_dir: Optional[str] = None,
    save_filtered_spectra_dir: Optional[str] = None,
    write_diagnostic_maps: bool = True,
    continue_on_error: bool = False,
    tilt_png_output_dir: Optional[str] = None,
    write_tilt_png: bool = True,
    extended_results_tsv: Optional[str] = None,
    output_ctffind: Optional[str] = None,
    output_avrot: Optional[str] = None,
    debug_output_dir: Optional[str] = None,
    timing_json: Optional[str] = None,
    worker_cpu_threads: int = 0,
    checkpoint_every: int = 0,
) -> list[CtfFitResult]:
    """Run one persistent process per device, one independent micrograph per worker."""
    program_started = time.perf_counter()
    program_timings: dict[str, float] = {}
    stage_started = time.perf_counter()
    paths = _expand_input_paths(input_paths)
    program_timings["input_path_expansion_s"] = float(
        time.perf_counter() - stage_started
    )
    stage_started = time.perf_counter()
    if not continue_on_error:
        for path in paths:
            _count_mrc_micrographs(path)
    program_timings["input_header_validation_s"] = float(
        time.perf_counter() - stage_started
    )
    if not devices:
        raise ValueError("At least one multi-device worker is required")

    layout = _resolve_aggregate_output_layout(
        output_star,
        ctf_output_dir,
        save_filtered_spectra_dir,
        tilt_png_output_dir,
        extended_results_tsv,
        output_ctffind,
        output_avrot,
        debug_output_dir,
        timing_json,
    )
    layout.output_path.parent.mkdir(parents=True, exist_ok=True)
    layout.ctf_dir.mkdir(parents=True, exist_ok=True)
    if config.fit_tilt and write_tilt_png:
        layout.tilt_png_dir.mkdir(parents=True, exist_ok=True)
    if config.debug:
        layout.debug_dir.mkdir(parents=True, exist_ok=True)
    if layout.spectra_dir is not None:
        layout.spectra_dir.mkdir(parents=True, exist_ok=True)
    _validate_parallel_output_collisions(
        paths, layout, config, write_tilt_png=write_tilt_png
    )

    worker_count = min(len(devices), len(paths))
    worker_devices = list(devices[:worker_count])
    if worker_cpu_threads < 0:
        raise ValueError("--worker-cpu-threads must be >= 0")
    if worker_cpu_threads == 0:
        logical_cpus = max(1, int(os.cpu_count() or 1))
        worker_cpu_threads = max(1, min(4, logical_cpus // worker_count))
    if checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")
    effective_checkpoint_every = (
        checkpoint_every if checkpoint_every > 0 else max(1, worker_count)
    )

    print(
        f"Multi-device mode: {worker_count} persistent worker(s), "
        "one micrograph per worker at a time"
    )
    print("Worker devices: " + ", ".join(worker_devices))
    print(
        f"Input files: {len(paths)}; worker CPU threads: {worker_cpu_threads}; "
        f"checkpoint every {effective_checkpoint_every} completed micrograph(s)"
    )

    options = _MultiDeviceRunOptions(
        output_star=str(layout.output_path),
        ctf_output_dir=str(layout.ctf_dir),
        save_filtered_spectra_dir=(
            str(layout.spectra_dir) if layout.spectra_dir is not None else None
        ),
        write_diagnostic_maps=write_diagnostic_maps,
        tilt_png_output_dir=str(layout.tilt_png_dir),
        write_tilt_png=write_tilt_png,
        extended_results_tsv=str(layout.extended_tsv_path),
        output_ctffind=str(layout.ctffind_text_path),
        output_avrot=str(layout.avrot_path),
        debug_output_dir=str(layout.debug_dir),
        timing_json=str(layout.timing_path),
    )

    context = mp.get_context("spawn")
    job_queue = context.Queue()
    result_queue = context.Queue()
    workers: list[mp.Process] = []
    scheduler_started = time.perf_counter()
    for worker_index, device_spec in enumerate(worker_devices):
        process = context.Process(
            target=_multi_device_worker_loop,
            args=(
                worker_index,
                device_spec,
                config,
                options,
                job_queue,
                result_queue,
                worker_cpu_threads,
            ),
            name=f"ctffind5-{device_spec.replace(':', '-')}",
        )
        process.start()
        workers.append(process)

    for job_index, source_file in enumerate(paths):
        job_queue.put((job_index, source_file))
    for _ in workers:
        job_queue.put(None)
    program_timings["multi_gpu_scheduler_startup_and_enqueue_s"] = float(
        time.perf_counter() - scheduler_started
    )

    completed: dict[int, CtfFitResult] = {}
    assignments: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    ready_workers: dict[int, dict[str, object]] = {}
    terminal_jobs = 0
    last_checkpoint_count = 0
    checkpoint_total = 0.0

    def ordered_results() -> list[CtfFitResult]:
        return [completed[index] for index in sorted(completed)]

    def scheduler_summary() -> dict[str, object]:
        per_device: dict[str, dict[str, object]] = {}
        per_worker: dict[str, dict[str, object]] = {}
        for assignment in assignments:
            device_spec = str(assignment["device"])
            worker_key = str(int(assignment["worker_index"]))
            entry = per_device.setdefault(
                device_spec,
                {"completed_micrographs": 0, "busy_wall_s": 0.0},
            )
            worker_entry = per_worker.setdefault(
                worker_key,
                {
                    "worker_index": int(assignment["worker_index"]),
                    "device": device_spec,
                    "completed_micrographs": 0,
                    "busy_wall_s": 0.0,
                },
            )
            for target in (entry, worker_entry):
                target["completed_micrographs"] = int(target["completed_micrographs"]) + 1
                target["busy_wall_s"] = float(target["busy_wall_s"]) + float(
                    assignment["job_wall_s"]
                )
        for collection in (per_device, per_worker):
            for entry in collection.values():
                count = int(entry["completed_micrographs"])
                entry["mean_job_wall_s"] = (
                    float(entry["busy_wall_s"]) / count if count else 0.0
                )
        return {
            "mode": "one_micrograph_per_device_worker",
            "worker_count": worker_count,
            "devices": worker_devices,
            "worker_cpu_threads": worker_cpu_threads,
            "checkpoint_every": effective_checkpoint_every,
            "ready_workers": [ready_workers[k] for k in sorted(ready_workers)],
            "per_device": per_device,
            "per_worker": per_worker,
            "assignments": sorted(assignments, key=lambda item: int(item["job_index"])),
            "errors": errors,
        }

    def write_checkpoint(force: bool = False) -> None:
        nonlocal last_checkpoint_count, checkpoint_total
        current = len(completed)
        if not force and current - last_checkpoint_count < effective_checkpoint_every:
            return
        results_now = ordered_results()
        checkpoint_started = time.perf_counter()
        _write_aggregate_outputs(
            layout,
            results_now,
            config,
            paths,
            include_ctf_image=write_diagnostic_maps,
        )
        checkpoint_total += time.perf_counter() - checkpoint_started
        last_checkpoint_count = current
        if config.timing:
            timing_values = dict(program_timings)
            timing_values["output_checkpoint_tables_write_s"] = float(checkpoint_total)
            timing_values["program_total_wall_s"] = float(
                time.perf_counter() - program_started
            )
            _write_timing_json(
                layout.timing_path,
                results_now,
                program_timings=timing_values,
                config=config,
                resolved_device=None,
                diagnostic_output=write_diagnostic_maps,
                runtime_settings_extra={
                    "multi_gpu": True,
                    "requested_device": "multi-device",
                    "resolved_device": worker_devices,
                    "worker_devices": worker_devices,
                    "worker_count": worker_count,
                    "worker_cpu_threads": worker_cpu_threads,
                    "micrographs_per_worker_at_a_time": 1,
                    "multi_gpu_timing_note": (
                        "Per-stage totals sum work across concurrent workers and may "
                        "exceed elapsed program wall time."
                    ),
                },
                extra_payload={"multi_gpu_scheduler": scheduler_summary()},
            )

    try:
        while terminal_jobs < len(paths):
            try:
                message = result_queue.get(timeout=0.5)
            except queue.Empty:
                if not any(process.is_alive() for process in workers):
                    missing = len(paths) - terminal_jobs
                    raise RuntimeError(
                        f"All multi-device workers exited with {missing} job(s) unreported"
                    )
                continue

            kind = str(message.get("kind", ""))
            if kind == "ready":
                ready_workers[int(message["worker_index"])] = {
                    "worker_index": int(message["worker_index"]),
                    "device": str(message["device"]),
                    "device_name": str(message.get("device_name", "")),
                }
                continue
            if kind == "stopped":
                continue
            if kind == "fatal":
                error_entry = {
                    "worker_index": int(message.get("worker_index", -1)),
                    "device": str(message.get("device", "")),
                    "source_file": None,
                    "error": str(message.get("error", "worker initialization failed")),
                }
                errors.append(error_entry)
                print(
                    f"ERROR: worker {error_entry['worker_index']} on "
                    f"{error_entry['device']} failed to initialize: {error_entry['error']}",
                    file=sys.stderr,
                )
                if not continue_on_error:
                    raise RuntimeError(str(error_entry["error"]))
                continue
            if kind not in {"result", "error"}:
                raise RuntimeError(f"Unknown multi-device worker message: {message!r}")

            terminal_jobs += 1
            job_index = int(message["job_index"])
            source_file = str(message["source_file"])
            worker_index = int(message["worker_index"])
            device_spec = str(message["device"])
            job_wall = float(message.get("job_wall_s", 0.0))
            if kind == "error":
                error_entry = {
                    "job_index": job_index,
                    "worker_index": worker_index,
                    "device": device_spec,
                    "source_file": source_file,
                    "error": str(message.get("error", "unknown worker error")),
                }
                errors.append(error_entry)
                print(
                    f"ERROR: [{terminal_jobs}/{len(paths)}] {Path(source_file).name} "
                    f"on {device_spec}: {error_entry['error']}",
                    file=sys.stderr,
                )
                if not continue_on_error:
                    detail = str(message.get("traceback", ""))
                    raise RuntimeError(
                        f"{source_file} failed on {device_spec}: {error_entry['error']}\n{detail}"
                    )
                write_checkpoint()
                continue

            result = CtfFitResult(**dict(message["result"]))
            completed[job_index] = result
            assignments.append(
                {
                    "job_index": job_index,
                    "micrograph": result.micrograph_name,
                    "source_file": source_file,
                    "worker_index": worker_index,
                    "device": device_spec,
                    "job_wall_s": job_wall,
                }
            )
            good = (
                f"{result.thon_rings_good_fit_resolution_A:.2f} A"
                if result.thon_rings_good_fit_resolution_A > 0.0
                else "undetermined"
            )
            tilt_text = (
                f", tilt={result.tilt_angle_deg:.2f} deg, axis={result.tilt_axis_deg:.2f} deg"
                if result.tilt_fitted else ""
            )
            thickness_text = (
                f", thickness={result.ice_thickness_A / 10.0:.1f} nm"
                if result.ice_thickness_fitted else ""
            )
            print(
                f"  [{terminal_jobs}/{len(paths)}] {Path(source_file).name} "
                f"on {device_spec}: dfU={result.defocus1_A:.1f}, "
                f"dfV={result.defocus2_A:.1f}, angle={result.astigmatism_angle_deg:.2f}, "
                f"CC={result.score:.5f}, maxres={good}{tilt_text}{thickness_text}, "
                f"wall={job_wall:.3f}s"
            )
            write_checkpoint()
    except BaseException:
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join(timeout=5.0)
        if completed:
            write_checkpoint(force=True)
        try:
            job_queue.close()
            result_queue.close()
        except Exception:
            pass
        raise

    for process in workers:
        process.join(timeout=30.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        if process.exitcode not in (0, None):
            warnings.warn(
                f"Worker {process.name} exited with code {process.exitcode}",
                RuntimeWarning,
            )

    try:
        job_queue.close()
        result_queue.close()
    except Exception:
        pass
    program_timings["multi_gpu_compute_and_collect_s"] = float(
        time.perf_counter() - scheduler_started
    )
    if len(completed) != last_checkpoint_count or not layout.output_path.exists():
        write_checkpoint(force=True)
    program_timings["output_checkpoint_tables_write_s"] = float(checkpoint_total)
    program_timings["program_total_wall_s"] = float(
        time.perf_counter() - program_started
    )
    final_results = ordered_results()
    if config.timing:
        _write_timing_json(
            layout.timing_path,
            final_results,
            program_timings=program_timings,
            config=config,
            resolved_device=None,
            diagnostic_output=write_diagnostic_maps,
            runtime_settings_extra={
                "multi_gpu": True,
                "requested_device": "multi-device",
                "resolved_device": worker_devices,
                "worker_devices": worker_devices,
                "worker_count": worker_count,
                "worker_cpu_threads": worker_cpu_threads,
                "micrographs_per_worker_at_a_time": 1,
                "multi_gpu_timing_note": (
                    "Per-stage totals sum work across concurrent workers and may "
                    "exceed elapsed program wall time."
                ),
            },
            extra_payload={"multi_gpu_scheduler": scheduler_summary()},
        )

    print(f"Wrote {len(final_results)} rows to {_relion_path(layout.output_path)}")
    if write_diagnostic_maps:
        print(f"Wrote one .ctf MRC per micrograph under {_relion_path(layout.ctf_dir)}")
    print(f"Wrote TSV results to {_relion_path(layout.extended_tsv_path)}")
    print(f"Wrote CTFFIND text to {_relion_path(layout.ctffind_text_path)}")
    print(f"Wrote avrot curves to {_relion_path(layout.avrot_path)}")
    if config.debug:
        print(f"Wrote debug outputs under {_relion_path(layout.debug_dir)}")
    if config.timing:
        print(f"Wrote timing JSON to {_relion_path(layout.timing_path)}")
    if config.fit_tilt and write_tilt_png and any(r.tilt_png_name for r in final_results):
        print(f"Wrote tilt PNG diagnostics under {_relion_path(layout.tilt_png_dir)}")
    if errors:
        print(
            f"Completed with {len(errors)} failed micrograph(s); "
            f"{len(final_results)} result(s) were written.",
            file=sys.stderr,
        )
    return final_results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CTFFIND5-PyTorch for independent 2-D MRC micrographs: validated "
            "CTFFIND4-style defocus/astigmatism fitting, optional CTFFIND5-style "
            "tilt correction, equi-phase averaging, and ice-thickness fitting."
        )
    )
    parser.add_argument(
        "inputs", nargs="*",
        help="Independent 2-D MRC files, glob patterns, or directories containing MRC files",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--self-test", action="store_true", help="Run numerical smoke tests and exit")
    parser.add_argument("--output", "--output-star", dest="output", default="micrographs_ctf.star")
    parser.add_argument("--output-tsv", "--extended-results-tsv", dest="extended_results_tsv", default=None)
    parser.add_argument("--output-ctffind", default=None)
    parser.add_argument("--output-avrot", default=None)
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
    parser.add_argument("--box-size", type=int, default=512)
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
    parser.add_argument("--phase-shift", "--phase-shift-rad", dest="phase_shift", type=float, default=0.0)
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
        help=(
            "Filtered spectra fitted together on the GPU. Very large batches "
            "can be slower when optimizer convergence varies between images."
        ),
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
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device", default="auto",
        help=(
            "Single-process PyTorch device: auto, cpu, cuda, or cuda:N. "
            "For one persistent worker per GPU, use --gpu-ids or --devices."
        ),
    )
    device_group.add_argument(
        "--devices", default=None, metavar="DEVICE,DEVICE,...",
        help=(
            "Enable the internal multi-device scheduler, for example "
            "--devices cuda:0,cuda:1,cuda:2,cuda:3. Each persistent worker "
            "uses one fixed device and processes one micrograph at a time."
        ),
    )
    device_group.add_argument(
        "--gpu-ids", default=None, metavar="IDS",
        help=(
            "CUDA shorthand for multi-GPU mode, for example --gpu-ids 0,1,2,3, "
            "--gpu-ids 0-3, or --gpu-ids all. IDs are logical after "
            "CUDA_VISIBLE_DEVICES is applied."
        ),
    )
    parser.add_argument(
        "--worker-cpu-threads", type=int, default=0,
        help=(
            "CPU threads assigned to each multi-device worker; 0 chooses an "
            "automatic value capped at 4."
        ),
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=0, metavar="N",
        help=(
            "In multi-device mode, rewrite aggregate STAR/TSV/avrot files after "
            "every N completed micrographs; 0 uses one scheduler wave."
        ),
    )
    parser.add_argument(
        "--save-filtered-spectra", default=None, metavar="DIRECTORY"
    )
    parser.add_argument(
        "--no-diagnostic-output", action="store_true",
        help="Do not write .ctf maps or the _rlnCtfImage STAR column",
    )
    parser.add_argument(
        "--fit-tilt", action="store_true",
        help=(
            "Run the prior-free CTFFIND5-style local-spectrum tilt search; "
            "does not require a nominal tilt angle"
        ),
    )
    parser.add_argument(
        "--tilt-tile-size", type=int, default=128,
        help="Local spectrum box size; CTFFIND5 uses 128 pixels",
    )
    parser.add_argument(
        "--tilt-tile-stride", type=int, default=64,
        help=("Compatibility option. Native CTFFIND5 uses approximately 50%% overlap; "
              "the validated frontend derives spacing from subsection geometry."),
    )
    parser.add_argument(
        "--tilt-search-pixel-size", type=float, default=None,
        help=(
            "Experimental override for local tilt sampling. Default: native CTFFIND5 "
            "40--10 A fitting band with nominal 5 A/pixel sampling."
        ),
    )
    parser.add_argument("--tilt-axis-step", type=float, default=10.0)
    parser.add_argument("--tilt-angle-step", type=float, default=5.0)
    parser.add_argument(
        "--tilt-max-angle", type=float, default=80.0,
        help="Largest non-negative tilt angle in the global search",
    )
    parser.add_argument(
        "--tilt-candidate-batch-size", type=int, default=32,
        help=(
            "Tilt candidates evaluated together. Larger values reduce launch "
            "overhead but use more memory."
        ),
    )
    parser.add_argument(
        "--tilt-tile-batch-size", type=int, default=48,
        help=(
            "Local-tile chunk size for tilt scoring and correction. The "
            "default preserves the 0.5.1 accumulation order."
        ),
    )
    parser.add_argument("--tilt-refine-maxiter", type=int, default=90)
    parser.add_argument("--tilt-refine-axis-half-range", type=float, default=20.0)
    parser.add_argument("--tilt-refine-angle-half-range", type=float, default=10.0)
    parser.add_argument("--tilt-refine-defocus-half-range", type=float, default=5000.0)
    parser.add_argument("--tilt-min-tiles", type=int, default=3)
    parser.add_argument(
        "--tilt-rms-mad-cutoff", type=float, default=0.0,
        help="Optional local-tile RMS outlier filter; 0 matches CTFFIND5 and disables it",
    )
    parser.add_argument("--tilt-png-dir", default=None)
    parser.add_argument("--no-tilt-png", action="store_true")

    parser.add_argument(
        "--estimate-thickness", action="store_true",
        help="Fit CTFFIND5 finite-thickness nodes; also works without --fit-tilt",
    )
    parser.add_argument("--thickness-min", type=float, default=300.0, help="Angstrom")
    parser.add_argument("--thickness-max", type=float, default=4000.0, help="Angstrom")
    parser.add_argument("--thickness-step", type=float, default=10.0, help="Angstrom")
    parser.add_argument("--thickness-low-resolution", type=float, default=30.0)
    parser.add_argument("--thickness-high-resolution", type=float, default=3.0)
    parser.add_argument(
        "--thickness-defocus-range", type=float, default=1000.0,
        help="Half-range in Angstrom for the joint 1-D mean-defocus search",
    )
    parser.add_argument(
        "--thickness-defocus-step", type=float, default=10.0,
        help="Mean-defocus step in Angstrom for the joint 1-D node search",
    )
    parser.add_argument("--no-thickness-2d-refine", action="store_true")
    parser.add_argument("--thickness-refine-maxiter", type=int, default=40)
    parser.add_argument("--thickness-rounded-square", action="store_true")
    parser.add_argument("--thickness-downweight-nodes", action="store_true")
    parser.add_argument(
        "--thickness-candidate-batch-size", type=int, default=1024,
        help=(
            "Joint 1-D thickness/defocus candidates evaluated together. "
            "Independent candidate scores make larger batches lossless."
        ),
    )
    parser.add_argument(
        "--thickness-refine-batch-size", type=int, default=32,
        help=(
            "Independent four-parameter thickness Powell refinements evaluated "
            "together. Reduce this value if the 2-D thickness stage runs out of VRAM."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Write per-micrograph debug JSON and filtered spectrum MRC")
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument(
        "--timing", action="store_true",
        help=(
            "Collect detailed stage timings and write timing JSON. CUDA is "
            "synchronized at timing boundaries, so timed runs are slightly slower."
        ),
    )
    parser.add_argument("--timing-json", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _run_self_test(device_spec: str = "cpu") -> None:
    """Deterministic tests for standard fitting, EPA/thickness, and CTFTilt parity."""
    device = _resolve_device(device_spec)
    size = 128
    pixel = 1.5
    config = CtffindConfig(
        pixel_size_A=pixel,
        box_size=size,
        minimum_resolution_A=30.0,
        maximum_resolution_A=5.0,
        minimum_defocus_A=10_000.0,
        maximum_defocus_A=25_000.0,
        defocus_search_step_A=100.0,
        astigmatism_tolerance_A=2_000.0,
        device=str(device),
        estimate_thickness=True,
        thickness_min_A=400.0,
        thickness_max_A=900.0,
        thickness_step_A=20.0,
        thickness_defocus_search_range_A=200.0,
        thickness_defocus_step_A=50.0,
        thickness_2d_refine=False,
        thickness_candidate_batch_size=256,
    )
    config.validate()
    estimator = TorchCtffindPowell(config)
    rng = np.random.default_rng(7)
    image = torch.as_tensor(
        rng.normal(size=(1, 256, 256)).astype(np.float32),
        device=estimator.device,
        dtype=estimator.dtype,
    )
    bundle = _ctffind_preprocess_bundle_batch(image, pixel, config)
    for tensor in (
        bundle.raw_amplitude,
        bundle.normalized_cross_capped,
        bundle.background,
        bundle.filtered_unmasked,
        bundle.filtered_masked,
    ):
        if tensor.shape != (1, size, size) or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("full-2D filtering self-test failed")

    d1, d2 = 16_000.0, 14_800.0
    angle = math.radians(27.0)
    wanted_thickness = 600.0
    coord = (
        torch.arange(size, device=estimator.device, dtype=estimator.dtype) - size // 2
    ) / (float(size) * pixel)
    fy, fx = torch.meshgrid(coord, coord, indexing="ij")
    freq2 = fx.square() + fy.square()
    azimuth = torch.atan2(fy, fx)
    model = _finite_thickness_power_model(
        freq2[None], azimuth[None],
        torch.tensor([d1], device=estimator.device, dtype=estimator.dtype),
        torch.tensor([d2], device=estimator.device, dtype=estimator.dtype),
        torch.tensor([angle], device=estimator.device, dtype=estimator.dtype),
        torch.tensor([wanted_thickness], device=estimator.device, dtype=estimator.dtype),
        estimator.wavelength_A,
        estimator.spherical_aberration_A,
        estimator.amplitude_phase_rad,
        config.fixed_phase_shift_rad,
        False,
    )[0]
    generator = torch.Generator(device=estimator.device).manual_seed(5)
    synthetic = model + 0.15 + 0.03 * torch.sqrt(freq2) + 0.005 * torch.randn(
        model.shape, generator=generator, device=estimator.device, dtype=estimator.dtype
    )
    epa = _compute_epa_statistics(
        synthetic, pixel, config, d1, d2, angle,
        estimator.wavelength_A, estimator.spherical_aberration_A,
        estimator.amplitude_phase_rad, config.fixed_phase_shift_rad,
        theoretical_thickness_A=None, node_mode=False, rounded_square=False,
    )
    result = estimator.estimate_ice_thickness(
        synthetic, pixel, d1, d2, angle, epa
    )
    if abs(result.thickness_A - wanted_thickness) > 40.0:
        raise RuntimeError(
            f"EPA thickness self-test failed: {result.thickness_A:.1f} A"
        )
    if abs(_tilt_axis_to_output_convention(20.0) - 160.0) > 1.0e-6:
        raise RuntimeError("tilt-axis convention self-test failed")
    # Validate the restored 0.4 global-Pearson local tilt objective on a
    # deterministic synthetic tilted CTF^2 field.  This isolates the frontend
    # from corrected-spectrum/full-2D filtering.
    compatibility = _make_v04_tilt_config(config)
    compatibility = replace(
        compatibility,
        tilt_low_resolution_A=40.0,
        tilt_high_resolution_A=10.0,
        tilt_candidate_batch_size=8,
        tilt_tile_batch_size=16,
    )
    local_size = 128
    local_pixel = 5.0
    coordinates = np.asarray(
        [(x, y) for y in (-600.0, 0.0, 600.0) for x in (-600.0, 0.0, 600.0)],
        dtype=np.float32,
    )
    _, _, _, freq2, az = _v04_frequency_grid(local_size, local_pixel, device, torch.float32)
    support = (freq2 > 1.0 / 40.0**2) & (freq2 < 1.0 / 10.0**2)
    truth_axis = 70.0
    truth_angle = 35.0
    truth_mean = 18_000.0
    gx_truth = math.sin(math.radians(truth_axis)) * math.tan(math.radians(truth_angle))
    gy_truth = math.cos(math.radians(truth_axis)) * math.tan(math.radians(truth_angle))
    local_mean = torch.as_tensor(
        truth_mean + gx_truth * coordinates[:, 0] + gy_truth * coordinates[:, 1],
        device=device,
        dtype=torch.float32,
    )
    astig_component = 0.5 * 800.0 * torch.cos(
        2.0 * (az - math.radians(27.0))
    )
    phase = (
        PI * _v04_electron_wavelength_A(300.0) * freq2[None]
        * (
            local_mean[:, None, None] + astig_component[None]
            - 0.5 * _v04_electron_wavelength_A(300.0)**2
            * freq2[None] * (2.7e7)
        )
        + _v04_amplitude_contrast_phase_rad(0.07)
    )
    synthetic_local = torch.sin(phase).square()
    # Deterministic tiny perturbation prevents an unrealistically exact tie.
    synthetic_local = synthetic_local + 1.0e-4 * torch.cos(
        torch.arange(local_size, device=device, dtype=torch.float32)[None, :, None]
    )
    candidates = np.asarray(
        [
            [truth_axis, truth_angle, truth_mean],
            [truth_axis + 20.0, truth_angle, truth_mean],
            [truth_axis, truth_angle - 15.0, truth_mean],
        ],
        dtype=np.float64,
    )
    score_context = _v04_make_tilt_score_context(
        synthetic_local,
        local_pixel,
        coordinates,
        truth_mean + 400.0,
        truth_mean - 400.0,
        27.0,
        compatibility,
        _v04_electron_wavelength_A(300.0),
    )
    scores = _v04_tilt_local_scores(candidates, score_context)
    if int(np.argmax(scores)) != 0 or float(scores[0]) < 0.999:
        raise RuntimeError(
            f"Restored CTFTilt local-score regression failed: {scores.tolist()}"
        )

    print(
        f"ctffind5_pytorch {VERSION} self-test passed on {device}: "
        f"thickness={result.thickness_A:.1f} A, "
        f"initial_cutoff={epa.good_fit_resolution_A:.3f} A"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    multi_devices = _parse_multi_device_specs(args.devices, args.gpu_ids)
    if multi_devices is None and ("," in args.device or args.device.count(":") > 1):
        parser.error(
            "--device accepts only one PyTorch device such as cuda:0. "
            "Use --gpu-ids 0,1 or --devices cuda:0,cuda:1 for multiple GPUs."
        )
    if args.self_test:
        if multi_devices is None:
            _run_self_test(args.device)
        else:
            for device_spec in multi_devices:
                _run_self_test(device_spec)
        return 0
    if not args.inputs:
        parser.error("at least one 2-D MRC input or glob is required")
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
        tilt_target_pixel_size_A=args.tilt_search_pixel_size,
        tilt_axis_step_deg=args.tilt_axis_step,
        tilt_angle_step_deg=args.tilt_angle_step,
        tilt_max_angle_deg=args.tilt_max_angle,
        tilt_candidate_batch_size=args.tilt_candidate_batch_size,
        tilt_tile_batch_size=args.tilt_tile_batch_size,
        tilt_refine_maxiter=args.tilt_refine_maxiter,
        tilt_refine_axis_half_range_deg=args.tilt_refine_axis_half_range,
        tilt_refine_angle_half_range_deg=args.tilt_refine_angle_half_range,
        tilt_refine_defocus_half_range_A=args.tilt_refine_defocus_half_range,
        tilt_min_tiles=args.tilt_min_tiles,
        tilt_rms_mad_cutoff=args.tilt_rms_mad_cutoff,
        estimate_thickness=args.estimate_thickness,
        thickness_min_A=args.thickness_min,
        thickness_max_A=args.thickness_max,
        thickness_step_A=args.thickness_step,
        thickness_low_resolution_A=args.thickness_low_resolution,
        thickness_high_resolution_A=args.thickness_high_resolution,
        thickness_defocus_search_range_A=args.thickness_defocus_range,
        thickness_defocus_step_A=args.thickness_defocus_step,
        thickness_2d_refine=not args.no_thickness_2d_refine,
        thickness_refine_maxiter=args.thickness_refine_maxiter,
        thickness_use_rounded_square=args.thickness_rounded_square,
        thickness_downweight_nodes=args.thickness_downweight_nodes,
        thickness_candidate_batch_size=args.thickness_candidate_batch_size,
        thickness_refine_batch_size=args.thickness_refine_batch_size,
        debug=args.debug,
        timing=args.timing,
    )
    config.validate()
    common_arguments = dict(
        output_star=args.output,
        ctf_output_dir=args.ctf_dir,
        save_filtered_spectra_dir=args.save_filtered_spectra,
        write_diagnostic_maps=not args.no_diagnostic_output,
        continue_on_error=args.continue_on_error,
        tilt_png_output_dir=args.tilt_png_dir,
        write_tilt_png=not args.no_tilt_png,
        extended_results_tsv=args.extended_results_tsv,
        output_ctffind=args.output_ctffind,
        output_avrot=args.output_avrot,
        debug_output_dir=args.debug_dir,
        timing_json=args.timing_json,
    )
    if multi_devices is None:
        fit_mrc_files(args.inputs, config, **common_arguments)
    else:
        fit_mrc_files_multi_device(
            args.inputs,
            config,
            multi_devices,
            worker_cpu_threads=args.worker_cpu_threads,
            checkpoint_every=args.checkpoint_every,
            **common_arguments,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("ERROR: interrupted by user", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if "--debug" in sys.argv:
            traceback.print_exc()
        raise SystemExit(1)
