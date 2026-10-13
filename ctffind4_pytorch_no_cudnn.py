"""
A wxWidgets-free PyTorch reimplementation of the fast-search path of
CTFFIND 4.1.8 for global micrograph CTF estimation.

No-cuDNN variant: cuDNN is disabled before tensor operations. CUDA execution
uses PyTorch's non-cuDNN backends; --device cuda still selects the GPU.
SciPy is not required; diagnostic extrema envelopes are interpolated without
Savitzky-Golay smoothing. CTF fitting algorithms and good-fit calculations
are retained.

Implemented
-----------
* MRC input (one 2-D micrograph per file; 3-D stacks are also accepted)
* RELION corrected_micrographs.star input via --input_star (plain-text reader)
* Batched PyTorch amplitude-spectrum generation and CTFFIND-like preprocessing
* Batched 1-D mean-defocus grid search
* Batched, derivative-free Powell direction-set refinement implemented in
  PyTorch; SciPy is not used for optimization
* Batched mirror/rotation estimation of the astigmatism angle
* Fixed additional phase shift in the CTF model; phase-search CLI is reserved
* One RELION-style ``.ctf`` diagnostic MRC per micrograph
* CTFFIND-style "Thon rings with good fit up to" resolution
* RELION 3.1 ``micrographs_ctf.star`` output written without starfile

Dependencies
------------
    numpy, torch, mrcfile

Example
-------
    python ctffind_pytorch_no_cudnn.py "MotionCorr/job003/*.mrc" \
        --pixel-size 1.06 --voltage 300 --cs 2.7 \
        --amplitude-contrast 0.07 --box-size 512 \
        --min-resolution 30 --max-resolution 5 \
        --min-defocus 5000 --max-defocus 50000 \
        --defocus-step 500 --preprocess-batch-size 4 --fit-batch-size 64 \
        --output micrographs_ctf.star --ctf-dir CtfFind/job005

    python ctffind_pytorch_no_cudnn.py \
        --input_star MotionCorr/job002/corrected_micrographs.star \
        --output CtfFind/job005/micrographs_ctf.star --device cuda

With --input_star, positional inputs are ignored. Explicit --pixel-size,
--voltage, --cs and --amplitude-contrast override STAR optics values; missing
values use the original defaults (or the MRC header for pixel size).
Run from the RELION project directory for project-relative micrograph paths.
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

# Scheme 1: keep conv2d/grid_sample, but dispatch without cuDNN on CUDA.
# Set this before any tensor operations.
# Do not query cuDNN's version: that can trigger the failing library loader.
torch.backends.cudnn.enabled = False

import torch.nn.functional as F


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
    # Display-only envelope: avoid a SciPy dependency in this standalone variant.
    smoothed = values
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

    This intentionally stops before the display-only interpolated envelope
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


def _iter_star_tokens(path: Path) -> Iterator[tuple[str, int, bool]]:
    """Yield (text, line number, quoted) without shell/backslash processing."""
    with path.open("r", encoding="utf-8-sig") as handle:
        lines = iter(enumerate(handle, 1))
        for line_number, line in lines:
            if line.startswith(";"):
                parts = [line[1:].rstrip("\r\n")]
                for _, continuation in lines:
                    if continuation.startswith(";"):
                        if continuation[1:].strip() and not continuation[1:].lstrip().startswith("#"):
                            raise ValueError(f"{path}:{line_number}: invalid multiline STAR value")
                        break
                    parts.append(continuation.rstrip("\r\n"))
                else:
                    raise ValueError(f"{path}:{line_number}: unterminated multiline STAR value")
                yield "\n".join(parts), line_number, True
                continue
            pos = 0
            while pos < len(line):
                if line[pos].isspace():
                    pos += 1
                    continue
                if line[pos] == "#":
                    break
                if line[pos] in ("'", '"'):
                    quote = line[pos]
                    start = pos = pos + 1
                    while pos < len(line):
                        if line[pos] == quote and (
                            pos + 1 == len(line) or line[pos + 1].isspace()
                            or line[pos + 1] == "#"
                        ):
                            break
                        pos += 1
                    else:
                        raise ValueError(f"{path}:{line_number}: unterminated quoted STAR value")
                    yield line[start:pos], line_number, True
                    pos += 1
                else:
                    start = pos
                    while pos < len(line) and not line[pos].isspace():
                        pos += 1
                    yield line[start:pos], line_number, False


def _star_control(token: tuple[str, int, bool]) -> bool:
    value, _, quoted = token
    value = value.lower()
    return not quoted and (
        value.startswith(("_", "data_", "save_"))
        or value in {"loop_", "stop_", "global_"}
    )


def _iter_star_rows(path: Path) -> Iterator[tuple[str, dict[str, str], int]]:
    """Read STAR loops by column name; a row may span physical text lines."""
    tokens = iter(_iter_star_tokens(path))
    token = next(tokens, None)
    block = ""
    while token is not None:
        value, line_number, quoted = token
        lower = value.lower()
        if not quoted and lower.startswith("data_"):
            block = lower[5:]
            token = next(tokens, None)
        elif not quoted and lower == "loop_":
            columns: list[str] = []
            token = next(tokens, None)
            while token is not None and not token[2] and token[0].startswith("_"):
                columns.append(token[0].lower())
                token = next(tokens, None)
            if not columns or len(set(columns)) != len(columns):
                raise ValueError(f"{path}:{line_number}: missing or duplicate STAR loop columns")
            row: list[str] = []
            row_line = line_number
            while token is not None and not _star_control(token):
                if not row:
                    row_line = token[1]
                row.append(token[0])
                if len(row) == len(columns):
                    yield block, dict(zip(columns, row)), row_line
                    row = []
                token = next(tokens, None)
            if row:
                raise ValueError(
                    f"{path}:{row_line}: incomplete STAR row in data_{block}: "
                    f"expected {len(columns)} values, found {len(row)}"
                )
        elif not quoted and lower.startswith("_"):
            scalar = next(tokens, None)
            if scalar is None or _star_control(scalar):
                raise ValueError(f"{path}:{line_number}: missing value for {value}")
            token = next(tokens, None)
        elif not quoted and lower == "stop_":
            token = next(tokens, None)
        else:
            raise ValueError(f"{path}:{line_number}: unexpected STAR token {value!r}")


@dataclass(frozen=True)
class _StarMicrographInput:
    config: CtffindConfig
    original_pixel_size_A: Optional[float] = None


def _resolve_star_micrograph(name: str, star_path: Path, line_number: int) -> str:
    if name in {"", ".", "?"}:
        raise ValueError(f"{star_path}:{line_number}: missing _rlnMicrographName")
    if "@" in name and name.split("@", 1)[0].isdigit():
        raise ValueError(
            f"{star_path}:{line_number}: indexed stack names are not supported in "
            "--input_star; supply a micrograph MRC filename"
        )
    path = Path(name)
    if path.is_absolute():
        candidates = [path]
    else:
        # RELION paths are normally relative to the project working directory.
        # Also accept STAR-relative files and project paths from nested job dirs.
        candidates = [Path.cwd() / path]
        candidates.extend(parent / path for parent in star_path.parents)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(
        f"{star_path}:{line_number}: micrograph not found: {name!r}. "
        "Run from the RELION project directory or use absolute paths in the STAR."
    )


def _read_micrograph_star(
    filename: str,
    config: CtffindConfig,
    overrides: Optional[dict[str, float]] = None,
) -> dict[str, _StarMicrographInput]:
    """Read RELION micrograph paths and optics using only Python text I/O.

    Explicit overrides take priority over STAR values, then config defaults.
    Motion statistics and per-micrograph metadata filenames are not CTF inputs.
    """
    path = Path(filename).resolve()
    optics: dict[int, dict[str, str]] = {}
    micrographs: list[tuple[dict[str, str], int]] = []

    def group_id(value: str, line_number: int) -> int:
        try:
            group = int(value)
            if group > 0:
                return group
        except ValueError:
            pass
        raise ValueError(f"{path}:{line_number}: invalid _rlnOpticsGroup {value!r}")

    for block, row, line_number in _iter_star_rows(path):
        if block == "optics":
            if "_rlnopticsgroup" not in row:
                raise ValueError(f"{path}:{line_number}: data_optics lacks _rlnOpticsGroup")
            group = group_id(row["_rlnopticsgroup"], line_number)
            if group in optics:
                raise ValueError(f"{path}:{line_number}: duplicate optics group {group}")
            optics[group] = row
        elif "_rlnmicrographname" in row:
            micrographs.append((row, line_number))
    if not micrographs:
        raise ValueError(f"{path}: no micrograph rows with _rlnMicrographName found")

    fields = {
        "pixel_size_A": "_rlnmicrographpixelsize",
        "acceleration_voltage_kV": "_rlnvoltage",
        "spherical_aberration_mm": "_rlnsphericalaberration",
        "amplitude_contrast": "_rlnamplitudecontrast",
    }
    overrides = overrides or {}
    unknown = set(overrides) - fields.keys()
    if unknown:
        raise ValueError(f"Unknown STAR microscope overrides: {sorted(unknown)}")

    def number(value: str | float, field: str, line_number: int) -> float:
        try:
            result = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid {field}: {value!r}") from exc
        if not math.isfinite(result):
            raise ValueError(f"{path}:{line_number}: non-finite {field}: {value!r}")
        if field in {"pixel_size_A", "original_pixel_size_A", "acceleration_voltage_kV"} and result <= 0:
            raise ValueError(f"{path}:{line_number}: {field} must be positive")
        if field == "spherical_aberration_mm" and result < 0:
            raise ValueError(f"{path}:{line_number}: Cs must be non-negative")
        if field == "amplitude_contrast" and not 0 <= result < 1:
            raise ValueError(f"{path}:{line_number}: amplitude contrast must satisfy 0 <= A < 1")
        return result

    inputs: dict[str, _StarMicrographInput] = {}
    for row, line_number in micrographs:
        if optics:
            if "_rlnopticsgroup" not in row:
                raise ValueError(f"{path}:{line_number}: micrograph lacks _rlnOpticsGroup")
            group = group_id(row["_rlnopticsgroup"], line_number)
            if group not in optics:
                raise ValueError(f"{path}:{line_number}: unknown optics group {group}")
            values = dict(optics[group])
        else:
            if "_rlnopticsgroup" in row:
                raise ValueError(f"{path}:{line_number}: _rlnOpticsGroup refers to missing data_optics")
            values = {}
        # Support older one-table micrograph STAR files as well.
        values.update({key: value for key, value in row.items() if value not in {"?", "."}})
        params = {}
        for field, column in fields.items():
            raw = overrides.get(field, values.get(column))
            if raw is None or raw in {"?", "."}:
                raw = getattr(config, field)
            if raw is not None:
                params[field] = number(raw, field, line_number)
        # Original (movie) pixel size is not a fallback for corrected pixel size:
        # motion correction may have binned the image. Use the MRC header instead.
        original = values.get("_rlnmicrographoriginalpixelsize")
        original_pixel = (
            number(original, "original_pixel_size_A", line_number)
            if original not in {None, "?", "."} else None
        )
        effective_config = replace(config, **params)
        effective_config.validate()
        source = _resolve_star_micrograph(row["_rlnmicrographname"], path, line_number)
        if source in inputs:
            raise ValueError(f"{path}:{line_number}: duplicate micrograph: {source}")
        inputs[source] = _StarMicrographInput(effective_config, original_pixel)
    return inputs


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
    config: Optional[CtffindConfig] = None


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


def _iter_micrograph_records(
    paths: Sequence[str],
    config: CtffindConfig,
    ctf_dir: Path,
    star_inputs: Optional[dict[str, _StarMicrographInput]] = None,
) -> Iterator[_MicrographRecord]:
    used_ctf_paths: dict[Path, str] = {}
    for path in paths:
        input_config = star_inputs[path].config if star_inputs else config
        with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
            data = mrc.data
            header_pixel = _pixel_size_from_mrc(mrc)
            pixel_size = input_config.pixel_size_A or header_pixel
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
                    config=input_config,
                )


def _iter_compatible_batches(
    records: Iterable[_MicrographRecord],
    batch_size: int,
) -> Iterator[list[_MicrographRecord]]:
    batch: list[_MicrographRecord] = []
    key = None
    for record in records:
        record_key = (record.array.shape, round(record.pixel_size_A, 8), record.config)
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
    star_inputs: Optional[dict[str, _StarMicrographInput]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    optics_keys: dict[tuple[float, float, float, float, float], int] = {}
    optics_rows: list[tuple[int, float, float, CtffindConfig]] = []
    row_groups: list[int] = []
    for result in results:
        star_input = star_inputs.get(result.source_file) if star_inputs else None
        input_config = star_input.config if star_input else config
        original_pixel = (
            star_input.original_pixel_size_A
            if star_input and star_input.original_pixel_size_A is not None
            else result.pixel_size_input_A
        )
        key = (
            round(result.pixel_size_input_A, 8),
            round(input_config.acceleration_voltage_kV, 8),
            round(input_config.spherical_aberration_mm, 8),
            round(input_config.amplitude_contrast, 8),
            round(original_pixel, 8),
        )
        group = optics_keys.get(key)
        if group is None:
            group = len(optics_rows) + 1
            optics_keys[key] = group
            optics_rows.append((group, result.pixel_size_input_A, original_pixel, input_config))
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
        for group, pixel, original_pixel, input_config in optics_rows:
            handle.write(
                f"opticsGroup{group} {group:d} {original_pixel:.6f} "
                f"{input_config.acceleration_voltage_kV:.6f} "
                f"{input_config.spherical_aberration_mm:.6f} "
                f"{input_config.amplitude_contrast:.6f} {pixel:.6f}\n"
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


def fit_mrc_files(
    input_paths: Sequence[str],
    config: CtffindConfig,
    output_star: str,
    ctf_output_dir: Optional[str] = None,
    save_filtered_spectra_dir: Optional[str] = None,
    write_diagnostic_maps: bool = True,
    continue_on_error: bool = False,
    input_star: Optional[str] = None,
    input_star_overrides: Optional[dict[str, float]] = None,
) -> list[CtfFitResult]:
    """Run a two-stage raw-image/preprocessed-spectrum GPU pipeline.

    input_star ignores input_paths and supplies per-micrograph optics parameters.
    input_star_overrides uses CtffindConfig microscope field names and takes
    priority over STAR values; config supplies defaults for missing STAR fields.
    """
    star_inputs = (
        _read_micrograph_star(input_star, config, input_star_overrides)
        if input_star is not None else None
    )
    paths = list(star_inputs) if star_inputs is not None else _expand_input_paths(input_paths)
    if input_star is not None:
        print(f"Input STAR: {input_star}; positional MRC inputs are ignored")
    active_config = star_inputs[paths[0]].config if star_inputs else config
    estimator = TorchCtffindPowell(active_config)
    output_path = Path(output_star).resolve()
    ctf_dir = (
        Path(ctf_output_dir).resolve()
        if ctf_output_dir is not None
        else output_path.parent
    )
    ctf_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir = None
    if save_filtered_spectra_dir:
        spectra_dir = Path(save_filtered_spectra_dir).resolve()
        spectra_dir.mkdir(parents=True, exist_ok=True)

    total_images = sum(_count_mrc_micrographs(path) for path in paths)
    results: list[CtfFitResult] = []
    processed = 0
    preprocessed = 0
    print(f"Device: {estimator.device}")
    print("cuDNN: disabled (CUDA operations use PyTorch non-cuDNN backends)")
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
            results.append(result)
            processed += 1
            good = (
                f"{result.thon_rings_good_fit_resolution_A:.2f} A"
                if result.thon_rings_good_fit_resolution_A > 0.0
                else "undetermined"
            )
            print(
                f"  [{processed}/{total_images}] {Path(result.source_file).name}: "
                f"dfU={result.defocus1_A:.1f}, dfV={result.defocus2_A:.1f}, "
                f"angle={result.astigmatism_angle_deg:.2f}, "
                f"CC={result.score:.5f}, maxres={good}"
            )
        _write_relion_star(
            output_path,
            results,
            config,
            include_ctf_image=write_diagnostic_maps,
            star_inputs=star_inputs,
        )

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

    records = _iter_micrograph_records(paths, config, ctf_dir, star_inputs)
    for raw_batch in _iter_compatible_batches(records, config.preprocess_batch_size):
        batch_config = raw_batch[0].config or config
        if batch_config != active_config:
            # Complete all pending spectra with the previous optics first.
            flush_all_pending()
            active_config = batch_config
            estimator = TorchCtffindPowell(batch_config)
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
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fast CTFFIND-4.1.8-like CTF estimation with batched PyTorch "
            "Powell optimization and RELION STAR/.ctf output."
        )
    )
    parser.add_argument(
        "inputs", nargs="*",
        help="MRC/MRCS files, glob patterns, or directories; ignored with --input_star",
    )
    parser.add_argument(
        "--input_star", "--input-star", default=None, metavar="STAR",
        help="RELION corrected_micrographs.star; replaces all positional MRC inputs",
    )
    parser.add_argument("--output", default="micrographs_ctf.star")
    parser.add_argument(
        "--ctf-dir", default=None,
        help="Directory for per-micrograph .ctf MRC files; default: STAR directory",
    )
    parser.add_argument(
        "--pixel-size", type=float, default=None,
        help="Input pixel size in Angstrom; overrides STAR optics, then falls back to MRC header",
    )
    parser.add_argument("--voltage", type=float, default=None,
                        help="Voltage in kV; overrides STAR optics; fallback: 300")
    parser.add_argument("--cs", type=float, default=None,
                        help="Spherical aberration in mm; overrides STAR optics; fallback: 2.7")
    parser.add_argument("--amplitude-contrast", type=float, default=None,
                        help="Amplitude contrast; overrides STAR optics; fallback: 0.07")
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
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.input_star is None and not args.inputs:
        parser.error("provide MRC inputs or --input_star corrected_micrographs.star")
    config = CtffindConfig(
        pixel_size_A=args.pixel_size,
        acceleration_voltage_kV=args.voltage if args.voltage is not None else 300.0,
        spherical_aberration_mm=args.cs if args.cs is not None else 2.7,
        amplitude_contrast=args.amplitude_contrast if args.amplitude_contrast is not None else 0.07,
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
    )
    config.validate()
    fit_mrc_files(
        args.inputs,
        config,
        output_star=args.output,
        ctf_output_dir=args.ctf_dir,
        save_filtered_spectra_dir=args.save_filtered_spectra,
        write_diagnostic_maps=not args.no_diagnostic_output,
        continue_on_error=args.continue_on_error,
        input_star=args.input_star,
        input_star_overrides={
            field: value for field, value in {
                "pixel_size_A": args.pixel_size,
                "acceleration_voltage_kV": args.voltage,
                "spherical_aberration_mm": args.cs,
                "amplitude_contrast": args.amplitude_contrast,
            }.items() if value is not None
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
