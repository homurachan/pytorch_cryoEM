#!/usr/bin/env python3
"""Bsoft blocfilt-compatible local-resolution filtering in Python/PyTorch.

This is an algorithmic port of the Bsoft ``blocfilt`` path:

  bsoft/src/blocfilt.cpp
  Bimage::local_filter / Bimage::local_filter_voxel in
  bsoft/src/img/Bimage_resolution.cpp
  Bimage::fspace_bandpass in bsoft/src/img/Bimage_fspace.cpp

The compatibility path intentionally preserves several non-obvious Bsoft
behaviours:

* one ``box^3`` patch is centered on every selected output voxel;
* the local-resolution value at that voxel is used as ``res_hi``;
* ``res_lo`` is 1e37 A, exactly as in ``local_filter_voxel``;
* the reciprocal-space edge width is fixed at 0.002 1/A;
* the filter edge is Bsoft's logistic/Fermi form using GOLDEN/width;
* only the center voxel of the inverse-transformed patch is retained;
* unprocessed voxels are zero, rather than copied from the input map;
* the Bsoft edge comparison is preserved, including its inclusive upper edge;
* patches extending outside the volume are zero-filled (Bsoft's default
  FILL_USER=0 extraction path).

The FFT implementation is PyTorch rather than FFTW, so bitwise identity with
Bsoft is not expected.  With float32 input, the transform uses complex64,
matching Bsoft's float-complex storage as closely as practical.

ResMap-PyTorch 2.0.0 compatibility
---------------------------------
The ``*_resmap.mrc`` output can be supplied directly with ``-Resolution``.
ResMap-PyTorch uses 100.0 A as the sentinel outside unresolved/masked regions.
Bsoft itself has no knowledge of this sentinel.  Use the original mask with
``-Mask`` for the closest Bsoft workflow, or explicitly add
``--mask-from-resmap`` to exclude voxels equal to the sentinel.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F


GOLDEN = 1.61803398874989484820458683436563811772
BANDPASS_WIDTH = 0.002  # reciprocal-space width, 1/A; hard-coded by Bsoft blocfilt
RES_LO = 1.0e37         # A; hard-coded by Bsoft local_filter_voxel


def _parse_triplet(text: str, cast=float) -> Tuple:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) == 1:
        value = cast(parts[0])
        return (value, value, value)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected one value or x,y,z")
    return tuple(cast(p) for p in parts)


def _parse_mask_spec(text: str) -> Tuple[str, int]:
    """Parse Bsoft-style ``mask.mrc,2``; level -1 means default behaviour."""
    value = str(text)
    if "," not in value:
        return value, -1
    path, maybe_level = value.rsplit(",", 1)
    try:
        level = int(maybe_level)
    except ValueError:
        # A comma can legally occur in a path.  In that case preserve the path.
        return value, -1
    return path, level


def _read_mrc(path: str) -> Tuple[np.ndarray, object, np.ndarray, Tuple[float, float, float]]:
    path = os.path.abspath(os.path.expanduser(path))
    with mrcfile.open(path, mode="r", permissive=True) as h:
        if h.data is None:
            raise ValueError(f"MRC file has no data: {path}")
        data = np.asarray(h.data).copy()
        header = h.header.copy()
        try:
            ext = np.asarray(h.extended_header).copy()
        except Exception:
            ext = np.asarray([], dtype=np.uint8)
        voxel = h.voxel_size
        sampling = (float(voxel.x), float(voxel.y), float(voxel.z))
    if data.ndim != 3:
        raise ValueError(f"expected a 3-D MRC volume, got shape {data.shape!r}: {path}")
    return data, header, ext, sampling


def _copy_scalar_header_fields(source: object, destination: object) -> None:
    # Matches the header-preserving logic in ResMap-PyTorch 2.0.0.
    for field in ("nxstart", "nystart", "nzstart", "mapc", "mapr", "maps", "ispg", "nversion"):
        try:
            destination[field] = source[field]
        except Exception:
            pass
    for axis in ("x", "y", "z"):
        try:
            setattr(destination.origin, axis, getattr(source.origin, axis))
        except Exception:
            pass
    for axis in ("alpha", "beta", "gamma"):
        try:
            setattr(destination.cellb, axis, getattr(source.cellb, axis))
        except Exception:
            pass
    try:
        destination.extra[:] = source.extra[:]
    except Exception:
        pass
    try:
        destination.nlabl = source.nlabl
        destination.label[:] = source.label[:]
    except Exception:
        pass


def _write_mrc(
    path: str,
    data: np.ndarray,
    source_header: object,
    source_extended_header: np.ndarray,
    sampling_xyz: Tuple[float, float, float],
) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data, dtype=np.float32, order="C")
    with mrcfile.new(path, overwrite=True) as h:
        h.set_data(arr)
        _copy_scalar_header_fields(source_header, h.header)
        sx, sy, sz = sampling_xyz
        if sx > 0 and sy > 0 and sz > 0:
            h.voxel_size = (sx, sy, sz)
        ext = np.asarray(source_extended_header)
        if ext.size:
            try:
                h.set_extended_header(ext.copy())
            except Exception:
                pass
        h.update_header_from_data()
        if sx > 0 and sy > 0 and sz > 0:
            h.voxel_size = (sx, sy, sz)
        h.update_header_stats()
        h.flush()
    return path


def _resolve_device(text: str) -> torch.device:
    dev = torch.device(text)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {text}")
    return dev


def _frequency_radius(
    box: int,
    sampling_xyz: Tuple[float, float, float],
    device: torch.device,
) -> torch.Tensor:
    """Return Bsoft-equivalent radial spatial frequency grid in 1/A.

    MRC/numpy arrays are Z,Y,X, while sampling is supplied as X,Y,Z.
    ``torch.fft.fftfreq`` has the same even/odd index ordering used explicitly
    by Bsoft's ``h=(size-1)/2; if (i>h) i-=size`` loops.
    """
    sx, sy, sz = sampling_xyz
    fz = torch.fft.fftfreq(box, d=sz, device=device, dtype=torch.float64)
    fy = torch.fft.fftfreq(box, d=sy, device=device, dtype=torch.float64)
    fx = torch.fft.fftfreq(box, d=sx, device=device, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(fz, fy, fx, indexing="ij")
    return torch.sqrt(xx * xx + yy * yy + zz * zz)


def _bsoft_filter_weights(
    freq_radius: torch.Tensor,
    resolutions: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the current Bsoft ``fspace_bandpass`` weight formula.

    Bsoft computes the filter in double precision and stores the multiplied
    complex values back to float.  We calculate weights in float64 and cast
    them to float32 immediately before applying them to complex64 spectra.
    """
    width = BANDPASS_WIDTH
    test_width = 10.0 * width
    a = GOLDEN / abs(width)
    slo = 1.0 / RES_LO

    s = freq_radius
    lo_delta = slo - s
    edge_lo = torch.where(
        lo_delta > test_width,
        torch.zeros_like(s),
        1.0 / (1.0 + torch.exp(a * lo_delta)),
    )

    # Bsoft: shi = 1/res_hi.  Do not silently clamp unusual values: this keeps
    # the behaviour transparent and makes bad resolution maps visible.
    res64 = resolutions.to(dtype=torch.float64)
    shi = 1.0 / res64
    hi_delta = s.unsqueeze(0) - shi[:, None, None, None]
    edge_hi = torch.where(
        hi_delta > test_width,
        torch.zeros_like(hi_delta),
        1.0 / (1.0 + torch.exp(a * hi_delta)),
    )

    return torch.minimum(edge_hi, edge_lo.unsqueeze(0))


def _build_active_mask(
    shape_zyx: Tuple[int, int, int],
    edge_xyz: Tuple[int, int, int],
    mask_values: Optional[np.ndarray],
    mask_level: int,
    resmap: np.ndarray,
    mask_from_resmap: bool,
    invalid_resmap_value: float,
    invalid_resmap_tolerance: float,
) -> np.ndarray:
    nz, ny, nx = shape_zyx
    ex, ey, ez = edge_xyz

    if mask_values is None:
        active = np.ones(shape_zyx, dtype=bool)
    else:
        # Exact Bsoft local_filter semantics: values below 0.99 are rejected.
        # This is stricter than the blocfilt help text's "all but zero" wording.
        active = np.asarray(mask_values >= 0.99, dtype=bool)
        if mask_level > 0:
            active &= np.abs(mask_values - float(mask_level)) <= 0.1

    z = np.arange(nz, dtype=np.int64)[:, None, None]
    y = np.arange(ny, dtype=np.int64)[None, :, None]
    x = np.arange(nx, dtype=np.int64)[None, None, :]

    # Preserve Bsoft's exact comparisons:
    #   if (coord < edge || coord > size-edge) dovox = 0;
    active &= (z >= ez) & (z <= nz - ez)
    active &= (y >= ey) & (y <= ny - ey)
    active &= (x >= ex) & (x <= nx - ex)

    if mask_from_resmap:
        valid = np.isfinite(resmap)
        valid &= np.abs(resmap - float(invalid_resmap_value)) > float(invalid_resmap_tolerance)
        active &= valid

    return active


def local_filter_bsoft_compatible(
    volume_zyx: np.ndarray,
    resmap_zyx: np.ndarray,
    sampling_xyz: Tuple[float, float, float],
    box: int,
    edge_xyz: Optional[Tuple[int, int, int]] = None,
    mask_values: Optional[np.ndarray] = None,
    mask_level: int = -1,
    device: str = "cpu",
    batch_size: int = 256,
    mask_from_resmap: bool = False,
    invalid_resmap_value: float = 100.0,
    invalid_resmap_tolerance: float = 1.0e-6,
    verbose: int = 1,
) -> Tuple[np.ndarray, int]:
    """Apply the Bsoft blocfilt local filter and return ``(output, nvox)``."""
    vol = np.asarray(volume_zyx, dtype=np.float32, order="C")
    resmap = np.asarray(resmap_zyx, dtype=np.float32, order="C")
    if vol.ndim != 3 or resmap.ndim != 3:
        raise ValueError("input map and local-resolution map must both be 3-D")
    if vol.shape != resmap.shape:
        raise ValueError(f"shape mismatch: input {vol.shape!r}, resolution {resmap.shape!r}")
    if mask_values is not None and np.asarray(mask_values).shape != vol.shape:
        raise ValueError(f"mask shape {np.asarray(mask_values).shape!r} != input shape {vol.shape!r}")
    if box < 1:
        raise ValueError("box size must be >= 1")
    if batch_size < 1:
        raise ValueError("batch size must be >= 1")
    if not all(math.isfinite(v) and v > 0.0 for v in sampling_xyz):
        raise ValueError(f"invalid sampling {sampling_xyz!r}; specify -sampling if needed")

    if edge_xyz is None:
        h = box // 2
        edge_xyz = (h, h, h)
    if any(e < 0 for e in edge_xyz):
        raise ValueError("edge values must be non-negative")

    active = _build_active_mask(
        vol.shape,
        edge_xyz,
        mask_values,
        mask_level,
        resmap,
        mask_from_resmap,
        invalid_resmap_value,
        invalid_resmap_tolerance,
    )
    active_flat = np.flatnonzero(active.reshape(-1))
    nvox = int(active_flat.size)
    out = np.zeros(vol.shape, dtype=np.float32)
    if nvox == 0:
        return out, 0

    dev = _resolve_device(device)
    h = box // 2
    tvol = torch.as_tensor(vol, dtype=torch.float32, device=dev)

    # Zero-fill outside the source map, matching the FILL_USER=0 extraction
    # overload reached by Bsoft's integer-valued start coordinate.
    padded = F.pad(tvol, (h, h, h, h, h, h), mode="constant", value=0.0)
    windows = padded.unfold(0, box, 1).unfold(1, box, 1).unfold(2, box, 1)
    freq_radius = _frequency_radius(box, sampling_xyz, dev)

    nz, ny, nx = vol.shape
    plane = ny * nx
    out_flat = out.reshape(-1)
    res_flat = resmap.reshape(-1)

    start_time = time.time()
    for begin in range(0, nvox, batch_size):
        end = min(nvox, begin + batch_size)
        flat = active_flat[begin:end]
        zz = flat // plane
        rem = flat - zz * plane
        yy = rem // nx
        xx = rem - yy * nx

        tz = torch.as_tensor(zz, dtype=torch.long, device=dev)
        ty = torch.as_tensor(yy, dtype=torch.long, device=dev)
        tx = torch.as_tensor(xx, dtype=torch.long, device=dev)

        # Advanced indexing materializes only this batch of patches.
        patches = windows[tz, ty, tx].contiguous()
        spectra = torch.fft.fftn(patches, dim=(-3, -2, -1), norm="ortho")

        local_res = torch.as_tensor(res_flat[flat], dtype=torch.float64, device=dev)
        weights64 = _bsoft_filter_weights(freq_radius, local_res)
        filtered_spec = spectra * weights64.to(dtype=spectra.real.dtype)
        filtered = torch.fft.ifftn(filtered_spec, dim=(-3, -2, -1), norm="ortho").real

        center_values = filtered[:, h, h, h].to(dtype=torch.float32).detach().cpu().numpy()
        out_flat[flat] = center_values

        if verbose:
            pct = 100.0 * end / nvox
            elapsed = time.time() - start_time
            print(f"Complete: {pct:7.3f} %  ({end}/{nvox}, {elapsed:.1f} s)", end="\r", flush=True)

        # Keep peak GPU memory bounded across many batches.
        del patches, spectra, local_res, weights64, filtered_spec, filtered

    if verbose:
        print()
    return out, nvox


def _self_test() -> None:
    """Small consistency test: batched path vs one-patch-at-a-time path."""
    rng = np.random.default_rng(7)
    vol = rng.normal(size=(8, 8, 8)).astype(np.float32)
    res = np.full((8, 8, 8), 5.0, dtype=np.float32)
    res[3:6, 3:6, 3:6] = 3.5
    mask = np.zeros_like(vol, dtype=np.float32)
    mask[2:7, 2:7, 2:7] = 1.0

    got, _ = local_filter_bsoft_compatible(
        vol, res, (1.25, 1.25, 1.25), box=4, edge_xyz=(2, 2, 2),
        mask_values=mask, device="cpu", batch_size=7, verbose=0,
    )
    ref, _ = local_filter_bsoft_compatible(
        vol, res, (1.25, 1.25, 1.25), box=4, edge_xyz=(2, 2, 2),
        mask_values=mask, device="cpu", batch_size=1, verbose=0,
    )
    err = float(np.max(np.abs(got - ref)))
    if err > 5e-6:
        raise RuntimeError(f"self-test failed: batch-vs-scalar max abs error {err}")
    print(f"self-test passed: max abs error = {err:.3g}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Python/PyTorch port of Bsoft blocfilt local-resolution filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", nargs="?", help="input density map (MRC/CCP4)")
    p.add_argument("output", nargs="?", help="output locally filtered map (MRC/CCP4)")

    # Bsoft spellings are retained as aliases where practical.
    p.add_argument("-Resolution", "--resolution", dest="resolution", help="local-resolution MRC map")
    p.add_argument("-Mask", "--mask", dest="mask", help="mask.mrc or Bsoft-style mask.mrc,LEVEL")
    p.add_argument("-box", "--box", dest="box", type=int, help="local filtering kernel size in voxels")
    p.add_argument("-edge", "--edge", dest="edge", help="excluded edge in voxels: N or X,Y,Z")
    p.add_argument("-sampling", "--sampling", dest="sampling", help="A/pixel: S or X,Y,Z; default input MRC header")
    p.add_argument("-verbose", "--verbose", dest="verbose", type=int, default=1, help="0 disables progress output")
    p.add_argument("-symmetry", "--symmetry", dest="symmetry", default="C1",
                   help="Bsoft symmetry option; only C1/no-symmetry is implemented exactly")

    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                   help="PyTorch device, e.g. cpu or cuda:0")
    p.add_argument("--batch-size", type=int, default=256, help="number of local boxes transformed together")
    p.add_argument("--mask-from-resmap", action="store_true",
                   help="exclude ResMap-PyTorch sentinel voxels from filtering")
    p.add_argument("--resmap-invalid-value", type=float, default=100.0,
                   help="sentinel used by --mask-from-resmap")
    p.add_argument("--resmap-invalid-tolerance", type=float, default=1.0e-6,
                   help="absolute tolerance around the sentinel")
    p.add_argument("--self-test", action="store_true", help="run a small internal consistency test and exit")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        _self_test()
        return 0

    if not args.input or not args.output:
        raise SystemExit("input and output MRC files are required")
    if not args.resolution:
        raise SystemExit("a local-resolution map is required: -Resolution FILE")
    if args.box is None or args.box < 1:
        raise SystemExit("a positive kernel size is required: -box N")

    sym = str(args.symmetry).strip().upper()
    if sym not in ("", "C1", "1"):
        raise SystemExit(
            "non-C1 -symmetry is intentionally not emulated: Bsoft constructs/dilates an "
            "asymmetric-unit level mask and replicates it after filtering. Use C1 or omit "
            "-symmetry for the exact local-filtering path implemented here."
        )

    vol, header, ext, header_sampling = _read_mrc(args.input)
    resmap, _, _, _ = _read_mrc(args.resolution)
    if vol.shape != resmap.shape:
        raise SystemExit(f"input/resolution shape mismatch: {vol.shape} vs {resmap.shape}")

    sampling = header_sampling if args.sampling is None else _parse_triplet(args.sampling, float)

    mask_values = None
    mask_level = -1
    if args.mask:
        mask_path, mask_level = _parse_mask_spec(args.mask)
        mask_values, _, _, _ = _read_mrc(mask_path)
        if mask_values.shape != vol.shape:
            raise SystemExit(f"mask shape mismatch: {mask_values.shape} vs {vol.shape}")

    edge = None if args.edge is None else _parse_triplet(args.edge, int)

    if args.verbose:
        print("Filtering by local resolution (Bsoft blocfilt compatibility path):")
        print(f"Input map:                       {os.path.abspath(args.input)}")
        print(f"Local resolution image:          {os.path.abspath(args.resolution)}")
        if args.mask:
            print(f"Mask:                            {os.path.abspath(mask_path)} ({mask_level})")
        print(f"Kernel size:                     {args.box}")
        print(f"Edge size:                       {edge if edge is not None else (args.box//2,)*3}")
        print(f"Sampling X,Y,Z:                  {sampling} A/pixel")
        print(f"Bandpass edge width:             {BANDPASS_WIDTH} 1/A")
        print(f"Low-resolution limit:            {RES_LO:g} A")
        print(f"Device / batch size:             {args.device} / {args.batch_size}")
        if args.mask_from_resmap:
            print(f"ResMap sentinel excluded:        {args.resmap_invalid_value} +/- {args.resmap_invalid_tolerance}")
        print()

    output, nvox = local_filter_bsoft_compatible(
        vol,
        resmap,
        tuple(float(v) for v in sampling),
        box=args.box,
        edge_xyz=edge,
        mask_values=mask_values,
        mask_level=mask_level,
        device=args.device,
        batch_size=args.batch_size,
        mask_from_resmap=args.mask_from_resmap,
        invalid_resmap_value=args.resmap_invalid_value,
        invalid_resmap_tolerance=args.resmap_invalid_tolerance,
        verbose=args.verbose,
    )

    output_path = _write_mrc(args.output, output, header, ext, tuple(float(v) for v in sampling))
    if args.verbose:
        finite = np.isfinite(output)
        print(f"Filtered voxels:                 {nvox}")
        if np.any(finite):
            print(f"Output min/max/mean:             {float(np.nanmin(output)):.6g} / "
                  f"{float(np.nanmax(output)):.6g} / {float(np.nanmean(output)):.6g}")
        print(f"Output written:                  {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
