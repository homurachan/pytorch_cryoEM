#!/usr/bin/env python3
"""Display one MRC micrograph with green circles from STAR or 2DTM TSV.

Dependencies: numpy, mrcfile, matplotlib (plus the Python standard library).
All settings are command-line options; run with --help for details.

STAR support: data blocks, loops, # comments, quoted single-line values,
arbitrary column order, and rows split across lines. Semicolon text fields
and save frames are deliberately unsupported. This is not a general CIF parser.
Use --star for particle tables with _rlnMicrographName, or --autopick-star
for a single micrograph's coordinates (all rows belong to the supplied --mrc).
TSV support: tab-separated header with x_pixel and y_pixel columns.
Optional x_angstrom/y_angstrom columns can supply the pixel size.
Coordinates and --radius refer to the input MRC BEFORE display binning.
"""

import argparse
import csv
import math
import posixpath
import shlex
import sys
from pathlib import Path

import mrcfile
import numpy as np


def warn(message):
    print(f"Warning: {message}", file=sys.stderr)


def star_tokens(path):
    """Stream tokens without treating Windows path backslashes as escapes."""
    with open(path, encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.startswith(";"):
                raise ValueError(f"STAR line {number}: multiline text is unsupported")
            if '\"' in line or "'" in line:
                lexer = shlex.shlex(line, posix=True)
                lexer.whitespace_split = True
                lexer.escape = ""
                try:
                    values = list(lexer)
                except ValueError as exc:
                    raise ValueError(f"STAR line {number}: {exc}") from exc
            else:
                values = line.partition("#")[0].split()
            for value in values:
                yield value, number


def control(value):
    value = value.lower()
    return (value.startswith(("_", "data_", "save_")) or
            value in ("loop_", "stop_", "global_"))


def star_rows(path):
    """Yield (block, row dictionary, starting line), streaming through loops."""
    tokens = iter(star_tokens(path))
    item = next(tokens, None)
    block = ""
    while item is not None:
        value, number = item
        if value.lower().startswith("data_"):
            block = value[5:].lower()
            item = next(tokens, None)
        elif value.lower() == "loop_":
            columns = []
            item = next(tokens, None)
            while item is not None and item[0].startswith("_"):
                columns.append(item[0].lower())
                item = next(tokens, None)
            if not columns or len(columns) != len(set(columns)):
                raise ValueError(f"STAR line {number}: missing or duplicate loop columns")
            while item is not None and not control(item[0]):
                row, start = [], item[1]
                for _ in columns:
                    if item is None or control(item[0]):
                        raise ValueError(f"STAR line {start}: incomplete row; "
                                         f"expected {len(columns)} values, got {len(row)}")
                    row.append(item[0])
                    item = next(tokens, None)
                yield block, dict(zip(columns, row)), start
        elif value.startswith("_"):
            # Scalar metadata outside loops is not needed for this viewer.
            item = next(tokens, None)
            if item is None or control(item[0]):
                raise ValueError(f"STAR line {number}: scalar value is missing")
            item = next(tokens, None)
        elif value.lower() == "stop_":
            item = next(tokens, None)
        else:
            raise ValueError(f"STAR line {number}: unexpected token {value!r}")


def normalized_path(value):
    return posixpath.normpath(str(value).replace("\\", "/"))


def read_coordinates(star, mrc_path, selector=None, *, autopick=False):
    """Read STAR coordinates; autopick assigns all coordinate rows to mrc_path."""
    queries = {normalized_path(selector)} if selector is not None else {
        normalized_path(mrc_path), normalized_path(Path(mrc_path).resolve())}
    basename = posixpath.basename(next(iter(queries)))
    candidates, optics = {}, {}
    required = {"_rlncoordinatex", "_rlncoordinatey"}
    for block, row, line in star_rows(star):
        if block == "optics" and "_rlnmicrographpixelsize" in row:
            group = row.get("_rlnopticsgroup")
            value = row["_rlnmicrographpixelsize"]
            if group and value not in ("?", "."):
                pixel = float(value)
                if not math.isfinite(pixel) or pixel <= 0:
                    raise ValueError(f"STAR line {line}: invalid micrograph pixel size")
                if group in optics and not np.isclose(optics[group], pixel):
                    raise ValueError(f"STAR line {line}: conflicting optics group {group}")
                optics[group] = pixel
        if not required.issubset(row):
            continue
        if autopick:
            name = normalized_path(mrc_path)
        elif "_rlnmicrographname" not in row:
            raise ValueError(f"STAR line {line}: missing _rlnMicrographName; "
                             "for a single-micrograph coordinate file, use "
                             "--autopick-star instead of --star together with --mrc")
        else:
            name = normalized_path(row["_rlnmicrographname"])

        try:
            x, y = float(row["_rlncoordinatex"]), float(row["_rlncoordinatey"])
        except ValueError as exc:
            raise ValueError(f"STAR line {line}: non-numeric X/Y coordinate") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"STAR line {line}: X/Y must be finite")
        points, groups = candidates.setdefault(name, ([], set()))
        points.append((x, y))
        groups.add(row.get("_rlnopticsgroup", "?"))

    exact = [name for name in candidates if name in queries]
    matches = exact or list(candidates)
    if not matches:
        if autopick:
            raise ValueError("Autopick STAR contains no particle coordinates; "
                             "expected _rlnCoordinateX and _rlnCoordinateY columns")
        raise ValueError("No matching particles. Check --mrc / --micrograph and "
                         "_rlnMicrographName, _rlnCoordinateX, _rlnCoordinateY. "
                         "Preserve STAR newlines: # starts an end-of-line comment.")
    if len(matches) > 1:
        raise ValueError("Ambiguous micrograph basename; use --micrograph with "
                         "one full STAR path:\n  " + "\n  ".join(matches))
    name = matches[0]
    if not exact:
        print(f"Matched by unique basename: {name}")
    points, groups = candidates[name]
    return np.asarray(points, dtype=np.float64), groups, optics, name


def read_tsv_coordinates(path):
    """Read all 2DTM pixel coordinates, independent of the target MRC filename.

    Infer A/pixel only if the optional Angstrom coordinates consistently equal
    pixel coordinates times one positive scale. Never rescale X/Y here, and
    never treat pixel_size_offset_angstrom as the image pixel size.
    """
    columns, points, angstrom_points = None, [], []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", strict=True)
        try:
            for values in reader:
                line = reader.line_num
                if not values or not any(v.strip() for v in values):
                    continue
                if values[0].lstrip().startswith("#"):
                    continue
                if columns is None:
                    columns = [v.strip().lower() for v in values]
                    if len(columns) != len(set(columns)) or "" in columns:
                        raise ValueError(f"TSV line {line}: empty or duplicate column names")
                    if not {"x_pixel", "y_pixel"}.issubset(columns):
                        raise ValueError("TSV header must contain tab-separated x_pixel and y_pixel columns")
                    continue
                if len(values) != len(columns):
                    raise ValueError(f"TSV line {line}: expected {len(columns)} columns, got {len(values)}")
                row = dict(zip(columns, values))
                try:
                    x, y = float(row["x_pixel"]), float(row["y_pixel"])
                except ValueError as exc:
                    raise ValueError(f"TSV line {line}: non-numeric x_pixel/y_pixel") from exc
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError(f"TSV line {line}: x_pixel/y_pixel must be finite")
                points.append((x, y))
                if {"x_angstrom", "y_angstrom"}.issubset(row):
                    try:
                        angstrom_points.append((float(row["x_angstrom"]), float(row["y_angstrom"])))
                    except ValueError:
                        angstrom_points.append((math.nan, math.nan))
        except csv.Error as exc:
            raise ValueError(f"TSV line {reader.line_num}: {exc}") from exc
    if not points:
        raise ValueError("TSV contains no particle coordinates")
    coordinates = np.asarray(points, dtype=np.float64)
    angpix = None
    if angstrom_points:
        angstrom = np.asarray(angstrom_points, dtype=np.float64)
        nonzero = coordinates != 0
        if np.isfinite(angstrom).all() and nonzero.any():
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                scale = float(np.median(angstrom[nonzero] / coordinates[nonzero]))
                if (math.isfinite(scale) and scale > 0 and
                        np.allclose(angstrom, coordinates * scale, rtol=1e-5, atol=1e-4)):
                    angpix = scale
        if angpix is None:
            warn("Cannot infer one pixel size from TSV Angstrom coordinates; "
                 "using --angpix or MRC header instead.")
    return coordinates, angpix


def read_micrograph(path):
    with mrcfile.open(path, mode="r") as mrc:
        data = mrc.data
        if data is None:
            raise ValueError("MRC contains no readable image data")
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        if data.ndim != 2 or np.iscomplexobj(data):
            raise ValueError(f"Expected a real 2D micrograph, got shape {data.shape}; "
                             "movie stacks and volumes are unsupported")
        image = np.array(data, dtype=np.float32, copy=True)
        header_pixel = (float(mrc.voxel_size.x), float(mrc.voxel_size.y))
    if not image.size or not np.isfinite(image).all():
        raise ValueError("MRC image is empty or contains NaN/Inf")
    return image, header_pixel


def choose_angpix(override, groups, optics, header_pixel, tsv_angpix=None):
    """Priority: command line > coordinate metadata > isotropic MRC header."""
    if override is not None:
        return override, "--angpix"
    if tsv_angpix is not None:
        if any(math.isfinite(v) and v > 0 and
               not math.isclose(v, tsv_angpix, rel_tol=0.01) for v in header_pixel):
            warn(f"TSV pixel size {tsv_angpix:g} differs from MRC header {header_pixel}; "
                 "using TSV. Use --angpix if the header is correct.")
        return tsv_angpix, "TSV coordinate units"
    values = [optics[g] for g in groups] if groups.issubset(optics) else []
    if values:
        if not np.allclose(values, values[0], rtol=1e-4, atol=0):
            raise ValueError("Matched particles have different pixel sizes; "
                             "specify the target MRC pixel size with --angpix")
        pixel = values[0]
        if any(math.isfinite(v) and v > 0 and
               not math.isclose(v, pixel, rel_tol=0.01) for v in header_pixel):
            warn(f"STAR pixel size {pixel:g} differs from MRC header {header_pixel}; "
                 "using STAR. Use --angpix if the header is correct.")
        return pixel, "STAR optics"
    if any(g not in optics and g not in ("?", ".") for g in groups):
        warn("Selected optics group lacks a micrograph pixel size; trying MRC header.")
    x, y = header_pixel
    if all(math.isfinite(v) and v > 0 for v in (x, y)) and math.isclose(x, y, rel_tol=1e-4):
        return x, "MRC header"
    return None, "unavailable"


def lowpass_filter(image, angpix, resolution):
    """Radial Fourier low-pass; cosine rolloff from 0.9/R to 1.1/R.

    R is in Angstroms. Amplitude is 0.5 at 1/R; this is an independent
    implementation, not a pixel-identical reproduction of RELION's filter.
    FFT implies periodic boundaries; filter BEFORE display binning.
    """
    height, width = image.shape
    cutoff = 1.0 / resolution
    fx = np.fft.rfftfreq(width, d=angpix)
    fy = np.fft.fftfreq(height, d=angpix)
    spectrum = np.fft.rfft2(image)
    # Row-wise masks avoid allocating a second full-sized frequency grid.
    for index, frequency_y in enumerate(fy):
        radial = np.hypot(fx, frequency_y)
        phase = np.clip((radial - 0.9 * cutoff) / (0.2 * cutoff), 0, 1)
        spectrum[index] *= 0.5 * (1 + np.cos(np.pi * phase))
    return np.fft.irfft2(spectrum, s=image.shape).astype(np.float32)


def bin_image(image, factor):
    """Mean each factor x factor block; discard incomplete bottom/right blocks."""
    height, width = image.shape
    kept_h, kept_w = height // factor * factor, width // factor * factor
    if not kept_h or not kept_w:
        raise ValueError("--bin is larger than the image")
    if (kept_h, kept_w) != (height, width):
        warn(f"Binning crops {height - kept_h} bottom rows and {width - kept_w} right columns.")
    if factor == 1:
        return image, (kept_h, kept_w)
    result = image[:kept_h, :kept_w].reshape(
        kept_h // factor, factor, kept_w // factor, factor).mean(axis=(1, 3))
    return result, (kept_h, kept_w)


def contrast_limits(image, sigma):
    if sigma > 0:
        mean, std = float(image.mean(dtype=np.float64)), float(image.std(dtype=np.float64))
        lo, hi = mean - sigma * std, mean + sigma * std
    else:
        lo, hi = float(image.min()), float(image.max())
    if hi <= lo:  # Constant images should still display as mid-gray.
        delta = max(abs(lo) * 1e-6, 1e-6)
        lo, hi = lo - delta, hi + delta
    return lo, hi


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite number > 0")
    return value


def finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    return value


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return value


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    coordinate_input = parser.add_mutually_exclusive_group(required=True)
    coordinate_input.add_argument("--star", type=Path, help="Particle STAR file with _rlnMicrographName")
    coordinate_input.add_argument("--autopick-star", type=Path, help="Single-micrograph STAR with _rlnCoordinateX/Y; reads all coordinate rows for --mrc without requiring _rlnMicrographName")
    coordinate_input.add_argument("--tsv", type=Path, help="2DTM TSV file with x_pixel/y_pixel; reads all rows regardless of MRC filename")
    parser.add_argument("--mrc", required=True, type=Path, help="Target unbinned 2D micrograph")
    parser.add_argument("--micrograph", help="With --star only: micrograph name if target was renamed; full path resolves duplicate basenames")
    parser.add_argument("--bin", type=positive_int, default=1, help="Integer block-mean binning factor (default: 1)")
    parser.add_argument("--lowpass", type=finite_float, default=0, metavar="ANGSTROM", help="Low-pass resolution; <= 0 disables it (default: 0)")
    parser.add_argument("--angpix", type=positive_float, help="Input MRC A/pixel; default: STAR optics or TSV Angstrom/pixel ratio, then MRC header")
    parser.add_argument("--sigma", "--sigma-contrast", "--sigma_contrast", type=finite_float, default=3, help="Mean +/- N std of processed image; 0 uses min/max (default: 3)")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--radius", type=positive_float, metavar="PIXELS", help="Circle radius in ORIGINAL MRC pixels (default: 100)")
    size.add_argument("--diameter", type=positive_float, metavar="ANGSTROM", help="Circle diameter in Angstroms, as in RELION")
    parser.add_argument("--coord-base", type=int, choices=(0, 1), default=0, help="Subtract 0 or 1 from input X/Y (default: 0)")
    parser.add_argument("--flip-y", action="store_true", help="Convert bottom-origin coordinates: y = height-1-y; image stays unchanged")
    parser.add_argument("--invert", action="store_true", help="Invert image grayscale")
    parser.add_argument("--linewidth", type=positive_float, default=1, help="Green circle line width in points (default: 1)")
    parser.add_argument("--figsize", nargs=2, type=positive_float, default=(10, 10), metavar=("WIDTH", "HEIGHT"), help="Figure size in inches (default: 10 10)")
    parser.add_argument("--dpi", type=positive_int, default=150, help="Figure/save resolution (default: 150)")
    parser.add_argument("--output", type=Path, help="Save figure, e.g. picks.png or picks.pdf")
    parser.add_argument("--no-show", action="store_true", help="Save without GUI; requires --output")
    return parser


def run(args):
    tsv_angpix = None
    if args.tsv is not None:
        coordinate_path = args.tsv
        coordinates, tsv_angpix = read_tsv_coordinates(coordinate_path)
        groups, optics, name = set(), {}, normalized_path(args.mrc)
    else:
        coordinate_path = args.autopick_star if args.autopick_star is not None else args.star
        coordinates, groups, optics, name = read_coordinates(
            coordinate_path, args.mrc, args.micrograph, autopick=args.autopick_star is not None)
    image, header_pixel = read_micrograph(args.mrc)
    height, width = image.shape
    if args.bin > min(height, width):
        raise ValueError("--bin is larger than the image")
    angpix, source = choose_angpix(args.angpix, groups, optics, header_pixel, tsv_angpix)
    if angpix is None and (args.lowpass > 0 or args.diameter is not None):
        raise ValueError("Pixel size is missing or anisotropic; supply --angpix in A/pixel")
    radius = args.diameter / (2 * angpix) if args.diameter is not None else (
        args.radius if args.radius is not None else 100.0)
    coordinates -= args.coord_base
    if args.flip_y:
        coordinates[:, 1] = height - 1 - coordinates[:, 1]
    if args.lowpass > 0:
        if args.lowpass < 2 * angpix:
            warn(f"--lowpass is below the input Nyquist resolution ({2 * angpix:g} A).")
        image = lowpass_filter(image, angpix, args.lowpass)
    image, (kept_h, kept_w) = bin_image(image, args.bin)
    lo, hi = contrast_limits(image, args.sigma)
    inside = ((coordinates[:, 0] >= -0.5) & (coordinates[:, 0] < kept_w - 0.5) &
              (coordinates[:, 1] >= -0.5) & (coordinates[:, 1] < kept_h - 0.5))
    if not inside.all():
        warn(f"{int((~inside).sum())} circle centers lie outside the displayed area; "
             "check coordinate scale/base/direction and bin cropping.")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")  # Must be set BEFORE importing pyplot.
    matplotlib.rcParams["toolbar"] = "None"
    from matplotlib import pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=args.figsize, dpi=args.dpi)
    # The axes remain in ORIGINAL pixel units. A binned pixel covers exactly
    # its source block, including its (bin-1)/2 pixel center displacement.
    # Thus neither coordinates nor radii need to be divided by --bin.
    ax.imshow(image, cmap="gray_r" if args.invert else "gray", vmin=lo, vmax=hi,
              origin="upper", extent=(-0.5, kept_w - 0.5, kept_h - 0.5, -0.5),
              interpolation="nearest", aspect="equal")
    circles = [Circle((x, y), radius) for x, y in coordinates]
    ax.add_collection(PatchCollection(circles, facecolors="none", edgecolors="lime",
                                     linewidths=args.linewidth))
    ax.set_xlim(-0.5, kept_w - 0.5)
    ax.set_ylim(kept_h - 0.5, -0.5)
    ax.set_axis_off()
    lp_text = f"{args.lowpass:g} A" if args.lowpass > 0 else "off"
    ax.set_title(f"{args.mrc.name}\n{len(coordinates)} particles | bin {args.bin} | "
                 f"lowpass {lp_text} | sigma {args.sigma:g}", fontsize=10)
    fig.tight_layout(pad=0.5)
    print(f"Coordinates: {coordinate_path}\nMicrograph: {name}\nImage: {width} x {height} -> {image.shape[1]} x {image.shape[0]} pixels")
    print(f"Particles: {len(coordinates)}; centers inside view: {int(inside.sum())}; radius: {radius:g} original pixels")
    print(f"Pixel size: {angpix if angpix is not None else 'unknown'} A/pixel ({source})")
    print(f"Display limits: [{lo:.6g}, {hi:.6g}]")
    if args.output is not None:
        if args.output.resolve() in (coordinate_path.resolve(), args.mrc.resolve()):
            raise ValueError("--output must differ from the input files")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved: {args.output.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.sigma < 0:
        parser.error("--sigma must be >= 0")
    if args.no_show and args.output is None:
        parser.error("--no-show requires --output")
    if args.star is None and args.micrograph is not None:
        parser.error("--micrograph applies only to --star; --autopick-star and --tsv read all coordinate rows")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
