#!/usr/bin/env python3
"""Display one MRC micrograph with green circles from STAR or 2DTM TSV.

Dependencies: numpy, mrcfile, matplotlib (plus the Python standard library).
All settings are command-line options; run with --help for details.

STAR support: data blocks, loops, # comments, quoted single-line values,
arbitrary column order, and rows split across lines. Semicolon text fields
and save frames are deliberately unsupported. This is not a general CIF parser.
Use --star for particle tables with _rlnMicrographName, or --autopick-star
for a single micrograph's coordinates (all rows belong to the supplied --mrc).
2DTM TSV support: peaks, two-pass candidates, results, summary and final picks.
Use the neighbouring PREFIX_metadata.json to map original and score-map pixels
through the actual search-grid crop. --mrc can be omitted for the recorded
input micrograph. --min-score defaults to positive_local_evidence for fine tables.
Coordinates and --radius refer to the input MRC BEFORE display binning.
"""

import argparse
import csv
import json
import math
import posixpath
import re
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
            if posixpath.basename(name) != basename:
                continue

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
    matches = exact or [name for name in candidates if posixpath.basename(name) == basename]
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


def tsv_table(path, score_column=None, min_score=None):
    """Keep both coordinate frames when supplied. Refined positions take priority."""
    columns, pairs, points, units = None, {}, {}, {}
    total = 0
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
                    for prefix, frame in (("local_original_", "original"), ("original_", "original"),
                                          ("output_", "output"), ("", "output")):
                        if frame not in pairs and all(prefix+a+"_pixel" in columns for a in "xy"):
                            pairs[frame] = prefix
                            points[frame] = []
                            if all(prefix+a+"_angstrom" in columns for a in "xy"):
                                units[frame] = []
                    if not pairs:
                        raise ValueError("No particle X/Y columns. Use a peaks, 2pass_candidates, "
                                         "2pass_results, 2pass_summary or 2pass_picks TSV.")
                    if min_score is not None and score_column is None:
                        score_column = next((c for c in ("positive_local_evidence", "zlocal_scaled_score",
                                                         "candidate_score", "score") if c in columns), None)
                        if score_column is None:
                            raise ValueError("No default score column for --min-score; specify --score-column")
                    if score_column is not None and score_column not in columns:
                        raise ValueError(f"TSV has no score column {score_column!r}")
                    continue
                if len(values) != len(columns):
                    raise ValueError(f"TSV line {line}: expected {len(columns)} columns, got {len(values)}")
                row = dict(zip(columns, values))
                total += 1
                if min_score is not None:
                    try:
                        score = float(row[score_column])
                    except ValueError as exc:
                        raise ValueError(f"TSV line {line}: non-numeric {score_column}") from exc
                    if not math.isfinite(score) or score < min_score:
                        continue
                for frame, prefix in pairs.items():
                    try:
                        xy = tuple(float(row[prefix+a+"_pixel"]) for a in "xy")
                    except ValueError as exc:
                        raise ValueError(f"TSV line {line}: non-numeric {prefix}X/Y") from exc
                    if not all(map(math.isfinite, xy)):
                        raise ValueError(f"TSV line {line}: {prefix}X/Y must be finite")
                    points[frame].append(xy)
                    if frame in units:
                        try:
                            units[frame].append(tuple(float(row[prefix+a+"_angstrom"]) for a in "xy"))
                        except ValueError:
                            units[frame].append((math.nan, math.nan))
        except csv.Error as exc:
            raise ValueError(f"TSV line {reader.line_num}: {exc}") from exc
    if columns is None:
        raise ValueError("TSV is missing its header")
    arrays = {k: np.asarray(v, dtype=np.float64).reshape(-1, 2) for k, v in points.items()}
    scales = {}
    for frame, values in units.items():
        coordinates = arrays[frame]
        angstrom = np.asarray(values, dtype=np.float64).reshape(-1, 2)
        nonzero = coordinates != 0
        if np.isfinite(angstrom).all() and nonzero.any():
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                scale = float(np.median(angstrom[nonzero] / coordinates[nonzero]))
                if (math.isfinite(scale) and scale > 0 and
                        np.allclose(angstrom, coordinates * scale, rtol=1e-5, atol=1e-4)):
                    scales[frame] = scale
        if values and frame not in scales:
            warn(f"Cannot infer a consistent {frame} pixel size from the TSV Angstrom columns.")
    return arrays, scales, total, score_column


def read_tsv_coordinates(path):
    """Return native coordinates and their pixel size when inferable."""
    arrays, scales, _, _ = tsv_table(path)
    frame = "original" if "original" in arrays else "output"
    return arrays[frame], scales.get(frame)


def match_metadata(tsv, explicit):
    path = explicit
    if path is None:
        name = re.fullmatch(r"(.+)_(?:peaks|2pass_(?:candidates|results|summary|picks_ple.+))", tsv.stem)
        if name is None:
            return None, None
        path = tsv.with_name(name[1]+"_metadata.json")
        if not path.is_file():
            return None, None
    with path.open(encoding="utf-8-sig") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict) or not {"sizing", "output_grid_shape", "output_pixel_size_angstrom"} <= meta.keys():
        raise ValueError(f"{path}: expected the full PREFIX_metadata.json from 2DTM")
    return meta, path


def tsv_overlay(arrays, scales, meta, image_shape, target_pixel, base, source_override, mrc_path):
    """Convert TSV centres to pixels of the displayed MRC, before display binning."""
    stem = mrc_path.stem.lower()
    named_mip = stem == "mip" or stem.endswith("_mip")
    if meta is None:
        if named_mip and "output" in arrays:
            frame = "output"
        elif len(arrays) == 2 and target_pixel and "output" in scales and math.isclose(
                target_pixel, scales["output"], rel_tol=1e-5):
            frame = "output"
        else:
            frame = "original" if "original" in arrays else "output"
        points = arrays[frame] - base
        source = source_override or scales.get(frame)
        if source and target_pixel and not math.isclose(source, target_pixel, rel_tol=1e-5):
            warn("No match metadata: assuming the two grids cover the same field from the same outer edge.")
            points = (points + .5) * (source / target_pixel) - .5
        elif source is None and frame == "original" and named_mip and len(points):
            raise ValueError("Refined coordinates are in input micrograph pixels. For this MIP supply "
                             "--metadata PREFIX_metadata.json (or --tsv-angpix for a same-field approximation).")
        return points, target_pixel or source, f"{frame} TSV pixels"

    sz = meta["sizing"]
    required = {"original_shape", "search_shape", "realized_binning_factor", "search_pixel_size_angstrom",
                "resampling_is_needed", "resizing_is_needed", "valid_lower_xy"}
    if not isinstance(sz, dict) or required.difference(sz):
        raise ValueError("Incomplete 2DTM sizing metadata; use the matching PREFIX_metadata.json")
    original = tuple(sz["original_shape"])
    output = tuple(meta["output_grid_shape"])
    search_xy = np.asarray(sz["search_shape"][::-1], dtype=int)
    centre = np.asarray(original[::-1], dtype=int) // 2
    factor = float(sz["realized_binning_factor"])
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Invalid binning factor in match metadata")
    cm = meta.get("two_pass", {}).get("coordinate_mapping", {})
    out_px = float(meta["output_pixel_size_angstrom"])
    in_px = float(cm["input_pixel_size_angstrom"] if "input_pixel_size_angstrom" in cm
                  else sz["search_pixel_size_angstrom"]/factor)
    rescaled = cm.get("pass1_output_rescaled", output == original and math.isclose(out_px, in_px, rel_tol=1e-5))
    if min(in_px, out_px) <= 0 or not all(map(math.isfinite, (in_px, out_px))):
        raise ValueError("Invalid pixel size in match metadata")
    if image_shape == original and image_shape == output:
        target = "output" if named_mip else "original"
    elif image_shape == original:
        target = "original"
    elif image_shape == output:
        target = "output"
    else:
        # E.g. an unbinned acquisition corresponding to the matched working MRC.
        if target_pixel is None or np.any(np.abs(np.asarray(image_shape)*target_pixel -
                                                  np.asarray(original)*in_px) > 2*max(in_px, target_pixel)):
            raise ValueError(f"MRC shape {image_shape} does not match original {original}, score map {output}, "
                             "or a same-field image at the supplied pixel size; check --mrc / --metadata / --angpix.")
        target = "resampled"
    frame = target if target in arrays else ("original" if "original" in arrays else "output")
    points = arrays[frame] - base
    if len(points) and not rescaled and ((frame == "output") != (target == "output")):
        if sz.get("rotated_by_90"):
            # DataSizer undoes its optional internal rotation before writing the map.
            search_xy = search_xy[::-1]
        lower = np.asarray(sz["valid_lower_xy"]) if sz["resampling_is_needed"] or sz["resizing_is_needed"] else 0
        map_centre = search_xy // 2 - lower
        if frame == "output":
            # Same truncation toward zero as cistem2dtm.two_pass._output_to_original.
            points = np.clip(np.trunc((points-map_centre)*factor)+centre, 0, np.asarray(original[::-1])-1)
        else:
            points = (points-centre)/factor + map_centre
    if target == "resampled":
        points = (points+.5) * (in_px/target_pixel) - .5
    expected = out_px if target == "output" else in_px
    return points, target_pixel or expected, f"{frame} TSV pixels -> {target} MRC pixels"


def read_micrograph(path):
    with mrcfile.open(path, mode="r", permissive=True) as mrc:
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
    """Use the existing STAR optics policy; pass empty groups for a TSV target."""
    if override is not None:
        return override, "--angpix"
    if tsv_angpix is not None:
        # Compatibility for callers asking for the TSV's own grid sampling.
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
    coordinate_input.add_argument("--tsv", type=Path, help="2DTM peaks, candidates, results, summary, or thresholded picks table")
    parser.add_argument("--mrc", type=Path, help="MRC to display (original micrograph, MIP or scaled MIP); TSV default: input_mrc in match metadata")
    parser.add_argument("--metadata", type=Path, help="Full PREFIX_metadata.json; normally found alongside the TSV")
    parser.add_argument("--tsv-angpix", type=positive_float, help="Source coordinate A/pixel for a TSV without match metadata")
    parser.add_argument("--min-score", type=finite_float, help="Keep TSV rows with score >= this value (default column for fine tables: positive_local_evidence)")
    parser.add_argument("--positive-local-evidence-threshold", type=finite_float, metavar="Z",
                        help="Shorthand for --score-column positive_local_evidence --min-score Z")
    parser.add_argument("--score-column", type=str.lower, help="Column for --min-score, e.g. penalized_local_score")
    parser.add_argument("--micrograph", help="With --star only: micrograph name if target was renamed; full path resolves duplicate basenames")
    parser.add_argument("--bin", type=positive_int, default=1, help="Integer block-mean binning factor (default: 1)")
    parser.add_argument("--lowpass", type=finite_float, default=0, metavar="ANGSTROM", help="Low-pass resolution; <= 0 disables it (default: 0)")
    parser.add_argument("--angpix", type=positive_float, help="Displayed MRC A/pixel. TSV: default to MRC header, then matching grid metadata. STAR: default to optics, then header.")
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
    mrc_path = args.mrc
    meta, meta_path = match_metadata(args.tsv, args.metadata) if args.tsv is not None else (None, None)
    if meta is not None and args.tsv_angpix is not None:
        warn("Using the matched grid geometry in metadata; --tsv-angpix is for standalone TSVs.")
    if mrc_path is None:
        if meta is None or not meta.get("input_mrc"):
            raise ValueError("Supply --mrc or a TSV with a neighbouring PREFIX_metadata.json naming its input_mrc")
        recorded = Path(normalized_path(meta["input_mrc"])).expanduser()
        mrc_path = next((p for p in (recorded, meta_path.parent / recorded, meta_path.parent / recorded.name)
                         if p.is_file()), None)
        if mrc_path is None:
            raise ValueError(f"Recorded micrograph {recorded} is unavailable here; pass --mrc with its current path")
    image, header_pixel = read_micrograph(mrc_path)
    height, width = image.shape
    if args.bin > min(height, width):
        raise ValueError("--bin is larger than the image")
    if args.tsv is not None:
        angpix, source = choose_angpix(args.angpix, set(), {}, header_pixel)
        coordinate_path = args.tsv
        arrays, scales, total, score_column = tsv_table(coordinate_path, args.score_column, args.min_score)
        coordinates, effective_pixel, mapping = tsv_overlay(
            arrays, scales, meta, (height, width), angpix, args.coord_base, args.tsv_angpix, mrc_path)
        if angpix is None:
            angpix, source = effective_pixel, "coordinate metadata" if effective_pixel else "unavailable"
        name = str(mrc_path)
        print(f"Mapping: {mapping}" + (f" ({meta_path})" if meta_path else ""))
        if args.min_score is not None:
            print(f"Score filter: {score_column} >= {args.min_score:g}; {len(coordinates)} of {total} rows")
    else:
        coordinate_path = args.autopick_star if args.autopick_star is not None else args.star
        coordinates, groups, optics, name = read_coordinates(
            coordinate_path, mrc_path, args.micrograph, autopick=args.autopick_star is not None)
        coordinates -= args.coord_base
        angpix, source = choose_angpix(args.angpix, groups, optics, header_pixel)
    if angpix is None and (args.lowpass > 0 or args.diameter is not None):
        raise ValueError("Pixel size is missing or anisotropic; supply --angpix in A/pixel")
    radius = args.diameter / (2 * angpix) if args.diameter is not None else (
        args.radius if args.radius is not None else 100.0)
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
    ax.set_title(f"{mrc_path.name}\n{len(coordinates)} particles | bin {args.bin} | "
                 f"lowpass {lp_text} | sigma {args.sigma:g}", fontsize=10)
    fig.tight_layout(pad=0.5)
    print(f"Coordinates: {coordinate_path}\nMicrograph: {name}\nImage: {width} x {height} -> {image.shape[1]} x {image.shape[0]} pixels")
    print(f"Particles: {len(coordinates)}; centers inside view: {int(inside.sum())}; radius: {radius:g} original pixels")
    print(f"Pixel size: {angpix if angpix is not None else 'unknown'} A/pixel ({source})")
    print(f"Display limits: [{lo:.6g}, {hi:.6g}]")
    if args.output is not None:
        if args.output.resolve() in (coordinate_path.resolve(), mrc_path.resolve(), meta_path.resolve() if meta_path else None):
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
    if args.positive_local_evidence_threshold is not None:
        if args.min_score is not None or args.score_column is not None:
            parser.error("Use either --positive-local-evidence-threshold or --min-score/--score-column")
        args.min_score = args.positive_local_evidence_threshold
        args.score_column = "positive_local_evidence"
    if args.sigma < 0:
        parser.error("--sigma must be >= 0")
    if args.no_show and args.output is None:
        parser.error("--no-show requires --output")
    if args.star is None and args.micrograph is not None:
        parser.error("--micrograph applies only to --star; --autopick-star and --tsv read all coordinate rows")
    if args.tsv is None and any(v is not None for v in (args.metadata, args.tsv_angpix, args.min_score, args.score_column)):
        parser.error("--metadata, --tsv-angpix, --min-score and --score-column apply to --tsv")
    if args.score_column is not None and args.min_score is None:
        parser.error("--score-column requires --min-score")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
