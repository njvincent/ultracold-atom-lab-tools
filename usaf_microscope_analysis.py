#!/usr/bin/env python3
"""
USAF / square target microscope resolution analysis.

Main modes
----------
1. stripe
   Analyze one manually selected USAF 1951 stripe/bar ROI.
   If --orientation is omitted, the code automatically detects whether the ROI
   contains vertical or horizontal bars.

2. auto-stripes
   Automatically detect multiple USAF stripe triplets in an image, then run the
   stripe analysis on each detected triplet.

3. edge
   Analyze one selected square/edge ROI using ESF -> Gaussian LSF -> MTF estimate.

4. square
   Analyze a full square ROI by automatically finding four edges.

Dependencies
------------
pip install numpy scipy matplotlib pandas tifffile

Example usage
-------------
Manual stripe analysis:
    python usaf_microscope_analysis.py stripe \
        --image usaf.tif \
        --dark dark.tif \
        --group 6 \
        --element 2 \
        --pixel-size-um 6.5 \
        --magnification 10 \
        --outdir stripe_G6E2

Automatic stripe detection:
    python usaf_microscope_analysis.py auto-stripes \
        --image usaf.tif \
        --dark dark.tif \
        --polarity dark \
        --pixel-size-um 6.5 \
        --magnification 10 \
        --outdir auto_usaf

Edge analysis:
    python usaf_microscope_analysis.py edge \
        --image square.tif \
        --orientation vertical \
        --pixel-size-um 6.5 \
        --magnification 10 \
        --outdir edge_left

Square analysis:
    python usaf_microscope_analysis.py square \
        --image square.tif \
        --pixel-size-um 6.5 \
        --magnification 10 \
        --outdir square_analysis
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    find_objects,
    gaussian_filter,
    gaussian_filter1d,
    label,
    rotate,
)
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import erf


# ============================================================
# Basic utilities
# ============================================================

@dataclass
class CameraCalibration:
    pixel_size_um: Optional[float] = None
    magnification: Optional[float] = None
    binning: int = 1

    @property
    def object_pixel_um(self) -> Optional[float]:
        if self.pixel_size_um is None or self.magnification is None:
            return None
        if self.magnification == 0:
            raise ValueError("magnification must be nonzero")
        return self.pixel_size_um * self.binning / self.magnification


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_tiff_image(
    path: str,
    dark_path: Optional[str] = None,
    flat_path: Optional[str] = None,
) -> np.ndarray:
    """
    Read TIFF image from Andor Solis export.

    Handles:
    - single-frame grayscale image
    - image stack: averaged over frames
    - RGB/RGBA image: averaged over color channels

    Optional:
    - dark subtraction
    - flat-field correction
    """
    img = tiff.imread(path).astype(np.float64)

    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img = img[..., :3].mean(axis=-1)
    elif img.ndim == 3:
        img = img.mean(axis=0)

    if img.ndim != 2:
        raise ValueError(f"Unsupported TIFF image shape: {img.shape}")

    dark = None
    if dark_path is not None:
        dark = tiff.imread(dark_path).astype(np.float64)

        if dark.ndim == 3 and dark.shape[-1] in (3, 4):
            dark = dark[..., :3].mean(axis=-1)
        elif dark.ndim == 3:
            dark = dark.mean(axis=0)

        if dark.shape != img.shape:
            raise ValueError(
                f"Dark frame shape {dark.shape} does not match image shape {img.shape}"
            )

        img = img - dark

    if flat_path is not None:
        flat = tiff.imread(flat_path).astype(np.float64)

        if flat.ndim == 3 and flat.shape[-1] in (3, 4):
            flat = flat[..., :3].mean(axis=-1)
        elif flat.ndim == 3:
            flat = flat.mean(axis=0)

        if flat.shape != img.shape:
            raise ValueError(
                f"Flat frame shape {flat.shape} does not match image shape {img.shape}"
            )

        if dark is not None:
            flat = flat - dark

        flat_norm = flat / np.nanmean(flat)
        flat_norm[flat_norm <= 0] = np.nan
        img = img / flat_norm
        img = np.nan_to_num(img, nan=np.nanmedian(img))

    return img


def parse_roi(roi_str: str) -> Tuple[int, int, int, int]:
    """Parse ROI string: x,y,w,h."""
    vals = [int(v.strip()) for v in roi_str.split(",")]
    if len(vals) != 4:
        raise ValueError("ROI must be x,y,w,h")
    x0, y0, w, h = vals
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be positive")
    return x0, y0, w, h


def pick_roi_interactive(
    img: np.ndarray,
    title: str = "Select ROI",
) -> Tuple[int, int, int, int]:
    """
    Interactive ROI selection using matplotlib.

    Draw a rectangle, then close the window.
    """
    from matplotlib.widgets import RectangleSelector

    state: Dict[str, Optional[Tuple[int, int, int, int]]] = {"roi": None}

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, cmap="gray", origin="upper")
    ax.set_title(title + "\nDraw rectangle, then close the window.")

    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        if x1 is None or x2 is None or y1 is None or y2 is None:
            return

        x0 = int(round(min(x1, x2)))
        y0 = int(round(min(y1, y2)))
        w = int(round(abs(x2 - x1)))
        h = int(round(abs(y2 - y1)))

        if w <= 0 or h <= 0:
            return

        state["roi"] = (x0, y0, w, h)
        print(f"Selected ROI: x={x0}, y={y0}, w={w}, h={h}")

    _selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )

    plt.show()

    if state["roi"] is None:
        raise RuntimeError("No ROI selected.")

    return state["roi"]


def get_roi(
    img: np.ndarray,
    roi_arg: Optional[str],
    title: str = "Select ROI",
) -> Tuple[int, int, int, int]:
    if roi_arg is not None:
        return parse_roi(roi_arg)
    return pick_roi_interactive(img, title=title)


def crop_roi(
    img: np.ndarray,
    roi: Tuple[int, int, int, int],
    angle_deg: float = 0.0,
) -> np.ndarray:
    x0, y0, w, h = roi
    H, W = img.shape

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x0 + w)
    y1 = min(H, y0 + h)

    crop = img[y0:y1, x0:x1].copy()

    if crop.size == 0:
        raise ValueError(f"Empty ROI after clipping: {roi}")

    if angle_deg != 0:
        crop = rotate(
            crop,
            angle_deg,
            reshape=False,
            order=1,
            mode="nearest",
        )

    return crop


def save_crop_plot(crop: np.ndarray, path: str, title: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(crop, cmap="gray", origin="upper")
    plt.title(title)
    plt.colorbar(label="counts")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ============================================================
# USAF stripe analysis
# ============================================================

def usaf_lpmm(group: int, element: int) -> float:
    """
    USAF 1951 resolving power formula:

        f = 2^(G + (E - 1)/6) lp/mm
    """
    return 2 ** (group + (element - 1) / 6)


def resolution_from_lpmm(lpmm: float) -> Dict[str, float]:
    period_um = 1000.0 / lpmm
    half_pitch_um = 1000.0 / (2.0 * lpmm)

    return {
        "lp_per_mm": lpmm,
        "line_pair_period_um": period_um,
        "half_pitch_um": half_pitch_um,
    }


def extract_bar_profile(crop: np.ndarray, orientation: str) -> np.ndarray:
    """
    orientation:
    - "vertical": vertical bars, intensity varies along x.
    - "horizontal": horizontal bars, intensity varies along y.
    """
    if orientation == "vertical":
        return crop.mean(axis=0)
    if orientation == "horizontal":
        return crop.mean(axis=1)
    raise ValueError("orientation must be 'vertical' or 'horizontal'")


def remove_slow_background(
    profile: np.ndarray,
    poly_order: int = 1,
) -> np.ndarray:
    """
    Remove slow illumination gradient by dividing by a low-order polynomial fit.
    """
    x = np.arange(len(profile))
    y = profile.astype(float)

    if len(y) <= poly_order + 2:
        return y

    coeff = np.polyfit(x, y, deg=poly_order)
    bg = np.polyval(coeff, x)

    if np.any(bg > 0):
        replacement = np.nanmedian(bg[bg > 0])
    else:
        replacement = 1.0

    bg[bg <= 0] = replacement

    corrected = y / bg * np.mean(bg)
    return corrected


def michelson_contrast(
    profile: np.ndarray,
    smooth_sigma: float = 1.0,
    percentile_low: float = 10,
    percentile_high: float = 90,
) -> float:
    """
    Robust Michelson contrast using percentiles rather than raw max/min.

        C = (Imax - Imin) / (Imax + Imin)
    """
    p = gaussian_filter1d(profile.astype(float), smooth_sigma)

    Imax = np.percentile(p, percentile_high)
    Imin = np.percentile(p, percentile_low)

    denom = Imax + Imin
    if denom == 0:
        return np.nan

    return abs((Imax - Imin) / denom)


def estimate_period_fft(profile: np.ndarray) -> float:
    """
    Estimate dominant period in pixels from FFT peak.
    """
    y = profile.astype(float)
    y = y - np.mean(y)

    if len(y) < 4:
        return np.nan

    window = np.hanning(len(y))
    yw = y * window

    fft = np.fft.rfft(yw)
    freqs = np.fft.rfftfreq(len(y), d=1.0)
    power = np.abs(fft) ** 2

    valid = freqs > 2.0 / len(y)

    if not np.any(valid):
        return np.nan

    idx = np.argmax(power[valid])
    freq_peak = freqs[valid][idx]

    if freq_peak <= 0:
        return np.nan

    return 1.0 / freq_peak


def fit_sine_fixed_period(
    profile: np.ndarray,
    period_px: float,
) -> Dict[str, object]:
    """
    Fit profile to:

        I(x) = a0 + a1*(x-xmean)
             + b*sin(2*pi*x/P)
             + c*cos(2*pi*x/P)

    The normalized fundamental modulation is:

        sqrt(b^2 + c^2) / a0
    """
    x = np.arange(len(profile), dtype=float)
    y = profile.astype(float)
    xmean = np.mean(x)

    def model(x_values, a0, a1, b, c):
        return (
            a0
            + a1 * (x_values - xmean)
            + b * np.sin(2 * np.pi * x_values / period_px)
            + c * np.cos(2 * np.pi * x_values / period_px)
        )

    p0 = [np.mean(y), 0.0, np.std(y), np.std(y)]

    popt, _pcov = curve_fit(
        model,
        x,
        y,
        p0=p0,
        maxfev=10000,
    )

    a0, _a1, b, c = popt
    baseline = a0
    amplitude = np.sqrt(b ** 2 + c ** 2)

    modulation = amplitude / baseline if baseline != 0 else np.nan

    yfit = model(x, *popt)

    return {
        "period_px": period_px,
        "sine_baseline": baseline,
        "sine_amplitude": amplitude,
        "fundamental_modulation": abs(modulation),
        "fit_profile": yfit,
        "fit_parameters": popt,
    }


def profile_periodic_score(
    profile: np.ndarray,
    expected_period_px: Optional[float] = None,
) -> Dict[str, float]:
    """
    Score how stripe-like a 1D profile is.

    If expected_period_px is available, use sine fitting at that expected period.
    Otherwise, estimate the dominant period by FFT.
    """
    profile_corr = remove_slow_background(profile, poly_order=1)
    profile_corr = gaussian_filter1d(profile_corr.astype(float), sigma=1.0)

    contrast = michelson_contrast(profile_corr)

    if expected_period_px is not None and np.isfinite(expected_period_px):
        period_px = expected_period_px
    else:
        period_px = estimate_period_fft(profile_corr)

    modulation = np.nan
    score = contrast

    if period_px is not None and np.isfinite(period_px) and 2 < period_px < len(profile_corr):
        try:
            sine_result = fit_sine_fixed_period(profile_corr, period_px)
            modulation = sine_result["fundamental_modulation"]

            if np.isfinite(modulation):
                score = modulation

        except Exception:
            pass

    return {
        "score": score,
        "contrast": contrast,
        "period_px": period_px,
        "fundamental_modulation": modulation,
    }


def auto_detect_stripe_orientation(
    crop: np.ndarray,
    expected_period_px: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """
    Automatically decide whether the selected USAF ROI contains vertical bars
    or horizontal bars.
    """
    diagnostics = {}

    for orientation in ["vertical", "horizontal"]:
        profile = extract_bar_profile(crop, orientation)
        diagnostics[orientation] = profile_periodic_score(
            profile,
            expected_period_px=expected_period_px,
        )

    v_score = diagnostics["vertical"]["score"]
    h_score = diagnostics["horizontal"]["score"]

    core = crop.copy()
    if crop.shape[0] > 10 and crop.shape[1] > 10:
        y0 = int(0.1 * crop.shape[0])
        y1 = int(0.9 * crop.shape[0])
        x0 = int(0.1 * crop.shape[1])
        x1 = int(0.9 * crop.shape[1])
        core = crop[y0:y1, x0:x1]

    gx = np.mean(np.abs(np.diff(core, axis=1)))
    gy = np.mean(np.abs(np.diff(core, axis=0)))

    gradient_orientation = "vertical" if gx > gy else "horizontal"
    orientation = "vertical" if v_score > h_score else "horizontal"

    max_score = max(v_score, h_score)
    if max_score > 0:
        relative_difference = abs(v_score - h_score) / max_score
        if relative_difference < 0.15:
            orientation = gradient_orientation

    diagnostics["gradient"] = {
        "x_gradient": gx,
        "y_gradient": gy,
        "gradient_orientation": gradient_orientation,
    }

    if verbose:
        print("\nAuto orientation detection:")
        print(f"  vertical score           = {v_score:.5g}")
        print(f"  horizontal score         = {h_score:.5g}")
        print(f"  x-gradient score         = {gx:.5g}")
        print(f"  y-gradient score         = {gy:.5g}")
        print(f"  gradient orientation     = {gradient_orientation}")
        print(f"  selected orientation     = {orientation}")

    return orientation, diagnostics


def analyze_stripe_roi(
    img: np.ndarray,
    roi: Tuple[int, int, int, int],
    orientation: Optional[str],
    outdir: str,
    group: Optional[int] = None,
    element: Optional[int] = None,
    lpmm: Optional[float] = None,
    period_px: Optional[float] = None,
    calibration: Optional[CameraCalibration] = None,
    target_contrast: float = 1.0,
    angle_deg: float = 0.0,
) -> pd.DataFrame:
    """
    Analyze one USAF stripe ROI.

    If orientation is None, automatically detects vertical/horizontal bars.
    """
    ensure_dir(outdir)

    crop = crop_roi(img, roi, angle_deg=angle_deg)
    save_crop_plot(crop, os.path.join(outdir, "stripe_roi.png"), "Stripe ROI")

    if calibration is None:
        calibration = CameraCalibration()

    # Determine lp/mm from USAF group/element if needed.
    if lpmm is None and group is not None and element is not None:
        lpmm = usaf_lpmm(group, element)

    # Determine expected period in pixels if possible.
    expected_period_px = period_px

    if expected_period_px is None and lpmm is not None:
        obj_px_um = calibration.object_pixel_um
        if obj_px_um is not None:
            expected_period_px = (1000.0 / lpmm) / obj_px_um

    # Auto-detect orientation if not provided.
    orientation_diag = None

    if orientation is None:
        orientation, orientation_diag = auto_detect_stripe_orientation(
            crop,
            expected_period_px=expected_period_px,
            verbose=True,
        )
    else:
        if orientation not in ["vertical", "horizontal"]:
            raise ValueError("orientation must be 'vertical', 'horizontal', or None")

    raw_profile = extract_bar_profile(crop, orientation)
    corrected_profile = remove_slow_background(raw_profile, poly_order=1)

    contrast = michelson_contrast(corrected_profile)
    ctf = contrast / target_contrast if target_contrast != 0 else np.nan

    if period_px is None:
        if expected_period_px is not None:
            period_px = expected_period_px
        else:
            period_px = estimate_period_fft(corrected_profile)

    sine_result = {}

    if period_px is not None and np.isfinite(period_px) and period_px > 1:
        try:
            sine_result = fit_sine_fixed_period(corrected_profile, period_px)
            fundamental_modulation = sine_result["fundamental_modulation"]

            target_fundamental_modulation = (4.0 / np.pi) * target_contrast

            fundamental_mtf_estimate = (
                fundamental_modulation / target_fundamental_modulation
                if target_fundamental_modulation != 0
                else np.nan
            )

        except Exception as exc:
            print(f"Warning: sine fit failed: {exc}")
            fundamental_modulation = np.nan
            fundamental_mtf_estimate = np.nan

    else:
        fundamental_modulation = np.nan
        fundamental_mtf_estimate = np.nan

    if lpmm is not None:
        res = resolution_from_lpmm(lpmm)
    else:
        res = {
            "lp_per_mm": np.nan,
            "line_pair_period_um": np.nan,
            "half_pitch_um": np.nan,
        }

    x = np.arange(len(corrected_profile))

    plt.figure(figsize=(7, 4))
    plt.plot(x, corrected_profile, label="corrected profile")

    if "fit_profile" in sine_result:
        plt.plot(x, sine_result["fit_profile"], "--", label="fundamental sine fit")

    plt.xlabel("pixel")
    plt.ylabel("intensity counts")
    plt.title(
        f"Stripe profile, orientation={orientation}\n"
        f"CTF={ctf:.3f}, fundamental MTF estimate={fundamental_mtf_estimate:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "stripe_profile_fit.png"), dpi=200)
    plt.close()

    result = {
        "mode": "stripe",
        "roi_x": roi[0],
        "roi_y": roi[1],
        "roi_w": roi[2],
        "roi_h": roi[3],
        "orientation": orientation,
        "angle_deg": angle_deg,
        "group": group,
        "element": element,
        "lp_per_mm": res["lp_per_mm"],
        "line_pair_period_um": res["line_pair_period_um"],
        "half_pitch_um": res["half_pitch_um"],
        "period_px": period_px,
        "target_contrast": target_contrast,
        "michelson_contrast": contrast,
        "ctf": ctf,
        "fundamental_modulation": fundamental_modulation,
        "fundamental_mtf_estimate": fundamental_mtf_estimate,
    }

    if orientation_diag is not None:
        result.update({
            "auto_vertical_score": orientation_diag["vertical"]["score"],
            "auto_horizontal_score": orientation_diag["horizontal"]["score"],
            "auto_vertical_contrast": orientation_diag["vertical"]["contrast"],
            "auto_horizontal_contrast": orientation_diag["horizontal"]["contrast"],
            "auto_x_gradient": orientation_diag["gradient"]["x_gradient"],
            "auto_y_gradient": orientation_diag["gradient"]["y_gradient"],
            "auto_gradient_orientation": orientation_diag["gradient"]["gradient_orientation"],
        })

    df = pd.DataFrame([result])
    df.to_csv(os.path.join(outdir, "stripe_result.csv"), index=False)

    print("\nStripe analysis result:")
    print(df.to_string(index=False))

    return df


# ============================================================
# Automatic USAF stripe extraction
# ============================================================

@dataclass
class BarCandidate:
    component_id: int
    orientation: str
    bbox: Tuple[int, int, int, int]  # x0, y0, w, h
    cx: float
    cy: float
    w: int
    h: int
    area: int
    fill_fraction: float


@dataclass
class StripeTriplet:
    orientation: str
    roi: Tuple[int, int, int, int]
    period_px: float
    bar_thickness_px: float
    bar_length_px: float
    score: float
    centers: List[Tuple[float, float]]
    polarity: str
    core_roi: Tuple[int, int, int, int]
    component_ids: List[int]
    span_to_length: float
    pitch_to_thickness: float
    local_bar_count: int = 3


def otsu_threshold(arr: np.ndarray, nbins: int = 256) -> float:
    """Simple Otsu threshold implementation."""
    data = arr[np.isfinite(arr)].ravel()

    if data.size == 0:
        raise ValueError("No finite pixels available for thresholding.")

    lo, hi = np.percentile(data, [1, 99])
    data = np.clip(data, lo, hi)

    hist, edges = np.histogram(data, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    total = hist.sum()
    if total == 0:
        return float(np.nanmean(data))

    cum_w = np.cumsum(hist)
    cum_m = np.cumsum(hist * centers)

    global_mean = cum_m[-1] / total

    denom = cum_w * (total - cum_w)
    valid = denom > 0

    between_var = np.zeros_like(centers)
    between_var[valid] = (global_mean * cum_w[valid] - cum_m[valid]) ** 2 / denom[valid]

    idx = int(np.argmax(between_var))
    return float(centers[idx])


def normalize_for_segmentation(
    img: np.ndarray,
    bg_sigma: float = 40.0,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """
    Normalize image for threshold-based segmentation.

    This removes slow illumination variation before thresholding.
    """
    x = img.astype(np.float64)
    x_smooth = gaussian_filter(x, smooth_sigma)

    bg = gaussian_filter(x_smooth, bg_sigma)
    bg_med = np.nanmedian(bg)

    if bg_med == 0 or not np.isfinite(bg_med):
        bg_med = 1.0

    bg[bg <= 0] = bg_med
    corrected = x_smooth / bg * bg_med

    lo, hi = np.percentile(corrected, [1, 99])

    if hi <= lo:
        return np.zeros_like(corrected)

    norm = (corrected - lo) / (hi - lo)
    norm = np.clip(norm, 0, 1)
    return norm


def make_feature_mask(
    img: np.ndarray,
    polarity: str = "dark",
    bg_sigma: float = 40.0,
) -> Tuple[np.ndarray, str, float, np.ndarray]:
    """
    Create binary mask for stripe bars.

    polarity:
    - "dark": dark bars on bright background
    - "bright": bright bars on dark background
    - "auto": choose the smaller foreground area between dark and bright
    """
    norm = normalize_for_segmentation(img, bg_sigma=bg_sigma)
    th = otsu_threshold(norm)

    dark_mask = norm < th
    bright_mask = norm > th

    if polarity == "dark":
        mask = dark_mask
        used_polarity = "dark"
    elif polarity == "bright":
        mask = bright_mask
        used_polarity = "bright"
    elif polarity == "auto":
        if dark_mask.mean() <= bright_mask.mean():
            mask = dark_mask
            used_polarity = "dark"
        else:
            mask = bright_mask
            used_polarity = "bright"
    else:
        raise ValueError("polarity must be 'dark', 'bright', or 'auto'")

    structure = np.ones((2, 2), dtype=bool)
    mask = binary_opening(mask, structure=structure)
    mask = binary_closing(mask, structure=structure)

    return mask, used_polarity, th, norm


def find_bar_candidates(
    mask: np.ndarray,
    min_area: int = 20,
    min_aspect: float = 2.0,
    min_fill_fraction: float = 0.25,
) -> List[BarCandidate]:
    """Find elongated connected components that could be individual USAF bars."""
    labeled, _n = label(mask, structure=np.ones((3, 3), dtype=int))
    slices = find_objects(labeled)

    candidates: List[BarCandidate] = []

    for i, sl in enumerate(slices):
        if sl is None:
            continue

        ys, xs = sl
        y0, y1 = ys.start, ys.stop
        x0, x1 = xs.start, xs.stop

        w = x1 - x0
        h = y1 - y0

        if w <= 0 or h <= 0:
            continue

        comp = labeled[ys, xs] == (i + 1)
        area = int(comp.sum())

        if area < min_area:
            continue

        fill_fraction = area / float(w * h)
        if fill_fraction < min_fill_fraction:
            continue

        aspect = max(w / h, h / w)
        if aspect < min_aspect:
            continue

        yy, xx = np.nonzero(comp)
        cx = x0 + float(xx.mean())
        cy = y0 + float(yy.mean())

        orientation = "vertical" if h >= w else "horizontal"

        candidates.append(
            BarCandidate(
                component_id=i + 1,
                orientation=orientation,
                bbox=(x0, y0, w, h),
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                area=area,
                fill_fraction=fill_fraction,
            )
        )

    return candidates


def _clip_roi(
    roi: Tuple[int, int, int, int],
    shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    x0, y0, w, h = roi
    H, W = shape

    x1 = x0 + w
    y1 = y0 + h

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)

    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _roi_iou(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b

    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)

    inter = iw * ih
    union = aw * ah + bw * bh - inter

    if union <= 0:
        return 0.0

    return inter / union


def _score_triplet(
    triple: Tuple[BarCandidate, BarCandidate, BarCandidate],
    orientation: str,
    img_shape: Tuple[int, int],
    polarity: str,
    margin_factor: float = 0.8,
    core_margin_factor: float = 0.25,
    max_span_to_length: float = 1.8,
) -> Optional[StripeTriplet]:
    """Check whether three bars form a plausible USAF stripe triplet."""
    bars = list(triple)

    if orientation == "vertical":
        bars = sorted(bars, key=lambda b: b.cx)
        centers_1d = np.array([b.cx for b in bars])
        perp_centers = np.array([b.cy for b in bars])
        thicknesses = np.array([b.w for b in bars], dtype=float)
        lengths = np.array([b.h for b in bars], dtype=float)
    elif orientation == "horizontal":
        bars = sorted(bars, key=lambda b: b.cy)
        centers_1d = np.array([b.cy for b in bars])
        perp_centers = np.array([b.cx for b in bars])
        thicknesses = np.array([b.h for b in bars], dtype=float)
        lengths = np.array([b.w for b in bars], dtype=float)
    else:
        raise ValueError("orientation must be vertical or horizontal")

    s1 = centers_1d[1] - centers_1d[0]
    s2 = centers_1d[2] - centers_1d[1]

    if s1 <= 0 or s2 <= 0:
        return None

    period_px = 0.5 * (s1 + s2)

    mean_thickness = float(np.mean(thicknesses))
    mean_length = float(np.mean(lengths))

    if mean_thickness <= 0 or mean_length <= 0:
        return None

    spacing_err = abs(s1 - s2) / period_px
    if spacing_err > 0.40:
        return None

    align_err = float(np.std(perp_centers) / max(mean_length, 1.0))
    if align_err > 0.35:
        return None

    thickness_similarity = float(np.std(thicknesses) / mean_thickness)
    length_similarity = float(np.std(lengths) / mean_length)

    if thickness_similarity > 0.70:
        return None
    if length_similarity > 0.70:
        return None

    # For an ideal square-wave element, the center-to-center pitch is close to
    # twice the bar width. Use broad bounds because thresholding/blur changes
    # apparent width.
    pitch_to_thickness = period_px / mean_thickness
    if not (1.1 <= pitch_to_thickness <= 6.0):
        return None

    pitch_err = abs(pitch_to_thickness - 2.0) / 2.0

    # A real USAF three-bar element has only three bars. If the chosen
    # bars span several neighboring elements, the center-to-center span
    # becomes too large compared with the individual bar length. This
    # catches the common false positive where three different elements are
    # incorrectly grouped into one large triplet.
    total_span = centers_1d[2] - centers_1d[0]
    span_to_length = float(total_span / max(mean_length, 1.0))
    if span_to_length > max_span_to_length:
        return None

    x0s = [b.bbox[0] for b in bars]
    y0s = [b.bbox[1] for b in bars]
    x1s = [b.bbox[0] + b.bbox[2] for b in bars]
    y1s = [b.bbox[1] + b.bbox[3] for b in bars]

    x0 = min(x0s)
    y0 = min(y0s)
    x1 = max(x1s)
    y1 = max(y1s)

    margin = int(max(4, margin_factor * period_px))
    core_margin = int(max(2, core_margin_factor * period_px))

    roi = _clip_roi(
        (x0 - margin, y0 - margin, (x1 - x0) + 2 * margin, (y1 - y0) + 2 * margin),
        img_shape,
    )

    # core_roi is deliberately tighter than roi. It is used only for validation:
    # if this tight region contains more than three same-orientation bars, then
    # the candidate probably merged several neighboring USAF elements.
    core_roi = _clip_roi(
        (
            x0 - core_margin,
            y0 - core_margin,
            (x1 - x0) + 2 * core_margin,
            (y1 - y0) + 2 * core_margin,
        ),
        img_shape,
    )

    score = 1.0 / (
        1.0
        + 3.0 * spacing_err
        + 2.0 * align_err
        + thickness_similarity
        + length_similarity
        + 0.5 * pitch_err
        + 0.5 * span_to_length
    )

    centers = [(float(b.cx), float(b.cy)) for b in bars]
    component_ids = [int(b.component_id) for b in bars]

    return StripeTriplet(
        orientation=orientation,
        roi=roi,
        period_px=float(period_px),
        bar_thickness_px=mean_thickness,
        bar_length_px=mean_length,
        score=float(score),
        centers=centers,
        polarity=polarity,
        core_roi=core_roi,
        component_ids=component_ids,
        span_to_length=span_to_length,
        pitch_to_thickness=float(pitch_to_thickness),
    )


def nonmax_suppress_triplets(
    triplets: List[StripeTriplet],
    iou_threshold: float = 0.25,
) -> List[StripeTriplet]:
    """Remove duplicate detections using ROI overlap."""
    triplets = sorted(triplets, key=lambda t: t.score, reverse=True)
    kept: List[StripeTriplet] = []

    for t in triplets:
        duplicate = False
        for k in kept:
            if _roi_iou(t.roi, k.roi) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(t)

    return kept


def _point_in_roi(x: float, y: float, roi: Tuple[int, int, int, int]) -> bool:
    x0, y0, w, h = roi
    return (x0 <= x < x0 + w) and (y0 <= y < y0 + h)


def count_same_orientation_bars_in_roi(
    candidates: List[BarCandidate],
    triplet: StripeTriplet,
    thickness_ratio_bounds: Tuple[float, float] = (0.25, 4.0),
    length_ratio_bounds: Tuple[float, float] = (0.25, 4.0),
) -> int:
    """
    Count same-orientation bar candidates whose centers lie inside the tight
    core ROI of a triplet.

    This is a guard against false positives where several neighboring USAF
    elements are grouped into one large triplet. A valid target element should
    contribute at most three same-orientation bars in this local region.
    """
    count = 0

    for c in candidates:
        if c.orientation != triplet.orientation:
            continue

        if not _point_in_roi(c.cx, c.cy, triplet.core_roi):
            continue

        if triplet.orientation == "vertical":
            thickness = c.w
            length = c.h
        else:
            thickness = c.h
            length = c.w

        thickness_ratio = thickness / max(triplet.bar_thickness_px, 1e-9)
        length_ratio = length / max(triplet.bar_length_px, 1e-9)

        if not (thickness_ratio_bounds[0] <= thickness_ratio <= thickness_ratio_bounds[1]):
            continue
        if not (length_ratio_bounds[0] <= length_ratio <= length_ratio_bounds[1]):
            continue

        count += 1

    return count


def filter_triplets_by_local_bar_count(
    triplets: List[StripeTriplet],
    candidates: List[BarCandidate],
    max_bars_per_roi: int = 3,
    verbose: bool = True,
) -> List[StripeTriplet]:
    """
    Reject triplets whose tight local ROI contains more than max_bars_per_roi
    same-orientation bars.

    For a standard USAF 1951 element, each stripe group contains exactly three
    bars. If the detector accidentally groups three neighboring elements into
    one giant triplet, its core ROI usually contains 6-9 bars and is rejected.
    """
    kept: List[StripeTriplet] = []
    rejected = 0

    for t in triplets:
        local_count = count_same_orientation_bars_in_roi(candidates, t)
        t.local_bar_count = int(local_count)

        if local_count <= max_bars_per_roi:
            kept.append(t)
        else:
            rejected += 1

    if verbose:
        print(
            f"  local bar-count filter: kept {len(kept)}, "
            f"rejected {rejected} with > {max_bars_per_roi} bars per ROI"
        )

    return kept


def detect_usaf_stripe_triplets(
    img: np.ndarray,
    polarity: str = "dark",
    min_area: int = 20,
    min_aspect: float = 2.0,
    max_candidates_per_orientation: int = 180,
    max_triplets: Optional[int] = None,
    bg_sigma: float = 40.0,
    max_bars_per_roi: int = 3,
    max_span_to_length: float = 1.8,
    verbose: bool = True,
) -> Tuple[List[StripeTriplet], Dict[str, object]]:
    """Automatically detect USAF stripe triplets in a full image."""
    mask, used_polarity, threshold, norm = make_feature_mask(
        img,
        polarity=polarity,
        bg_sigma=bg_sigma,
    )

    candidates = find_bar_candidates(
        mask,
        min_area=min_area,
        min_aspect=min_aspect,
    )

    if verbose:
        print("\nAuto stripe detection:")
        print(f"  polarity used: {used_polarity}")
        print(f"  threshold: {threshold:.4g}")
        print(f"  bar candidates: {len(candidates)}")

    all_triplets: List[StripeTriplet] = []

    for orientation in ["vertical", "horizontal"]:
        cands = [c for c in candidates if c.orientation == orientation]
        cands = sorted(cands, key=lambda c: c.area, reverse=True)

        if len(cands) > max_candidates_per_orientation:
            cands = cands[:max_candidates_per_orientation]

        if verbose:
            print(f"  {orientation} candidates used: {len(cands)}")

        for triple in combinations(cands, 3):
            t = _score_triplet(
                triple,
                orientation=orientation,
                img_shape=img.shape,
                polarity=used_polarity,
                max_span_to_length=max_span_to_length,
            )
            if t is not None:
                all_triplets.append(t)

    all_triplets = filter_triplets_by_local_bar_count(
        all_triplets,
        candidates,
        max_bars_per_roi=max_bars_per_roi,
        verbose=verbose,
    )

    triplets = nonmax_suppress_triplets(all_triplets, iou_threshold=0.25)

    if max_triplets is not None:
        triplets = triplets[:max_triplets]

    if verbose:
        print(f"  triplets detected after NMS: {len(triplets)}")

    debug = {
        "mask": mask,
        "norm": norm,
        "threshold": threshold,
        "polarity": used_polarity,
        "candidates": candidates,
    }

    return triplets, debug


def nearest_usaf_group_element(
    lpmm: float,
    group_min: int = -2,
    group_max: int = 9,
) -> Dict[str, float]:
    """Estimate nearest USAF group/element from measured lp/mm."""
    best = None

    for g in range(group_min, group_max + 1):
        for e in range(1, 7):
            f = usaf_lpmm(g, e)
            rel_log_error = abs(np.log(f / lpmm))

            candidate = {
                "nearest_group": g,
                "nearest_element": e,
                "nearest_lpmm": f,
                "relative_error": abs(f - lpmm) / lpmm,
                "log_error": rel_log_error,
            }

            if best is None or rel_log_error < best["log_error"]:
                best = candidate

    return best if best is not None else {
        "nearest_group": np.nan,
        "nearest_element": np.nan,
        "nearest_lpmm": np.nan,
        "relative_error": np.nan,
        "log_error": np.nan,
    }


def plot_detected_triplets(
    img: np.ndarray,
    triplets: List[StripeTriplet],
    path: str,
    labels: Optional[List[str]] = None,
    title: str = "Detected USAF stripe triplets",
) -> None:
    """Save diagnostic plot showing detected stripe ROIs.

    Parameters
    ----------
    labels:
        Optional list of display labels, one per triplet. If omitted, use a
        single unified numbering 0, 1, 2, ...
    """
    from matplotlib.patches import Rectangle

    annotation_color = "#00E5FF"  # single high-contrast cyan

    if labels is None:
        labels = [str(i) for i in range(len(triplets))]
    if len(labels) != len(triplets):
        raise ValueError("labels must have the same length as triplets")

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.imshow(img, cmap="gray", origin="upper")

    for t, label in zip(triplets, labels):
        x0, y0, w, h = t.roi
        lw = float(np.clip(0.8 + 0.015 * min(w, h), 1.2, 2.5))
        rect = Rectangle(
            (x0, y0),
            w,
            h,
            fill=False,
            edgecolor=annotation_color,
            linewidth=lw,
        )
        ax.add_patch(rect)

        fs = float(np.clip(3.5 + 0.04 * min(w, h), 5.0, 8.0))

        if y0 >= 10:
            tx = x0 + 1
            ty = y0 - 2
            va = "bottom"
        else:
            tx = x0 + 1
            ty = y0 + 2
            va = "top"

        ax.text(
            tx,
            ty,
            str(label),
            fontsize=fs,
            color="white",
            fontweight="bold",
            verticalalignment=va,
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor=annotation_color,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.95,
            ),
            clip_on=True,
            zorder=5,
        )

    ax.set_title(title + "\n(see CSV for full details)")
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


def analyze_auto_stripes(
    img: np.ndarray,
    outdir: str,
    calibration: Optional[CameraCalibration] = None,
    polarity: str = "dark",
    target_contrast: float = 1.0,
    min_area: int = 20,
    min_aspect: float = 2.0,
    max_candidates_per_orientation: int = 180,
    max_triplets: Optional[int] = 50,
    bg_sigma: float = 40.0,
    max_bars_per_roi: int = 3,
    max_span_to_length: float = 1.8,
) -> pd.DataFrame:
    """Detect and analyze all visible USAF stripe triplets in the image.

    This unified-numbering version guarantees that the label shown on the
    diagnostic image is exactly the same as the label saved in the CSV output.
    """
    ensure_dir(outdir)

    if calibration is None:
        calibration = CameraCalibration()

    triplets, debug = detect_usaf_stripe_triplets(
        img,
        polarity=polarity,
        min_area=min_area,
        min_aspect=min_aspect,
        max_candidates_per_orientation=max_candidates_per_orientation,
        max_triplets=max_triplets,
        bg_sigma=bg_sigma,
        max_bars_per_roi=max_bars_per_roi,
        max_span_to_length=max_span_to_length,
        verbose=True,
    )

    # Stable visual order: top-to-bottom, then left-to-right, then orientation.
    triplets = sorted(
        triplets,
        key=lambda t: (t.roi[1], t.roi[0], 0 if t.orientation == "vertical" else 1),
    )

    unified_labels = [str(i) for i in range(len(triplets))]

    # Main diagnostic image: labels match CSV exactly.
    plot_detected_triplets(
        img,
        triplets,
        os.path.join(outdir, "auto_detected_stripes.png"),
        labels=unified_labels,
        title="Detected USAF stripe triplets (unified labels)",
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(debug["mask"], cmap="gray", origin="upper")
    plt.title(f"Segmentation mask, polarity={debug['polarity']}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "auto_segmentation_mask.png"), dpi=200)
    plt.close()

    rows = []
    obj_px_um = calibration.object_pixel_um

    for i, t in enumerate(triplets):
        stripe_dir = os.path.join(outdir, f"stripe_{i:03d}_{t.orientation}")
        ensure_dir(stripe_dir)

        measured_lpmm = None
        nearest_group = None
        nearest_element = None
        nearest_lpmm = None
        nearest_relative_error = None

        if obj_px_um is not None and t.period_px > 0:
            period_um = t.period_px * obj_px_um
            measured_lpmm = 1000.0 / period_um

            nearest = nearest_usaf_group_element(measured_lpmm)
            nearest_group = int(nearest["nearest_group"])
            nearest_element = int(nearest["nearest_element"])
            nearest_lpmm = nearest["nearest_lpmm"]
            nearest_relative_error = nearest["relative_error"]

        df_one = analyze_stripe_roi(
            img=img,
            roi=t.roi,
            orientation=t.orientation,
            outdir=stripe_dir,
            group=nearest_group,
            element=nearest_element,
            lpmm=measured_lpmm,
            period_px=t.period_px,
            calibration=calibration,
            target_contrast=target_contrast,
            angle_deg=0.0,
        )

        row = df_one.iloc[0].to_dict()
        row.update({
            "auto_index": i,
            "display_label": str(i),
            "detection_score": t.score,
            "detected_period_px": t.period_px,
            "detected_bar_thickness_px": t.bar_thickness_px,
            "detected_bar_length_px": t.bar_length_px,
            "detected_span_to_length": t.span_to_length,
            "detected_pitch_to_thickness": t.pitch_to_thickness,
            "detected_local_bar_count": t.local_bar_count,
            "detected_core_roi": str(t.core_roi),
            "detected_component_ids": str(t.component_ids),
            "detected_polarity": t.polarity,
            "detected_centers": str(t.centers),
            "measured_lpmm_from_spacing": measured_lpmm,
            "nearest_group": nearest_group,
            "nearest_element": nearest_element,
            "nearest_lpmm": nearest_lpmm,
            "nearest_relative_error": nearest_relative_error,
            "output_dir": stripe_dir,
        })
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(outdir, "auto_stripes_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nAuto stripe summary:")
    if len(summary) > 0:
        cols = [
            "auto_index",
            "display_label",
            "orientation",
            "roi_x",
            "roi_y",
            "roi_w",
            "roi_h",
            "detected_period_px",
            "detected_local_bar_count",
            "ctf",
            "fundamental_mtf_estimate",
            "detection_score",
        ]
        cols = [c for c in cols if c in summary.columns]
        print(summary[cols].to_string(index=False))
    else:
        print("No stripe triplets detected.")

    print(f"\nSaved summary to: {summary_path}")

    return summary


# ============================================================
# Edge / square analysis
# ============================================================

def extract_edge_profile(crop: np.ndarray, orientation: str) -> np.ndarray:
    """
    orientation:
    - "vertical": vertical edge, intensity changes along x.
    - "horizontal": horizontal edge, intensity changes along y.
    """
    if orientation == "vertical":
        return crop.mean(axis=0)
    if orientation == "horizontal":
        return crop.mean(axis=1)
    raise ValueError("orientation must be 'vertical' or 'horizontal'")


def edge_model(x, y0, slope, amp, x0, sigma):
    """
    Error-function edge spread model:

        y = y0 + slope*(x-xmean)
          + amp/2 * [1 + erf((x-x0)/(sqrt(2)*sigma))]
    """
    xmean = np.mean(x)
    return (
        y0
        + slope * (x - xmean)
        + amp * 0.5 * (1 + erf((x - x0) / (np.sqrt(2) * sigma)))
    )


def fit_edge_profile(
    profile: np.ndarray,
    smooth_sigma: float = 1.0,
) -> Dict[str, object]:
    """
    Fit an edge spread function with an error function.

    Under Gaussian blur approximation, fitted sigma is the LSF Gaussian sigma
    in pixels.
    """
    y = gaussian_filter1d(profile.astype(float), smooth_sigma)
    x = np.arange(len(y), dtype=float)

    n_edge = max(3, len(y) // 5)

    y_start = np.percentile(y[:n_edge], 50)
    y_end = np.percentile(y[-n_edge:], 50)

    amp0 = y_end - y_start
    y00 = min(y_start, y_end)
    x00 = len(y) / 2
    sigma0 = max(1.0, len(y) / 20)
    slope0 = 0.0

    p0 = [y00, slope0, amp0, x00, sigma0]

    lower = [-np.inf, -np.inf, -np.inf, 0.0, 0.2]
    upper = [np.inf, np.inf, np.inf, float(len(y)), float(len(y))]

    popt, _pcov = curve_fit(
        edge_model,
        x,
        y,
        p0=p0,
        bounds=(lower, upper),
        maxfev=20000,
    )

    yfit = edge_model(x, *popt)

    y0, slope, amp, x0, sigma = popt
    sigma = abs(sigma)

    edge_width_10_90_px = 2.563103131 * sigma
    lsf_fwhm_px = 2.354820045 * sigma

    dx = np.mean(np.diff(x))
    lsf = np.gradient(yfit, dx)

    return {
        "x": x,
        "profile": y,
        "fit": yfit,
        "lsf": lsf,
        "y0": y0,
        "slope": slope,
        "amp": amp,
        "edge_center_px": x0,
        "sigma_px": sigma,
        "edge_width_10_90_px": edge_width_10_90_px,
        "lsf_fwhm_px": lsf_fwhm_px,
    }


def gaussian_mtf_from_sigma(
    sigma_um: float,
    max_lpmm: Optional[float] = None,
    n: int = 500,
) -> Dict[str, np.ndarray]:
    """
    Gaussian MTF:

        MTF(f) = exp[-2*pi^2*sigma_mm^2*f^2]

    f is in lp/mm, sigma is object-space Gaussian sigma.
    """
    sigma_mm = sigma_um / 1000.0

    if sigma_mm <= 0:
        raise ValueError("sigma_um must be positive")

    if max_lpmm is None:
        mtf10 = np.sqrt(np.log(10)) / (np.sqrt(2) * np.pi * sigma_mm)
        max_lpmm = 1.5 * mtf10

    f = np.linspace(0, max_lpmm, n)
    mtf = np.exp(-2 * np.pi ** 2 * sigma_mm ** 2 * f ** 2)

    return {
        "frequency_lpmm": f,
        "mtf": mtf,
    }


def edge_resolution_metrics_px_to_um(
    edge_result: Dict[str, object],
    calibration: CameraCalibration,
) -> Dict[str, float]:
    """Convert pixel-space edge metrics to object-space units."""
    obj_px_um = calibration.object_pixel_um

    metrics = {
        "object_pixel_um": obj_px_um,
        "sigma_um": np.nan,
        "edge_width_10_90_um": np.nan,
        "lsf_fwhm_um": np.nan,
        "mtf50_lpmm": np.nan,
        "mtf10_lpmm": np.nan,
        "half_pitch_at_mtf50_um": np.nan,
        "half_pitch_at_mtf10_um": np.nan,
    }

    if obj_px_um is None:
        return metrics

    sigma_um = edge_result["sigma_px"] * obj_px_um
    sigma_mm = sigma_um / 1000.0

    mtf50_lpmm = np.sqrt(np.log(2)) / (np.sqrt(2) * np.pi * sigma_mm)
    mtf10_lpmm = np.sqrt(np.log(10)) / (np.sqrt(2) * np.pi * sigma_mm)

    metrics.update({
        "sigma_um": sigma_um,
        "edge_width_10_90_um": edge_result["edge_width_10_90_px"] * obj_px_um,
        "lsf_fwhm_um": edge_result["lsf_fwhm_px"] * obj_px_um,
        "mtf50_lpmm": mtf50_lpmm,
        "mtf10_lpmm": mtf10_lpmm,
        "half_pitch_at_mtf50_um": 1000.0 / (2 * mtf50_lpmm),
        "half_pitch_at_mtf10_um": 1000.0 / (2 * mtf10_lpmm),
    })

    return metrics


def analyze_edge_array(
    crop: np.ndarray,
    orientation: str,
    calibration: Optional[CameraCalibration] = None,
    edge_name: str = "edge",
    outdir: Optional[str] = None,
) -> Dict[str, object]:
    """Analyze an already-cropped edge image."""
    profile = extract_edge_profile(crop, orientation)
    fit = fit_edge_profile(profile)

    if calibration is None:
        calibration = CameraCalibration()

    metrics = edge_resolution_metrics_px_to_um(fit, calibration)

    result = {
        "edge_name": edge_name,
        "orientation": orientation,
        "sigma_px": fit["sigma_px"],
        "edge_width_10_90_px": fit["edge_width_10_90_px"],
        "lsf_fwhm_px": fit["lsf_fwhm_px"],
        **metrics,
    }

    if outdir is not None:
        ensure_dir(outdir)

        save_crop_plot(
            crop,
            os.path.join(outdir, f"{edge_name}_crop.png"),
            f"{edge_name} crop",
        )

        x = fit["x"]

        plt.figure(figsize=(7, 4))
        plt.plot(x, fit["profile"], label="ESF profile")
        plt.plot(x, fit["fit"], "--", label="erf fit")
        plt.xlabel("pixel")
        plt.ylabel("intensity counts")
        plt.title(
            f"{edge_name}: sigma={fit['sigma_px']:.2f} px, "
            f"10-90={fit['edge_width_10_90_px']:.2f} px"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{edge_name}_esf_fit.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(x, fit["lsf"])
        plt.xlabel("pixel")
        plt.ylabel("dI/dx")
        plt.title(f"{edge_name}: LSF")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{edge_name}_lsf.png"), dpi=200)
        plt.close()

        if np.isfinite(metrics["sigma_um"]):
            mtf_curve = gaussian_mtf_from_sigma(metrics["sigma_um"])

            plt.figure(figsize=(7, 4))
            plt.plot(mtf_curve["frequency_lpmm"], mtf_curve["mtf"])
            plt.axhline(0.5, linestyle="--", label="MTF50")
            plt.axhline(0.1, linestyle=":", label="MTF10")
            plt.xlabel("spatial frequency [lp/mm]")
            plt.ylabel("MTF")
            plt.title(
                f"{edge_name}: MTF50={metrics['mtf50_lpmm']:.2f} lp/mm, "
                f"MTF10={metrics['mtf10_lpmm']:.2f} lp/mm"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{edge_name}_mtf.png"), dpi=200)
            plt.close()

    return result


def analyze_edge_roi(
    img: np.ndarray,
    roi: Tuple[int, int, int, int],
    orientation: str,
    outdir: str,
    calibration: Optional[CameraCalibration] = None,
    angle_deg: float = 0.0,
) -> pd.DataFrame:
    ensure_dir(outdir)

    crop = crop_roi(img, roi, angle_deg=angle_deg)

    result = analyze_edge_array(
        crop,
        orientation=orientation,
        calibration=calibration,
        edge_name="edge",
        outdir=outdir,
    )

    result.update({
        "mode": "edge",
        "roi_x": roi[0],
        "roi_y": roi[1],
        "roi_w": roi[2],
        "roi_h": roi[3],
        "angle_deg": angle_deg,
    })

    df = pd.DataFrame([result])
    df.to_csv(os.path.join(outdir, "edge_result.csv"), index=False)

    print("\nEdge analysis result:")
    print(df.to_string(index=False))

    return df


# ============================================================
# Full square analysis
# ============================================================

def find_two_edge_positions_1d(
    profile: np.ndarray,
    min_separation_fraction: float = 0.25,
) -> Tuple[int, int]:
    """Given a 1D projection across a square, find the two strongest edges."""
    p = gaussian_filter1d(profile.astype(float), sigma=2.0)
    g = np.abs(np.gradient(p))

    min_distance = max(3, int(len(profile) * min_separation_fraction))

    peaks, _props = find_peaks(
        g,
        distance=min_distance,
        prominence=np.std(g) * 0.5,
    )

    if len(peaks) >= 2:
        heights = g[peaks]
        best = peaks[np.argsort(heights)[-2:]]
        best = np.sort(best)
        return int(best[0]), int(best[1])

    idx_sorted = np.argsort(g)[::-1]
    chosen = []

    for idx in idx_sorted:
        if all(abs(idx - c) >= min_distance for c in chosen):
            chosen.append(idx)
        if len(chosen) == 2:
            break

    if len(chosen) < 2:
        raise RuntimeError("Could not find two square edges.")

    chosen = np.sort(chosen)
    return int(chosen[0]), int(chosen[1])


def analyze_square_roi(
    img: np.ndarray,
    roi: Tuple[int, int, int, int],
    outdir: str,
    calibration: Optional[CameraCalibration] = None,
    angle_deg: float = 0.0,
    edge_window_px: int = 30,
    central_fraction: float = 0.6,
) -> pd.DataFrame:
    """
    Analyze a full square ROI.

    The ROI should contain the full square plus margin. The code detects four
    edges and fits each one.
    """
    ensure_dir(outdir)

    square = crop_roi(img, roi, angle_deg=angle_deg)
    save_crop_plot(square, os.path.join(outdir, "square_roi.png"), "Square ROI")

    h, w = square.shape

    x_projection = square.mean(axis=0)
    y_projection = square.mean(axis=1)

    left_x, right_x = find_two_edge_positions_1d(x_projection)
    top_y, bottom_y = find_two_edge_positions_1d(y_projection)

    cx0 = int(w * (0.5 - central_fraction / 2))
    cx1 = int(w * (0.5 + central_fraction / 2))
    cy0 = int(h * (0.5 - central_fraction / 2))
    cy1 = int(h * (0.5 + central_fraction / 2))

    edge_results = []

    def edge_crop_vertical(x_center: int, name: str):
        x0 = max(0, x_center - edge_window_px)
        x1 = min(w, x_center + edge_window_px)
        sub = square[cy0:cy1, x0:x1]

        return analyze_edge_array(
            sub,
            orientation="vertical",
            calibration=calibration,
            edge_name=name,
            outdir=outdir,
        )

    def edge_crop_horizontal(y_center: int, name: str):
        y0 = max(0, y_center - edge_window_px)
        y1 = min(h, y_center + edge_window_px)
        sub = square[y0:y1, cx0:cx1]

        return analyze_edge_array(
            sub,
            orientation="horizontal",
            calibration=calibration,
            edge_name=name,
            outdir=outdir,
        )

    edge_results.append(edge_crop_vertical(left_x, "left_edge"))
    edge_results.append(edge_crop_vertical(right_x, "right_edge"))
    edge_results.append(edge_crop_horizontal(top_y, "top_edge"))
    edge_results.append(edge_crop_horizontal(bottom_y, "bottom_edge"))

    df = pd.DataFrame(edge_results)
    df.insert(0, "mode", "square")
    df.insert(1, "angle_deg", angle_deg)

    df.to_csv(os.path.join(outdir, "square_edges_result.csv"), index=False)

    plt.figure(figsize=(6, 5))
    plt.imshow(square, cmap="gray", origin="upper")
    plt.axvline(left_x, linestyle="--", label="left edge")
    plt.axvline(right_x, linestyle="--", label="right edge")
    plt.axhline(top_y, linestyle=":", label="top edge")
    plt.axhline(bottom_y, linestyle=":", label="bottom edge")
    plt.legend()
    plt.title("Detected square edges")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "square_detected_edges.png"), dpi=200)
    plt.close()

    print("\nSquare edge analysis result:")
    print(df.to_string(index=False))

    return df


# ============================================================
# Command line interface
# ============================================================

def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", required=True, help="Input TIFF image.")
    parser.add_argument("--dark", default=None, help="Optional dark frame TIFF.")
    parser.add_argument("--flat", default=None, help="Optional flat-field TIFF.")
    parser.add_argument(
        "--roi",
        default=None,
        help="ROI as x,y,w,h. If omitted, interactive selector is used for manual modes.",
    )
    parser.add_argument("--outdir", default="analysis_output", help="Output directory.")
    parser.add_argument(
        "--angle-deg",
        type=float,
        default=0.0,
        help="Rotate selected ROI or search image by this angle in degrees.",
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="Camera physical pixel size in um.",
    )
    parser.add_argument(
        "--magnification",
        type=float,
        default=None,
        help="Microscope magnification.",
    )
    parser.add_argument("--binning", type=int, default=1, help="Camera binning.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Microscope resolution analysis from TIFF images."
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Stripe mode.
    p_stripe = subparsers.add_parser(
        "stripe",
        help="Analyze one USAF stripe/bar ROI.",
    )
    add_common_args(p_stripe)

    p_stripe.add_argument(
        "--orientation",
        choices=["vertical", "horizontal"],
        default=None,
        help=(
            "Optional. If omitted, orientation is auto-detected. "
            "vertical = vertical bars; horizontal = horizontal bars."
        ),
    )
    p_stripe.add_argument("--group", type=int, default=None, help="USAF group number.")
    p_stripe.add_argument("--element", type=int, default=None, help="USAF element number.")
    p_stripe.add_argument(
        "--lpmm",
        type=float,
        default=None,
        help="Spatial frequency in lp/mm. Overrides group/element if provided.",
    )
    p_stripe.add_argument(
        "--period-px",
        type=float,
        default=None,
        help="Known stripe period in pixels.",
    )
    p_stripe.add_argument(
        "--target-contrast",
        type=float,
        default=1.0,
        help="Intrinsic target Michelson contrast. Use 1.0 for ideal high-contrast USAF.",
    )

    # Auto-stripes mode.
    p_auto = subparsers.add_parser(
        "auto-stripes",
        help="Automatically detect and analyze multiple USAF stripe triplets.",
    )
    add_common_args(p_auto)

    p_auto.add_argument(
        "--polarity",
        choices=["dark", "bright", "auto"],
        default="dark",
        help=(
            "Stripe polarity. Use dark for dark bars on bright background; "
            "bright for bright bars on dark background; auto to choose automatically."
        ),
    )
    p_auto.add_argument(
        "--target-contrast",
        type=float,
        default=1.0,
        help="Intrinsic target Michelson contrast.",
    )
    p_auto.add_argument(
        "--min-area",
        type=int,
        default=20,
        help="Minimum connected-component area for a single bar.",
    )
    p_auto.add_argument(
        "--min-aspect",
        type=float,
        default=2.0,
        help="Minimum aspect ratio for a component to be considered a bar.",
    )
    p_auto.add_argument(
        "--max-candidates-per-orientation",
        type=int,
        default=180,
        help="Max connected-component bar candidates per orientation before triplet grouping.",
    )
    p_auto.add_argument(
        "--max-triplets",
        type=int,
        default=50,
        help="Maximum number of detected stripe triplets to analyze.",
    )
    p_auto.add_argument(
        "--bg-sigma",
        type=float,
        default=40.0,
        help="Gaussian sigma for slow-background normalization.",
    )
    p_auto.add_argument(
        "--max-bars-per-roi",
        type=int,
        default=3,
        help=(
            "Reject auto-detected stripe ROIs containing more than this many "
            "same-orientation bar candidates in the tight local region. "
            "Use 3 for standard USAF elements."
        ),
    )
    p_auto.add_argument(
        "--max-span-to-length",
        type=float,
        default=1.8,
        help=(
            "Reject triplets whose total center span is too large compared "
            "with individual bar length. Lower values are stricter."
        ),
    )

    # Edge mode.
    p_edge = subparsers.add_parser(
        "edge",
        help="Analyze one square edge ROI.",
    )
    add_common_args(p_edge)

    p_edge.add_argument(
        "--orientation",
        choices=["vertical", "horizontal"],
        required=True,
        help=(
            "vertical = vertical edge, intensity changes along x; "
            "horizontal = horizontal edge, intensity changes along y."
        ),
    )

    # Square mode.
    p_square = subparsers.add_parser(
        "square",
        help="Analyze full square ROI and fit four edges.",
    )
    add_common_args(p_square)

    p_square.add_argument(
        "--edge-window-px",
        type=int,
        default=30,
        help="Half-window size around each detected edge.",
    )
    p_square.add_argument(
        "--central-fraction",
        type=float,
        default=0.6,
        help="Central fraction used to avoid square corners.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    img = read_tiff_image(
        args.image,
        dark_path=args.dark,
        flat_path=args.flat,
    )

    calibration = CameraCalibration(
        pixel_size_um=args.pixel_size_um,
        magnification=args.magnification,
        binning=args.binning,
    )

    if calibration.object_pixel_um is not None:
        print(f"Object-space pixel size: {calibration.object_pixel_um:.5g} um/pixel")
    else:
        print("No full calibration provided. Pixel-space results will still be computed.")

    if args.mode == "stripe":
        roi = get_roi(img, args.roi, title="Select USAF stripe ROI")

        analyze_stripe_roi(
            img=img,
            roi=roi,
            orientation=args.orientation,
            outdir=args.outdir,
            group=args.group,
            element=args.element,
            lpmm=args.lpmm,
            period_px=args.period_px,
            calibration=calibration,
            target_contrast=args.target_contrast,
            angle_deg=args.angle_deg,
        )

    elif args.mode == "auto-stripes":
        # Optional --roi restricts the automatic search region.
        if args.roi is not None:
            search_roi = parse_roi(args.roi)
            work_img = crop_roi(img, search_roi, angle_deg=args.angle_deg)
            print(f"Auto-stripes search restricted to ROI: {search_roi}")
        else:
            if args.angle_deg != 0:
                work_img = rotate(
                    img,
                    args.angle_deg,
                    reshape=False,
                    order=1,
                    mode="nearest",
                )
            else:
                work_img = img

        analyze_auto_stripes(
            img=work_img,
            outdir=args.outdir,
            calibration=calibration,
            polarity=args.polarity,
            target_contrast=args.target_contrast,
            min_area=args.min_area,
            min_aspect=args.min_aspect,
            max_candidates_per_orientation=args.max_candidates_per_orientation,
            max_triplets=args.max_triplets,
            bg_sigma=args.bg_sigma,
            max_bars_per_roi=args.max_bars_per_roi,
            max_span_to_length=args.max_span_to_length,
        )

    elif args.mode == "edge":
        roi = get_roi(img, args.roi, title="Select edge ROI")

        analyze_edge_roi(
            img=img,
            roi=roi,
            orientation=args.orientation,
            outdir=args.outdir,
            calibration=calibration,
            angle_deg=args.angle_deg,
        )

    elif args.mode == "square":
        roi = get_roi(img, args.roi, title="Select full square ROI")

        analyze_square_roi(
            img=img,
            roi=roi,
            outdir=args.outdir,
            calibration=calibration,
            angle_deg=args.angle_deg,
            edge_window_px=args.edge_window_px,
            central_fraction=args.central_fraction,
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()