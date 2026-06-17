#!/usr/bin/env python3
"""
USAF / square target microscope resolution analysis.

Main modes
----------
1. stripe
   Analyze one manually selected USAF 1951 stripe/bar ROI.
   If --orientation is omitted, the code automatically detects whether the ROI
   contains vertical or horizontal bars. The selected 1D PSF model is fit
   directly to an ideal finite three-bar object convolved with that PSF.

2. auto-stripes
   Automatically detect multiple USAF stripe triplets in an image, then run the
   stripe analysis on each detected triplet.

3. edge
   Analyze one selected square/edge ROI using ESF -> Gaussian LSF -> MTF estimate.

4. square
   Analyze one USAF square ROI. The square side length is fixed from the
   supplied group/element geometry, and Gaussian sigma is fit separately along
   the horizontal and vertical image axes.

5. auto-squares
   Automatically detect one or more USAF square targets, then run square
   analysis on each detected ROI.

Dependencies
------------
pip install numpy scipy matplotlib pandas tifffile

Stripe PSF models
-----------------
gaussian
    Fit a Gaussian line-spread function. Report sigma and the RMS-matched
    Rayleigh-equivalent resolution:

        resolution = 2.898785 * sigma

airy
    Fit a 1D Airy kernel. Report r0, the Airy first-zero radius and
    Rayleigh-style resolution.

both
    Fit and plot both models for comparison.

When USAF group/element or lp/mm and object-space calibration are available,
the bar width is fixed automatically to the nominal USAF value. Otherwise the
width is fitted, unless --convolution-width-px or --fix-convolution-width is
used. See USAF_MICROSCOPE_ANALYSIS.md for details.

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
        --psf-model gaussian \
        --outdir stripe_G6E2

Automatic stripe detection:
    python usaf_microscope_analysis.py auto-stripes \
        --image usaf.tif \
        --dark dark.tif \
        --polarity dark \
        --pixel-size-um 6.5 \
        --magnification 10 \
        --psf-model airy \
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
        --group 6 \
        --outdir square_analysis

Automatic square detection:
    python usaf_microscope_analysis.py auto-squares \
        --image usaf.tif \
        --group 6 \
        --polarity dark \
        --outdir auto_squares
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
from scipy.optimize import curve_fit, least_squares
from scipy.signal import fftconvolve, find_peaks
from scipy.special import erf, j1


GAUSSIAN_RAYLEIGH_EQUIVALENT_FACTOR = 2.898785
USAF_SQUARE_SIDE_BAR_WIDTHS = 5.0


# ============================================================
# Basic utilities
# ============================================================

@dataclass
class CameraCalibration:
    pixel_size_um: Optional[float] = None
    magnification: Optional[float] = None
    binning: int = 1
    object_pixel_size_um: Optional[float] = None

    @property
    def object_pixel_um(self) -> Optional[float]:
        if self.object_pixel_size_um is not None:
            if self.object_pixel_size_um <= 0:
                raise ValueError("object pixel size must be positive")
            return self.object_pixel_size_um
        if self.pixel_size_um is None:
            return None
        if self.pixel_size_um <= 0:
            raise ValueError("pixel size must be positive")
        if self.magnification is None:
            # If magnification is omitted, treat --pixel-size-um as the
            # calibrated object-space image pixel size.
            return self.pixel_size_um
        if self.magnification == 0:
            raise ValueError("magnification must be nonzero")
        return self.pixel_size_um * self.binning / self.magnification


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def calibration_with_object_pixel_um(
    calibration: Optional[CameraCalibration],
    object_pixel_um: float,
) -> CameraCalibration:
    """Return a calibration that uses an inferred or supplied object pixel size."""
    if calibration is None:
        calibration = CameraCalibration()
    return CameraCalibration(
        pixel_size_um=calibration.pixel_size_um,
        magnification=calibration.magnification,
        binning=calibration.binning,
        object_pixel_size_um=float(object_pixel_um),
    )


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

    fig, ax = plt.subplots(figsize=(11, 8))
    finite_pixels = img[np.isfinite(img)]
    if finite_pixels.size > 0:
        lo, hi = np.percentile(finite_pixels, [1, 99])
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        lo, hi = None, None
    ax.imshow(img, cmap="gray", origin="upper", vmin=lo, vmax=hi)
    ax.set_title(
        title
        + "\nScroll to zoom, right-drag to pan, draw ROI with left mouse. "
        + "Press Enter when done."
    )
    initial_xlim = ax.get_xlim()
    initial_ylim = ax.get_ylim()

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

    selector_kwargs = {
        "useblit": True,
        "button": [1],
        "minspanx": 2,
        "minspany": 2,
        "spancoords": "pixels",
        "interactive": True,
        "props": {
            "facecolor": "none",
            "edgecolor": "tab:red",
            "linewidth": 0.8,
            "alpha": 0.95,
        },
        "handle_props": {
            "markeredgecolor": "tab:red",
            "markerfacecolor": "white",
            "markersize": 4,
            "markeredgewidth": 0.8,
        },
    }

    try:
        _selector = RectangleSelector(ax, onselect, **selector_kwargs)
    except TypeError:
        # Older matplotlib versions used rectprops instead of props and may not
        # accept handle styling.
        selector_kwargs["rectprops"] = selector_kwargs.pop("props")
        selector_kwargs.pop("handle_props", None)
        _selector = RectangleSelector(ax, onselect, **selector_kwargs)

    pan_state: Dict[str, object] = {"press": None}

    def on_scroll(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        scale = 1.0 / 1.5 if event.button == "up" else 1.5
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_width = (xlim[1] - xlim[0]) * scale
        y_height = (ylim[1] - ylim[0]) * scale
        x_rel = (event.xdata - xlim[0]) / (xlim[1] - xlim[0])
        y_rel = (event.ydata - ylim[0]) / (ylim[1] - ylim[0])

        ax.set_xlim(event.xdata - x_width * x_rel, event.xdata + x_width * (1 - x_rel))
        ax.set_ylim(event.ydata - y_height * y_rel, event.ydata + y_height * (1 - y_rel))
        fig.canvas.draw_idle()

    def on_button_press(event):
        if event.inaxes == ax and event.button == 3:
            pan_state["press"] = (
                event.xdata,
                event.ydata,
                ax.get_xlim(),
                ax.get_ylim(),
            )

    def on_motion(event):
        press = pan_state.get("press")
        if press is None or event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        x_press, y_press, xlim, ylim = press
        dx = event.xdata - x_press
        dy = event.ydata - y_press
        ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        fig.canvas.draw_idle()

    def on_button_release(event):
        if event.button == 3:
            pan_state["press"] = None

    def on_key(event):
        if event.key in ("enter", "return") and state["roi"] is not None:
            plt.close(fig)
        elif event.key in ("escape", "backspace"):
            state["roi"] = None
            _selector.set_visible(False)
            fig.canvas.draw_idle()
        elif event.key == "r":
            ax.set_xlim(initial_xlim)
            ax.set_ylim(initial_ylim)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_button_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_button_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

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


def usaf_square_lpmm(group: int) -> float:
    """Return the USAF frequency used to size the single square for a group."""
    return usaf_lpmm(group, 2)


def resolution_from_lpmm(lpmm: float) -> Dict[str, float]:
    period_um = 1000.0 / lpmm
    half_pitch_um = 1000.0 / (2.0 * lpmm)

    return {
        "lp_per_mm": lpmm,
        "line_pair_period_um": period_um,
        "half_pitch_um": half_pitch_um,
    }


def nominal_bar_width_px_from_lpmm(
    lpmm: Optional[float],
    calibration: Optional[CameraCalibration],
) -> float:
    """Return the nominal USAF bar width in image pixels when calibration allows it."""
    if lpmm is None or not np.isfinite(lpmm) or lpmm <= 0:
        return np.nan
    if calibration is None or calibration.object_pixel_um is None:
        return np.nan

    obj_px_um = calibration.object_pixel_um
    if not np.isfinite(obj_px_um) or obj_px_um <= 0:
        return np.nan

    line_pair_period_um = 1000.0 / lpmm
    return line_pair_period_um / (2.0 * obj_px_um)


def nominal_usaf_square_side_um_from_lpmm(lpmm: Optional[float]) -> float:
    """Return the standard square side length for a USAF element in object-space um."""
    if lpmm is None or not np.isfinite(lpmm) or lpmm <= 0:
        return np.nan
    bar_width_um = 1000.0 / (2.0 * lpmm)
    return USAF_SQUARE_SIDE_BAR_WIDTHS * bar_width_um


def nominal_usaf_square_side_px_from_lpmm(
    lpmm: Optional[float],
    calibration: Optional[CameraCalibration],
) -> float:
    """Return the standard square side length in image pixels when calibration allows it."""
    side_um = nominal_usaf_square_side_um_from_lpmm(lpmm)
    if not np.isfinite(side_um):
        return np.nan
    if calibration is None or calibration.object_pixel_um is None:
        return np.nan
    obj_px_um = calibration.object_pixel_um
    if not np.isfinite(obj_px_um) or obj_px_um <= 0:
        return np.nan
    return side_um / obj_px_um


def width_bounds_around_nominal(
    nominal_w_px: float,
    tolerance_fraction: float = 0.20,
) -> Optional[Tuple[float, float]]:
    """Build conservative bounds around a known/nominal bar width."""
    if not np.isfinite(nominal_w_px) or nominal_w_px <= 0:
        return None

    tolerance = max(0.25, tolerance_fraction * nominal_w_px)
    lower = max(0.5, nominal_w_px - tolerance)
    upper = nominal_w_px + tolerance
    if upper <= lower:
        upper = lower + max(0.25, 0.1 * nominal_w_px)
    return float(lower), float(upper)


def extract_bar_profile(
    crop: np.ndarray,
    orientation: str,
    averaging_band: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    orientation:
    - "vertical": vertical bars, intensity varies along x.
    - "horizontal": horizontal bars, intensity varies along y.
    """
    if orientation == "vertical":
        if averaging_band is not None:
            y0, y1 = averaging_band
            crop = crop[y0:y1, :]
        return crop.mean(axis=0)
    if orientation == "horizontal":
        if averaging_band is not None:
            x0, x1 = averaging_band
            crop = crop[:, x0:x1]
        return crop.mean(axis=1)
    raise ValueError("orientation must be 'vertical' or 'horizontal'")


def _longest_true_run(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return [start, end) for the longest contiguous True run."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return None

    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    lengths = ends - starts
    idx = int(np.argmax(lengths))
    return int(starts[idx]), int(ends[idx])


def estimate_stripe_averaging_band(
    crop: np.ndarray,
    orientation: str,
    min_fraction: float = 0.12,
    padding_px: int = 0,
    threshold_fraction: float = 0.55,
) -> Dict[str, object]:
    """
    Find the part of the ROI to average along the stripe direction.

    Rows/columns that are outside the stripe bars tend to have little variation
    across the bars and gaps. Trimming them avoids diluting the profile with
    surrounding dark/background area.
    """
    if orientation == "vertical":
        axis_length = crop.shape[0]
        variation = np.percentile(crop, 90, axis=1) - np.percentile(crop, 10, axis=1)
        band_axis = "y"
    elif orientation == "horizontal":
        axis_length = crop.shape[1]
        variation = np.percentile(crop, 90, axis=0) - np.percentile(crop, 10, axis=0)
        band_axis = "x"
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    if axis_length <= 4 or not np.any(np.isfinite(variation)):
        return {
            "band": (0, axis_length),
            "band_axis": band_axis,
            "band_fraction": 1.0,
            "trim_applied": False,
        }

    variation = np.nan_to_num(variation.astype(float), nan=0.0)
    smooth_sigma = max(0.75, min(3.0, axis_length / 80.0))
    smoothed = gaussian_filter1d(variation, sigma=smooth_sigma)

    lo = float(np.percentile(smoothed, 20))
    hi = float(np.percentile(smoothed, 90))
    if hi <= lo:
        return {
            "band": (0, axis_length),
            "band_axis": band_axis,
            "band_fraction": 1.0,
            "trim_applied": False,
        }

    threshold = lo + threshold_fraction * (hi - lo)
    mask = smoothed >= threshold
    run = _longest_true_run(mask)

    if run is None:
        band = (0, axis_length)
    else:
        start, end = run
        start = max(0, start - padding_px)
        end = min(axis_length, end + padding_px)
        band = (start, end)

    min_len = max(3, int(round(min_fraction * axis_length)))
    if band[1] - band[0] < min_len:
        center = int(np.argmax(smoothed))
        half = max(1, min_len // 2)
        start = max(0, center - half)
        end = min(axis_length, start + min_len)
        start = max(0, end - min_len)
        band = (start, end)

    fraction = (band[1] - band[0]) / float(axis_length)
    trim_applied = band != (0, axis_length)

    return {
        "band": band,
        "band_axis": band_axis,
        "band_fraction": fraction,
        "trim_applied": trim_applied,
    }


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


def blurred_bar_profile(
    x: np.ndarray,
    center: float,
    width: float,
    sigma: float,
) -> np.ndarray:
    """
    Ideal rectangular bar convolved with a Gaussian line-spread function.
    """
    x = np.asarray(x, dtype=float)
    width = float(width)
    sigma = max(float(sigma), 0.1)
    denom = np.sqrt(2.0) * sigma

    return 0.5 * (
        erf((x - center + 0.5 * width) / denom)
        - erf((x - center - 0.5 * width) / denom)
    )


def three_bar_convolution_model(
    x: np.ndarray,
    b0: float,
    b1: float,
    A: float,
    x0: float,
    w: float,
    sigma: float,
) -> np.ndarray:
    """
    Three ideal USAF bars convolved with a Gaussian LSF.

    The fitted sigma is converted to a Rayleigh-equivalent resolution using the
    configured RMS-matching factor.
    """
    x = np.asarray(x, dtype=float)
    signal = np.zeros_like(x, dtype=float)

    for j in (-1, 0, 1):
        signal += blurred_bar_profile(x, x0 + 2.0 * j * w, w, sigma)

    return b0 + b1 * x + A * signal


def parameter_standard_errors(
    jacobian: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    """
    Approximate 1-sigma parameter uncertainties from the least-squares Jacobian.

    The covariance estimate is scaled by the residual variance.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    n_obs, n_params = jacobian.shape
    dof = n_obs - n_params
    if dof <= 0:
        return np.full(n_params, np.nan)

    residual_variance = float(np.sum(residuals ** 2) / dof)
    covariance = np.linalg.pinv(jacobian.T @ jacobian) * residual_variance
    diagonal = np.diag(covariance)
    return np.sqrt(np.maximum(diagonal, 0.0))


def format_value_with_uncertainty(value: float, uncertainty: float) -> str:
    """Format value and 1-sigma uncertainty compactly, for example 0.78(4)."""
    if not np.isfinite(value):
        return "nan"
    if not np.isfinite(uncertainty) or uncertainty <= 0:
        return f"{value:.4g}"

    decimal_places = max(0, -int(np.floor(np.log10(uncertainty))))
    if decimal_places > 6:
        return f"{value:.4g}({uncertainty:.1g})"

    uncertainty_digits = int(round(uncertainty * 10**decimal_places))
    if uncertainty_digits >= 10:
        decimal_places = max(0, decimal_places - 1)
        uncertainty_digits = int(round(uncertainty * 10**decimal_places))

    value_text = f"{value:.{decimal_places}f}"
    return f"{value_text}({uncertainty_digits})"


def format_usaf_label(
    group: Optional[int],
    element: Optional[int] = None,
    square: bool = False,
) -> str:
    """Return a compact USAF label for plot annotations."""
    if group is None:
        return "USAF: unspecified"
    if square:
        return f"USAF G{group} square (E2 scale)"
    if element is None:
        return f"USAF G{group}"
    return f"USAF G{group}E{element}"


def three_bar_ideal_profile(
    x: np.ndarray,
    x0: float,
    w: float,
) -> np.ndarray:
    """Unblurred ideal three-bar object profile used for diagnostics."""
    x = np.asarray(x, dtype=float)
    signal = np.zeros_like(x, dtype=float)

    for j in (-1, 0, 1):
        center = x0 + 2.0 * j * w
        signal += (np.abs(x - center) < 0.5 * w).astype(float)

    return signal


def airy_kernel_1d(
    offsets: np.ndarray,
    r0: float,
) -> np.ndarray:
    """
    Normalized 1D Airy kernel with first zero at |x| = r0.

        A(x; r0) = [2*J1(3.83170597*|x|/r0) / (3.83170597*|x|/r0)]^2
    """
    offsets = np.asarray(offsets, dtype=float)
    if not np.isfinite(r0) or r0 <= 0:
        raise ValueError("r0 must be positive")

    z = 3.83170597 * np.abs(offsets) / float(r0)
    kernel = np.ones_like(z, dtype=float)
    nonzero = z > np.finfo(float).eps
    kernel[nonzero] = (2.0 * j1(z[nonzero]) / z[nonzero]) ** 2

    normalization = float(np.sum(kernel))
    if normalization <= np.finfo(float).eps:
        raise ValueError("Airy kernel normalization is zero")

    return kernel / normalization


def three_bar_airy_convolution_model(
    x: np.ndarray,
    b0: float,
    b1: float,
    A: float,
    x0: float,
    w: float,
    r0: float,
) -> np.ndarray:
    """
    Three ideal USAF bars convolved with a normalized 1D Airy kernel.

    Evaluate the convolution on a finer internal grid, then sample it at the
    measured positions. This keeps x0 and w effectively continuous during the
    fit instead of snapping bar edges to the measured pixel centers.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        raise ValueError("Airy convolution model requires at least two x samples")

    dx = float(np.mean(np.diff(x)))
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("x must be increasing for Airy convolution model")

    oversample = 8
    fine_dx = dx / oversample
    kernel_half_width = max(12.0 * float(r0), 4.0 * dx)
    padding = kernel_half_width + dx
    fine_x = np.arange(
        x[0] - padding,
        x[-1] + padding + 0.5 * fine_dx,
        fine_dx,
        dtype=float,
    )

    # Average each ideal rectangle over the fine-grid cell. The fractional
    # edge cells make the numerical Airy model responsive to subpixel shifts.
    ideal = np.zeros_like(fine_x, dtype=float)
    cell_left = fine_x - 0.5 * fine_dx
    cell_right = fine_x + 0.5 * fine_dx
    for j in (-1, 0, 1):
        center = x0 + 2.0 * j * w
        bar_left = center - 0.5 * w
        bar_right = center + 0.5 * w
        ideal += np.maximum(
            0.0,
            np.minimum(cell_right, bar_right) - np.maximum(cell_left, bar_left),
        ) / fine_dx

    kernel_radius = int(np.ceil(kernel_half_width / fine_dx))
    offsets = np.arange(-kernel_radius, kernel_radius + 1, dtype=float) * fine_dx
    kernel = airy_kernel_1d(offsets, r0)
    blurred = fftconvolve(ideal, kernel, mode="same")

    return b0 + b1 * x + A * np.interp(x, fine_x, blurred)


def _empty_convolution_result(message: str = "disabled") -> Dict[str, object]:
    return {
        "conv_fit_success": False,
        "conv_sigma_px": np.nan,
        "conv_sigma_um": np.nan,
        "conv_sigma_mm": np.nan,
        "conv_sigma_uncertainty_px": np.nan,
        "conv_sigma_uncertainty_um": np.nan,
        "conv_sigma_uncertainty_mm": np.nan,
        "conv_rayleigh_equivalent_resolution_px": np.nan,
        "conv_rayleigh_equivalent_resolution_um": np.nan,
        "conv_rayleigh_equivalent_resolution_mm": np.nan,
        "conv_rayleigh_equivalent_resolution_uncertainty_px": np.nan,
        "conv_rayleigh_equivalent_resolution_uncertainty_um": np.nan,
        "conv_rayleigh_equivalent_resolution_uncertainty_mm": np.nan,
        "conv_object_pixel_um": np.nan,
        "conv_object_pixel_mm": np.nan,
        "conv_frequency_scale_source": "none",
        "conv_w_px": np.nan,
        "conv_w_um": np.nan,
        "conv_w_mm": np.nan,
        "conv_width_fixed": False,
        "conv_stripe_spacing_um": np.nan,
        "conv_width_bound_lower_px": np.nan,
        "conv_width_bound_upper_px": np.nan,
        "conv_width_bound_lower_um": np.nan,
        "conv_width_bound_upper_um": np.nan,
        "conv_width_bound_lower_mm": np.nan,
        "conv_width_bound_upper_mm": np.nan,
        "conv_x0_px": np.nan,
        "conv_x0_um": np.nan,
        "conv_x0_mm": np.nan,
        "conv_amplitude": np.nan,
        "conv_background_offset": np.nan,
        "conv_background_slope": np.nan,
        "conv_background_slope_per_px": np.nan,
        "conv_background_slope_per_um": np.nan,
        "conv_background_slope_per_mm": np.nan,
        "conv_fit_rmse": np.nan,
        "conv_fit_chi_squared": np.nan,
        "conv_fit_r2": np.nan,
        "conv_fit_message": message,
        "conv_fit_profile": None,
        "conv_ideal_profile": None,
    }


def _estimate_three_bar_initial_guess(
    x: np.ndarray,
    y: np.ndarray,
    initial_width: Optional[float] = None,
) -> Dict[str, float]:
    """Estimate robust starting values for the three-bar convolution fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n < 6:
        raise ValueError("profile too short for three-bar convolution fit")

    finite = np.isfinite(y)
    if not np.any(finite):
        raise ValueError("profile contains no finite values")

    y_finite = y[finite]
    y_med = float(np.nanmedian(y_finite))
    y_min = float(np.nanmin(y_finite))
    y_max = float(np.nanmax(y_finite))
    y_ptp = max(y_max - y_min, np.finfo(float).eps)

    if initial_width is not None and np.isfinite(initial_width) and initial_width > 0:
        w0 = float(initial_width)
    else:
        period_guess = estimate_period_fft(y)
        if np.isfinite(period_guess) and period_guess > 1:
            w0 = 0.5 * float(period_guess)
        else:
            w0 = max(1.0, (float(np.nanmax(x)) - float(np.nanmin(x)) + 1.0) / 6.0)

    w0 = float(np.clip(w0, 0.75, max(1.0, n)))

    smooth_sigma = max(0.75, min(2.0, 0.15 * w0))
    y_smooth = gaussian_filter1d(y.astype(float), sigma=smooth_sigma)

    bright_contrast = float(np.nanmax(y_smooth) - np.nanmedian(y_smooth))
    dark_contrast = float(np.nanmedian(y_smooth) - np.nanmin(y_smooth))
    polarity_sign = 1.0 if bright_contrast >= dark_contrast else -1.0
    feature_signal = polarity_sign * (y_smooth - np.nanmedian(y_smooth))

    min_distance = max(1, int(round(1.2 * w0)))
    peaks, props = find_peaks(
        feature_signal,
        distance=min_distance,
        prominence=max(0.05 * y_ptp, np.finfo(float).eps),
    )

    x0 = 0.5 * (float(np.nanmin(x)) + float(np.nanmax(x)))
    if len(peaks) >= 3:
        prominences = props.get("prominences", np.ones(len(peaks)))
        selected = peaks[np.argsort(prominences)[-3:]]
        selected = np.sort(selected)
        centers = x[selected]
        x0 = float(np.median(centers))

        spacings = np.diff(centers)
        if initial_width is None and len(spacings) > 0:
            spacing_guess = float(np.nanmedian(spacings))
            if np.isfinite(spacing_guess) and spacing_guess > 1:
                w0 = 0.5 * spacing_guess

    elif len(peaks) > 0:
        strongest = peaks[int(np.argmax(feature_signal[peaks]))]
        x0 = float(x[strongest])

    slope = 0.0
    if n > 2:
        try:
            slope = float(np.polyfit(x[finite], y[finite], deg=1)[0])
        except Exception:
            slope = 0.0

    amplitude = polarity_sign * y_ptp

    return {
        "b0": y_med,
        "b1": slope,
        "A": amplitude,
        "x0": x0,
        "w": w0,
        "sigma": max(0.5, 0.15 * w0),
    }


def fit_three_bar_convolution(
    profile: np.ndarray,
    x: Optional[np.ndarray] = None,
    initial_width: Optional[float] = None,
    fixed_width: Optional[float] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, object]:
    """
    Fit a finite three-bar object convolved with a Gaussian LSF.

    The amplitude is allowed to be positive or negative, so the same model works
    for bright bars on dark background and dark bars on bright background.
    """
    y = np.asarray(profile, dtype=float)
    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.asarray(x, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and profile must have the same length")
    if len(y) < 6:
        raise ValueError("profile too short for three-bar convolution fit")

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 6:
        raise ValueError("not enough finite profile samples")

    x_fit = x[finite]
    y_fit_data = y[finite]
    guess = _estimate_three_bar_initial_guess(x_fit, y_fit_data, initial_width)

    y_min = float(np.nanmin(y_fit_data))
    y_max = float(np.nanmax(y_fit_data))
    y_ptp = max(y_max - y_min, np.finfo(float).eps)

    x_min = float(np.nanmin(x_fit))
    x_max = float(np.nanmax(x_fit))
    x_span = max(x_max - x_min, 1.0)

    if fixed_width is not None and np.isfinite(fixed_width) and fixed_width > 0:
        fit_w = float(fixed_width)
    else:
        fit_w = None

    w0 = fit_w if fit_w is not None else guess["w"]
    w_lower = max(0.5, 0.5 * w0)
    w_upper = min(max(w_lower * 1.01, 1.5 * w0), max(x_span, w_lower * 1.01))
    sigma_upper = max(1.0, min(2.0 * x_span, 4.0 * max(w0, 1.0)))

    default_bounds = {
        "b0": (y_min - 3.0 * y_ptp, y_max + 3.0 * y_ptp),
        "b1": (-10.0 * y_ptp / x_span, 10.0 * y_ptp / x_span),
        "A": (-5.0 * y_ptp, 5.0 * y_ptp),
        "x0": (x_min, x_max),
        "w": (w_lower, w_upper),
        "sigma": (0.1, sigma_upper),
    }
    if bounds is not None:
        default_bounds.update(bounds)

    if fit_w is None:
        p0 = np.array([
            guess["b0"],
            guess["b1"],
            guess["A"],
            guess["x0"],
            np.clip(guess["w"], *default_bounds["w"]),
            np.clip(guess["sigma"], *default_bounds["sigma"]),
        ])
        lower = np.array([default_bounds[k][0] for k in ("b0", "b1", "A", "x0", "w", "sigma")])
        upper = np.array([default_bounds[k][1] for k in ("b0", "b1", "A", "x0", "w", "sigma")])

        def residuals(params):
            return three_bar_convolution_model(x_fit, *params) - y_fit_data

    else:
        p0 = np.array([
            guess["b0"],
            guess["b1"],
            guess["A"],
            guess["x0"],
            np.clip(guess["sigma"], *default_bounds["sigma"]),
        ])
        lower = np.array([default_bounds[k][0] for k in ("b0", "b1", "A", "x0", "sigma")])
        upper = np.array([default_bounds[k][1] for k in ("b0", "b1", "A", "x0", "sigma")])

        def residuals(params):
            b0, b1, A, x0, sigma = params
            return three_bar_convolution_model(x_fit, b0, b1, A, x0, fit_w, sigma) - y_fit_data

    p0 = np.clip(p0, lower, upper)
    opt = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss="linear",
        max_nfev=20000,
    )

    if fit_w is None:
        b0, b1, A, x0, w, sigma = opt.x
    else:
        b0, b1, A, x0, sigma = opt.x
        w = fit_w

    if not opt.success:
        raise RuntimeError(opt.message)
    if w <= 0 or sigma <= 0 or x0 < x_min or x0 > x_max:
        raise RuntimeError("fit returned unphysical parameters")

    y_model = three_bar_convolution_model(x, b0, b1, A, x0, w, sigma)
    ideal = b0 + b1 * x + A * three_bar_ideal_profile(x, x0, w)

    residual = y - y_model
    residual_finite = residual[np.isfinite(residual)]
    rmse = float(np.sqrt(np.mean(residual_finite ** 2))) if residual_finite.size else np.nan

    y_centered = y_fit_data - np.mean(y_fit_data)
    sst = float(np.sum(y_centered ** 2))
    sse = float(np.sum((y_fit_data - three_bar_convolution_model(x_fit, b0, b1, A, x0, w, sigma)) ** 2))
    r2 = np.nan if sst <= np.finfo(float).eps else float(1.0 - sse / sst)
    parameter_errors = parameter_standard_errors(opt.jac, residuals(opt.x))
    sigma_error = float(parameter_errors[-1])

    return {
        "success": True,
        "message": opt.message,
        "b0": float(b0),
        "b1": float(b1),
        "A": float(A),
        "x0": float(x0),
        "w": float(w),
        "sigma": float(sigma),
        "sigma_error": sigma_error,
        "fit_profile": y_model,
        "ideal_profile": ideal,
        "rmse": rmse,
        "chi_squared": sse,
        "r2": r2,
        "nfev": opt.nfev,
        "fixed_width": fit_w is not None,
    }


def fit_three_bar_airy_convolution(
    profile: np.ndarray,
    x: Optional[np.ndarray] = None,
    initial_width: Optional[float] = None,
    fixed_width: Optional[float] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, object]:
    """Fit the finite three-bar object convolved with a normalized 1D Airy PSF."""
    y = np.asarray(profile, dtype=float)
    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.asarray(x, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and profile must have the same length")
    if len(y) < 6:
        raise ValueError("profile too short for three-bar Airy fit")

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 6:
        raise ValueError("not enough finite profile samples")

    x_fit = x[finite]
    y_fit_data = y[finite]
    guess = _estimate_three_bar_initial_guess(x_fit, y_fit_data, initial_width)

    y_min = float(np.nanmin(y_fit_data))
    y_max = float(np.nanmax(y_fit_data))
    y_ptp = max(y_max - y_min, np.finfo(float).eps)
    x_min = float(np.nanmin(x_fit))
    x_max = float(np.nanmax(x_fit))
    x_span = max(x_max - x_min, 1.0)

    if fixed_width is not None and np.isfinite(fixed_width) and fixed_width > 0:
        fit_w = float(fixed_width)
    else:
        fit_w = None

    w0 = fit_w if fit_w is not None else guess["w"]
    w_lower = max(0.5, 0.5 * w0)
    w_upper = min(max(w_lower * 1.01, 1.5 * w0), max(x_span, w_lower * 1.01))
    r0_upper = max(1.0, min(2.0 * x_span, 6.0 * max(w0, 1.0)))

    default_bounds = {
        "b0": (y_min - 3.0 * y_ptp, y_max + 3.0 * y_ptp),
        "b1": (-10.0 * y_ptp / x_span, 10.0 * y_ptp / x_span),
        "A": (-5.0 * y_ptp, 5.0 * y_ptp),
        "x0": (x_min, x_max),
        "w": (w_lower, w_upper),
        "r0": (0.1, r0_upper),
    }
    if bounds is not None:
        default_bounds.update(bounds)

    r00 = max(0.5, 0.75 * w0)
    if fit_w is None:
        p0 = np.array([
            guess["b0"],
            guess["b1"],
            guess["A"],
            guess["x0"],
            np.clip(guess["w"], *default_bounds["w"]),
            np.clip(r00, *default_bounds["r0"]),
        ])
        keys = ("b0", "b1", "A", "x0", "w", "r0")
        lower = np.array([default_bounds[k][0] for k in keys])
        upper = np.array([default_bounds[k][1] for k in keys])

        def residuals(params):
            return three_bar_airy_convolution_model(x_fit, *params) - y_fit_data

    else:
        p0 = np.array([
            guess["b0"],
            guess["b1"],
            guess["A"],
            guess["x0"],
            np.clip(r00, *default_bounds["r0"]),
        ])
        keys = ("b0", "b1", "A", "x0", "r0")
        lower = np.array([default_bounds[k][0] for k in keys])
        upper = np.array([default_bounds[k][1] for k in keys])

        def residuals(params):
            b0, b1, A, x0, r0 = params
            return (
                three_bar_airy_convolution_model(x_fit, b0, b1, A, x0, fit_w, r0)
                - y_fit_data
            )

    p0 = np.clip(p0, lower, upper)
    r0_guesses = np.clip(
        np.asarray([0.25, 0.5, 0.75, 1.0, 1.5]) * w0,
        *default_bounds["r0"],
    )
    r0_guesses = np.unique(np.append(r0_guesses, p0[-1]))
    opt = None
    best_sse = np.inf
    for r0_guess in r0_guesses:
        trial_p0 = p0.copy()
        trial_p0[-1] = r0_guess
        trial_opt = least_squares(
            residuals,
            trial_p0,
            bounds=(lower, upper),
            loss="linear",
            max_nfev=20000,
        )
        trial_sse = float(np.sum(residuals(trial_opt.x) ** 2))
        if opt is None or trial_sse < best_sse:
            opt = trial_opt
            best_sse = trial_sse

    if fit_w is None:
        b0, b1, A, x0, w, r0 = opt.x
    else:
        b0, b1, A, x0, r0 = opt.x
        w = fit_w

    if not opt.success:
        raise RuntimeError(opt.message)
    if w <= 0 or r0 <= 0 or x0 < x_min or x0 > x_max:
        raise RuntimeError("Airy fit returned unphysical parameters")

    y_model = three_bar_airy_convolution_model(x, b0, b1, A, x0, w, r0)
    ideal = b0 + b1 * x + A * three_bar_ideal_profile(x, x0, w)
    residual = y - y_model
    residual_finite = residual[np.isfinite(residual)]
    rmse = float(np.sqrt(np.mean(residual_finite ** 2))) if residual_finite.size else np.nan

    y_centered = y_fit_data - np.mean(y_fit_data)
    sst = float(np.sum(y_centered ** 2))
    sse = float(
        np.sum(
            (
                y_fit_data
                - three_bar_airy_convolution_model(x_fit, b0, b1, A, x0, w, r0)
            )
            ** 2
        )
    )
    r2 = np.nan if sst <= np.finfo(float).eps else float(1.0 - sse / sst)

    return {
        "success": True,
        "message": opt.message,
        "b0": float(b0),
        "b1": float(b1),
        "A": float(A),
        "x0": float(x0),
        "w": float(w),
        "r0": float(r0),
        "fit_profile": y_model,
        "ideal_profile": ideal,
        "rmse": rmse,
        "chi_squared": sse,
        "r2": r2,
        "nfev": opt.nfev,
        "fixed_width": fit_w is not None,
    }


def _empty_airy_result(message: str = "disabled") -> Dict[str, object]:
    return {
        "airy_fit_success": False,
        "airy_r0_px": np.nan,
        "airy_r0_um": np.nan,
        "airy_r0_mm": np.nan,
        "airy_rayleigh_resolution_um": np.nan,
        "airy_object_pixel_um": np.nan,
        "airy_frequency_scale_source": "none",
        "airy_w_px": np.nan,
        "airy_w_um": np.nan,
        "airy_width_fixed": False,
        "airy_x0_px": np.nan,
        "airy_x0_um": np.nan,
        "airy_amplitude": np.nan,
        "airy_background_offset": np.nan,
        "airy_background_slope_per_px": np.nan,
        "airy_background_slope_per_um": np.nan,
        "airy_fit_rmse": np.nan,
        "airy_fit_chi_squared": np.nan,
        "airy_fit_r2": np.nan,
        "airy_fit_message": message,
        "airy_fit_profile": None,
        "airy_ideal_profile": None,
    }


def fit_airy_psf_profile(
    profile: np.ndarray,
    x: Optional[np.ndarray] = None,
    initial_width: Optional[float] = None,
    fixed_width: Optional[float] = None,
    width_bounds: Optional[Tuple[float, float]] = None,
    pixel_size_um: Optional[float] = None,
    nominal_lpmm: Optional[float] = None,
) -> Dict[str, object]:
    """Fit the 1D Airy PSF model and report its first-zero radius."""
    result = _empty_airy_result("not run")
    fit_bounds = None
    if width_bounds is not None:
        lo, hi = width_bounds
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo > 0:
            fit_bounds = {"w": (float(lo), float(hi))}

    try:
        fit = fit_three_bar_airy_convolution(
            profile,
            x=x,
            initial_width=initial_width,
            fixed_width=fixed_width,
            bounds=fit_bounds,
        )

        w_px = fit["w"]
        f0_cyc_per_px = 1.0 / (2.0 * w_px) if w_px > 0 else np.nan
        object_pixel_um = np.nan
        frequency_scale_source = "pixel"
        if pixel_size_um is not None and np.isfinite(pixel_size_um) and pixel_size_um > 0:
            object_pixel_um = float(pixel_size_um)
            frequency_scale_source = "calibration"
        elif nominal_lpmm is not None and np.isfinite(nominal_lpmm) and nominal_lpmm > 0:
            object_pixel_um = f0_cyc_per_px * 1000.0 / nominal_lpmm
            frequency_scale_source = "group_element"

        r0_um = np.nan
        r0_mm = np.nan
        w_um = np.nan
        x0_um = np.nan
        slope_per_um = np.nan
        if np.isfinite(object_pixel_um) and object_pixel_um > 0:
            r0_um = fit["r0"] * object_pixel_um
            r0_mm = r0_um / 1000.0
            w_um = w_px * object_pixel_um
            x0_um = fit["x0"] * object_pixel_um
            slope_per_um = fit["b1"] / object_pixel_um

        result.update({
            "airy_fit_success": True,
            "airy_r0_px": fit["r0"],
            "airy_r0_um": r0_um,
            "airy_r0_mm": r0_mm,
            "airy_rayleigh_resolution_um": r0_um,
            "airy_object_pixel_um": object_pixel_um,
            "airy_frequency_scale_source": frequency_scale_source,
            "airy_w_px": w_px,
            "airy_w_um": w_um,
            "airy_width_fixed": fit["fixed_width"],
            "airy_x0_px": fit["x0"],
            "airy_x0_um": x0_um,
            "airy_amplitude": fit["A"],
            "airy_background_offset": fit["b0"],
            "airy_background_slope_per_px": fit["b1"],
            "airy_background_slope_per_um": slope_per_um,
            "airy_fit_rmse": fit["rmse"],
            "airy_fit_chi_squared": fit["chi_squared"],
            "airy_fit_r2": fit["r2"],
            "airy_fit_message": fit["message"],
            "airy_fit_profile": fit["fit_profile"],
            "airy_ideal_profile": fit["ideal_profile"],
        })

    except Exception as exc:
        result["airy_fit_message"] = str(exc)

    return result


def fit_gaussian_psf_profile(
    profile: np.ndarray,
    x: Optional[np.ndarray] = None,
    initial_width: Optional[float] = None,
    fixed_width: Optional[float] = None,
    width_bounds: Optional[Tuple[float, float]] = None,
    pixel_size_um: Optional[float] = None,
    nominal_lpmm: Optional[float] = None,
) -> Dict[str, object]:
    """
    Fit the ideal three-bar object convolved with a Gaussian LSF.

    Report the fitted Gaussian sigma and the RMS-matched Rayleigh-equivalent
    resolution. This is a real-space PSF-model fit, not a sampled MTF estimate.
    """
    result = _empty_convolution_result("not run")

    fit_bounds = None
    if width_bounds is not None:
        lo, hi = width_bounds
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo > 0:
            fit_bounds = {"w": (float(lo), float(hi))}
            result["conv_width_bound_lower_px"] = float(lo)
            result["conv_width_bound_upper_px"] = float(hi)

    try:
        fit = fit_three_bar_convolution(
            profile,
            x=x,
            initial_width=initial_width,
            fixed_width=fixed_width,
            bounds=fit_bounds,
        )

        sigma_px = fit["sigma"]
        sigma_uncertainty_px = fit["sigma_error"]
        w_px = fit["w"]
        freq_cyc_per_px = 1.0 / (2.0 * w_px) if w_px > 0 else np.nan

        object_pixel_um = np.nan
        frequency_scale_source = "pixel"
        if pixel_size_um is not None and np.isfinite(pixel_size_um) and pixel_size_um > 0:
            object_pixel_um = float(pixel_size_um)
            frequency_scale_source = "calibration"
        elif (
            nominal_lpmm is not None
            and np.isfinite(nominal_lpmm)
            and nominal_lpmm > 0
            and np.isfinite(freq_cyc_per_px)
            and freq_cyc_per_px > 0
        ):
            object_pixel_um = freq_cyc_per_px * 1000.0 / nominal_lpmm
            frequency_scale_source = "group_element"

        sigma_um = np.nan
        sigma_mm = np.nan
        sigma_uncertainty_um = np.nan
        sigma_uncertainty_mm = np.nan
        # RMS-matched Airy/Rayleigh-equivalent resolution for a Gaussian PSF.
        rayleigh_equivalent_resolution_px = (
            GAUSSIAN_RAYLEIGH_EQUIVALENT_FACTOR * sigma_px
        )
        rayleigh_equivalent_resolution_um = np.nan
        rayleigh_equivalent_resolution_mm = np.nan
        rayleigh_equivalent_resolution_uncertainty_px = (
            GAUSSIAN_RAYLEIGH_EQUIVALENT_FACTOR * sigma_uncertainty_px
        )
        rayleigh_equivalent_resolution_uncertainty_um = np.nan
        rayleigh_equivalent_resolution_uncertainty_mm = np.nan
        object_pixel_mm = np.nan
        w_um = np.nan
        w_mm = np.nan
        stripe_spacing_um = np.nan
        width_bound_lower_um = np.nan
        width_bound_upper_um = np.nan
        width_bound_lower_mm = np.nan
        width_bound_upper_mm = np.nan
        x0_um = np.nan
        x0_mm = np.nan
        background_slope_per_um = np.nan
        background_slope_per_mm = np.nan
        if np.isfinite(object_pixel_um) and object_pixel_um > 0:
            sigma_um = sigma_px * object_pixel_um
            sigma_mm = sigma_um / 1000.0
            sigma_uncertainty_um = sigma_uncertainty_px * object_pixel_um
            sigma_uncertainty_mm = sigma_uncertainty_um / 1000.0
            rayleigh_equivalent_resolution_um = (
                rayleigh_equivalent_resolution_px * object_pixel_um
            )
            rayleigh_equivalent_resolution_mm = (
                rayleigh_equivalent_resolution_um / 1000.0
            )
            rayleigh_equivalent_resolution_uncertainty_um = (
                rayleigh_equivalent_resolution_uncertainty_px * object_pixel_um
            )
            rayleigh_equivalent_resolution_uncertainty_mm = (
                rayleigh_equivalent_resolution_uncertainty_um / 1000.0
            )
            object_pixel_mm = object_pixel_um / 1000.0
            w_um = w_px * object_pixel_um
            w_mm = w_px * object_pixel_mm
            stripe_spacing_um = 2.0 * w_um
            x0_um = fit["x0"] * object_pixel_um
            x0_mm = fit["x0"] * object_pixel_mm
            background_slope_per_um = fit["b1"] / object_pixel_um
            background_slope_per_mm = fit["b1"] / object_pixel_mm
            if np.isfinite(result["conv_width_bound_lower_px"]):
                width_bound_lower_um = (
                    result["conv_width_bound_lower_px"] * object_pixel_um
                )
                width_bound_lower_mm = (
                    result["conv_width_bound_lower_px"] * object_pixel_mm
                )
            if np.isfinite(result["conv_width_bound_upper_px"]):
                width_bound_upper_um = (
                    result["conv_width_bound_upper_px"] * object_pixel_um
                )
                width_bound_upper_mm = (
                    result["conv_width_bound_upper_px"] * object_pixel_mm
                )

        result.update({
            "conv_fit_success": True,
            "conv_sigma_px": sigma_px,
            "conv_sigma_um": sigma_um,
            "conv_sigma_mm": sigma_mm,
            "conv_sigma_uncertainty_px": sigma_uncertainty_px,
            "conv_sigma_uncertainty_um": sigma_uncertainty_um,
            "conv_sigma_uncertainty_mm": sigma_uncertainty_mm,
            "conv_rayleigh_equivalent_resolution_px": rayleigh_equivalent_resolution_px,
            "conv_rayleigh_equivalent_resolution_um": rayleigh_equivalent_resolution_um,
            "conv_rayleigh_equivalent_resolution_mm": rayleigh_equivalent_resolution_mm,
            "conv_rayleigh_equivalent_resolution_uncertainty_px": rayleigh_equivalent_resolution_uncertainty_px,
            "conv_rayleigh_equivalent_resolution_uncertainty_um": rayleigh_equivalent_resolution_uncertainty_um,
            "conv_rayleigh_equivalent_resolution_uncertainty_mm": rayleigh_equivalent_resolution_uncertainty_mm,
            "conv_object_pixel_um": object_pixel_um,
            "conv_object_pixel_mm": object_pixel_mm,
            "conv_frequency_scale_source": frequency_scale_source,
            "conv_w_px": w_px,
            "conv_w_um": w_um,
            "conv_w_mm": w_mm,
            "conv_width_fixed": fit["fixed_width"],
            "conv_stripe_spacing_um": stripe_spacing_um,
            "conv_width_bound_lower_px": result["conv_width_bound_lower_px"],
            "conv_width_bound_upper_px": result["conv_width_bound_upper_px"],
            "conv_width_bound_lower_um": width_bound_lower_um,
            "conv_width_bound_upper_um": width_bound_upper_um,
            "conv_width_bound_lower_mm": width_bound_lower_mm,
            "conv_width_bound_upper_mm": width_bound_upper_mm,
            "conv_x0_px": fit["x0"],
            "conv_x0_um": x0_um,
            "conv_x0_mm": x0_mm,
            "conv_amplitude": fit["A"],
            "conv_background_offset": fit["b0"],
            "conv_background_slope": fit["b1"],
            "conv_background_slope_per_px": fit["b1"],
            "conv_background_slope_per_um": background_slope_per_um,
            "conv_background_slope_per_mm": background_slope_per_mm,
            "conv_fit_rmse": fit["rmse"],
            "conv_fit_chi_squared": fit["chi_squared"],
            "conv_fit_r2": fit["r2"],
            "conv_fit_message": fit["message"],
            "conv_fit_profile": fit["fit_profile"],
            "conv_ideal_profile": fit["ideal_profile"],
        })

    except Exception as exc:
        result["conv_fit_message"] = str(exc)

    return result


def fit_single_square_axis_profile(
    profile: np.ndarray,
    fixed_side_px: float,
    pixel_size_um: Optional[float] = None,
) -> Dict[str, object]:
    """
    Fit one axis projection of a square target with a fixed top-hat side length
    convolved with a Gaussian LSF.
    """
    y = np.asarray(profile, dtype=float)
    x = np.arange(len(y), dtype=float)
    result = _empty_convolution_result("not run")

    if not np.isfinite(fixed_side_px) or fixed_side_px <= 0:
        result["conv_fit_message"] = "fixed square side length is unavailable"
        return result

    try:
        finite = np.isfinite(y)
        if finite.sum() < 6:
            raise ValueError("not enough finite profile samples")

        x_fit = x[finite]
        y_fit = y[finite]
        y_min = float(np.nanmin(y_fit))
        y_max = float(np.nanmax(y_fit))
        y_med = float(np.nanmedian(y_fit))
        y_ptp = max(y_max - y_min, np.finfo(float).eps)
        x_min = float(np.nanmin(x_fit))
        x_max = float(np.nanmax(x_fit))
        x_span = max(x_max - x_min, 1.0)

        smooth_sigma = max(0.75, min(3.0, 0.03 * len(y)))
        y_smooth = gaussian_filter1d(y.astype(float), smooth_sigma)
        bright_contrast = float(np.nanmax(y_smooth) - np.nanmedian(y_smooth))
        dark_contrast = float(np.nanmedian(y_smooth) - np.nanmin(y_smooth))
        polarity_sign = 1.0 if bright_contrast >= dark_contrast else -1.0
        feature_signal = polarity_sign * (y_smooth - np.nanmedian(y_smooth))
        if np.any(np.isfinite(feature_signal)):
            weights = np.maximum(feature_signal, 0.0)
            if np.sum(weights) > np.finfo(float).eps:
                x0_guess = float(np.sum(x * weights) / np.sum(weights))
            else:
                x0_guess = 0.5 * (x_min + x_max)
        else:
            x0_guess = 0.5 * (x_min + x_max)

        slope_guess = 0.0
        try:
            slope_guess = float(np.polyfit(x_fit, y_fit, deg=1)[0])
        except Exception:
            slope_guess = 0.0

        center_guess_from_edges = None
        try:
            edge0, edge1 = find_two_edge_positions_1d(y_smooth)
            center_guess_from_edges = 0.5 * (edge0 + edge1)
            x0_guess = float(center_guess_from_edges)
        except Exception:
            pass

        x0_margin = max(2.0, 0.35 * fixed_side_px)
        if center_guess_from_edges is not None:
            x0_lower = max(x_min, x0_guess - x0_margin)
            x0_upper = min(x_max, x0_guess + x0_margin)
            if x0_upper <= x0_lower:
                x0_lower, x0_upper = x_min, x_max
        else:
            x0_lower, x0_upper = x_min, x_max

        sigma0 = max(0.35, 0.04 * fixed_side_px)
        sigma_upper = max(0.75, min(0.75 * x_span, 0.35 * fixed_side_px))
        lower = np.array([
            y_min - y_ptp,
            -2.0 * y_ptp / x_span,
            -2.5 * y_ptp,
            x0_lower,
            0.1,
        ])
        upper = np.array([
            y_max + y_ptp,
            2.0 * y_ptp / x_span,
            2.5 * y_ptp,
            x0_upper,
            sigma_upper,
        ])
        p0 = np.array([
            y_med,
            slope_guess,
            polarity_sign * y_ptp,
            x0_guess,
            sigma0,
        ])
        p0 = np.clip(p0, lower, upper)

        def model(xvals, b0, b1, A, x0, sigma):
            return b0 + b1 * xvals + A * blurred_bar_profile(
                xvals,
                x0,
                fixed_side_px,
                sigma,
            )

        def residuals(params):
            return model(x_fit, *params) - y_fit

        opt = least_squares(
            residuals,
            p0,
            bounds=(lower, upper),
            loss="linear",
            max_nfev=20000,
        )

        if not opt.success:
            raise RuntimeError(opt.message)

        b0, b1, A, x0_fit, sigma_px = opt.x
        if sigma_px <= 0 or x0_fit < x_min or x0_fit > x_max:
            raise RuntimeError("fit returned unphysical parameters")

        y_model = model(x, b0, b1, A, x0_fit, sigma_px)
        ideal = b0 + b1 * x + A * (
            np.abs(x - x0_fit) < 0.5 * fixed_side_px
        ).astype(float)
        residual = y - y_model
        residual_finite = residual[np.isfinite(residual)]
        rmse = float(np.sqrt(np.mean(residual_finite ** 2))) if residual_finite.size else np.nan
        sse = float(np.sum(residuals(opt.x) ** 2))
        sst = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
        r2 = np.nan if sst <= np.finfo(float).eps else float(1.0 - sse / sst)
        parameter_errors = parameter_standard_errors(opt.jac, residuals(opt.x))
        sigma_uncertainty_px = float(parameter_errors[-1])

        object_pixel_um = np.nan
        object_pixel_mm = np.nan
        sigma_um = np.nan
        sigma_mm = np.nan
        sigma_uncertainty_um = np.nan
        sigma_uncertainty_mm = np.nan
        side_um = np.nan
        side_mm = np.nan
        x0_um = np.nan
        x0_mm = np.nan
        slope_per_um = np.nan
        slope_per_mm = np.nan
        if pixel_size_um is not None and np.isfinite(pixel_size_um) and pixel_size_um > 0:
            object_pixel_um = float(pixel_size_um)
            object_pixel_mm = object_pixel_um / 1000.0
            sigma_um = sigma_px * object_pixel_um
            sigma_mm = sigma_um / 1000.0
            sigma_uncertainty_um = sigma_uncertainty_px * object_pixel_um
            sigma_uncertainty_mm = sigma_uncertainty_um / 1000.0
            side_um = fixed_side_px * object_pixel_um
            side_mm = side_um / 1000.0
            x0_um = x0_fit * object_pixel_um
            x0_mm = x0_fit * object_pixel_mm
            slope_per_um = b1 / object_pixel_um
            slope_per_mm = b1 / object_pixel_mm

        rayleigh_px = GAUSSIAN_RAYLEIGH_EQUIVALENT_FACTOR * sigma_px
        rayleigh_unc_px = GAUSSIAN_RAYLEIGH_EQUIVALENT_FACTOR * sigma_uncertainty_px
        rayleigh_um = rayleigh_px * object_pixel_um if np.isfinite(object_pixel_um) else np.nan
        rayleigh_mm = rayleigh_um / 1000.0 if np.isfinite(rayleigh_um) else np.nan
        rayleigh_unc_um = (
            rayleigh_unc_px * object_pixel_um if np.isfinite(object_pixel_um) else np.nan
        )
        rayleigh_unc_mm = (
            rayleigh_unc_um / 1000.0 if np.isfinite(rayleigh_unc_um) else np.nan
        )

        result.update({
            "conv_fit_success": True,
            "conv_sigma_px": float(sigma_px),
            "conv_sigma_um": sigma_um,
            "conv_sigma_mm": sigma_mm,
            "conv_sigma_uncertainty_px": sigma_uncertainty_px,
            "conv_sigma_uncertainty_um": sigma_uncertainty_um,
            "conv_sigma_uncertainty_mm": sigma_uncertainty_mm,
            "conv_rayleigh_equivalent_resolution_px": rayleigh_px,
            "conv_rayleigh_equivalent_resolution_um": rayleigh_um,
            "conv_rayleigh_equivalent_resolution_mm": rayleigh_mm,
            "conv_rayleigh_equivalent_resolution_uncertainty_px": rayleigh_unc_px,
            "conv_rayleigh_equivalent_resolution_uncertainty_um": rayleigh_unc_um,
            "conv_rayleigh_equivalent_resolution_uncertainty_mm": rayleigh_unc_mm,
            "conv_object_pixel_um": object_pixel_um,
            "conv_object_pixel_mm": object_pixel_mm,
            "conv_frequency_scale_source": "calibration" if np.isfinite(object_pixel_um) else "none",
            "conv_w_px": float(fixed_side_px),
            "conv_w_um": side_um,
            "conv_w_mm": side_mm,
            "conv_width_fixed": True,
            "conv_x0_px": float(x0_fit),
            "conv_x0_um": x0_um,
            "conv_x0_mm": x0_mm,
            "conv_amplitude": float(A),
            "conv_background_offset": float(b0),
            "conv_background_slope": float(b1),
            "conv_background_slope_per_px": float(b1),
            "conv_background_slope_per_um": slope_per_um,
            "conv_background_slope_per_mm": slope_per_mm,
            "conv_fit_rmse": rmse,
            "conv_fit_chi_squared": sse,
            "conv_fit_r2": r2,
            "conv_fit_message": opt.message,
            "conv_fit_profile": y_model,
            "conv_ideal_profile": ideal,
        })

    except Exception as exc:
        result["conv_fit_message"] = str(exc)

    return result


# Backward-compatible name for notebooks written before the PSF-fit cleanup.
fit_convolution_mtf_profile = fit_gaussian_psf_profile


def profile_periodic_score(
    profile: np.ndarray,
    expected_period_px: Optional[float] = None,
) -> Dict[str, float]:
    """
    Score how stripe-like a 1D profile is.

    This lightweight score is used only for orientation detection. It measures
    the normalized projection at the expected or FFT-estimated stripe period;
    it is not reported as a resolution or MTF result.
    """
    profile_corr = remove_slow_background(profile, poly_order=1)
    profile_corr = gaussian_filter1d(profile_corr.astype(float), sigma=1.0)
    centered = profile_corr - np.mean(profile_corr)
    rms = float(np.sqrt(np.mean(centered ** 2)))

    if expected_period_px is not None and np.isfinite(expected_period_px):
        period_px = expected_period_px
    else:
        period_px = estimate_period_fft(profile_corr)

    score = rms

    if period_px is not None and np.isfinite(period_px) and 2 < period_px < len(profile_corr):
        x = np.arange(len(centered), dtype=float)
        projection = np.sum(centered * np.exp(-2j * np.pi * x / period_px))
        amplitude = 2.0 * abs(projection) / len(centered)
        score = float(amplitude / max(rms, np.finfo(float).eps))

    return {
        "score": score,
        "period_px": period_px,
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


def _rotate_crop_for_analysis(crop: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate an already-cropped image while keeping its array size fixed."""
    if not np.isfinite(angle_deg) or abs(angle_deg) < 1e-9:
        return crop
    return rotate(
        crop,
        float(angle_deg),
        reshape=False,
        order=1,
        mode="nearest",
    )


def _scan_rotation_angle(
    crop: np.ndarray,
    score_func,
    max_angle_deg: float = 8.0,
    step_deg: float = 0.5,
) -> Tuple[float, float]:
    """Return the angle with the highest supplied alignment score."""
    if max_angle_deg <= 0 or step_deg <= 0:
        return 0.0, float(score_func(crop))

    angles = np.arange(
        -abs(float(max_angle_deg)),
        abs(float(max_angle_deg)) + 0.5 * float(step_deg),
        float(step_deg),
    )
    best_angle = 0.0
    best_score = -np.inf

    for angle in angles:
        rotated = _rotate_crop_for_analysis(crop, float(angle))
        try:
            score = float(score_func(rotated))
        except Exception:
            score = -np.inf
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    if not np.isfinite(best_score):
        return 0.0, np.nan
    return best_angle, best_score


def estimate_stripe_rotation_angle(
    crop: np.ndarray,
    expected_period_px: Optional[float] = None,
    max_angle_deg: float = 8.0,
    step_deg: float = 0.5,
) -> Tuple[float, float]:
    """Estimate small deskew angle for a stripe ROI from projection periodicity."""
    def score(rotated_crop: np.ndarray) -> float:
        scores = []
        for orientation in ("vertical", "horizontal"):
            profile = extract_bar_profile(rotated_crop, orientation)
            scores.append(
                profile_periodic_score(
                    profile,
                    expected_period_px=expected_period_px,
                )["score"]
            )
        return float(np.nanmax(scores))

    return _scan_rotation_angle(
        crop,
        score,
        max_angle_deg=max_angle_deg,
        step_deg=step_deg,
    )


def _projection_two_edge_score(profile: np.ndarray) -> float:
    """Score how sharp the two dominant edges are in a 1D projection."""
    p = gaussian_filter1d(np.asarray(profile, dtype=float), sigma=2.0)
    if len(p) < 6:
        return 0.0
    g = np.abs(np.gradient(p))
    min_distance = max(3, int(len(p) * 0.25))
    peaks, _props = find_peaks(
        g,
        distance=min_distance,
        prominence=max(np.std(g) * 0.25, np.finfo(float).eps),
    )

    if len(peaks) >= 2:
        chosen = peaks[np.argsort(g[peaks])[-2:]]
        return float(np.sum(g[chosen]))

    chosen = []
    for idx in np.argsort(g)[::-1]:
        if all(abs(int(idx) - int(c)) >= min_distance for c in chosen):
            chosen.append(int(idx))
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        return 0.0
    return float(np.sum(g[chosen]))


def estimate_square_rotation_angle(
    crop: np.ndarray,
    max_angle_deg: float = 8.0,
    step_deg: float = 0.5,
) -> Tuple[float, float]:
    """Estimate small deskew angle for a square ROI from projection edge sharpness."""
    def score(rotated_crop: np.ndarray) -> float:
        x_projection = rotated_crop.mean(axis=0)
        y_projection = rotated_crop.mean(axis=1)
        return (
            _projection_two_edge_score(x_projection)
            + _projection_two_edge_score(y_projection)
        )

    return _scan_rotation_angle(
        crop,
        score,
        max_angle_deg=max_angle_deg,
        step_deg=step_deg,
    )


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
    angle_deg: float = 0.0,
    enable_convolution_mtf: bool = True,
    convolution_initial_width_px: Optional[float] = None,
    convolution_fixed_width_px: Optional[float] = None,
    fix_convolution_width: bool = False,
    psf_model: str = "gaussian",
    auto_trim_profile_band: bool = True,
    auto_rotate: bool = True,
    auto_rotate_max_deg: float = 8.0,
    auto_rotate_step_deg: float = 0.5,
) -> pd.DataFrame:
    """
    Analyze one USAF stripe ROI.

    If orientation is None, automatically detects vertical/horizontal bars.
    """
    ensure_dir(outdir)

    crop = crop_roi(img, roi, angle_deg=angle_deg)

    if calibration is None:
        calibration = CameraCalibration()
    if psf_model not in ("gaussian", "airy", "both"):
        raise ValueError("psf_model must be 'gaussian', 'airy', or 'both'")

    input_period_px = period_px

    # Determine lp/mm from USAF group/element if needed.
    if lpmm is None and group is not None and element is not None:
        lpmm = usaf_lpmm(group, element)

    # Determine expected period in pixels if possible.
    expected_period_px = period_px
    nominal_w_px = nominal_bar_width_px_from_lpmm(lpmm, calibration)
    nominal_width_source = "none"

    if expected_period_px is None and lpmm is not None:
        obj_px_um = calibration.object_pixel_um
        if obj_px_um is not None:
            expected_period_px = (1000.0 / lpmm) / obj_px_um

    if np.isfinite(nominal_w_px):
        nominal_width_source = "group_element_or_lpmm_calibration"
    elif input_period_px is not None and np.isfinite(input_period_px) and input_period_px > 0:
        nominal_w_px = 0.5 * float(input_period_px)
        nominal_width_source = "input_period_px"

    nominal_width_bounds = width_bounds_around_nominal(nominal_w_px)

    auto_rotation_angle_deg = 0.0
    auto_rotation_score = np.nan
    if auto_rotate:
        auto_rotation_angle_deg, auto_rotation_score = estimate_stripe_rotation_angle(
            crop,
            expected_period_px=expected_period_px,
            max_angle_deg=auto_rotate_max_deg,
            step_deg=auto_rotate_step_deg,
        )
        crop = _rotate_crop_for_analysis(crop, auto_rotation_angle_deg)
        print(
            "Auto stripe rotation: "
            f"{auto_rotation_angle_deg:.4g} deg "
            f"(score={auto_rotation_score:.4g})"
        )

    analysis_angle_deg = angle_deg + auto_rotation_angle_deg
    save_crop_plot(
        crop,
        os.path.join(outdir, "stripe_roi.png"),
        f"Stripe ROI - {format_usaf_label(group, element)}",
    )

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

    band_diag = {
        "band": (0, crop.shape[0] if orientation == "vertical" else crop.shape[1]),
        "band_axis": "y" if orientation == "vertical" else "x",
        "band_fraction": 1.0,
        "trim_applied": False,
    }
    if auto_trim_profile_band:
        band_diag = estimate_stripe_averaging_band(crop, orientation)

    raw_profile = extract_bar_profile(
        crop,
        orientation,
        averaging_band=band_diag["band"],
    )
    corrected_profile = remove_slow_background(raw_profile, poly_order=1)

    if period_px is None:
        if expected_period_px is not None:
            period_px = expected_period_px
        else:
            period_px = estimate_period_fft(corrected_profile)

    conv_result = _empty_convolution_result()
    airy_result = _empty_airy_result()
    if enable_convolution_mtf:
        conv_initial_width_px = convolution_initial_width_px
        if np.isfinite(nominal_w_px):
            conv_initial_width_px = nominal_w_px
        if conv_initial_width_px is None:
            conv_initial_width_px = (
                0.5 * period_px
                if period_px is not None and np.isfinite(period_px) and period_px > 0
                else None
            )

        conv_fixed_width_px = convolution_fixed_width_px
        if (
            conv_fixed_width_px is None
            and nominal_width_source == "group_element_or_lpmm_calibration"
        ):
            # When USAF geometry and calibration provide the nominal bar width,
            # keep it fixed so the PSF width is not traded against target width.
            conv_fixed_width_px = nominal_w_px
        elif conv_fixed_width_px is None and fix_convolution_width:
            conv_fixed_width_px = conv_initial_width_px

        if psf_model in ("gaussian", "both"):
            conv_result = fit_gaussian_psf_profile(
                raw_profile,
                x=np.arange(len(raw_profile), dtype=float),
                initial_width=conv_initial_width_px,
                fixed_width=conv_fixed_width_px,
                width_bounds=nominal_width_bounds if conv_fixed_width_px is None else None,
                pixel_size_um=calibration.object_pixel_um,
                nominal_lpmm=lpmm,
            )

        if psf_model in ("airy", "both"):
            airy_result = fit_airy_psf_profile(
                raw_profile,
                x=np.arange(len(raw_profile), dtype=float),
                initial_width=conv_initial_width_px,
                fixed_width=conv_fixed_width_px,
                width_bounds=nominal_width_bounds if conv_fixed_width_px is None else None,
                pixel_size_um=calibration.object_pixel_um,
                nominal_lpmm=lpmm,
            )

    if lpmm is not None:
        res = resolution_from_lpmm(lpmm)
    else:
        res = {
            "lp_per_mm": np.nan,
            "line_pair_period_um": np.nan,
            "half_pitch_um": np.nan,
        }

    if auto_trim_profile_band and band_diag["trim_applied"]:
        b0, b1 = band_diag["band"]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(crop, cmap="gray", origin="upper")
        if orientation == "vertical":
            ax.axhspan(b0, b1, facecolor="tab:green", alpha=0.18)
            ax.axhline(b0, color="tab:green", linewidth=0.8)
            ax.axhline(b1, color="tab:green", linewidth=0.8)
        else:
            ax.axvspan(b0, b1, facecolor="tab:green", alpha=0.18)
            ax.axvline(b0, color="tab:green", linewidth=0.8)
            ax.axvline(b1, color="tab:green", linewidth=0.8)
        ax.set_title("Stripe ROI averaging band")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "stripe_profile_averaging_band.png"), dpi=200)
        plt.close(fig)

    if enable_convolution_mtf:
        x_raw = np.arange(len(raw_profile), dtype=float)
        object_pixel_um = conv_result["conv_object_pixel_um"]
        if not np.isfinite(object_pixel_um):
            object_pixel_um = airy_result["airy_object_pixel_um"]
        if not np.isfinite(object_pixel_um) and calibration.object_pixel_um is not None:
            object_pixel_um = calibration.object_pixel_um

        if np.isfinite(object_pixel_um) and object_pixel_um > 0:
            x_conv_plot = x_raw * object_pixel_um
            x_conv_label = "object-space position (um)"
        else:
            x_conv_plot = x_raw
            x_conv_label = "position (pixel; no physical scale available)"

        fig, (ax_profile, ax_resid) = plt.subplots(
            2,
            1,
            figsize=(7, 5),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax_profile.plot(
            x_conv_plot,
            raw_profile,
            linestyle="none",
            marker="o",
            markersize=4,
            color="tab:blue",
            label="measured profile",
        )

        gaussian_success = conv_result["conv_fit_success"]
        airy_success = airy_result["airy_fit_success"]
        if gaussian_success or airy_success:
            x_theory_raw = np.linspace(x_raw[0], x_raw[-1], 1200)
            if np.isfinite(object_pixel_um) and object_pixel_um > 0:
                x_theory_plot = x_theory_raw * object_pixel_um
            else:
                x_theory_plot = x_theory_raw

            if psf_model == "airy" and airy_success:
                ideal_result = airy_result
                ideal_prefix = "airy"
            elif gaussian_success:
                ideal_result = conv_result
                ideal_prefix = "conv"
            else:
                ideal_result = airy_result
                ideal_prefix = "airy"

            ideal_profile_theory = (
                ideal_result[f"{ideal_prefix}_background_offset"]
                + ideal_result[f"{ideal_prefix}_background_slope_per_px"] * x_theory_raw
                + ideal_result[f"{ideal_prefix}_amplitude"]
                * three_bar_ideal_profile(
                    x_theory_raw,
                    ideal_result[f"{ideal_prefix}_x0_px"],
                    ideal_result[f"{ideal_prefix}_w_px"],
                )
            )
            title_lines = []
            if psf_model == "both":
                ax_profile.plot(
                    x_theory_plot,
                    ideal_profile_theory,
                    ":",
                    drawstyle="steps-mid",
                    color="0.35",
                    label="unblurred model",
                )
            else:
                ax_profile.plot(
                    x_theory_plot,
                    ideal_profile_theory,
                    "-",
                    color="tab:green",
                    label="unblurred model",
                )

            if gaussian_success and psf_model in ("gaussian", "both"):
                gaussian_theory = three_bar_convolution_model(
                    x_theory_raw,
                    conv_result["conv_background_offset"],
                    conv_result["conv_background_slope_per_px"],
                    conv_result["conv_amplitude"],
                    conv_result["conv_x0_px"],
                    conv_result["conv_w_px"],
                    conv_result["conv_sigma_px"],
                )
                ax_profile.plot(
                    x_theory_plot,
                    gaussian_theory,
                    "-",
                    color="tab:orange",
                    label="Gaussian fit",
                )
                ax_resid.plot(
                    x_conv_plot,
                    raw_profile - conv_result["conv_fit_profile"],
                    linestyle="none" if psf_model == "gaussian" else "-",
                    marker="o" if psf_model == "gaussian" else None,
                    markersize=4,
                    color="tab:blue" if psf_model == "gaussian" else "tab:orange",
                    label="residual" if psf_model == "gaussian" else "Gaussian residual",
                )
                if np.isfinite(conv_result["conv_sigma_um"]):
                    sigma_text = format_value_with_uncertainty(
                        conv_result["conv_sigma_um"],
                        conv_result["conv_sigma_uncertainty_um"],
                    )
                    resolution_text = format_value_with_uncertainty(
                        conv_result["conv_rayleigh_equivalent_resolution_um"],
                        conv_result[
                            "conv_rayleigh_equivalent_resolution_uncertainty_um"
                        ],
                    )
                    title_lines.append(
                        "Gaussian-model fit: "
                        f"σ={sigma_text} um\n"
                        "Rayleigh-equivalent resolution=2.898785σ="
                        f"{resolution_text} um"
                    )
                else:
                    sigma_text = format_value_with_uncertainty(
                        conv_result["conv_sigma_px"],
                        conv_result["conv_sigma_uncertainty_px"],
                    )
                    resolution_text = format_value_with_uncertainty(
                        conv_result["conv_rayleigh_equivalent_resolution_px"],
                        conv_result[
                            "conv_rayleigh_equivalent_resolution_uncertainty_px"
                        ],
                    )
                    title_lines.append(
                        "Gaussian-model fit: "
                        f"σ={sigma_text} px\n"
                        "Rayleigh-equivalent resolution=2.898785σ="
                        f"{resolution_text} px"
                    )

            if airy_success and psf_model in ("airy", "both"):
                airy_theory = three_bar_airy_convolution_model(
                    x_theory_raw,
                    airy_result["airy_background_offset"],
                    airy_result["airy_background_slope_per_px"],
                    airy_result["airy_amplitude"],
                    airy_result["airy_x0_px"],
                    airy_result["airy_w_px"],
                    airy_result["airy_r0_px"],
                )
                ax_profile.plot(
                    x_theory_plot,
                    airy_theory,
                    "-",
                    color="tab:orange" if psf_model == "airy" else "tab:green",
                    label="Airy fit",
                )
                ax_resid.plot(
                    x_conv_plot,
                    raw_profile - airy_result["airy_fit_profile"],
                    linestyle="none" if psf_model == "airy" else "-",
                    marker="o" if psf_model == "airy" else None,
                    markersize=4,
                    color="tab:blue" if psf_model == "airy" else "tab:green",
                    label="residual" if psf_model == "airy" else "Airy residual",
                )
                if np.isfinite(airy_result["airy_r0_um"]):
                    title_lines.append(
                        "Airy-model fit: "
                        f"r0={airy_result['airy_r0_um']:.4g} um, "
                        f"Rayleigh resolution={airy_result['airy_r0_um']:.4g} um"
                    )
                else:
                    title_lines.append(
                        "Airy-model fit: "
                        f"r0 Rayleigh resolution={airy_result['airy_r0_px']:.4g} px"
                    )

            if group is not None and element is not None:
                # A named USAF element has a standard line-pair period. Keep
                # the measured pixel period in the CSV, but label the plot
                # with the nominal target geometry.
                spacing_um = 1000.0 / usaf_lpmm(group, element)
                spacing_label = f"stripe spacing = {spacing_um:.4g} um"
            else:
                if period_px is not None and np.isfinite(period_px) and period_px > 0:
                    spacing_px = float(period_px)
                elif np.isfinite(nominal_w_px) and nominal_w_px > 0:
                    spacing_px = 2.0 * float(nominal_w_px)
                else:
                    spacing_px = 2.0 * ideal_result[f"{ideal_prefix}_w_px"]
                if np.isfinite(object_pixel_um) and object_pixel_um > 0:
                    spacing_label = f"stripe spacing = {spacing_px * object_pixel_um:.4g} um"
                else:
                    spacing_label = f"stripe spacing = {spacing_px:.4g} px"
            fig.text(
                0.5,
                0.01,
                f"{format_usaf_label(group, element)}; {spacing_label}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            ax_profile.set_title("\n".join(title_lines))
        else:
            ax_profile.set_title(
                f"Three-bar PSF fit failed ({psf_model}): "
                f"Gaussian: {conv_result['conv_fit_message']}; "
                f"Airy: {airy_result['airy_fit_message']}"
            )
            ax_resid.plot(
                x_conv_plot,
                np.zeros_like(x_raw),
                linestyle="-",
                label="residual",
            )

        ax_profile.set_ylabel("intensity counts")
        ax_profile.legend(loc="lower right")
        ax_resid.axhline(0, color="k", linewidth=0.8)
        ax_resid.set_xlabel(x_conv_label)
        ax_resid.set_ylabel("residual")
        ax_resid.legend(loc="lower right")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(
            os.path.join(outdir, "stripe_psf_fit.png"),
            dpi=200,
        )
        plt.close(fig)

    result = {
        "mode": "stripe",
        "roi_x": roi[0],
        "roi_y": roi[1],
        "roi_w": roi[2],
        "roi_h": roi[3],
        "orientation": orientation,
        "angle_deg": analysis_angle_deg,
        "input_angle_deg": angle_deg,
        "auto_rotation_angle_deg": auto_rotation_angle_deg,
        "auto_rotation_score": auto_rotation_score,
        "psf_model": psf_model,
        "group": group,
        "element": element,
        "lp_per_mm": res["lp_per_mm"],
        "line_pair_period_um": res["line_pair_period_um"],
        "half_pitch_um": res["half_pitch_um"],
        "period_px": period_px,
        "nominal_w_px": nominal_w_px,
        "nominal_width_source": nominal_width_source,
        "nominal_width_bound_lower_px": (
            nominal_width_bounds[0] if nominal_width_bounds is not None else np.nan
        ),
        "nominal_width_bound_upper_px": (
            nominal_width_bounds[1] if nominal_width_bounds is not None else np.nan
        ),
        "profile_band_axis": band_diag["band_axis"],
        "profile_band_start_px": band_diag["band"][0],
        "profile_band_end_px": band_diag["band"][1],
        "profile_band_fraction": band_diag["band_fraction"],
        "profile_band_trim_applied": band_diag["trim_applied"],
    }
    result.update({
        key: value
        for key, value in conv_result.items()
        if key not in ("conv_fit_profile", "conv_ideal_profile")
    })
    result.update({
        key: value
        for key, value in airy_result.items()
        if key not in ("airy_fit_profile", "airy_ideal_profile")
    })
    if orientation_diag is not None:
        result.update({
            "auto_vertical_score": orientation_diag["vertical"]["score"],
            "auto_horizontal_score": orientation_diag["horizontal"]["score"],
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


@dataclass
class SquareCandidate:
    component_id: int
    roi: Tuple[int, int, int, int]
    bbox: Tuple[int, int, int, int]
    cx: float
    cy: float
    w: int
    h: int
    area: int
    fill_fraction: float
    aspect_ratio: float
    side_px: float
    score: float
    polarity: str


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
    margin_factor: float = 0.20,
    core_margin_factor: float = 0.05,
    max_span_to_length: float = 1.5,
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
    if spacing_err > 0.25:
        return None

    align_err = float(np.std(perp_centers) / max(mean_length, 1.0))
    if align_err > 0.20:
        return None

    thickness_similarity = float(np.std(thicknesses) / mean_thickness)
    length_similarity = float(np.std(lengths) / mean_length)

    if thickness_similarity > 0.45:
        return None
    if length_similarity > 0.45:
        return None

    # For an ideal square-wave element, the center-to-center pitch is close to
    # twice the bar width. Keep this fairly strict so neighboring structures
    # are less likely to be folded into the ROI.
    pitch_to_thickness = period_px / mean_thickness
    if not (1.25 <= pitch_to_thickness <= 4.0):
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

    across_margin = int(max(1, margin_factor * period_px))
    along_margin = int(max(0, core_margin_factor * period_px))
    core_margin = int(max(1, core_margin_factor * period_px))

    # The profile is averaged along the stripe direction. Keep that direction
    # tight to the common bar support so dark surround/background does not
    # dilute the averaged profile, while preserving some context across bars.
    if orientation == "vertical":
        common_y0 = max(y0s)
        common_y1 = min(y1s)
        if common_y1 - common_y0 >= 0.65 * mean_length:
            roi_y0, roi_y1 = common_y0, common_y1
        else:
            center_y = float(np.median([b.cy for b in bars]))
            support_len = 0.75 * float(np.min(lengths))
            roi_y0 = int(round(center_y - 0.5 * support_len))
            roi_y1 = int(round(center_y + 0.5 * support_len))

        roi = _clip_roi(
            (
                x0 - across_margin,
                int(round(roi_y0)) - along_margin,
                (x1 - x0) + 2 * across_margin,
                int(round(roi_y1 - roi_y0)) + 2 * along_margin,
            ),
            img_shape,
        )

    else:
        common_x0 = max(x0s)
        common_x1 = min(x1s)
        if common_x1 - common_x0 >= 0.65 * mean_length:
            roi_x0, roi_x1 = common_x0, common_x1
        else:
            center_x = float(np.median([b.cx for b in bars]))
            support_len = 0.75 * float(np.min(lengths))
            roi_x0 = int(round(center_x - 0.5 * support_len))
            roi_x1 = int(round(center_x + 0.5 * support_len))

        roi = _clip_roi(
            (
                int(round(roi_x0)) - along_margin,
                y0 - across_margin,
                int(round(roi_x1 - roi_x0)) + 2 * along_margin,
                (y1 - y0) + 2 * across_margin,
            ),
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
    max_span_to_length: float = 1.5,
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
    min_area: int = 20,
    min_aspect: float = 2.0,
    max_candidates_per_orientation: int = 180,
    max_triplets: Optional[int] = 50,
    bg_sigma: float = 40.0,
    max_bars_per_roi: int = 3,
    max_span_to_length: float = 1.5,
    enable_convolution_mtf: bool = True,
    convolution_fixed_width_px: Optional[float] = None,
    fix_convolution_width: bool = False,
    psf_model: str = "gaussian",
    auto_trim_profile_band: bool = True,
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
            lpmm=nearest_lpmm,
            period_px=t.period_px,
            calibration=calibration,
            angle_deg=0.0,
            enable_convolution_mtf=enable_convolution_mtf,
            convolution_initial_width_px=t.bar_thickness_px,
            convolution_fixed_width_px=convolution_fixed_width_px,
            fix_convolution_width=fix_convolution_width,
            psf_model=psf_model,
            auto_trim_profile_band=auto_trim_profile_band,
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
            "psf_model",
            "roi_x",
            "roi_y",
            "roi_w",
            "roi_h",
            "detected_period_px",
            "detected_local_bar_count",
            "nominal_w_px",
            "nominal_width_source",
            "conv_w_um",
            "conv_width_fixed",
            "conv_stripe_spacing_um",
            "conv_width_bound_lower_um",
            "conv_width_bound_upper_um",
            "conv_sigma_um",
            "conv_sigma_uncertainty_um",
            "conv_rayleigh_equivalent_resolution_um",
            "conv_rayleigh_equivalent_resolution_uncertainty_um",
            "conv_fit_rmse",
            "airy_r0_um",
            "airy_rayleigh_resolution_um",
            "airy_width_fixed",
            "airy_fit_rmse",
            "detection_score",
        ]
        cols = [c for c in cols if c in summary.columns]
        print(summary[cols].to_string(index=False))
    else:
        print("No stripe triplets detected.")

    print(f"\nSaved summary to: {summary_path}")

    return summary


# ============================================================
# Automatic USAF square extraction
# ============================================================

def find_square_candidates(
    mask: np.ndarray,
    img: np.ndarray,
    polarity: str,
    expected_side_px: Optional[float] = None,
    min_area: int = 20,
    max_aspect_error: float = 0.35,
    min_fill_fraction: float = 0.25,
    roi_margin_fraction: float = 0.35,
) -> List[SquareCandidate]:
    """Find compact connected components that could be USAF square targets."""
    labeled, _n = label(mask, structure=np.ones((3, 3), dtype=int))
    slices = find_objects(labeled)
    candidates: List[SquareCandidate] = []

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

        aspect_ratio = max(w / h, h / w)
        aspect_error = aspect_ratio - 1.0
        if aspect_error > max_aspect_error:
            continue

        yy, xx = np.nonzero(comp)
        cx = x0 + float(xx.mean())
        cy = y0 + float(yy.mean())
        measured_side_px = 0.5 * (w + h)

        side_error = 0.0
        roi_side = measured_side_px
        if (
            expected_side_px is not None
            and np.isfinite(expected_side_px)
            and expected_side_px > 0
        ):
            side_error = abs(measured_side_px - expected_side_px) / expected_side_px
            # Thresholded blurred objects can be narrower than the true square.
            # Keep this permissive while still ranking nominal-size matches first.
            if side_error > 0.75:
                continue
            roi_side = expected_side_px

        margin = max(2.0, roi_margin_fraction * roi_side)
        roi_size = int(round(roi_side + 2.0 * margin))
        roi = _clip_roi(
            (
                int(round(cx - 0.5 * roi_size)),
                int(round(cy - 0.5 * roi_size)),
                roi_size,
                roi_size,
            ),
            img.shape,
        )

        roi_crop = crop_roi(img, roi)
        roi_contrast = float(np.nanpercentile(roi_crop, 95) - np.nanpercentile(roi_crop, 5))
        contrast_scale = max(abs(float(np.nanmedian(img))), 1.0)
        contrast_score = roi_contrast / contrast_scale

        score = 1.0 / (
            1.0
            + 2.5 * aspect_error
            + 2.0 * side_error
            + abs(fill_fraction - 0.75)
            + 0.2 / max(contrast_score, 1e-6)
        )

        candidates.append(
            SquareCandidate(
                component_id=i + 1,
                roi=roi,
                bbox=(x0, y0, w, h),
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                area=area,
                fill_fraction=float(fill_fraction),
                aspect_ratio=float(aspect_ratio),
                side_px=float(measured_side_px),
                score=float(score),
                polarity=polarity,
            )
        )

    return candidates


def nonmax_suppress_squares(
    squares: List[SquareCandidate],
    iou_threshold: float = 0.25,
) -> List[SquareCandidate]:
    """Remove duplicate square detections using ROI overlap."""
    squares = sorted(squares, key=lambda s: s.score, reverse=True)
    kept: List[SquareCandidate] = []
    for s in squares:
        if all(_roi_iou(s.roi, k.roi) <= iou_threshold for k in kept):
            kept.append(s)
    return kept


def detect_usaf_squares(
    img: np.ndarray,
    polarity: str = "dark",
    expected_side_px: Optional[float] = None,
    min_area: int = 20,
    max_aspect_error: float = 0.35,
    min_fill_fraction: float = 0.25,
    max_squares: Optional[int] = None,
    bg_sigma: float = 40.0,
    verbose: bool = True,
) -> Tuple[List[SquareCandidate], Dict[str, object]]:
    """Automatically detect compact USAF square targets in an image."""
    mask, used_polarity, threshold, norm = make_feature_mask(
        img,
        polarity=polarity,
        bg_sigma=bg_sigma,
    )

    candidates = find_square_candidates(
        mask=mask,
        img=img,
        polarity=used_polarity,
        expected_side_px=expected_side_px,
        min_area=min_area,
        max_aspect_error=max_aspect_error,
        min_fill_fraction=min_fill_fraction,
    )
    squares = nonmax_suppress_squares(candidates, iou_threshold=0.25)
    if max_squares is not None:
        squares = squares[:max_squares]
    squares = sorted(squares, key=lambda s: (s.roi[1], s.roi[0], -s.score))

    if verbose:
        print("\nAuto square detection:")
        print(f"  polarity used: {used_polarity}")
        print(f"  threshold: {threshold:.4g}")
        print(f"  square candidates before NMS: {len(candidates)}")
        print(f"  square candidates after NMS: {len(squares)}")

    debug = {
        "mask": mask,
        "norm": norm,
        "threshold": threshold,
        "polarity": used_polarity,
        "candidates": candidates,
    }
    return squares, debug


def plot_detected_squares(
    img: np.ndarray,
    squares: List[SquareCandidate],
    path: str,
    labels: Optional[List[str]] = None,
    title: str = "Detected USAF squares",
) -> None:
    """Save diagnostic plot showing detected square ROIs."""
    from matplotlib.patches import Rectangle

    annotation_color = "#00E5FF"
    if labels is None:
        labels = [str(i) for i in range(len(squares))]
    if len(labels) != len(squares):
        raise ValueError("labels must have the same length as squares")

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.imshow(img, cmap="gray", origin="upper")

    for s, label_text in zip(squares, labels):
        x0, y0, w, h = s.roi
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
        tx = x0 + 1
        ty = y0 - 2 if y0 >= 10 else y0 + 2
        va = "bottom" if y0 >= 10 else "top"
        ax.text(
            tx,
            ty,
            str(label_text),
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


def autocrop_square_roi_from_projection(
    img: np.ndarray,
    expected_side_px: Optional[float] = None,
    margin_fraction: float = 0.35,
) -> Tuple[Tuple[int, int, int, int], Dict[str, float]]:
    """
    Auto-crop one square inside a rough user-selected region.

    This is deliberately local and projection-based: after the user narrows the
    search zone, the strongest pair of vertical and horizontal square edges is
    enough to center a tight ROI without relying on global segmentation.
    """
    x_projection = img.mean(axis=0)
    y_projection = img.mean(axis=1)

    left_x, right_x = find_two_edge_positions_1d(x_projection)
    top_y, bottom_y = find_two_edge_positions_1d(y_projection)

    measured_w = max(1.0, float(right_x - left_x))
    measured_h = max(1.0, float(bottom_y - top_y))
    measured_side_px = 0.5 * (measured_w + measured_h)

    if (
        expected_side_px is not None
        and np.isfinite(expected_side_px)
        and expected_side_px > 0
    ):
        side_px = float(expected_side_px)
    else:
        side_px = measured_side_px

    cx = 0.5 * (left_x + right_x)
    cy = 0.5 * (top_y + bottom_y)
    margin = max(2.0, margin_fraction * side_px)
    roi_size = int(round(side_px + 2.0 * margin))
    roi = _clip_roi(
        (
            int(round(cx - 0.5 * roi_size)),
            int(round(cy - 0.5 * roi_size)),
            roi_size,
            roi_size,
        ),
        img.shape,
    )

    diagnostics = {
        "left_x": float(left_x),
        "right_x": float(right_x),
        "top_y": float(top_y),
        "bottom_y": float(bottom_y),
        "center_x": float(cx),
        "center_y": float(cy),
        "measured_width_px": measured_w,
        "measured_height_px": measured_h,
        "measured_side_px": measured_side_px,
        "crop_side_px": float(side_px),
        "roi_margin_px": float(margin),
    }
    return roi, diagnostics


def plot_projection_autocrop(
    img: np.ndarray,
    roi: Tuple[int, int, int, int],
    diagnostics: Dict[str, float],
    path: str,
) -> None:
    """Save a diagnostic plot for the rough-zone projection autocrop."""
    from matplotlib.patches import Rectangle

    x0, y0, w, h = roi
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img, cmap="gray", origin="upper")
    rect = Rectangle(
        (x0, y0),
        w,
        h,
        fill=False,
        edgecolor="#00E5FF",
        linewidth=1.8,
    )
    ax.add_patch(rect)
    ax.axvline(diagnostics["left_x"], color="tab:orange", linestyle="--", linewidth=1.0)
    ax.axvline(diagnostics["right_x"], color="tab:orange", linestyle="--", linewidth=1.0)
    ax.axhline(diagnostics["top_y"], color="tab:green", linestyle="--", linewidth=1.0)
    ax.axhline(diagnostics["bottom_y"], color="tab:green", linestyle="--", linewidth=1.0)
    ax.set_title("Projection-based square autocrop")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def analyze_auto_squares(
    img: np.ndarray,
    outdir: str,
    group: Optional[int] = None,
    element: Optional[int] = None,
    lpmm: Optional[float] = None,
    calibration: Optional[CameraCalibration] = None,
    polarity: str = "dark",
    min_area: int = 20,
    max_aspect_error: float = 0.35,
    min_fill_fraction: float = 0.25,
    max_squares: Optional[int] = 50,
    bg_sigma: float = 40.0,
) -> pd.DataFrame:
    """Detect and analyze visible USAF square targets in the image."""
    ensure_dir(outdir)
    if calibration is None:
        calibration = CameraCalibration()
    if lpmm is None and group is not None:
        lpmm = usaf_square_lpmm(group)

    square_side_um = nominal_usaf_square_side_um_from_lpmm(lpmm)
    if not np.isfinite(square_side_um):
        raise ValueError(
            "Auto-square detection requires --group or --lpmm so the standard "
            "USAF square side length is known."
        )
    expected_side_px = nominal_usaf_square_side_px_from_lpmm(lpmm, calibration)

    squares, debug = detect_usaf_squares(
        img=img,
        polarity=polarity,
        expected_side_px=expected_side_px if np.isfinite(expected_side_px) else None,
        min_area=min_area,
        max_aspect_error=max_aspect_error,
        min_fill_fraction=min_fill_fraction,
        max_squares=max_squares,
        bg_sigma=bg_sigma,
        verbose=True,
    )

    labels = [str(i) for i in range(len(squares))]
    plot_detected_squares(
        img,
        squares,
        os.path.join(outdir, "auto_detected_squares.png"),
        labels=labels,
        title="Detected USAF squares (unified labels)",
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(debug["mask"], cmap="gray", origin="upper")
    plt.title(f"Segmentation mask, polarity={debug['polarity']}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "auto_square_segmentation_mask.png"), dpi=200)
    plt.close()

    rows = []
    for i, s in enumerate(squares):
        square_dir = os.path.join(outdir, f"square_{i:03d}")
        calibration_one = calibration
        if calibration.object_pixel_um is None and np.isfinite(s.side_px) and s.side_px > 0:
            calibration_one = calibration_with_object_pixel_um(
                calibration,
                square_side_um / s.side_px,
            )
        df_one = analyze_square_roi(
            img=img,
            roi=s.roi,
            outdir=square_dir,
            group=group,
            element=element,
            lpmm=lpmm,
            calibration=calibration_one,
            angle_deg=0.0,
        )
        row = df_one.iloc[0].to_dict()
        row.update({
            "auto_index": i,
            "display_label": str(i),
            "detection_score": s.score,
            "detected_side_px": s.side_px,
            "detected_bbox": str(s.bbox),
            "detected_component_id": s.component_id,
            "detected_polarity": s.polarity,
            "detected_center_x": s.cx,
            "detected_center_y": s.cy,
            "detected_area": s.area,
            "detected_fill_fraction": s.fill_fraction,
            "detected_aspect_ratio": s.aspect_ratio,
            "output_dir": square_dir,
        })
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(outdir, "auto_squares_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nAuto square summary:")
    if len(summary) > 0:
        cols = [
            "auto_index",
            "display_label",
            "roi_x",
            "roi_y",
            "roi_w",
            "roi_h",
            "square_side_px",
            "horizontal_conv_sigma_um",
            "horizontal_conv_rayleigh_equivalent_resolution_um",
            "vertical_conv_sigma_um",
            "vertical_conv_rayleigh_equivalent_resolution_um",
            "detection_score",
        ]
        cols = [c for c in cols if c in summary.columns]
        print(summary[cols].to_string(index=False))
    else:
        print("No square targets detected.")

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
    group: Optional[int] = None,
    element: Optional[int] = None,
    lpmm: Optional[float] = None,
    calibration: Optional[CameraCalibration] = None,
    angle_deg: float = 0.0,
    auto_rotate: bool = False,
    auto_rotate_max_deg: float = 8.0,
    auto_rotate_step_deg: float = 0.5,
    reported_angle_deg: Optional[float] = None,
    reported_auto_rotation_angle_deg: float = 0.0,
    reported_auto_rotation_score: float = np.nan,
) -> pd.DataFrame:
    """
    Analyze a full USAF square ROI with a fixed nominal square side length.

    The x projection reports the horizontal-axis blur sigma, and the y
    projection reports the vertical-axis blur sigma.
    """
    ensure_dir(outdir)

    square = crop_roi(img, roi, angle_deg=angle_deg)
    local_auto_rotation_angle_deg = 0.0
    local_auto_rotation_score = np.nan
    if auto_rotate:
        local_auto_rotation_angle_deg, local_auto_rotation_score = estimate_square_rotation_angle(
            square,
            max_angle_deg=auto_rotate_max_deg,
            step_deg=auto_rotate_step_deg,
        )
        square = _rotate_crop_for_analysis(square, local_auto_rotation_angle_deg)
        print(
            "Auto square rotation: "
            f"{local_auto_rotation_angle_deg:.4g} deg "
            f"(score={local_auto_rotation_score:.4g})"
        )

    total_auto_rotation_angle_deg = (
        reported_auto_rotation_angle_deg + local_auto_rotation_angle_deg
    )
    total_auto_rotation_score = (
        local_auto_rotation_score
        if np.isfinite(local_auto_rotation_score)
        else reported_auto_rotation_score
    )
    analysis_angle_deg = (
        reported_angle_deg
        if reported_angle_deg is not None
        else angle_deg + local_auto_rotation_angle_deg
    )
    save_crop_plot(
        square,
        os.path.join(outdir, "square_roi.png"),
        f"Square ROI - {format_usaf_label(group, square=True)}",
    )

    if calibration is None:
        calibration = CameraCalibration()

    if lpmm is None and group is not None:
        lpmm = usaf_square_lpmm(group)

    square_side_um = nominal_usaf_square_side_um_from_lpmm(lpmm)
    square_side_px = nominal_usaf_square_side_px_from_lpmm(lpmm, calibration)
    if not np.isfinite(square_side_px) and np.isfinite(square_side_um):
        try:
            _projection_roi, projection_diag = autocrop_square_roi_from_projection(
                square,
                expected_side_px=None,
            )
            measured_side_px = projection_diag["measured_side_px"]
            if np.isfinite(measured_side_px) and measured_side_px > 0:
                inferred_object_pixel_um = square_side_um / measured_side_px
                calibration = calibration_with_object_pixel_um(
                    calibration,
                    inferred_object_pixel_um,
                )
                square_side_px = nominal_usaf_square_side_px_from_lpmm(lpmm, calibration)
                print(
                    "Inferred object-space pixel size from group square: "
                    f"{inferred_object_pixel_um:.6g} um/pixel"
                )
        except Exception:
            pass
    square_side_source = "group_or_lpmm_calibration" if np.isfinite(square_side_px) else "none"
    if not np.isfinite(square_side_px):
        raise ValueError(
            "Square mode requires --group or --lpmm so the standard USAF "
            "square side length is known."
        )

    try:
        _profile_roi, profile_diag = autocrop_square_roi_from_projection(
            square,
            expected_side_px=square_side_px,
        )
    except Exception:
        profile_diag = {
            "center_x": 0.5 * (square.shape[1] - 1),
            "center_y": 0.5 * (square.shape[0] - 1),
        }

    band_fraction = 0.60
    band_width_px = max(3, int(round(band_fraction * square_side_px)))
    center_x = float(profile_diag["center_x"])
    center_y = float(profile_diag["center_y"])
    x_band0 = max(0, int(round(center_x - 0.5 * band_width_px)))
    x_band1 = min(square.shape[1], int(round(center_x + 0.5 * band_width_px)))
    y_band0 = max(0, int(round(center_y - 0.5 * band_width_px)))
    y_band1 = min(square.shape[0], int(round(center_y + 0.5 * band_width_px)))
    if x_band1 <= x_band0:
        x_band0, x_band1 = 0, square.shape[1]
    if y_band1 <= y_band0:
        y_band0, y_band1 = 0, square.shape[0]

    x_profile = square[y_band0:y_band1, :].mean(axis=0)
    y_profile = square[:, x_band0:x_band1].mean(axis=1)
    horizontal_fit = fit_single_square_axis_profile(
        x_profile,
        fixed_side_px=square_side_px,
        pixel_size_um=calibration.object_pixel_um,
    )
    vertical_fit = fit_single_square_axis_profile(
        y_profile,
        fixed_side_px=square_side_px,
        pixel_size_um=calibration.object_pixel_um,
    )

    if lpmm is not None:
        res = resolution_from_lpmm(lpmm)
    else:
        res = {
            "lp_per_mm": np.nan,
            "line_pair_period_um": np.nan,
            "half_pitch_um": np.nan,
        }

    def prefixed_fit(prefix: str, fit: Dict[str, object]) -> Dict[str, object]:
        return {
            f"{prefix}_{key}": value
            for key, value in fit.items()
            if key not in ("conv_fit_profile", "conv_ideal_profile")
        }

    result = {
        "mode": "square",
        "roi_x": roi[0],
        "roi_y": roi[1],
        "roi_w": roi[2],
        "roi_h": roi[3],
        "orientation": "square",
        "angle_deg": analysis_angle_deg,
        "input_angle_deg": angle_deg,
        "auto_rotation_angle_deg": total_auto_rotation_angle_deg,
        "auto_rotation_score": total_auto_rotation_score,
        "psf_model": "gaussian",
        "group": group,
        "element": element,
        "lp_per_mm": res["lp_per_mm"],
        "line_pair_period_um": res["line_pair_period_um"],
        "half_pitch_um": res["half_pitch_um"],
        "period_px": (
            res["line_pair_period_um"] / calibration.object_pixel_um
            if np.isfinite(res["line_pair_period_um"])
            and calibration.object_pixel_um is not None
            and calibration.object_pixel_um > 0
            else np.nan
        ),
        "nominal_w_px": square_side_px,
        "nominal_width_source": square_side_source,
        "nominal_width_bound_lower_px": square_side_px,
        "nominal_width_bound_upper_px": square_side_px,
        "square_side_px": square_side_px,
        "square_side_um": square_side_um,
        "square_side_mm": square_side_um / 1000.0 if np.isfinite(square_side_um) else np.nan,
        "square_side_bar_widths": USAF_SQUARE_SIDE_BAR_WIDTHS,
        "profile_band_axis": "xy",
        "profile_band_start_px": 0,
        "profile_band_end_px": np.nan,
        "profile_band_fraction": 1.0,
        "profile_band_trim_applied": False,
        "horizontal_profile_band_axis": "y",
        "horizontal_profile_band_start_px": y_band0,
        "horizontal_profile_band_end_px": y_band1,
        "vertical_profile_band_axis": "x",
        "vertical_profile_band_start_px": x_band0,
        "vertical_profile_band_end_px": x_band1,
        "square_profile_band_fraction": band_fraction,
    }
    result.update(prefixed_fit("horizontal", horizontal_fit))
    result.update(prefixed_fit("vertical", vertical_fit))

    df = pd.DataFrame([result])
    df.to_csv(os.path.join(outdir, "square_result.csv"), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes[0, 0].imshow(square, cmap="gray", origin="upper")
    axes[0, 0].set_title("Square ROI")
    axes[0, 0].axhspan(y_band0, y_band1, facecolor="tab:orange", alpha=0.12)
    axes[0, 0].axvspan(x_band0, x_band1, facecolor="tab:green", alpha=0.12)
    axes[0, 0].axvline(horizontal_fit["conv_x0_px"], color="tab:orange", linestyle="--")
    axes[0, 0].axhline(vertical_fit["conv_x0_px"], color="tab:green", linestyle="--")

    axis_specs = [
        (axes[0, 1], x_profile, horizontal_fit, "horizontal", "x position"),
        (axes[1, 0], y_profile, vertical_fit, "vertical", "y position"),
    ]
    for ax, profile, fit, label, xlabel in axis_specs:
        x_axis = np.arange(len(profile), dtype=float)
        plot_x = x_axis
        x_label = f"{xlabel} (pixel)"
        if calibration.object_pixel_um is not None and calibration.object_pixel_um > 0:
            plot_x = x_axis * calibration.object_pixel_um
            x_label = f"{xlabel} (object-space um)"
        ax.plot(plot_x, profile, "o", markersize=3, label="measured profile")
        if fit["conv_fit_success"]:
            x_dense = np.linspace(x_axis[0], x_axis[-1], 1200)
            plot_x_dense = x_dense
            if calibration.object_pixel_um is not None and calibration.object_pixel_um > 0:
                plot_x_dense = x_dense * calibration.object_pixel_um
            ideal_profile_dense = (
                fit["conv_background_offset"]
                + fit["conv_background_slope_per_px"] * x_dense
                + fit["conv_amplitude"]
                * (np.abs(x_dense - fit["conv_x0_px"]) < 0.5 * fit["conv_w_px"]).astype(float)
            )
            fit_profile_dense = (
                fit["conv_background_offset"]
                + fit["conv_background_slope_per_px"] * x_dense
                + fit["conv_amplitude"]
                * blurred_bar_profile(
                    x_dense,
                    fit["conv_x0_px"],
                    fit["conv_w_px"],
                    fit["conv_sigma_px"],
                )
            )
            ax.plot(
                plot_x_dense,
                ideal_profile_dense,
                ":",
                color="0.35",
                label="unblurred square",
            )
            ax.plot(
                plot_x_dense,
                fit_profile_dense,
                "-",
                color="tab:orange",
                linewidth=2.0,
                label="Gaussian fit",
            )
            sigma_key = "conv_sigma_um" if np.isfinite(fit["conv_sigma_um"]) else "conv_sigma_px"
            unc_key = (
                "conv_sigma_uncertainty_um"
                if sigma_key == "conv_sigma_um"
                else "conv_sigma_uncertainty_px"
            )
            unit = "um" if sigma_key == "conv_sigma_um" else "px"
            res_key = (
                "conv_rayleigh_equivalent_resolution_um"
                if unit == "um"
                else "conv_rayleigh_equivalent_resolution_px"
            )
            res_unc_key = (
                "conv_rayleigh_equivalent_resolution_uncertainty_um"
                if unit == "um"
                else "conv_rayleigh_equivalent_resolution_uncertainty_px"
            )
            sigma_text = format_value_with_uncertainty(fit[sigma_key], fit[unc_key])
            resolution_text = format_value_with_uncertainty(fit[res_key], fit[res_unc_key])
            ax.set_title(
                f"{label}: sigma={sigma_text} {unit}, "
                f"resolution={resolution_text} {unit}"
            )
        else:
            ax.set_title(f"{label} fit failed: {fit['conv_fit_message']}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("intensity counts")
        ax.legend(loc="best")

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        0.9,
        "Gaussian square fit\n"
        f"{format_usaf_label(group, square=True)}\n"
        f"fixed side = {square_side_um:.4g} um\n"
        "Rayleigh-equivalent resolution = 2.898785 sigma",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "square_psf_fit.png"), dpi=200)
    plt.close(fig)

    print("\nSquare analysis result:")
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
        help=(
            "ROI as x,y,w,h. For square mode this is a rough search zone that "
            "will be auto-cropped unless --manual-roi is used."
        ),
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
        help=(
            "Pixel size in um. If --magnification is omitted, this is treated "
            "as object-space um/image pixel; otherwise it is camera pixel size."
        ),
    )
    parser.add_argument(
        "--object-pixel-size-um",
        type=float,
        default=None,
        help=(
            "Direct calibrated object-space pixel size in um/image pixel. "
            "Overrides --pixel-size-um/--magnification."
        ),
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
        "--psf-model",
        choices=["gaussian", "airy", "both"],
        default="gaussian",
        help="PSF model for the real-space three-bar fit.",
    )
    p_stripe.add_argument(
        "--convolution-width-px",
        type=float,
        default=None,
        help="Known bar width in pixels for the PSF fit; if set, width is fixed.",
    )
    p_stripe.add_argument(
        "--fix-convolution-width",
        action="store_true",
        help=(
            "Fix PSF-fit bar width to the available estimate. Nominal USAF width "
            "is already fixed automatically when group/element and calibration are known."
        ),
    )
    p_stripe.add_argument(
        "--no-auto-trim-profile-band",
        action="store_true",
        help="Disable automatic trimming of the stripe-direction averaging band.",
    )
    p_stripe.set_defaults(auto_rotate=True)
    p_stripe.add_argument(
        "--auto-rotate",
        dest="auto_rotate",
        action="store_true",
        help="Automatically estimate and correct small ROI rotation before fitting.",
    )
    p_stripe.add_argument(
        "--no-auto-rotate",
        dest="auto_rotate",
        action="store_false",
        help="Disable automatic ROI rotation correction.",
    )
    p_stripe.add_argument(
        "--auto-rotate-max-deg",
        type=float,
        default=8.0,
        help="Maximum absolute angle searched by automatic rotation correction.",
    )
    p_stripe.add_argument(
        "--auto-rotate-step-deg",
        type=float,
        default=0.5,
        help="Angle step for automatic rotation correction.",
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
        "--psf-model",
        choices=["gaussian", "airy", "both"],
        default="gaussian",
        help="PSF model for the real-space three-bar fit.",
    )
    p_auto.add_argument(
        "--convolution-width-px",
        type=float,
        default=None,
        help="Known bar width in pixels for the PSF fit; if set, width is fixed for every ROI.",
    )
    p_auto.add_argument(
        "--fix-convolution-width",
        action="store_true",
        help=(
            "Fix PSF-fit bar width to the detected estimate when no nominal USAF "
            "width is available. Nominal USAF width is fixed automatically."
        ),
    )
    p_auto.add_argument(
        "--no-auto-trim-profile-band",
        action="store_true",
        help="Disable automatic trimming of the stripe-direction averaging band.",
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
        default=1.5,
        help=(
            "Reject triplets whose total center span is too large compared "
            "with individual bar length. Lower values are stricter."
        ),
    )

    # Auto-squares mode.
    p_auto_square = subparsers.add_parser(
        "auto-squares",
        help="Automatically detect and analyze multiple USAF square targets.",
    )
    add_common_args(p_auto_square)
    p_auto_square.add_argument("--group", type=int, default=None, help="USAF group number.")
    p_auto_square.add_argument(
        "--element",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p_auto_square.add_argument(
        "--lpmm",
        type=float,
        default=None,
        help="Spatial frequency in lp/mm. Overrides group/element if provided.",
    )
    p_auto_square.add_argument(
        "--polarity",
        choices=["dark", "bright", "auto"],
        default="dark",
        help=(
            "Square polarity. Use dark for dark squares on bright background; "
            "bright for bright squares on dark background; auto to choose automatically."
        ),
    )
    p_auto_square.add_argument(
        "--min-area",
        type=int,
        default=20,
        help="Minimum connected-component area for a square candidate.",
    )
    p_auto_square.add_argument(
        "--max-aspect-error",
        type=float,
        default=0.35,
        help="Maximum allowed deviation of square component aspect ratio from 1.",
    )
    p_auto_square.add_argument(
        "--min-fill-fraction",
        type=float,
        default=0.25,
        help="Minimum foreground fill fraction inside a candidate bounding box.",
    )
    p_auto_square.add_argument(
        "--max-squares",
        type=int,
        default=50,
        help="Maximum number of detected square targets to analyze.",
    )
    p_auto_square.add_argument(
        "--bg-sigma",
        type=float,
        default=40.0,
        help="Gaussian sigma for slow-background normalization.",
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
        help="Analyze one USAF square ROI with fixed side length and Gaussian PSF fits.",
    )
    add_common_args(p_square)

    p_square.add_argument(
        "--group",
        type=int,
        default=None,
        help="USAF group number.",
    )
    p_square.add_argument(
        "--element",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p_square.add_argument(
        "--lpmm",
        type=float,
        default=None,
        help="Spatial frequency in lp/mm. Overrides group/element if provided.",
    )
    p_square.add_argument(
        "--auto-roi",
        action="store_true",
        help=(
            "Deprecated: square mode now auto-crops the best square inside the "
            "rough selected ROI by default."
        ),
    )
    p_square.add_argument(
        "--manual-roi",
        action="store_true",
        help="Use the manual/interactive ROI selector instead of automatic square ROI detection.",
    )
    p_square.add_argument(
        "--polarity",
        choices=["dark", "bright", "auto"],
        default="dark",
        help=argparse.SUPPRESS,
    )
    p_square.add_argument(
        "--min-area",
        type=int,
        default=20,
        help=argparse.SUPPRESS,
    )
    p_square.add_argument(
        "--max-aspect-error",
        type=float,
        default=0.35,
        help=argparse.SUPPRESS,
    )
    p_square.add_argument(
        "--min-fill-fraction",
        type=float,
        default=0.25,
        help=argparse.SUPPRESS,
    )
    p_square.add_argument(
        "--bg-sigma",
        type=float,
        default=40.0,
        help=argparse.SUPPRESS,
    )
    p_square.set_defaults(auto_rotate=True)
    p_square.add_argument(
        "--auto-rotate",
        dest="auto_rotate",
        action="store_true",
        help="Automatically estimate and correct small square rotation before autocropping/fitting.",
    )
    p_square.add_argument(
        "--no-auto-rotate",
        dest="auto_rotate",
        action="store_false",
        help="Disable automatic square rotation correction.",
    )
    p_square.add_argument(
        "--auto-rotate-max-deg",
        type=float,
        default=8.0,
        help="Maximum absolute angle searched by automatic rotation correction.",
    )
    p_square.add_argument(
        "--auto-rotate-step-deg",
        type=float,
        default=0.5,
        help="Angle step for automatic rotation correction.",
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
        object_pixel_size_um=args.object_pixel_size_um,
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
            angle_deg=args.angle_deg,
            convolution_fixed_width_px=args.convolution_width_px,
            fix_convolution_width=args.fix_convolution_width,
            psf_model=args.psf_model,
            auto_trim_profile_band=not args.no_auto_trim_profile_band,
            auto_rotate=args.auto_rotate,
            auto_rotate_max_deg=args.auto_rotate_max_deg,
            auto_rotate_step_deg=args.auto_rotate_step_deg,
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
            min_area=args.min_area,
            min_aspect=args.min_aspect,
            max_candidates_per_orientation=args.max_candidates_per_orientation,
            max_triplets=args.max_triplets,
            bg_sigma=args.bg_sigma,
            max_bars_per_roi=args.max_bars_per_roi,
            max_span_to_length=args.max_span_to_length,
            convolution_fixed_width_px=args.convolution_width_px,
            fix_convolution_width=args.fix_convolution_width,
            psf_model=args.psf_model,
            auto_trim_profile_band=not args.no_auto_trim_profile_band,
        )

    elif args.mode == "auto-squares":
        # Optional --roi restricts the automatic search region.
        if args.roi is not None:
            search_roi = parse_roi(args.roi)
            work_img = crop_roi(img, search_roi, angle_deg=args.angle_deg)
            print(f"Auto-squares search restricted to ROI: {search_roi}")
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

        analyze_auto_squares(
            img=work_img,
            outdir=args.outdir,
            group=args.group,
            element=args.element,
            lpmm=args.lpmm,
            calibration=calibration,
            polarity=args.polarity,
            min_area=args.min_area,
            max_aspect_error=args.max_aspect_error,
            min_fill_fraction=args.min_fill_fraction,
            max_squares=args.max_squares,
            bg_sigma=args.bg_sigma,
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
        if not args.manual_roi:
            search_roi = get_roi(
                img,
                args.roi,
                title="Select rough square search zone",
            )
            work_img = crop_roi(img, search_roi, angle_deg=args.angle_deg)
            print(f"Square auto-crop search zone: {search_roi}")

            auto_rotation_angle_deg = 0.0
            auto_rotation_score = np.nan
            if args.auto_rotate:
                auto_rotation_angle_deg, auto_rotation_score = estimate_square_rotation_angle(
                    work_img,
                    max_angle_deg=args.auto_rotate_max_deg,
                    step_deg=args.auto_rotate_step_deg,
                )
                work_img = _rotate_crop_for_analysis(work_img, auto_rotation_angle_deg)
                print(
                    "Auto square rotation: "
                    f"{auto_rotation_angle_deg:.4g} deg "
                    f"(score={auto_rotation_score:.4g})"
                )

            lpmm = args.lpmm
            if lpmm is None and args.group is not None:
                lpmm = usaf_square_lpmm(args.group)
            square_side_um = nominal_usaf_square_side_um_from_lpmm(lpmm)
            if not np.isfinite(square_side_um):
                raise ValueError(
                    "square mode requires --group or --lpmm so the standard "
                    "USAF square side length is known."
                )
            expected_side_px = nominal_usaf_square_side_px_from_lpmm(lpmm, calibration)

            ensure_dir(args.outdir)
            roi, autocrop_diag = autocrop_square_roi_from_projection(
                work_img,
                expected_side_px=expected_side_px if np.isfinite(expected_side_px) else None,
            )
            calibration_for_square = calibration
            if calibration.object_pixel_um is None:
                measured_side_px = autocrop_diag["measured_side_px"]
                if not np.isfinite(measured_side_px) or measured_side_px <= 0:
                    raise RuntimeError("Could not infer calibration from square projection.")
                inferred_object_pixel_um = square_side_um / measured_side_px
                calibration_for_square = calibration_with_object_pixel_um(
                    calibration,
                    inferred_object_pixel_um,
                )
                print(
                    "Inferred object-space pixel size from group square: "
                    f"{inferred_object_pixel_um:.6g} um/pixel"
                )
            plot_projection_autocrop(
                work_img,
                roi,
                autocrop_diag,
                os.path.join(args.outdir, "auto_detected_square_roi.png"),
            )
            img_for_square = work_img
            angle_for_square = 0.0
            reported_angle_for_square = args.angle_deg + auto_rotation_angle_deg
            reported_auto_rotation_for_square = auto_rotation_angle_deg
            reported_auto_rotation_score_for_square = auto_rotation_score
            print(f"Auto-cropped square ROI inside search zone: {roi}")
            print(
                "Projection edge span: "
                f"width={autocrop_diag['measured_width_px']:.4g} px, "
                f"height={autocrop_diag['measured_height_px']:.4g} px"
            )
        else:
            roi = get_roi(img, args.roi, title="Select full square ROI")
            img_for_square = img
            angle_for_square = args.angle_deg
            reported_angle_for_square = None
            reported_auto_rotation_for_square = 0.0
            reported_auto_rotation_score_for_square = np.nan
            calibration_for_square = calibration
            lpmm = args.lpmm
            if lpmm is None and args.group is not None:
                lpmm = usaf_square_lpmm(args.group)
            square_side_um = nominal_usaf_square_side_um_from_lpmm(lpmm)
            if calibration.object_pixel_um is None and np.isfinite(square_side_um):
                manual_crop = crop_roi(img_for_square, roi, angle_deg=angle_for_square)
                _manual_roi, manual_diag = autocrop_square_roi_from_projection(
                    manual_crop,
                    expected_side_px=None,
                )
                measured_side_px = manual_diag["measured_side_px"]
                if np.isfinite(measured_side_px) and measured_side_px > 0:
                    inferred_object_pixel_um = square_side_um / measured_side_px
                    calibration_for_square = calibration_with_object_pixel_um(
                        calibration,
                        inferred_object_pixel_um,
                    )
                    print(
                        "Inferred object-space pixel size from group square: "
                        f"{inferred_object_pixel_um:.6g} um/pixel"
                    )

        analyze_square_roi(
            img=img_for_square,
            roi=roi,
            outdir=args.outdir,
            group=args.group,
            element=args.element,
            lpmm=args.lpmm,
            calibration=calibration_for_square,
            angle_deg=angle_for_square,
            auto_rotate=args.auto_rotate and args.manual_roi,
            auto_rotate_max_deg=args.auto_rotate_max_deg,
            auto_rotate_step_deg=args.auto_rotate_step_deg,
            reported_angle_deg=reported_angle_for_square,
            reported_auto_rotation_angle_deg=reported_auto_rotation_for_square,
            reported_auto_rotation_score=reported_auto_rotation_score_for_square,
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
