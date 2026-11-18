"""Isotropic content scale of the box interior between causal event-count images (pure numpy).

For a query with current box region ``R`` (continuous pixel coordinates, clipped to the sensor)
the event count image of the window ``[t - tau, t)`` is compared with the images of the windows
ending ``lag`` earlier.  For each lag the log scale ``rho`` is searched such that

    E_now(x) ~= E_prev(c_prev + (x - c_now) / exp(rho) + d)

i.e. the box interior content is ``exp(rho)`` times larger now.  Magnifying by ``s`` about a
centre is the same as area-binning a region ``s`` times smaller into the same grid, so each scale
candidate costs one lookup in the integral image of the earlier window.  The translation ``d`` is
searched exhaustively by normalized cross-correlation (FFT) of the central template.  Nothing
here reads box edges as a scale signal: the box only selects the region and the centre prior.

Under constant closing speed ``exp(rho(lag)) = 1 + k * lag`` with ``k = 1/TTC``, so the content
expansion rate is the score-weighted least-squares slope of ``exp(rho) - 1`` on ``lag`` through
the origin (the analogue of the inverse-size fit of ``geometry_robust``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from operational.phase_geometry import geometry

FloatArray = NDArray[np.float64]

SENSOR_WIDTH = 1280
SENSOR_HEIGHT = 720
TAU_US = 30_000
LAGS_US = (100_000, 200_000, 400_000, 600_000)
SHORT_LAG_MAX_US = 200_000
GRID = 64
TEMPLATE = 40
K_SEARCH = (-1.0, 3.0)
COARSE_STEP = 0.04
FINE_STEP = 0.01
MIN_SCORE = 0.15
MIN_TEMPLATE_EVENTS = 200
LOG_EVENTS_NORMALIZER = 6.0
PHASE_CLIP = geometry.LEARNED_PHASE_CLIP
PHASE_SCALE = geometry.PHASE_SCALE

RAW_C_FIELDS = (
    "k_content",
    "phi_content",
    "k_short",
    "phi_short",
    "n_valid",
    "mean_score",
    "mean_gain",
    "rms_fit",
    "log_events",
    "valid",
    "short_valid",
    *(f"rho_{lag // 1000}" for lag in LAGS_US),
    *(f"score_{lag // 1000}" for lag in LAGS_US),
    *(f"lag_valid_{lag // 1000}" for lag in LAGS_US),
)
C_FIELDS = (
    "phi_content",
    "phi_short",
    "n_valid",
    "mean_score",
    "mean_gain",
    "rms_fit",
    "log_events",
    "valid",
    "short_valid",
)


# ------------------------------------------------------------------------- event images


def integral_image(
    x: NDArray[Any], y: NDArray[Any], width: int = SENSOR_WIDTH, height: int = SENSOR_HEIGHT
) -> NDArray[np.int64]:
    """Integral ``I[j, i]`` = number of events with ``row < j`` and ``col < i`` (``[H+1, W+1]``).

    Polarities are summed; events outside the sensor are rejected.
    """
    xi = np.asarray(x, dtype=np.int64)
    yi = np.asarray(y, dtype=np.int64)
    keep = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    counts = np.bincount(yi[keep] * width + xi[keep], minlength=width * height)
    out = np.zeros((height + 1, width + 1), dtype=np.int64)
    out[1:, 1:] = counts.reshape(height, width).cumsum(0).cumsum(1)
    return out


def _lattice(integral: NDArray[Any], xs: NDArray[Any], ys: NDArray[Any]) -> FloatArray:
    """Bilinear interpolation of the integral image at ``ys[..., :, None] x xs[..., None, :]``.

    ``xs`` ``[..., A]`` and ``ys`` ``[..., B]`` are continuous coordinates; the result is the exact
    event mass of ``[0, x) x [0, y)`` when events are spread uniformly inside their pixel.
    """
    height, width = integral.shape[0] - 1, integral.shape[1] - 1
    xc = np.clip(np.asarray(xs, dtype=np.float64), 0.0, float(width))
    yc = np.clip(np.asarray(ys, dtype=np.float64), 0.0, float(height))
    ix = np.minimum(np.floor(xc).astype(np.int64), width - 1)
    iy = np.minimum(np.floor(yc).astype(np.int64), height - 1)
    fx = (xc - ix)[..., None, :]
    fy = (yc - iy)[..., :, None]
    iyc, ixc = iy[..., :, None], ix[..., None, :]
    i00 = integral[iyc, ixc]
    i01 = integral[iyc, ixc + 1]
    i10 = integral[iyc + 1, ixc]
    i11 = integral[iyc + 1, ixc + 1]
    return (1 - fy) * ((1 - fx) * i00 + fx * i01) + fy * ((1 - fx) * i10 + fx * i11)


def bin_regions(
    integral: NDArray[Any], centers: NDArray[Any], half_sizes: NDArray[Any], bins: int
) -> FloatArray:
    """Event mass of ``K`` rectangles (``centre +- half_size``) on a ``bins x bins`` grid each.

    ``centers`` and ``half_sizes`` are ``[K, 2]`` in (x, y) order; returns ``[K, bins, bins]``.
    """
    c = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    h = np.asarray(half_sizes, dtype=np.float64).reshape(-1, 2)
    edges = np.linspace(-1.0, 1.0, bins + 1)
    xs = c[:, :1] + h[:, :1] * edges[None, :]
    ys = c[:, 1:] + h[:, 1:] * edges[None, :]
    s = _lattice(integral, xs, ys)
    return np.maximum(s[:, 1:, 1:] - s[:, :-1, 1:] - s[:, 1:, :-1] + s[:, :-1, :-1], 0.0)


def prepare(mass: NDArray[Any]) -> FloatArray:
    """Square-root (Poisson-stabilizing, gain-covariant) and a separable ``[1,4,6,4,1]`` blur."""
    v = np.sqrt(np.maximum(np.asarray(mass, dtype=np.float64), 0.0))
    for axis in (-1, -2):
        n = v.shape[axis]
        pad = [(0, 0)] * v.ndim
        pad[axis] = (2, 2)
        p = np.pad(v, pad, mode="edge")
        parts = [np.take(p, np.arange(i, i + n), axis=axis) for i in range(5)]
        v = (parts[0] + 4 * parts[1] + 6 * parts[2] + 4 * parts[3] + parts[4]) / 16.0
    return v


# ------------------------------------------------------------------------------ matching


@dataclass(frozen=True)
class Template:
    """Zero-mean template spectrum for FFT normalized cross-correlation."""

    spectrum: NDArray[np.complex128]
    norm: float
    size: int
    events: float


def make_template(values: NDArray[Any], events: float, grid: int = GRID) -> Template:
    """Template from a prepared ``[t, t]`` image (``norm`` 0 marks an unusable template)."""
    t = np.asarray(values, dtype=np.float64)
    t0 = t - t.mean()
    norm = float(np.sqrt((t0 * t0).sum()))
    spectrum = np.fft.rfft2(t0, s=(grid, grid))
    return Template(spectrum=spectrum, norm=norm, size=t.shape[0], events=float(events))


def ncc_maps(images: NDArray[Any], template: Template) -> FloatArray:
    """Normalized cross-correlation of ``[K, G, G]`` images with the template at every valid
    offset; returns ``[K, G-t+1, G-t+1]`` (0 where the image window has no variance)."""
    img = np.asarray(images, dtype=np.float64)
    k, g = img.shape[0], img.shape[-1]
    t = template.size
    n = t * t
    span = g - t + 1
    corr = np.fft.irfft2(np.fft.rfft2(img) * np.conj(template.spectrum)[None], s=(g, g))
    corr = corr[:, :span, :span]
    c1 = np.zeros((k, g + 1, g + 1))
    c2 = np.zeros((k, g + 1, g + 1))
    c1[:, 1:, 1:] = img.cumsum(1).cumsum(2)
    c2[:, 1:, 1:] = (img * img).cumsum(1).cumsum(2)

    def window(c: FloatArray) -> FloatArray:
        return c[:, t:, t:] - c[:, :span, t:] - c[:, t:, :span] + c[:, :span, :span]

    s1, s2 = window(c1), window(c2)
    var = np.maximum(s2 - s1 * s1 / n, 0.0)
    den = template.norm * np.sqrt(var)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-9 * max(template.norm, 1e-12), corr / den, 0.0)
    return np.clip(out, -1.0, 1.0)


def rho_bounds(lag_s: float) -> tuple[float, float]:
    """Searched log-scale range for a lag: ``log(1 + k * lag)`` for ``k`` in ``K_SEARCH``."""
    lo = math.log(max(1.0 + K_SEARCH[0] * lag_s, 0.05))
    hi = math.log(1.0 + K_SEARCH[1] * lag_s)
    return lo, hi


@dataclass(frozen=True)
class LagEstimate:
    """Best log scale of one lag, its NCC score, the score at ``rho = 0`` and flags."""

    rho: float
    score: float
    score_zero: float
    boundary: bool
    usable: bool


def _scores(
    prev_integral: NDArray[Any],
    template: Template,
    center_prev: NDArray[Any],
    half_region: NDArray[Any],
    rhos: NDArray[Any],
    grid: int,
) -> FloatArray:
    half = np.asarray(half_region, dtype=np.float64)[None, :] / np.exp(np.asarray(rhos))[:, None]
    centers = np.repeat(np.asarray(center_prev, dtype=np.float64)[None, :], len(rhos), axis=0)
    images = prepare(bin_regions(prev_integral, centers, half, grid))
    return ncc_maps(images, template).reshape(len(rhos), -1).max(axis=1)


def estimate_lag(
    prev_integral: NDArray[Any],
    template: Template,
    center_prev: NDArray[Any],
    half_region: NDArray[Any],
    lag_s: float,
    grid: int = GRID,
) -> LagEstimate:
    """Coarse grid (``COARSE_STEP``), fine grid (``FINE_STEP``) and a parabolic refinement."""
    if template.norm <= 0.0 or template.events < MIN_TEMPLATE_EVENTS:
        return LagEstimate(0.0, 0.0, 0.0, False, False)
    lo, hi = rho_bounds(lag_s)
    coarse = np.arange(lo, hi + 1e-12, COARSE_STEP)
    rhos = np.concatenate((coarse, [0.0]))
    scores = _scores(prev_integral, template, center_prev, half_region, rhos, grid)
    score_zero = float(scores[-1])
    coarse_scores = scores[:-1]
    best = int(np.argmax(coarse_scores))
    boundary = best in (0, len(coarse) - 1)
    fine = coarse[best] + np.arange(-4, 5) * FINE_STEP
    fine = fine[(fine >= lo - 1e-12) & (fine <= hi + 1e-12)]
    fine_scores = _scores(prev_integral, template, center_prev, half_region, fine, grid)
    j = int(np.argmax(fine_scores))
    rho = float(fine[j])
    if 0 < j < len(fine) - 1:
        a, b, c = fine_scores[j - 1], fine_scores[j], fine_scores[j + 1]
        curvature = a - 2 * b + c
        if curvature < 0:
            rho += float(np.clip(0.5 * (a - c) / curvature, -0.5, 0.5)) * FINE_STEP
    score = float(fine_scores[j])
    usable = (score >= MIN_SCORE) and not boundary
    return LagEstimate(rho, score, score_zero, boundary, usable)


# ------------------------------------------------------------------------------- per query


@dataclass(frozen=True)
class Region:
    """Continuous half-open rectangle ``[x0, x1) x [y0, y1)``."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def empty(self) -> bool:
        """True when the rectangle has no area (or is not finite)."""
        values = (self.x0, self.y0, self.x1, self.y1)
        return not all(math.isfinite(v) for v in values) or self.x1 <= self.x0 or self.y1 <= self.y0

    @property
    def center(self) -> FloatArray:
        """Centre ``(x, y)``."""
        return np.asarray([(self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0])

    @property
    def half(self) -> FloatArray:
        """Half sizes ``(w/2, h/2)``."""
        return np.asarray([(self.x1 - self.x0) / 2.0, (self.y1 - self.y0) / 2.0])

    def intersect(self, other: Region) -> Region:
        """Intersection (possibly empty)."""
        return Region(
            max(self.x0, other.x0),
            max(self.y0, other.y0),
            min(self.x1, other.x1),
            min(self.y1, other.y1),
        )


SENSOR = Region(0.0, 0.0, float(SENSOR_WIDTH), float(SENSOR_HEIGHT))


def box_region(box: Sequence[float] | NDArray[Any], frame: Region = SENSOR) -> Region:
    """Pixel-inclusive xyxy box as a continuous region ``[x1, x2+1) x [y1, y2+1)`` in the frame."""
    b = np.asarray(box, dtype=np.float64).reshape(4)
    return Region(float(b[0]), float(b[1]), float(b[2]) + 1.0, float(b[3]) + 1.0).intersect(frame)


def crop_region(crop: Sequence[float] | NDArray[Any]) -> Region:
    """Original-image region seen by a ``geometry_robust`` virtual crop ``(x0, y0, zoom)``."""
    x0, y0, z = (float(v) for v in np.asarray(crop, dtype=np.float64).reshape(3))
    return Region(x0, y0, x0 + SENSOR_WIDTH / z, y0 + SENSOR_HEIGHT / z).intersect(SENSOR)


def empty_raw() -> dict[str, float]:
    """All-zero raw content fields (no estimate)."""
    return {name: 0.0 for name in RAW_C_FIELDS}


def query_content(
    current: NDArray[Any] | None,
    previous: Mapping[int, NDArray[Any] | None],
    region: Region,
    prev_centers: Mapping[int, NDArray[Any] | None],
    grid: int = GRID,
    template_bins: int = TEMPLATE,
) -> dict[str, float]:
    """Raw content-scale fields (``RAW_C_FIELDS``) for one query.

    ``current``/``previous[lag]`` are integral images of the windows ending at the query and
    ``lag`` microseconds earlier (``None`` if unavailable); ``prev_centers[lag]`` is the centre
    prior in the earlier window (``None`` -> the current region centre).
    """
    out = empty_raw()
    if current is None or region.empty:
        return out
    center, half = region.center, region.half
    fraction = template_bins / grid
    template_half = half * fraction
    t_mass = bin_regions(current, center[None], template_half[None], template_bins)[0]
    template = make_template(prepare(t_mass), float(t_mass.sum()), grid)
    out["log_events"] = math.log10(1.0 + template.events) / LOG_EVENTS_NORMALIZER
    estimates: dict[int, LagEstimate] = {}
    for lag in LAGS_US:
        prev = previous.get(lag)
        if prev is None:
            continue
        c_prev = prev_centers.get(lag)
        c = center if c_prev is None else np.asarray(c_prev, dtype=np.float64)
        estimates[lag] = estimate_lag(prev, template, c, half, lag * 1e-6, grid)
    for lag, e in estimates.items():
        tag = lag // 1000
        out[f"rho_{tag}"] = e.rho
        out[f"score_{tag}"] = e.score
        out[f"lag_valid_{tag}"] = float(e.usable)
    usable = {lag: e for lag, e in estimates.items() if e.usable}
    if usable:
        out["n_valid"] = float(len(usable))
        out["mean_score"] = float(np.mean([e.score for e in usable.values()]))
        out["mean_gain"] = float(np.mean([e.score - e.score_zero for e in usable.values()]))
        k, rms = fit_rate(usable)
        out["k_content"], out["rms_fit"], out["valid"] = k, rms, 1.0
        out["phi_content"] = float(geometry.slope_to_phase(k))
        short = {lag: e for lag, e in usable.items() if lag <= SHORT_LAG_MAX_US}
        if short:
            k_short, _ = fit_rate(short)
            out["k_short"], out["short_valid"] = k_short, 1.0
            out["phi_short"] = float(geometry.slope_to_phase(k_short))
    return out


def fit_rate(estimates: Mapping[int, LagEstimate]) -> tuple[float, float]:
    """Score-weighted LS slope of ``exp(rho) - 1`` on lag (s) through the origin, and its RMS."""
    lags = np.asarray([lag * 1e-6 for lag in estimates], dtype=np.float64)
    y = np.expm1(np.asarray([e.rho for e in estimates.values()], dtype=np.float64))
    w = np.asarray([max(e.score, 1e-6) for e in estimates.values()], dtype=np.float64)
    k = float((w * lags * y).sum() / (w * lags * lags).sum())
    rms = float(np.sqrt((w * (y - k * lags) ** 2).sum() / w.sum()))
    return k, rms


def learned_c(raw: NDArray[Any]) -> NDArray[np.float32]:
    """Project a raw content table ``[N, len(RAW_C_FIELDS)]`` to the learned inputs C."""
    r = {name: np.asarray(raw, dtype=np.float64)[:, i] for i, name in enumerate(RAW_C_FIELDS)}
    columns = [
        np.clip(r["phi_content"], -PHASE_CLIP, PHASE_CLIP) / PHASE_SCALE,
        np.clip(r["phi_short"], -PHASE_CLIP, PHASE_CLIP) / PHASE_SCALE,
        r["n_valid"] / len(LAGS_US),
        r["mean_score"],
        r["mean_gain"],
        np.clip(10.0 * r["rms_fit"], 0.0, 5.0),
        r["log_events"],
        r["valid"],
        r["short_valid"],
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def raw_c_column(raw: NDArray[Any], name: str) -> FloatArray:
    """One named raw content column."""
    return np.asarray(raw, dtype=np.float64)[:, RAW_C_FIELDS.index(name)]


def window_key_ms(end_us: int) -> int:
    """Causal window key: the window ends at ``floor(end_us / 1000) * 1000 <= end_us``."""
    return int(end_us) // 1000


__all__ = [
    "C_FIELDS",
    "LAGS_US",
    "RAW_C_FIELDS",
    "SENSOR",
    "TAU_US",
    "LagEstimate",
    "Region",
    "Template",
    "bin_regions",
    "box_region",
    "crop_region",
    "empty_raw",
    "estimate_lag",
    "fit_rate",
    "integral_image",
    "learned_c",
    "make_template",
    "ncc_maps",
    "prepare",
    "query_content",
    "raw_c_column",
    "rho_bounds",
    "window_key_ms",
]
