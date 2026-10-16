"""
Occlusion simulation - the core of the "occlusion-robust" claim.

Standard segmentation training teaches a model to find visible road pixels.
If you want a model that infers road *continuity* under a tree canopy, you must
show it images where the road is hidden but the label still says "road".
That is exactly what this module manufactures: the occluder is painted onto the
image only, never onto the mask.

Four occluder families, matching the failure modes named in the problem statement:
    tree canopy      - blobby, textured, green, hard-edged
    building shadow  - directional, low-luminance, sharp-edged polygon
    cloud / haze     - large, soft-edged, brightening + contrast-crushing
    vehicles         - small, high-contrast, on-road rectangles
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np


@dataclass
class OcclusionConfig:
    tree_prob: float = 0.7
    shadow_prob: float = 0.5
    cloud_prob: float = 0.25
    vehicle_prob: float = 0.6
    # Fraction of road pixels we are willing to hide in one sample.
    max_road_occlusion: float = 0.45
    sun_azimuth_deg: Tuple[float, float] = (0.0, 360.0)
    rng_seed: int | None = None
    _rng: np.random.Generator = field(default=None, repr=False)

    def rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.rng_seed)
        return self._rng


def _perlin_like(shape, rng, scale: int = 8) -> np.ndarray:
    """Cheap smooth noise field (upsampled white noise) for organic blob shapes."""
    small = rng.random((max(shape[0] // scale, 2), max(shape[1] // scale, 2))).astype(np.float32)
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def tree_canopy(shape, rng, coverage: float = 0.15) -> np.ndarray:
    """Soft blobby alpha map imitating a tree line or canopy patch."""
    field_ = _perlin_like(shape, rng, scale=rng.integers(6, 16))
    thresh = np.quantile(field_, 1.0 - coverage)
    alpha = np.clip((field_ - thresh) * 6.0, 0, 1)
    return cv2.GaussianBlur(alpha, (0, 0), sigmaX=2.0)


def building_shadow(shape, rng, azimuth_deg: float, n: int = 3) -> np.ndarray:
    """Directional hard-edged quadrilaterals cast off building footprints."""
    alpha = np.zeros(shape, np.float32)
    h, w = shape
    theta = np.deg2rad(azimuth_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    for _ in range(n):
        x, y = rng.integers(0, w), rng.integers(0, h)
        bw, bh = rng.integers(15, 60), rng.integers(15, 60)
        length = rng.integers(20, 90)
        pts = np.array(
            [
                [x, y],
                [x + bw, y],
                [x + bw + dx * length, y + bh + dy * length],
                [x + dx * length, y + bh + dy * length],
            ],
            np.int32,
        )
        cv2.fillPoly(alpha, [pts], 1.0)
    return cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.0)


def cloud(shape, rng, coverage: float = 0.2) -> np.ndarray:
    """Large soft alpha region - the only occluder that brightens rather than darkens."""
    field_ = _perlin_like(shape, rng, scale=rng.integers(20, 40))
    thresh = np.quantile(field_, 1.0 - coverage)
    alpha = np.clip((field_ - thresh) * 3.0, 0, 1)
    return cv2.GaussianBlur(alpha, (0, 0), sigmaX=8.0)


def vehicles(shape, rng, road_mask: np.ndarray, n: int = 25) -> np.ndarray:
    """Small rectangles placed *on* the road - the hardest occluder because it
    sits exactly where the positive evidence should be."""
    alpha = np.zeros(shape, np.float32)
    ys, xs = np.nonzero(road_mask)
    if len(ys) == 0:
        return alpha
    idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    for i in idx:
        y, x = int(ys[i]), int(xs[i])
        vh, vw = int(rng.integers(2, 5)), int(rng.integers(3, 8))
        cv2.rectangle(alpha, (x - vw // 2, y - vh // 2), (x + vw // 2, y + vh // 2), 1.0, -1)
    return alpha


def apply_occlusions(
    image: np.ndarray,
    road_mask: np.ndarray,
    cfg: OcclusionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (occluded_image, occlusion_alpha).

    The road mask is deliberately returned unchanged by the caller: the label
    still marks road under the occluder, which is what forces the network to
    learn contextual completion instead of pure appearance matching.
    """
    cfg = cfg or OcclusionConfig()
    rng = cfg.rng()
    img = image.astype(np.float32).copy()
    if img.ndim == 2:
        img = img[..., None]
    shape = img.shape[:2]
    total_alpha = np.zeros(shape, np.float32)

    if rng.random() < cfg.tree_prob:
        a = tree_canopy(shape, rng, coverage=float(rng.uniform(0.08, 0.28)))
        green = np.array([0.12, 0.30, 0.10], np.float32)[: img.shape[-1]]
        img = img * (1 - a[..., None] * 0.85) + a[..., None] * 0.85 * green
        total_alpha = np.maximum(total_alpha, a)

    if rng.random() < cfg.shadow_prob:
        az = float(rng.uniform(*cfg.sun_azimuth_deg))
        a = building_shadow(shape, rng, az, n=int(rng.integers(2, 6)))
        img = img * (1 - a[..., None] * 0.65)  # shadows darken, keep some detail
        total_alpha = np.maximum(total_alpha, a * 0.8)

    if rng.random() < cfg.cloud_prob:
        a = cloud(shape, rng, coverage=float(rng.uniform(0.1, 0.35)))
        img = img * (1 - a[..., None]) + a[..., None] * 0.92  # bright, low contrast
        total_alpha = np.maximum(total_alpha, a)

    if rng.random() < cfg.vehicle_prob:
        a = vehicles(shape, rng, road_mask, n=int(rng.integers(10, 40)))
        colors = rng.random((img.shape[-1],)).astype(np.float32)
        img = img * (1 - a[..., None]) + a[..., None] * colors
        total_alpha = np.maximum(total_alpha, a)

    # Safety valve: if we hid too much road, blend some of the clean image back.
    road = road_mask.astype(bool)
    if road.any():
        hidden = float((total_alpha[road] > 0.5).mean())
        if hidden > cfg.max_road_occlusion:
            keep = cfg.max_road_occlusion / hidden
            img = img * keep + image.astype(np.float32) * (1 - keep)
            total_alpha *= keep

    return np.clip(img, 0, 1), total_alpha


def occlusion_report(alpha: np.ndarray, road_mask: np.ndarray) -> dict:
    """Diagnostics used in the evaluation table: how much road was hidden, and
    therefore how much of the model's recall came from contextual inference."""
    road = road_mask.astype(bool)
    occ = alpha > 0.5
    return {
        "scene_occluded_frac": float(occ.mean()),
        "road_occluded_frac": float(occ[road].mean()) if road.any() else 0.0,
        "road_visible_px": int((road & ~occ).sum()),
        "road_hidden_px": int((road & occ).sum()),
    }
