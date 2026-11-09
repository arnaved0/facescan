"""Analytic test head (signed distance function), in head space, metres.

Head space matches ARKit's face anchor: +y up, +z out of the face, origin
near the centre of the head. Used by make_synthetic_scan.py and
evaluate_synthetic.py to test the pipeline without an iPhone.
"""

from __future__ import annotations

import numpy as np


def _ellipsoid(p, c, r):
    q = (p - c) / r
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm((p - c) / (r * r), axis=-1)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-9)


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def _smax(a, b, k):
    return -_smin(-a, -b, k)


def _capsule_y(p, c, r, y0, y1):
    q = p - c
    y = np.clip(q[:, 1], y0, y1)
    q = q - np.stack([np.zeros_like(y), y, np.zeros_like(y)], axis=1)
    return np.linalg.norm(q, axis=1) - r


def head_sdf(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    d = _ellipsoid(p, np.array([0, 0.02, -0.01]), np.array([0.075, 0.10, 0.095]))
    d = _smin(d, _ellipsoid(p, np.array([0, -0.045, 0.025]), np.array([0.058, 0.06, 0.065])), 0.02)
    d = _smin(d, _ellipsoid(p, np.array([0, -0.005, 0.083]), np.array([0.011, 0.024, 0.022])), 0.008)
    for sx in (-1, 1):
        d = _smin(d, _ellipsoid(p, np.array([sx * 0.077, 0.0, -0.012]), np.array([0.008, 0.028, 0.017])), 0.006)
        # eye sockets
        eye = np.linalg.norm(p - np.array([sx * 0.032, 0.028, 0.084]), axis=1) - 0.013
        d = _smax(d, -eye, 0.006)
    d = _smin(d, _capsule_y(p, np.array([0, 0, -0.02]), 0.05, -0.25, -0.06), 0.02)
    return d


def sdf_gradient(p: np.ndarray, h: float = 1e-5) -> np.ndarray:
    g = np.zeros_like(p)
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        g[:, i] = (head_sdf(p + e) - head_sdf(p - e)) / (2 * h)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
