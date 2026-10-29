"""Loader + quick renderer for the 8i 'redandblack' sequence (vox10).

The dataset is NOT included in this repo (8i license forbids redistribution).
Get it from http://plenodb.jpeg.org/pc/8ilabs and point REDBLACK_DIR at the
folder that contains Ply/, or pass --data to the scripts.
"""
import os

import numpy as np
import pandas as pd

DEFAULT_DIR = os.environ.get(
    "REDBLACK_DIR",
    os.path.join(os.path.dirname(__file__), "..", "redandblack", "redandblack"))


def load_frame(frame=1450, data_dir=DEFAULT_DIR, cache_dir=None):
    """Return (P int64 Nx3, C uint8 Nx3) for one frame. Caches as .npz."""
    cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    npz = os.path.join(cache_dir, f"redandblack_{frame}.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        return z["P"], z["C"]
    path = os.path.join(data_dir, "Ply", f"redandblack_vox10_{frame:04d}.ply")
    with open(path, "rb") as f:
        n_header = 0
        for line in f:
            n_header += 1
            if line.strip() == b"end_header":
                break
    a = pd.read_csv(path, sep=" ", header=None, skiprows=n_header,
                    usecols=range(6)).to_numpy()
    P, C = a[:, :3].astype(np.int64), a[:, 3:6].astype(np.uint8)
    np.savez(npz, P=P, C=C)
    return P, C


def render(P, C, size=1024, view="front", bg=255, splat=1):
    """Orthographic z-buffer render (numpy only). y is up in 8i data.
    view: 'front' looks along -z, 'side' looks along -x. Returns HxWx3 uint8."""
    P = np.asarray(P)
    if view == "front":
        u, v, d = P[:, 0], P[:, 1], P[:, 2]
    else:
        u, v, d = P[:, 2], P[:, 1], -P[:, 0]
    img = np.full((size, size, 3), bg, np.uint8)
    zbuf = np.full((size, size), -np.inf)
    for dy in range(splat):
        for dx in range(splat):
            x = np.clip(u + dx, 0, size - 1)
            y = np.clip(size - 1 - v - dy, 0, size - 1)
            # nearest point (largest d) wins: sort so the last write is nearest
            o = np.argsort(d, kind="stable")
            xi, yi, di = x[o], y[o], d[o]
            better = di >= zbuf[yi, xi]
            xi, yi, di, co = xi[better], yi[better], di[better], C[o][better]
            zbuf[yi, xi] = di
            img[yi, xi] = co
    return img


def crop(imgs, bg=255, margin=8):
    """Crop a list of same-size renders to the union bounding box of content."""
    m = np.zeros(imgs[0].shape[:2], bool)
    for im in imgs:
        m |= (im != bg).any(-1)
    ys, xs = np.nonzero(m)
    y0, y1 = max(ys.min() - margin, 0), ys.max() + margin
    x0, x1 = max(xs.min() - margin, 0), xs.max() + margin
    return [im[y0:y1, x0:x1] for im in imgs]


# V-PCC settings for 10-bit 8i data (close to TMC2's lossy configuration)
RATES = {               # rate point: (geometry JPEG q, texture JPEG q)
    "R1": (25, 8), "R2": (50, 20), "R3": (70, 40), "R4": (85, 60), "R5": (95, 80)}


def vpcc_config(rate="R3", **kw):
    from vpcc import Config
    gq, tq = RATES[rate]
    base = dict(bits=10, block=8, canvas_width=1280, min_patch_points=8,
                max_passes=6, occ_precision=4, raw_points=False,
                geo_quality=gq, tex_quality=tq)
    base.update(kw)
    return Config(**base)


def mbps(n_bytes, fps=30):
    """Bitrate if every frame cost n_bytes, at the paper's 30 fps."""
    return n_bytes * 8 * fps / 1e6
