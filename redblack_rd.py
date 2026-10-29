"""Rate-distortion comparison on 8i redandblack, in the style of the paper's
results: bitrate (Mbps at 30 fps) vs. D1 / D2 geometry PSNR and luma PSNR,
toy V-PCC vs. an octree+JPEG anchor (anchor.py).

    python redblack_rd.py                     # frames 1450 1500 1550
    python redblack_rd.py --frames 1450       # quicker
"""
import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import anchor
from redblack import RATES, crop, load_frame, mbps, render, vpcc_config
from vpcc import decode, encode, estimate_normals, psnr_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, nargs="+", default=[1450, 1500, 1550])
ap.add_argument("--out", default="out/redandblack")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# octree depth (9 = half resolution, 10 = lossless geometry), colour JPEG q
ANCHOR_RATES = {"A1": (7, 50), "A2": (8, 60), "A3": (9, 75), "A4": (9, 90),
                "A5": (10, 60), "A6": (10, 90)}

rows, decoded = [], {}
for f in args.frames:
    P, C = load_frame(f)
    normals, _ = estimate_normals(P.astype(float), 16)
    runs = [("V-PCC", r) for r in RATES] + [("anchor", r) for r in ANCHOR_RATES]
    for codec, r in runs:
        t = time.time()
        if codec == "V-PCC":
            cfg = vpcc_config(r)
            s, _ = encode(P, C, cfg)
            Q, D = decode(s, cfg)
        else:
            depth, q = ANCHOR_RATES[r]
            s, _ = anchor.encode(P, C, bits=10, depth=depth, quality=q)
            Q, D = anchor.decode(s)
        m = psnr_metrics(P, C, Q, D, 10, normals)
        rows.append(dict(frame=f, codec=codec, rate=r, bytes=len(s),
                         bpp=8 * len(s) / len(P), mbps=mbps(len(s)), **m))
        if f == args.frames[0]:
            decoded[(codec, r)] = (Q, D, mbps(len(s)), m)
        print(f"frame {f} {codec:6} {r}: {mbps(len(s)):6.1f} Mbps  "
              f"{8 * len(s) / len(P):5.2f} bpp  D1 {m['geo_psnr']:5.2f}  "
              f"D2 {m['d2_psnr']:5.2f}  Y {m['y_psnr']:5.2f}  ({time.time() - t:.0f}s)")

with open(f"{args.out}/rd_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)


def average(codec, rates):
    out = []
    for r in rates:
        sel = [x for x in rows if x["codec"] == codec and x["rate"] == r]
        out.append({k: np.mean([x[k] for x in sel])
                    for k in ("mbps", "geo_psnr", "d2_psnr", "y_psnr")})
    return out


V, A = average("V-PCC", RATES), average("anchor", ANCHOR_RATES)

# ------------------------------------------------------------ RD figure --
fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, key, label in zip(axs, ("geo_psnr", "d2_psnr", "y_psnr"),
                          ("D1 geometry PSNR (dB)", "D2 geometry PSNR (dB)",
                           "Luma (Y) PSNR (dB)")):
    for pts, names, style, lab in ((V, list(RATES), "s-", "toy V-PCC (2D video)"),
                                   (A, list(ANCHOR_RATES), "o--", "octree anchor (3D)")):
        fin = [p for p in pts if np.isfinite(p[key])]       # lossless = inf dB
        names = [n for n, p in zip(names, pts) if np.isfinite(p[key])]
        x, y = [p["mbps"] for p in fin], [p[key] for p in fin]
        ax.plot(x, y, style, label=lab)
        for xi, yi, n in zip(x, y, names):
            ax.annotate(n, (xi, yi), textcoords="offset points", xytext=(4, -10), fontsize=7)
    if key != "y_psnr":
        ax.text(0.98, 0.03, "A5, A6: lossless geometry (not shown)",
                transform=ax.transAxes, ha="right", fontsize=7, color="gray")
    ax.set_xscale("log")
    ax.set(xlabel="bitrate (Mbps @ 30 fps, log)", ylabel=label)
    ax.grid(alpha=0.3, which="both")
axs[0].legend()
fig.suptitle(f"redandblack, mean of frames {args.frames}  (data (c) 8i Labs)")
plt.tight_layout()
plt.savefig(f"{args.out}/rd_curves.png", dpi=130)


# Bitrate saving at equal quality: for each anchor point, what would V-PCC
# need for the same PSNR? (log-rate interpolation on V-PCC's curve)
def rate_at(curve, key, target):
    q = np.array([p[key] for p in curve]); r = np.log([p["mbps"] for p in curve])
    o = np.argsort(q)
    if not q.min() <= target <= q.max():
        return None
    return float(np.exp(np.interp(target, q[o], r[o])))


print("\nBitrate V-PCC needs to match each anchor point "
      "(saving > 0: V-PCC is cheaper; < 0: anchor is cheaper):")
for name, a in zip(ANCHOR_RATES, A):
    msg = f"  {name} ({a['mbps']:6.1f} Mbps):"
    for key, lab in (("geo_psnr", "D1"), ("y_psnr", "Y")):
        r = rate_at(V, key, a[key]) if np.isfinite(a[key]) else None
        msg += (f"   {lab} {a[key]:.1f} dB -> {r:6.1f} Mbps (saving {1 - r / a['mbps']:+.0%})"
                if r else f"   {lab} {a[key]:.1f} dB -> out of V-PCC range")
    print(msg)

# ------------------------------------------------- visual comparison figure --
P, C = load_frame(args.frames[0])
# pairs at similar bitrates: low (~30 Mbps) and high (~100 Mbps)
show = [("original", None), ("V-PCC", "R1"), ("anchor", "A4"),
        ("V-PCC", "R5"), ("anchor", "A6")]
full, titles = [], []
for codec, r in show:
    if codec == "original":
        full.append(render(P, C)); titles.append("original\n1181 Mbps (uncompressed)")
    else:
        Q, D, rate, m = decoded[(codec, r)]
        # draw each point as big as its voxel (anchor at depth 9 = 2x2 voxels)
        splat = 1 << (10 - ANCHOR_RATES[r][0]) if codec == "anchor" else 1
        full.append(render(Q, D, splat=splat))
        titles.append(f"{codec} {r}\n{rate:.0f} Mbps, Y {m['y_psnr']:.1f} dB")
crops = crop(full)
# zoom on the face: rows/cols around the highest skin-coloured points
top = P[P[:, 1] > P[:, 1].max() - 140]
cx, cy = int(np.median(top[:, 0])), 1023 - int(np.median(top[:, 1]))
zoom = [im[max(cy - 90, 0):cy + 90, max(cx - 90, 0):cx + 90] for im in full]

fig, axs = plt.subplots(2, len(show), figsize=(3 * len(show), 8.5),
                        gridspec_kw=dict(height_ratios=[2, 1]))
for i in range(len(show)):
    axs[0, i].imshow(crops[i]); axs[0, i].set_title(titles[i], fontsize=10)
    axs[1, i].imshow(zoom[i], interpolation="nearest")
    for a in axs[:, i]:
        a.axis("off")
fig.suptitle(f"redandblack frame {args.frames[0]}: decoded quality  (data (c) 8i Labs)")
plt.tight_layout()
plt.savefig(f"{args.out}/visual_compare.png", dpi=120)
print(f"\nSaved {args.out}/rd_curves.png, visual_compare.png, rd_results.csv")
