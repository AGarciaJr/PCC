"""V-PCC on one frame of 8i 'redandblack': run encoder + decoder and save a
figure of every stage of the pipeline (paper Figs. 11/12), plus the
individual geometry / texture / occupancy images for slides.

    python redblack_demo.py                     # frame 1450, rate R3
    python redblack_demo.py --frame 1500 --rate R5
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from redblack import RATES, crop, load_frame, mbps, render, vpcc_config
from vpcc import decode, encode, estimate_normals, psnr_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--frame", type=int, default=1450)
ap.add_argument("--rate", default="R3", choices=list(RATES))
ap.add_argument("--out", default="out/redandblack")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

P, C = load_frame(args.frame)
N = len(P)
cfg = vpcc_config(args.rate)
print(f"redandblack frame {args.frame}: {N} points, 10-bit grid; rate {args.rate} "
      f"(geo q={cfg.geo_quality}, tex q={cfg.tex_quality})")

t = time.time(); stream, info = encode(P, C, cfg); t_enc = time.time() - t
t = time.time(); Q, D = decode(stream, cfg); t_dec = time.time() - t
normals, _ = estimate_normals(P.astype(float), 16)
m = psnr_metrics(P, C, Q, D, cfg.bits, normals)

raw_bytes = N * (3 * 10 + 24) / 8
print(f"\nPatches: {len(info['patches'])}   canvas {info['occ'].shape[1]}x"
      f"{info['occ'].shape[0]} ({info['occ'].mean():.0%} occupied)")
print("Bitstream (bytes):")
for k, v in info["sizes"].items():
    print(f"  {k:<11}{v:>9}  ({v / len(stream):.0%})")
print(f"  {'TOTAL':<11}{len(stream):>9}")
print(f"Bits/point {8 * len(stream) / N:.2f}   {raw_bytes / len(stream):.0f}x smaller "
      f"than raw   = {mbps(len(stream)):.1f} Mbps at 30 fps "
      f"(raw: {mbps(raw_bytes):.0f} Mbps)")
print(f"D1 {m['geo_psnr']:.2f} dB  D2 {m['d2_psnr']:.2f} dB  Y {m['y_psnr']:.2f} dB")
print(f"Decoded points {len(Q)}   enc {t_enc:.1f}s  dec {t_dec:.1f}s")

# ------------------------------------------------------------ slide images --
pal = (plt.cm.tab10(np.arange(6))[:, :3] * 255).astype(np.uint8)
pm = info["pmap"]
patch_img = np.full(pm.shape + (3,), 255, np.uint8)
rng = np.random.default_rng(0)
pcols = rng.integers(40, 230, (len(info["patches"]), 3)).astype(np.uint8)
patch_img[pm >= 0] = pcols[pm[pm >= 0]]

imgs = {
    "occupancy.png": (info["occ_dec"] * 255).astype(np.uint8),
    "geometry_D0_raw.png": info["G"][0],
    "geometry_D0_padded.png": info["G_pad"][0],
    "geometry_D1_padded.png": info["G_pad"][1],
    "texture_T0_raw.png": np.where(info["occ_dec"][..., None], info["T"][0], 0).astype(np.uint8),
    "texture_T0_padded.png": info["T_pad"][0],
    "patches_packed.png": patch_img,
}
for name, im in imgs.items():
    Image.fromarray(im).save(f"{args.out}/{name}")

r_orig, r_seg, r_dec = crop([render(P, C), render(P, pal[info["labels"]]),
                             render(Q, D)])
for name, im in (("render_original.png", r_orig), ("render_planes.png", r_seg),
                 ("render_decoded.png", r_dec)):
    Image.fromarray(im).save(f"{args.out}/{name}")

# ------------------------------------------------------- one-glance figure --
fig = plt.figure(figsize=(18, 9))
panels = [
    (r_orig, f"1. Input: {N/1e3:.0f}K points (vox10)"),
    (r_seg, "2. Patch generation:\nnormal -> 6 projection planes"),
    (patch_img, f"3. Packing: {len(info['patches'])} patches"),
    (imgs["occupancy.png"], f"4. Occupancy map ({cfg.occ_precision}x{cfg.occ_precision} blocks)"),
    (info["G_pad"][0], "5. Geometry image D0 (padded)"),
    (info["T_pad"][0], "6. Texture image T0 (padded)"),
    (r_dec, f"8. Decoded: {len(Q)/1e3:.0f}K points\n"
            f"D1 {m['geo_psnr']:.1f} dB, Y {m['y_psnr']:.1f} dB"),
]
for i, (im, title) in enumerate(panels):
    ax = fig.add_subplot(2, 4, i + 1 if i < 6 else 8)
    ax.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest")
    ax.set_title(title, fontsize=10); ax.axis("off")

ax = fig.add_subplot(2, 4, 7)
names = list(info["sizes"])
vals = np.array([info["sizes"][k] for k in names]) / 1e3
ax.barh(names, vals, color="#4c72b0")
for y, v in enumerate(vals):
    ax.text(v, y, f" {v:.0f} KB", va="center", fontsize=8)
ax.invert_yaxis(); ax.set_xlabel("KB")
ax.set_title(f"7. Bitstream: {len(stream)/1e3:.0f} KB = {8*len(stream)/N:.2f} bpp\n"
             f"= {mbps(len(stream)):.0f} Mbps @30fps (raw {mbps(raw_bytes):.0f})", fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.suptitle(f"Toy V-PCC on 8i redandblack, frame {args.frame}, rate {args.rate}  "
             "(data (c) 8i Labs)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{args.out}/pipeline_{args.rate}.png", dpi=120)
print(f"\nSaved figures to {args.out}/")
