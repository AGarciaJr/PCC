"""
toy_vpcc / vpcc.py
==================
A from-scratch, single-frame, *simplified* re-implementation of MPEG V-PCC
(Video-based Point Cloud Compression), following Section VII of

    S. Schwarz et al., "Emerging MPEG Standards for Point Cloud Compression",
    IEEE JETCAS, vol. 9, no. 1, pp. 133-148, 2019.
    doi:10.1109/JETCAS.2018.2885981

Every function is named after a box in the paper's block diagrams
(Fig. 11 = encoder, Fig. 12 = decoder). Each docstring starts with a
"Paper:" line saying which section / figure / equation it implements.

  ENCODER (Fig. 11)              paper    DECODER (Fig. 12)            paper
  -----------------              -----    -----------------            -----
  patch_generation()      VII-A, Fig. 13  demultiplex()                VII
  packing()                      VII-A    video_decompress()           VII
  geometry_image_generation()    VII-B    occupancy_map_decompress()   VII-D
  texture_image_generation()     VII-B    aux_patch_info_decompress()  VII-C
  image_padding()                VII-B    geometry_reconstruction()    VII-E, Eqs. 12-14
  video_compress()               VII      smoothing()                  VII-E
  occupancy_map_compress()       VII-D    texture_reconstruction()     VII-E
  aux_patch_info_compress()      VII-C
  multiplex()                    VII      psnr_metrics()               IV-B (D1, D2, PSNR)

Paper's one-line summary of V-PCC (Sec. VII, 1st paragraph): turn the point
cloud into two videos (geometry + texture), code them with an existing video
codec, and send a small amount of metadata (occupancy map + patch info, 5-20%
of the bitstream) alongside.

Simplifications vs. the paper / the real standard (TMC2):
  * one frame (no temporal prediction); "video codec" = JPEG/PNG per frame via
    Pillow, standing in for HEVC.
  * occupancy map is down-sampled to occ_precision x occ_precision blocks
    (like TMC2) but then zlib'd instead of video-coded.
  * patch depth range limited to 256 levels (8-bit geometry images), as TMC2.
  * patches are never rotated during packing; aux info is a zlib'd int16 table.
  * smoothing = centroid of k nearest neighbours for points on patch borders.
"""
from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# The 6 projection planes = "the six unit cube oriented planes" of Sec. VII-A.
# (axis, sign): points whose normal points toward +x are "seen" from the +x
# side, and so on.
ORIENTS = [(0, -1), (1, -1), (2, -1), (0, +1), (1, +1), (2, +1)]
PLANE_NORMALS = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                          [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)


@dataclass
class Config:
    bits: int = 7                 # voxel grid is 2^bits per side
    knn: int = 16                 # neighbours for normal estimation / refinement
    refine_iters: int = 10        # iterations of plane-label smoothing (VII-A)
    refine_lambda: float = 2.0    # weight of neighbour vote vs. normal agreement
    surface_thickness: int = 4    # tau in Sec. VII-B: far layer is within [D0, D0+tau]
    max_passes: int = 2           # re-run patch generation on missed points
    min_patch_points: int = 16    # smaller leftovers go to the raw-points patch
    block: int = 4                # T in Sec. VII-A/C (paper's example: 16)
    canvas_width: int = 256       # W in Sec. VII-A (H is found by packing)
    geo_quality: int = 95         # JPEG quality for geometry; 0 = lossless PNG
    tex_quality: int = 75         # JPEG quality for texture;  0 = lossless PNG
    occ_precision: int = 1        # B0 in Sec. VII-D (1 = lossless; 2 or 4 typical)
    raw_points: bool = True       # code missed points? (TMC2 lossy mode: no)
    smooth: bool = True           # Sec. VII-E smoothing on/off
    smooth_k: int = 8
    smooth_threshold: float = 1.0 # only move a border point if it is this far off

    @property
    def max_coord(self) -> int:
        return (1 << self.bits) - 1


# =============================================================================
#                                ENCODER
# =============================================================================

# ---------------------------------------------------------------- patch gen --
def estimate_normals(P, k):
    """Paper: Sec. VII-A, Fig. 13 step 1, "normal at every point is estimated
    as described in [50]" (Hoppe et al., SIGGRAPH 1992).
    As in [50]: PCA over the k nearest neighbours; the normal is the
    smallest-eigenvalue eigenvector of the local covariance. Orientation here
    is simplified (point away from the centroid, fine for star-shaped
    objects and roughly right for a standing person); [50] propagates
    orientation over a minimum spanning tree instead."""
    _, idx = cKDTree(P).query(P, k)
    nb = P[idx] - P[idx].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vecs = np.linalg.eigh(cov)
    n = vecs[:, :, 0]
    flip = ((P - P.mean(0)) * n).sum(1) < 0
    n[flip] *= -1
    return n, idx


def segment_planes(normals, nbr_idx, cfg):
    """Paper: Sec. VII-A, Fig. 13 steps 2-3 (initial clustering + refinement).
    Initial clustering: each point goes to the plane whose normal maximises
    the dot product with the point normal.
    Refinement: the paper says the cluster index is updated iteratively
    "based on its normal and the cluster indexes of its nearest neighbors".
    Here: score = n·n_plane + λ·(fraction of neighbours with that label).
    The exact weighting is our choice (the paper doesn't give a formula).
    It gives smoother, larger patches."""
    score0 = normals @ PLANE_NORMALS.T            # N x 6
    lab = score0.argmax(1)
    for _ in range(cfg.refine_iters):
        nl = lab[nbr_idx]                         # N x k neighbour labels
        votes = np.stack([(nl == c).mean(1) for c in range(6)], 1)
        lab = (score0 + cfg.refine_lambda * votes).argmax(1)
    return lab


def connected_patches(P, lab, mask):
    """Paper: Sec. VII-A, Fig. 13 last step, patches are extracted with "a
    connected component extraction procedure".
    Here: two points are connected if they are 26-neighbours on the voxel
    grid (distance <= sqrt(3)) and have the same plane label."""
    ids = np.flatnonzero(mask)
    pairs = cKDTree(P[ids]).query_pairs(r=1.75, output_type="ndarray")
    pairs = pairs[lab[ids[pairs[:, 0]]] == lab[ids[pairs[:, 1]]]]
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                   shape=(len(ids), len(ids)))
    n, cc = connected_components(g, directed=False)
    order = np.argsort(cc, kind="stable")
    splits = np.cumsum(np.bincount(cc, minlength=n))[:-1]
    return [ids[s] for s in np.split(order, splits)]


def project_patch(P, pidx, orient, cfg):
    """Paper: Sec. VII-A (patches are mapped to 2D "through a simple orthogonal
    projection") and Sec. VII-B (two layers per patch).
    For every pixel (u, v), H(u, v) is the set of the patch's points that
    project onto it. The near layer stores the lowest depth D0; the far layer
    stores the highest depth within [D0, D0 + tau], tau = surface thickness.
    The patch's 3D offset (d0, u0, v0) is (d0, s0, r0) in the paper's
    Eqs. 12-14; its size (w, h) gives the 2D bounding box.
    Returns the patch record (D0/D1 depth maps + mask) and a mask of which of
    its points are exactly representable by the two layers."""
    a, s = ORIENTS[orient]
    t, b = (a + 1) % 3, (a + 2) % 3                # tangent / bitangent axes
    Q = P[pidx]
    depth = Q[:, a] if s < 0 else cfg.max_coord - Q[:, a]
    # Geometry images are 8-bit, so a patch may only span 256 depth levels
    # (TMC2 does the same). Points deeper than that stay for the next pass.
    d0 = int(depth.min())
    inrange = depth - d0 <= 255
    full = np.zeros(len(Q), bool)
    Q, depth = Q[inrange], depth[inrange]
    u, v = Q[:, t], Q[:, b]
    u0, v0 = int(u.min()), int(v.min())
    U, V, D = u - u0, v - v0, depth - d0
    w, h = int(U.max()) + 1, int(V.max()) + 1

    BIG = 1 << 20
    D0 = np.full((h, w), BIG, np.int64)
    np.minimum.at(D0, (V, U), D)
    near = D0[V, U]
    ok = D <= near + cfg.surface_thickness
    D1 = np.full((h, w), -1, np.int64)
    np.maximum.at(D1, (V[ok], U[ok]), D[ok])
    occ = D0 < BIG
    D0[~occ] = 0
    D1[~occ] = 0
    full[inrange] = (D == near) | (D == D1[V, U])
    return dict(orient=orient, u0=u0, v0=v0, d0=d0, w=w, h=h,
                occ=occ, D0=D0, D1=D1), full


def patch_generation(P, cfg):
    """Paper: Fig. 11 "Patch generation" box, detailed in Sec. VII-A / Fig. 13:
    normals -> 6-plane clustering -> refinement -> connected components.
    The paper's goal is an injective mapping (every point gets its own
    pixel) so reconstruction can be lossless. Points hidden behind the two
    layers break that, so (like TMC2) we re-run patch extraction on those
    'missed' points, max_passes times. Whatever is still left is the
    'raw / missed points patch' (a TMC2 detail, not in the paper)."""
    normals, nbr = estimate_normals(P.astype(float), cfg.knn)
    lab = segment_planes(normals, nbr, cfg)
    remaining = np.ones(len(P), bool)
    patches = []
    for _ in range(cfg.max_passes):
        if not remaining.any():
            break
        for comp in connected_patches(P, lab, remaining):
            if len(comp) < cfg.min_patch_points:
                continue
            patch, rep = project_patch(P, comp, int(lab[comp[0]]), cfg)
            patches.append(patch)
            remaining[comp[rep]] = False
    raw_idx = np.flatnonzero(remaining)    # "missed points patch", coded raw
    return patches, lab, raw_idx


# ------------------------------------------------------------------ packing --
def packing(patches, cfg):
    """Paper: Fig. 11 "Packing" box, Sec. VII-A (last two paragraphs).
    As in the paper:
      - every T x T block of the canvas belongs to at most one patch;
      - canvas width W is fixed; each patch goes in the first free spot found
        by an exhaustive search in raster-scan order;
      - if nothing fits, the height H is doubled and the search repeats; at
        the end H is cut down to the rows actually used.
    Our addition: patches are inserted largest first, and a summed-area
    table checks every candidate position in one numpy operation."""
    B, Wb = cfg.block, cfg.canvas_width // cfg.block
    grid = np.zeros((4 * Wb, Wb), bool)            # grows downward if needed
    for i in sorted(range(len(patches)),
                    key=lambda i: -patches[i]["w"] * patches[i]["h"]):
        p = patches[i]
        bw, bh = -(-p["w"] // B), -(-p["h"] // B)
        assert bw <= Wb, "patch wider than canvas; increase canvas_width"
        while True:
            # summed-area table -> every window sum in one shot
            S = np.pad(grid.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            win = S[bh:, bw:] - S[:-bh, bw:] - S[bh:, :-bw] + S[:-bh, :-bw]
            free = np.argwhere(win == 0)
            if len(free):
                by, bx = free[0]                   # first fit, row-major
                break
            grid = np.vstack([grid, np.zeros_like(grid)])
        grid[by:by + bh, bx:bx + bw] = True
        p["bx"], p["by"] = int(bx), int(by)
    used_rows = np.flatnonzero(grid.any(1)).max() + 1
    return cfg.canvas_width, int(used_rows * B)    # canvas W, H


# ------------------------------------------------------- image generation --
def geometry_image_generation(patches, W, H, cfg):
    """Paper: Fig. 11 "Geometry image generation" box, Sec. VII-B, Fig. 14
    (left). Writes each packed patch's depths into two 8-bit images, near
    layer D0 and far layer D1, plus the binary occupancy map (Sec. VII-D:
    "whether it belongs to the empty space or to the point cloud") and a
    per-pixel patch index (the decoder rebuilds it from the aux info)."""
    G = np.zeros((2, H, W), np.uint8)
    occ = np.zeros((H, W), bool)
    pmap = np.full((H, W), -1, np.int32)
    for i, p in enumerate(patches):
        y, x = p["by"] * cfg.block, p["bx"] * cfg.block
        sl = (slice(y, y + p["h"]), slice(x, x + p["w"]))
        m = p["occ"]
        G[0][sl][m] = p["D0"][m]
        G[1][sl][m] = p["D1"][m]
        occ[sl] |= m
        pmap[sl][m] = i
    return G, occ, pmap


def texture_image_generation(rec_pts, layer, ys, xs, P, C, occ, cfg):
    """Paper: Fig. 11 "Texture image generation" box, Sec. VII-B, Fig. 14
    (right). In Fig. 11 this box takes the *reconstructed, smoothed*
    geometry as input, not the original. So we re-colour each decoded point
    from its nearest original point, then write the colours into two
    texture images, one per layer."""
    _, nn = cKDTree(P).query(rec_pts)
    cols = C[nn]
    H, W = occ.shape
    T = np.zeros((2, H, W, 3), np.uint8)
    T[0][ys[layer == 0], xs[layer == 0]] = cols[layer == 0]
    T[1] = T[0]                         # pixels where D1==D0 reuse near colour
    T[1][ys[layer == 1], xs[layer == 1]] = cols[layer == 1]
    return T


# ----------------------------------------------------------------- padding --
def _push_pull(img, w):
    """Recursive push-pull: average the filled pixels into a half-size image
    (push), fill that recursively, then use it for the empty pixels here
    (pull). O(pixels) no matter how large the empty areas are."""
    H, W = w.shape
    if (w > 0).all() or min(H, W) == 1:
        if not (w > 0).any():
            return np.zeros_like(img)
        out = img.copy()
        out[w == 0] = (img * w[..., None]).sum((0, 1)) / w.sum()
        return out
    h2, w2 = -(-H // 2), -(-W // 2)
    pi = np.zeros((h2 * 2, w2 * 2, img.shape[2]), np.float32)
    pw = np.zeros((h2 * 2, w2 * 2), np.float32)
    pi[:H, :W] = img * w[..., None]
    pw[:H, :W] = w
    si = pi.reshape(h2, 2, w2, 2, -1).sum((1, 3))
    sw = pw.reshape(h2, 2, w2, 2).sum((1, 3))
    small = np.where(sw[..., None] > 0, si / np.maximum(sw, 1e-9)[..., None], 0)
    small = _push_pull(small, (sw > 0).astype(np.float32))
    up = small.repeat(2, 0).repeat(2, 1)[:H, :W]
    return np.where(w[..., None] > 0, img, up)


def image_padding(img, occ):
    """Paper: Fig. 11 "Image padding" box, Sec. VII-B (last paragraph).
    Goal (same as the paper): fill the empty space between patches so the
    image is "piecewise smooth", which is cheap for a video codec.
    Method (DIFFERENT from the paper): the paper pads each T x T block
    separately (copy the previous block's last row/column into empty
    blocks; iteratively average neighbours inside mixed blocks). We use
    push-pull (multi-resolution fill), which later TMC2 versions use for
    texture. On the toy blob this cut the stream from 8.5 to 5.8 bits/point
    compared with simple neighbour dilation."""
    x = img.astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    out = _push_pull(x, occ.astype(np.float32))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8).squeeze()


# ------------------------------------------------------ video compression --
def video_compress(frames, quality):
    """Paper: Fig. 11 "Video compression" box, Sec. VII (1st paragraph): the
    images are coded using existing video codecs (the paper names MPEG-4 AVC,
    HEVC and AV1). BIGGEST SIMPLIFICATION here: each frame is coded on its own
    with JPEG (quality > 0) or lossless PNG (quality == 0), so there is no
    inter-frame prediction. Sec. VIII reports separate all-intra (AI) and
    random-access (RA) results; this toy is closest to all-intra."""
    out = []
    for f in frames:
        buf = io.BytesIO()
        if quality > 0:
            Image.fromarray(f).save(buf, "JPEG", quality=quality)
        else:
            Image.fromarray(f).save(buf, "PNG", optimize=True)
        out.append(buf.getvalue())
    return out


def occupancy_map_downsample(occ, p):
    """Paper: Sec. VII-D, occupancy map "precision of B0 x B0 blocks". A
    sub-block is 1 if it has at least one non-padded pixel. B0 = 1 is
    lossless; the paper notes B0 = 2 or 4 look fine and save many bits.
    Cost: the decoder rebuilds extra points at patch edges from padded
    geometry values (which smoothing, Sec. VII-E, then cleans up)."""
    H, W = occ.shape
    assert H % p == 0 and W % p == 0, "canvas must be a multiple of occ_precision"
    return occ.reshape(H // p, p, W // p, p).any((1, 3))


def occupancy_map_compress(occ, p=1):
    """Paper: Fig. 11 "Occupancy map compression" box, Sec. VII-D.
    The paper skips empty T x T blocks (known from the aux info), then codes
    each remaining block's B0 x B0 sub-blocks with a full/non-full flag, one
    of 4 traversal orders (Fig. 15) and run-length + arithmetic coding.
    Simplified here: the B0 x B0 map is bit-packed and zlib'd."""
    low = occupancy_map_downsample(occ, p)
    H, W = low.shape
    return struct.pack("<HHB", H, W, p) + zlib.compress(np.packbits(low).tobytes(), 9)


# Sec. VII-C: per patch, "the index of its projection plane, its 3D location,
# and its 2D bounding box". orient = plane index; (u0, v0, d0) = 3D location
# (s0, r0, d0 in Eqs. 12-14); (bx, by, w, h) = 2D bounding box on the canvas.
AUX_FIELDS = ["orient", "u0", "v0", "d0", "bx", "by", "w", "h"]


def aux_patch_info_compress(patches):
    """Paper: Fig. 11 "Auxiliary patch-info compression" box, Sec. VII-C.
    The paper predicts and arithmetic-codes the patch table, and also codes
    a block-to-patch map (each T x T block's patch, as an index into a
    candidate list). Simplified here: an int16 table, zlib'd. We don't send
    the block-to-patch map: the decoder rebuilds it from the bounding boxes
    and the occupancy map (patch_index_map)."""
    tab = np.array([[p[k] for k in AUX_FIELDS] for p in patches], np.int16)
    return zlib.compress(tab.tobytes(), 9)


def raw_points_compress(pts, cols):
    """Not in the paper (a TMC2 detail): points no patch could represent,
    the 'missed / raw points patch'. Needed for lossless coding, the
    paper's "injective mapping" goal in Sec. VII-A.
    Here: stored losslessly as uint16 xyz + rgb, zlib'd."""
    return zlib.compress(pts.astype(np.uint16).tobytes() + cols.tobytes(), 9)


def raw_points_decompress(b):
    buf = zlib.decompress(b)
    n = len(buf) // 9
    pts = np.frombuffer(buf[:6 * n], np.uint16).reshape(n, 3).astype(np.int64)
    return pts, np.frombuffer(buf[6 * n:], np.uint8).reshape(n, 3)


def multiplex(sections):
    """Paper: Fig. 11 "Multiplexer" box, Sec. VII (1st paragraph): the video
    bitstreams and the metadata are multiplexed into one V-PCC bitstream.
    Here: [n_sections][len_i][bytes_i]..."""
    out = struct.pack("<B", len(sections))
    for s in sections:
        out += struct.pack("<I", len(s)) + s
    return out


def encode(P, C, cfg: Config):
    """Paper: the whole encoder of Fig. 11, boxes in the same order.
    Returns the bitstream and a dict of intermediate images/stats for
    visualisation."""
    patches, labels, raw_idx = patch_generation(P, cfg)
    W, H = packing(patches, cfg)
    G, occ, pmap = geometry_image_generation(patches, W, H, cfg)
    G_pad = np.stack([image_padding(g, occ) for g in G])
    geo_bits = video_compress(G_pad, cfg.geo_quality)
    occ_bits = occupancy_map_compress(occ, cfg.occ_precision)
    occ_dec = occupancy_map_decompress(occ_bits)    # what the decoder will see

    # Fig. 11 also has "Geometry reconstruction" and "Smoothing" boxes INSIDE
    # the encoder (closed loop): texture must match the geometry the
    # *decoder* will see, not the original.
    G_dec = video_decompress(geo_bits)
    aux = np.array([[p[k] for k in AUX_FIELDS] for p in patches], np.int64)
    rec, layer, ys, xs = geometry_reconstruction(G_dec, occ_dec, aux, cfg)
    if cfg.smooth:
        rec = smoothing(rec, ys, xs, occ_dec, patch_index_map(aux, occ_dec, cfg), cfg)

    T = texture_image_generation(rec, layer, ys, xs, P, C, occ_dec, cfg)
    T_pad = np.stack([image_padding(t, occ_dec) for t in T])
    tex_bits = video_compress(T_pad, cfg.tex_quality)

    if not cfg.raw_points:
        raw_idx = raw_idx[:0]
    aux_bits = aux_patch_info_compress(patches)
    raw_bits = raw_points_compress(P[raw_idx], C[raw_idx])
    stream = multiplex(geo_bits + tex_bits + [occ_bits, aux_bits, raw_bits])
    info = dict(patches=patches, labels=labels, n_raw=len(raw_idx), occ=occ,
                occ_dec=occ_dec,
                pmap=pmap, G=G, G_pad=G_pad, T=T, T_pad=T_pad,
                sizes=dict(geometry=sum(map(len, geo_bits)),
                           texture=sum(map(len, tex_bits)),
                           occupancy=len(occ_bits), aux=len(aux_bits),
                           raw_points=len(raw_bits)))
    return stream, info


# =============================================================================
#                                DECODER
# =============================================================================

def demultiplex(stream):
    """Paper: Fig. 12 "De-multiplexer" box (inverse of multiplex())."""
    n, = struct.unpack_from("<B", stream, 0)
    off, out = 1, []
    for _ in range(n):
        L, = struct.unpack_from("<I", stream, off)
        out.append(stream[off + 4: off + 4 + L])
        off += 4 + L
    return out


def video_decompress(frames):
    """Paper: Fig. 12 "Video decompression" box (JPEG/PNG here, see
    video_compress)."""
    return np.stack([np.array(Image.open(io.BytesIO(b))) for b in frames])


def occupancy_map_decompress(b):
    """Paper: Fig. 12 "Occupancy map decompression" box, Sec. VII-D.
    Each B0 x B0 sub-block is expanded back to full resolution."""
    H, W, p = struct.unpack_from("<HHB", b, 0)
    bits = np.unpackbits(np.frombuffer(zlib.decompress(b[5:]), np.uint8))
    low = bits[:H * W].reshape(H, W).astype(bool)
    return low.repeat(p, 0).repeat(p, 1)


def aux_patch_info_decompress(b):
    """Paper: Fig. 12 "Auxiliary patch-info decompression" box, Sec. VII-C."""
    return np.frombuffer(zlib.decompress(b), np.int16).reshape(-1, len(AUX_FIELDS)).astype(np.int64)


def patch_index_map(aux, occ, cfg):
    """Which patch each occupied pixel belongs to. This plays the role of the
    paper's block-to-patch map (Sec. VII-C), rebuilt from the aux table."""
    pmap = np.full(occ.shape, -1, np.int32)
    for i, (_, _, _, _, bx, by, w, h) in enumerate(aux):
        y, x = by * cfg.block, bx * cfg.block
        sub = pmap[y:y + h, x:x + w]
        sub[occ[y:y + h, x:x + w]] = i
    return pmap


def geometry_reconstruction(G, occ, aux, cfg):
    """Paper: Fig. 12 "Geometry reconstruction" box, Sec. VII-E, Eqs. 12-14.
    For each occupied pixel (u, v) of a patch with 3D location (d0, s0, r0)
    and 2D bounding-box corner (u0, v0):
        depth             d(u, v) = d0 + g(u, v)      (12)  g = geometry image
        tangential shift  s(u, v) = s0 - u0 + u       (13)
        bi-tangential     r(u, v) = r0 - v0 + v       (14)
    Naming: our aux fields u0, v0 hold the paper's s0, r0 (3D location); the
    paper's bounding-box corner (u0, v0) is our (bx*T, by*T) = (x0, y0).
    So in the code: depth = d0 + dd (12); q[:, t] = u0 + xx (13);
    q[:, b] = v0 + yy (14), where xx = u - x0 and yy = v - y0.
    Every occupied pixel gives a near-layer point (D0) and, if D1 != D0, a
    far-layer point. Returns points plus the (layer, y, x) pixel each came
    from (needed for the texture lookup)."""
    pts, lay, Y, X = [], [], [], []
    for orient, u0, v0, d0, bx, by, w, h in aux:
        a, s = ORIENTS[orient]
        t, b = (a + 1) % 3, (a + 2) % 3
        y0, x0 = by * cfg.block, bx * cfg.block
        yy, xx = np.nonzero(occ[y0:y0 + h, x0:x0 + w])
        g0 = G[0, y0 + yy, x0 + xx].astype(np.int64)
        g1 = np.maximum(G[1, y0 + yy, x0 + xx].astype(np.int64), g0)
        for li, dd in ((0, g0), (1, g1)):
            keep = np.ones_like(dd, bool) if li == 0 else dd != g0
            depth = d0 + dd[keep]
            q = np.empty((keep.sum(), 3), np.int64)
            q[:, a] = depth if s < 0 else cfg.max_coord - depth
            q[:, t] = u0 + xx[keep]
            q[:, b] = v0 + yy[keep]
            pts.append(q); lay.append(np.full(len(q), li))
            Y.append(y0 + yy[keep]); X.append(x0 + xx[keep])
    pts = np.clip(np.concatenate(pts), 0, cfg.max_coord)
    return pts, np.concatenate(lay), np.concatenate(Y), np.concatenate(X)


def smoothing(pts, ys, xs, occ, pmap, cfg):
    """Paper: Fig. 11/12 "Smoothing" box, Sec. VII-E: smoothing "moves
    boundary points to the centroid of their nearest neighbors" to hide
    discontinuities at patch boundaries caused by compression.
    Here: a point is on a boundary if a pixel in its 3x3 neighbourhood
    belongs to another patch (or empty space). It moves to the centroid of
    its k nearest reconstructed points, only if more than
    smooth_threshold away."""
    H, W = occ.shape
    pad = np.pad(pmap, 1, constant_values=-1)
    border = np.zeros((H, W), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            border |= pad[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx] != pmap
    border &= occ
    sel = border[ys, xs]
    _, nn = cKDTree(pts).query(pts[sel], cfg.smooth_k)
    centroid = pts[nn].mean(1)
    far = np.linalg.norm(centroid - pts[sel], axis=1) > cfg.smooth_threshold
    out = pts.copy()
    idx = np.flatnonzero(sel)[far]
    out[idx] = np.rint(centroid[far]).astype(np.int64)
    return out


def texture_reconstruction(T, layer, ys, xs):
    """Paper: Fig. 12 "Texture reconstruction" box, Sec. VII-E: each point
    takes the colour of the texture pixel (same layer, same u, v) that its
    geometry came from."""
    return T[layer, ys, xs]


def decode(stream, cfg: Config):
    """Paper: the whole decoder of Fig. 12, boxes in the same order."""
    sec = demultiplex(stream)
    G = video_decompress(sec[0:2])
    T = video_decompress(sec[2:4])
    occ = occupancy_map_decompress(sec[4])
    aux = aux_patch_info_decompress(sec[5])
    pts, layer, ys, xs = geometry_reconstruction(G, occ, aux, cfg)
    if cfg.smooth:
        pts = smoothing(pts, ys, xs, occ, patch_index_map(aux, occ, cfg), cfg)
    cols = texture_reconstruction(T, layer, ys, xs)
    rp, rc = raw_points_decompress(sec[6])
    return np.vstack([pts, rp]), np.vstack([cols, rc])


# =============================================================================
#                                METRICS
# =============================================================================

def _luma(c):
    return c.astype(float) @ np.array([0.2126, 0.7152, 0.0722])


def psnr_metrics(P, C, Q, D, bits, normals=None):
    """Paper: Sec. IV-B "Evaluation Methodology" (metrics from refs [39] Tian
    et al., ICIP 2017 and [40] Mekuria et al., VCIP 2017); used in the RD
    curves of Fig. 17.
      geo_psnr = D1, point-to-point: MSE between each point and its closest
                 point in the other cloud.
      d2_psnr  = D2, point-to-plane: the same error projected onto the
                 reference surface normal (needs `normals` of P).
      y_psnr   = the D1-style nearest-neighbour match applied to colour
                 (luma only here; the paper uses Y, U and V).
    Symmetric: the worse of A->B and B->A. As the paper says, geometry PSNR
    peak comes from the 3D volume resolution (2^bits - 1 here); colour peak
    is 255."""
    peak = (1 << bits) - 1
    d_ab, i_ab = cKDTree(Q).query(P)
    d_ba, i_ba = cKDTree(P).query(Q)
    mse_g = max((d_ab ** 2).mean(), (d_ba ** 2).mean())
    mse_y = max(((_luma(C) - _luma(D[i_ab])) ** 2).mean(),
                ((_luma(D) - _luma(C[i_ba])) ** 2).mean())
    ps = lambda mse, p: float("inf") if mse == 0 else 10 * np.log10(p / mse)
    out = dict(geo_psnr=ps(mse_g, 3 * peak ** 2), y_psnr=ps(mse_y, 255 ** 2))
    if normals is not None:
        e_ab = (((Q[i_ab] - P) * normals).sum(1)) ** 2
        e_ba = (((Q - P[i_ba]) * normals[i_ba]).sum(1)) ** 2
        out["d2_psnr"] = ps(max(e_ab.mean(), e_ba.mean()), 3 * peak ** 2)
    return out
