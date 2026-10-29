"""
anchor.py - a simplified octree + JPEG point cloud codec, used as the
baseline ("anchor") V-PCC is compared against.

Modelled on the MPEG PCC anchor of
    R. Mekuria, K. Blom, P. Cesar, "Design, Implementation and Evaluation of a
    Point Cloud Codec for Tele-Immersive Video", IEEE TCSVT, 2017.

  geometry : quantise to a 2^depth grid, then code the octree breadth-first as
             one occupancy byte per node (which of the 8 children exist),
             entropy-coded with LZMA (stand-in for the range coder).
  colour   : one colour per voxel, laid out in octree (Morton) order into a 2D
             image and coded with JPEG.
It works in 3D directly, so it cannot reuse 2D video tools (no motion
compensation, no intra prediction across the surface): that's the gap V-PCC
closes by projecting to 2D first.
"""
import io
import lzma
import struct

import numpy as np
from PIL import Image


def _morton(Q, depth):
    m = np.zeros(len(Q), np.int64)
    for i in range(depth):
        for a in range(3):
            m |= ((Q[:, a] >> i) & 1) << (3 * i + (2 - a))
    return m


def _unmorton(m, depth):
    Q = np.zeros((len(m), 3), np.int64)
    for i in range(depth):
        for a in range(3):
            Q[:, a] |= ((m >> (3 * i + (2 - a))) & 1) << i
    return Q


def octree_encode(M, depth):
    """Sorted unique Morton codes -> breadth-first occupancy bytes."""
    out = []
    for lvl in range(depth):
        node = M >> (3 * (depth - lvl))
        child = (M >> (3 * (depth - lvl - 1))) & 7
        bits = np.zeros(len(M), np.uint8) | (1 << child).astype(np.uint8)
        starts = np.flatnonzero(np.r_[True, node[1:] != node[:-1]])
        out.append(np.bitwise_or.reduceat(bits, starts))
    return np.concatenate(out).tobytes()


def octree_decode(buf, depth):
    occ = np.frombuffer(buf, np.uint8)
    nodes, off = np.zeros(1, np.int64), 0
    for _ in range(depth):
        b = occ[off:off + len(nodes)]
        off += len(nodes)
        has = (b[:, None] >> np.arange(8)) & 1           # nodes x 8 children
        pi, ci = np.nonzero(has)
        nodes = (nodes[pi] << 3) | ci                     # stays Morton-sorted
    return nodes


def encode(P, C, bits=10, depth=8, quality=75, width=512):
    shift = bits - depth
    M = _morton(P >> shift, depth)
    uk, inv = np.unique(M, return_inverse=True)
    cnt = np.bincount(inv)
    col = np.stack([np.bincount(inv, C[:, c]) / cnt for c in range(3)], 1)
    geo = lzma.compress(octree_encode(uk, depth), preset=9)
    n = len(uk)
    h = -(-n // width)
    img = np.zeros((h * width, 3), np.uint8)
    img[:n] = np.rint(col)
    img[n:] = img[n - 1]                                  # pad tail
    buf = io.BytesIO()
    Image.fromarray(img.reshape(h, width, 3)).save(buf, "JPEG", quality=quality)
    tex = buf.getvalue()
    head = struct.pack("<BBBHI", bits, depth, 0, width, len(geo))
    return head + geo + tex, dict(geometry=len(geo), texture=len(tex))


def decode(stream):
    bits, depth, _, width, lg = struct.unpack_from("<BBBHI", stream, 0)
    off = struct.calcsize("<BBBHI")
    M = octree_decode(lzma.decompress(stream[off:off + lg]), depth)
    img = np.array(Image.open(io.BytesIO(stream[off + lg:]))).reshape(-1, 3)
    shift = bits - depth
    Q = (_unmorton(M, depth) << shift) + ((1 << shift) >> 1)   # voxel centres
    return Q, img[:len(M)]
