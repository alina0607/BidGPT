"""Bake a drawn cat into a sprite frame: its face warped to fill the frame's head exactly, its body covered with
texture taken from its own face.

Face: a piecewise affine warp (skimage) from the frame to the face, pinned at the eyes and mouth and at 50 outline
pairs: the 6 ear points are moved onto each outline, and each stretch of outline between them is divided evenly by
length, so ear meets ear and the face's edge lands on the head's edge, whatever shape the head is drawn in.

Body: game code, not the model (design-v2 §14.4): small patches of the cat's own fur (eyes, mouth and the face's
outline line left out), shrunk to the head's scale, are laid in a grid; each place takes, of a few random
patches, the one whose overlap with what is already laid differs least (a simplified image quilting, Efros &
Freeman 2001), and overlaps are feathered.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt
from scipy.ndimage import label as cc_label
from skimage.measure import find_contours
from skimage.transform import PiecewiseAffineTransform, resize, warp

PAPER = 245                                    # a face pixel this light in every channel is outside the cat


def silhouette(face: np.ndarray) -> np.ndarray:
    """The cat's own face: the largest non-white piece, holes filled."""
    m = face.min(-1) < PAPER
    lab, n = cc_label(m)
    if n == 0:
        return m
    big = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return binary_fill_holes(big)


def _outline(mask: np.ndarray, anchors: np.ndarray, per_arc: list[int]) -> np.ndarray:
    """Points on the mask's outline, (x, y): the outline point nearest each anchor, and between consecutive anchors
    (the last back to the first) per_arc[i] points spaced evenly by arc length."""
    c = max(find_contours(np.pad(mask, 1).astype(float), 0.5), key=len)[:, ::-1] - 1
    if np.cross(c[1:] - c[:-1], c[:-1] - c.mean(0)).sum() < 0:          # one direction round, whichever the input
        c = c[::-1]
    idx = [int(np.argmin(np.linalg.norm(c - a, axis=1))) for a in anchors]
    seg = np.linalg.norm(np.diff(c, axis=0, append=c[:1]), axis=1)
    out = []
    for i, k in enumerate(per_arc):
        a, b = idx[i], idx[(i + 1) % len(idx)]
        run = np.r_[a:b] if b > a else np.r_[a:len(c), 0:b]
        s = np.r_[0, np.cumsum(seg[run])]
        t = np.linspace(0, s[-1], k + 2)[:-1]
        pos = np.r_[run, b]
        out.append(np.c_[np.interp(t, s, c[pos, 0]), np.interp(t, s, c[pos, 1])])
    return np.vstack(out)


# the 6 ear points round the outline (outer, tip, inner; inner, tip, outer), then the jaw back to the start
EAR_ORDER = [3, 4, 5, 6, 7, 8]
PER_ARC = [4, 4, 6, 4, 4, 24]


def warp_face(face: np.ndarray, face_pts: np.ndarray, head: np.ndarray, head_pts: np.ndarray) -> np.ndarray:
    """RGB face (H, W, 3), its 9 points (x, y); the head mask and its 9 points on the frame → the frame's head pixels
    (frame-sized RGB float; only meaningful inside head)."""
    sil = silhouette(face)
    cf = face_pts[:3].mean(0)                                       # eyes and mouth: the middle of the face
    ch = head_pts[:3].mean(0)
    src = np.vstack([head_pts[:3], _outline(head, head_pts[EAR_ORDER], PER_ARC)])     # frame points …
    dst = np.vstack([face_pts[:3], _outline(sil, face_pts[EAR_ORDER], PER_ARC)])      # … and where they come from
    far = 4 * max(head.shape)                                       # pin the far field so nothing folds
    box = np.array([[-far, -far], [far, -far], [-far, far], [far, far]], float)
    tf = PiecewiseAffineTransform()
    tf.estimate(np.vstack([src, ch + box]), np.vstack([dst, cf + box * face.shape[0] / head.shape[0]]))
    out = warp(face.astype(np.float64) / 255, tf, output_shape=head.shape, mode="edge") * 255
    # pixels still outside the cat's face (thin slivers at the edge), paper-white inside it (the unpainted gaps
    # the drawing leaves at the ear tips), or off the face's canvas (an ear cut off by the crop, which mode="edge"
    # smears into streaks) take the nearest face colour
    fw = warp(sil.astype(float), tf, output_shape=head.shape, mode="constant", cval=0) > 0.5
    fw &= out.min(-1) < PAPER
    yy, xx = np.mgrid[:head.shape[0], :head.shape[1]]
    at = tf(np.c_[xx.ravel(), yy.ravel()]).reshape(head.shape + (2,))
    fw &= (at >= 0).all(-1) & (at[..., 0] <= face.shape[1] - 1) & (at[..., 1] <= face.shape[0] - 1)
    if fw.any() and (head & ~fw).any():
        _, (iy, ix) = distance_transform_edt(~fw, return_indices=True)
        miss = head & ~fw
        out[miss] = out[iy[miss], ix[miss]]
    return out


def fur_source(face: np.ndarray, face_pts: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """The face shrunk to the sprite head's scale, and where on it there is plain fur (not eyes, mouth or the
    outline line)."""
    sil = silhouette(face)
    d = np.linalg.norm(face_pts[1] - face_pts[0])
    yy, xx = np.mgrid[:face.shape[0], :face.shape[1]]
    ok = binary_erosion(sil, iterations=max(2, int(0.08 * face.shape[0])))
    for (x, y), r in zip(face_pts[:3], (0.35 * d, 0.35 * d, 0.45 * d)):
        ok &= (xx - x) ** 2 + (yy - y) ** 2 > r * r
    size = (max(8, round(face.shape[0] * scale)), max(8, round(face.shape[1] * scale)))
    small = resize(face.astype(np.float64), size + (3,), anti_aliasing=True, preserve_range=True)
    ok_small = resize(ok.astype(float), size, anti_aliasing=False) > 0.9
    return small, ok_small


def quilt(src: np.ndarray, ok: np.ndarray, shape: tuple[int, int], rng: np.random.Generator,
          patch: int = 10, overlap: int = 3, tries: int = 24) -> np.ndarray:
    """An RGB texture of the given shape laid from patches of src whose pixels are all ok."""
    h, w = shape
    ys, xs = np.nonzero(ok[:ok.shape[0] - patch + 1, :ok.shape[1] - patch + 1])
    cand = [(y, x) for y, x in zip(ys, xs) if ok[y:y + patch, x:x + patch].all()]
    if not cand:                                                    # too little plain fur: its mean colour
        return np.broadcast_to(src[ok].mean(0) if ok.any() else src.reshape(-1, 3).mean(0), (h, w, 3)).copy()
    out = np.zeros((h + patch, w + patch, 3))
    wsum = np.zeros((h + patch, w + patch, 1))
    ramp = np.minimum(1, (np.arange(patch) + 1) / (overlap + 1))
    feather = np.minimum.outer(np.minimum(ramp, ramp[::-1]), np.minimum(ramp, ramp[::-1]))[..., None]
    step = patch - overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            have = wsum[y:y + patch, x:x + patch] > 0
            best, best_err = None, np.inf
            for i in rng.choice(len(cand), min(tries, len(cand)), replace=False):
                cy, cx = cand[i]
                p = src[cy:cy + patch, cx:cx + patch]
                if rng.random() < 0.5:
                    p = p[:, ::-1]
                if not have.any():
                    best = p
                    break
                cur = out[y:y + patch, x:x + patch] / np.maximum(wsum[y:y + patch, x:x + patch], 1e-6)
                err = (((p - cur) ** 2) * have).sum()
                if err < best_err:
                    best, best_err = p, err
            out[y:y + patch, x:x + patch] += best * feather
            wsum[y:y + patch, x:x + patch] += feather
    return (out / np.maximum(wsum, 1e-6))[:h, :w]


def bake(frame_rgba: np.ndarray, head: np.ndarray, body: np.ndarray, head_pts: np.ndarray,
         face: np.ndarray, face_pts: np.ndarray, head_scale: float, rng: np.random.Generator) -> np.ndarray:
    """A sprite frame (RGBA uint8) with the cat painted in: face over head, fur texture over body; the frame's
    outline and transparency stay, and so do the lines and shading drawn inside the body."""
    img = frame_rgba.astype(np.float64).copy()
    img[head, :3] = warp_face(face, face_pts, head, head_pts)[head]
    if body.any():
        # quilted at the face's own size and then shrunk, so a small head still has enough fur to choose from
        src, ok = fur_source(face, face_pts, 1.0)
        ys, xs = np.nonzero(body)
        y0, x0 = ys.min(), xs.min()
        h, w = ys.max() - y0 + 1, xs.max() - x0 + 1
        big = quilt(src, ok, (math.ceil(h / head_scale), math.ceil(w / head_scale)), rng)
        tex = resize(big, (h, w, 3), anti_aliasing=True, preserve_range=True)
        # the frame's grey against its plain fill: 1 on the fill, below 1 on lines and shadow
        lum = frame_rgba[..., :3].astype(np.float64).mean(-1)
        shade = lum[body] / np.median(lum[body])
        img[ys, xs, :3] = tex[ys - y0, xs - x0] * shade[:, None]
    return img.clip(0, 255).astype(np.uint8)
