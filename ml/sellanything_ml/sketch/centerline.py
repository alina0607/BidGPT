"""Ink (bold contour lines) → centreline strokes with a width each: skeleton → paths between end and junction
pixels (crossing number) → drop short spurs → join at junctions where they run on smoothly → bridge small aligned
gaps → simplify to TOL px. Width = twice the median distance from the path to the edge of the ink. A blob whose
skeleton is too short to keep becomes a dot (a stroke of one point) of the blob's equal-area diameter."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label
from skimage.measure import approximate_polygon
from skimage.morphology import skeletonize

MIN_LEN = 6
JOIN_DEG = 45
GAP = 6
GAP_DEG = 40
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]       # clockwise around a pixel


def trace(skel: np.ndarray) -> list[tuple[np.ndarray, bool, bool]]:
    """Pixel paths (y, x) between end and junction pixels, plus closed loops. A pixel is an end or a junction by
    its crossing number (ink runs met going once around it: 1 = end, 3+ = junction), not by its neighbour count,
    which also counts the corner of every diagonal staircase and cut lines into specks."""
    h, w = skel.shape
    pad = np.pad(skel, 1)
    ring = np.stack([pad[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx] for dy, dx in RING])
    crossings = (ring & ~np.roll(ring, 1, axis=0)).sum(0)
    node = skel & (crossings != 2)
    seen = np.zeros_like(skel)

    def step(p):
        y, x = p
        cand = [(y + dy, x + dx) for dy, dx in NB if 0 <= y + dy < h and 0 <= x + dx < w and skel[y + dy, x + dx]
                and not seen[y + dy, x + dx]]
        cand.sort(key=lambda q: abs(q[0] - y) + abs(q[1] - x))           # straight neighbours before diagonal
        return cand

    def walk(start, first):
        path = [start, first]
        while not node[path[-1]]:
            seen[path[-1]] = True
            nxt = [q for q in step(path[-1]) if q != path[-2]]
            if not nxt:
                break
            path.append(nxt[0])
        return path

    paths = []
    for p in zip(*np.nonzero(node)):
        for q in step(p):
            if not node[q] or q > p:                                   # a node–node edge only once
                paths.append(walk(p, q))
    for p in zip(*np.nonzero(skel & ~node)):                          # loops with no end or junction (an eye)
        if not seen[p]:
            seen[p] = True
            nxt = step(p)
            if nxt:
                paths.append([p] + walk(p, nxt[0])[1:] + [p])
    free = lambda q: bool(crossings[q] <= 1)                            # noqa: E731
    return [(np.array(pt, float)[:, ::-1], free(pt[0]) and pt[0] != pt[-1], free(pt[-1]) and pt[0] != pt[-1])
            for pt in paths]                                               # (x, y) points, start free, end free


def length(p):
    return np.linalg.norm(np.diff(p, axis=0), axis=1).sum()


def clean(paths):
    """Drop short paths with a free end, then join paths at junctions: two that meet alone become one, and among
    three or more the pair that bends least, if under JOIN_DEG."""
    ps = [[p, a, b] for p, a, b in paths if length(p) >= MIN_LEN or not (a or b)]

    def heading(p, at_start):                                              # direction leaving the junction
        q = p if at_start else p[::-1]
        v = q[min(5, len(q) - 1)] - q[0]
        return v / (np.linalg.norm(v) + 1e-9)

    while True:
        ends = {}
        for i, (p, a, b) in enumerate(ps):
            if (p[0] == p[-1]).all() and len(p) > 2:
                continue                                                   # closed loop
            if not a:
                ends.setdefault(tuple(p[0]), []).append((i, True))
            if not b:
                ends.setdefault(tuple(p[-1]), []).append((i, False))
        best = None
        for j, es in ends.items():
            for x in range(len(es)):
                for y in range(x + 1, len(es)):
                    (i1, s1), (i2, s2) = es[x], es[y]
                    bend = 180 - np.degrees(np.arccos(np.clip(heading(ps[i1][0], s1) @ heading(ps[i2][0], s2), -1, 1)))
                    if (len(es) == 2 or bend < JOIN_DEG) and (best is None or bend < best[0]):
                        best = (bend, (i1, s1), (i2, s2))
        if best is None:
            return [(p, a, b) for p, a, b in ps]
        (_, (i1, s1), (i2, s2)) = best
        p1, a1, b1 = ps[i1]
        p2, a2, b2 = ps[i2]
        if s1:                                                             # make p1 end at the junction
            p1, a1, b1 = p1[::-1], b1, a1
        if i1 == i2:
            ps[i1] = [np.vstack([p1, p1[:1]]), False, False]
            continue
        if not s2:                                                         # and p2 start there
            p2, a2, b2 = p2[::-1], b2, a2
        ps[i1] = [np.vstack([p1, p2[1:]]), a1, b2]
        del ps[i2]


def bridge(paths):
    """Join free ends closer than GAP whose headings both point across the gap (within GAP_DEG), closest first."""
    ps = [[p, a, b] for p, a, b in paths]

    def tip(p, at_start):                                                 # end point and outward heading
        q = p if at_start else p[::-1]
        v = q[0] - q[min(5, len(q) - 1)]
        return q[0], v / (np.linalg.norm(v) + 1e-9)

    cos_max = np.cos(np.radians(GAP_DEG))
    while True:
        tips = [(i, st, *tip(p, st)) for i, (p, a, b) in enumerate(ps) for st, free in ((True, a), (False, b)) if free]
        best = None
        for x in range(len(tips)):
            for y in range(x + 1, len(tips)):
                (i1, s1, e1, h1), (i2, s2, e2, h2) = tips[x], tips[y]
                d = e2 - e1
                dist = np.linalg.norm(d)
                if i1 == i2 or dist > GAP or dist == 0:
                    continue
                u = d / dist
                if h1 @ u >= cos_max and h2 @ -u >= cos_max and (best is None or dist < best[0]):
                    best = (dist, (i1, s1), (i2, s2))
        if best is None:
            return [p for p, _, _ in ps]
        _, (i1, s1), (i2, s2) = best
        p1, a1, b1 = ps[i1]
        p2, a2, b2 = ps[i2]
        if s1:
            p1, a1, b1 = p1[::-1], b1, a1
        if not s2:
            p2, a2, b2 = p2[::-1], b2, a2
        ps[i1] = [np.vstack([p1, p2]), a1, b2]
        del ps[i2]


def strokes(mask: np.ndarray, tol: float = 1.5, min_area: float = 12) -> list[tuple[np.ndarray, float]]:
    """(points (x, y), width) for the ink in mask, longest first."""
    if not mask.any():
        return []
    dt = distance_transform_edt(mask)
    paths = bridge(clean(trace(skeletonize(mask))))
    out = []
    for p in paths:
        ij = np.clip(np.round(p[:, ::-1]).astype(int), 0, np.array(mask.shape) - 1)
        out.append((approximate_polygon(p, tol), max(1.0, 2 * float(np.median(dt[ij[:, 0], ij[:, 1]])))))
    lab, n = cc_label(mask)
    hit = np.zeros(n + 1, bool)
    for p, _ in out:
        ij = np.clip(np.round(p[:, ::-1]).astype(int), 0, np.array(mask.shape) - 1)
        hit[lab[ij[:, 0], ij[:, 1]]] = True
    for i in range(1, n + 1):
        area = (lab == i).sum()
        if not hit[i] and area >= min_area:
            cy, cx = np.argwhere(lab == i).mean(0)
            out.append((np.array([[cx, cy]]), 2 * np.sqrt(area / np.pi)))
    return sorted(out, key=lambda s: -length(s[0]) if len(s[0]) > 1 else 0)
