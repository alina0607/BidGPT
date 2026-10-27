"""Is L8's fur messier than its training paintings? (owner, 2026-10-21) Two measures on the three hair layers (fur,
dark hairs, light hairs), per layer:

  neighbours  for every hair stroke, the mean difference in direction to its 6 nearest hair strokes of the same layer
              (direction is a line, so the difference is folded to 0-90 degrees). Real fur flows: neighbours point the
              same way and the difference is small. Strokes pointing anywhere at random would give 45 degrees.
  same cat    for paintings L8 drew from a cat's real plan: the difference in direction between each generated hair and
              the nearest hair of that cat's training painting (same layer). Low means L8 knows which way a cat's fur
              flows at each place on the face (the plan gives the face's shape and angle, not the fur).

  photo       against the photo's own fur direction (catbrush.orientation: along the local structure, the direction
              the repaint's strokes start from), only where that direction is clear (coherence of the structure tensor
              at least 0.5): the difference to each hair of the training painting and of L8's painting from the real
              plan; and, as how smoothly real fur flows, the neighbour difference of the photo's own direction sampled
              at the training hairs.

Paintings: the L8 training paintings of the first 64 validation cats; L8's paintings of the same 64 cats from their real
plans (tools/eye_upper_bound_l8.py, temperature 1); and 64 free L8 samples (Kaggle, temperature 1, seed 0). CPU.

    PYTHONPATH=.:tools python tools/fur_flow_l8.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import torch
import torch.nn.functional as F

from sellanything_ml.sketch import brushtokens_l as L

ML = Path(__file__).resolve().parents[1]
CATS8 = ML / "data" / "processed" / "brush_l" / "cats8"
K = 6


def hair(tokens: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Layer -> (centres (n, 2), directions (n,) in radians over [0, pi)) of the hair strokes."""
    out = {}
    lays = L.decode_layers(tokens)
    for k in L.DETAIL:
        if k < len(lays) and len(lays[k]):
            v = L.stroke_values(lays[k], k)
            out[k] = (v["xy"], np.mod(v["angle"], np.pi))
    return out


def fold(d: np.ndarray) -> np.ndarray:
    d = np.abs(d) % np.pi
    return np.degrees(np.minimum(d, np.pi - d))


def neighbours(h: dict) -> dict[int, float]:
    res = {}
    for k, (xy, a) in h.items():
        if len(xy) <= K:
            continue
        _, idx = cKDTree(xy).query(xy, K + 1)
        res[k] = float(fold(a[idx[:, 1:]] - a[:, None]).mean())
    return res


def same_cat(gen: dict, real: dict) -> dict[int, float]:
    res = {}
    for k in gen:
        if k in real and len(real[k][0]):
            _, idx = cKDTree(real[k][0]).query(gen[k][0], 1)
            res[k] = float(fold(gen[k][1] - real[k][1][idx]).mean())
    return res


def photo_field(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The photo's direction (as catbrush.orientation) and how clear it is (0-1), (H, W) each."""
    g = torch.from_numpy(target.astype(np.float32)).mean(-1)[None, None]
    kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=g.dtype) / 8
    gx, gy = F.conv2d(g, kx, padding=1), F.conv2d(g, kx.transpose(2, 3), padding=1)
    blur = lambda t: F.avg_pool2d(t, 9, 1, 4)                           # noqa: E731
    jxx, jyy, jxy = blur(gx * gx), blur(gy * gy), blur(gx * gy)
    ang = (0.5 * torch.atan2(2 * jxy, jxx - jyy) + np.pi / 2)[0, 0].numpy() % np.pi
    coh = (torch.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-9))[0, 0].numpy()
    return ang, coh


def vs_photo(h: dict, field: tuple) -> dict[int, float]:
    ang, coh = field
    res = {}
    for k, (xy, a) in h.items():
        ix, iy = np.clip(xy[:, 0].astype(int), 0, ang.shape[1] - 1), np.clip(xy[:, 1].astype(int), 0, ang.shape[0] - 1)
        m = coh[iy, ix] >= 0.5
        if m.sum() > K:
            res[k] = float(fold(a[m] - ang[iy, ix][m]).mean())
    return res


def photo_neighbours(h: dict, field: tuple) -> dict[int, float]:
    ang, coh = field
    res = {}
    for k, (xy, _) in h.items():
        ix, iy = np.clip(xy[:, 0].astype(int), 0, ang.shape[1] - 1), np.clip(xy[:, 1].astype(int), 0, ang.shape[0] - 1)
        res[k] = neighbours({k: (xy, ang[iy, ix])}).get(k)
    return {k: v for k, v in res.items() if v is not None}


def summary(rows: list[dict]) -> dict:
    out = {}
    for k in L.DETAIL:
        v = np.array([r[k] for r in rows if k in r])
        out[str(k)] = [round(float(v.mean()), 2), round(float(v.std() / np.sqrt(len(v))), 2), int(len(v))]
    return out


def main() -> None:
    g = np.load(ML / "runs/eye_upper_bound_l8/gen_t1.0_s3.npz")
    names = [str(x) for x in g["names"]]
    gen_plan = [g["tokens"][i][:g["lengths"][i]] for i in range(len(names))]
    real = [np.load(CATS8 / f"{n}.npz")["tokens"] for n in names]
    f = np.load(ML / "runs/brushgpt/L8/samples_t1.0_s0.npz")
    free = [f["tokens"][i][:f["lengths"][i]] for i in range(len(f["lengths"]))]
    real_h = [hair(t) for t in real]
    plan_h = [hair(t) for t in gen_plan]
    free_h = [hair(t) for t in free]
    def repaint_target(n: str) -> np.ndarray:
        f_ = next(p for root in ("kaggle_t4_run1/out", "kaggle_t4_run2/out", "mac")
                  for p in (ML / "data/processed/repaint" / root).glob(f"shard_*/{n}.npz"))
        return np.load(f_)["target"] / 255
    fields = [photo_field(repaint_target(n)) for n in names]
    rng = np.random.default_rng(0)
    rand_h = [{k: (xy, rng.uniform(0, np.pi, len(xy))) for k, (xy, _) in h.items()} for h in real_h]
    res = {
        "neighbours_deg": {"training": summary([neighbours(h) for h in real_h]),
                           "l8_real_plan": summary([neighbours(h) for h in plan_h]),
                           "l8_free": summary([neighbours(h) for h in free_h]),
                           "random_directions": summary([neighbours(h) for h in rand_h])},
        "same_cat_deg": {"l8_real_plan_vs_training": summary([same_cat(p, r) for p, r in zip(plan_h, real_h)]),
                         "random_vs_training": summary([same_cat(p, r) for p, r in zip(rand_h, real_h)])},
        "photo_deg": {"training_vs_photo": summary([vs_photo(h, fl) for h, fl in zip(real_h, fields)]),
                      "l8_real_plan_vs_photo": summary([vs_photo(h, fl) for h, fl in zip(plan_h, fields)]),
                      "random_vs_photo": summary([vs_photo(h, fl) for h, fl in zip(rand_h, fields)]),
                      "photo_direction_neighbours": summary([photo_neighbours(h, fl) for h, fl in zip(real_h, fields)])},
        "hair_strokes": {"training": [int(np.mean([len(h[k][0]) for h in real_h if k in h])) for k in L.DETAIL],
                         "l8_real_plan": [int(np.mean([len(h[k][0]) for h in plan_h if k in h])) for k in L.DETAIL],
                         "l8_free": [int(np.mean([len(h[k][0]) for h in free_h if k in h])) for k in L.DETAIL]},
    }
    out = ML / "runs/brushgpt/L8/fur_flow.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
