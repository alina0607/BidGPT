"""Which AFHQ cats have a finished repaint, on which machine, and which are still to paint (owner, 2026-10-19).

Every cat of the repaint inputs (tools/repaint_prep.py) is looked up in the output folders given (Mac, Kaggle sessions):
a file counts only if it loads and has all eight layers. A cat painted on more than one machine is kept once (the
first folder given wins); the others are listed as duplicates. The cats still to paint are written as source__id
lists: --mac-keep cats are left to the Mac (its current shard, from the last cat backwards, as tools/run_repaint_mac.sh
paints them) and the rest are split evenly between Kaggle's two GPUs (todo_gpu0.txt, todo_gpu1.txt).

    PYTHONPATH=.:tools python tools/repaint_merge.py --done data/processed/repaint/mac data/processed/repaint/kaggle_t4_run1/out \
        --todo-out data/kaggle/repaint_afhq/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
INPUTS = ML / "data" / "processed" / "repaint" / "inputs"
LAYERS = 8


def complete(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as z:
            return int(z["layers"]) == LAYERS and all(f"l{k}_xy" in z.files for k in range(LAYERS)) and "painting" in z.files
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--done", type=Path, nargs="+", required=True, help="repaint output folders (shard_NNN/ inside)")
    ap.add_argument("--source", default="afhq")
    ap.add_argument("--mac-shards", default="9", help="shards the Mac is painting backwards now (comma list)")
    ap.add_argument("--mac-keep", type=int, default=0, help="how many of the still-to-paint cats of the Mac's shards it keeps")
    ap.add_argument("--todo-out", type=Path, default=None)
    ap.add_argument("--check", action="store_true", help="open every file (slow) instead of trusting names")
    a = ap.parse_args()
    index = json.loads((INPUTS / "index.json").read_text())
    wanted = []                                                    # (shard, position, name) in paint order
    for k, sh in enumerate(index["shards"]):
        z = np.load(INPUTS / sh["shard"])
        wanted += [(k, i, f"{s}__{c}") for i, (s, c) in enumerate(zip(z["source"], z["id"])) if str(s) == a.source]
    found: dict[str, Path] = {}
    dup, bad = [], []
    per_folder = {}
    for root in a.done:
        n = 0
        for f in sorted(root.glob("shard_*/*.npz")):
            if f.name.endswith(".tmp.npz"):
                continue
            name = f.stem
            if a.check and not complete(f):
                bad.append(str(f))
                continue
            n += 1
            if name in found:
                dup.append(name)
            else:
                found[name] = f
        per_folder[str(root)] = n
    todo = [(k, i, n) for k, i, n in wanted if n not in found]
    mac_shards = {int(s) for s in a.mac_shards.split(",") if s}
    mac_side = sorted([t for t in todo if t[0] in mac_shards], key=lambda t: (t[0], -t[1]))[:a.mac_keep]
    rest = [t for t in todo if t not in mac_side]
    gpu = [rest[0::2], rest[1::2]]
    summary = {"wanted": len(wanted), "found": len(found), "todo": len(todo), "mac_keeps": len(mac_side),
               "kaggle": [len(g) for g in gpu], "duplicates": len(dup), "bad": len(bad), "per_folder": per_folder,
               "todo_by_shard": {int(s): sum(1 for t in todo if t[0] == s) for s in sorted({t[0] for t in todo})}}
    print(json.dumps(summary, indent=1))
    if bad:
        print("bad files:", bad[:20])
    if a.todo_out:
        a.todo_out.mkdir(parents=True, exist_ok=True)
        for g in (0, 1):
            (a.todo_out / f"todo_gpu{g}.txt").write_text("\n".join(n for *_, n in gpu[g]) + "\n")
        (a.todo_out / "todo_mac.txt").write_text("\n".join(n for *_, n in mac_side) + "\n")


if __name__ == "__main__":
    main()
