"""Package the CAT-dataset cats of the digital repaint (tools/repaint.py, shards 11-30) for Google Colab's free GPU
(owner, 2026-10-20: start the CAT repaint on Colab now; whether to paint all of it is decided after the data-size
experiment L5h / L5 / L5m).

    PYTHONPATH=. python tools/prepare_colab_repaint.py

writes data/colab_repaint/: colab_repaint_pkg.zip (the repaint code exactly as on Kaggle, and the input shards) and
repaint_cat_colab.ipynb. The owner puts both in Google Drive at MyDrive/sell_anything_colab/repaint_cat/ and runs the
notebook; paintings go to .../repaint_cat/out/shard_NNN/ on Drive, one file per cat, written whole (a temporary file
renamed), so a disconnected session loses at most the cat it was painting and a new session skips every cat already
there. Colab goes through the shards from the front; another machine can take them from the back (--reverse
--reverse-cats) and the two meet in the middle. Paintings come back with tools/repaint_merge.py --check.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
OUT = ML / "data" / "colab_repaint"
INPUTS = ML / "data" / "processed" / "repaint" / "inputs"
SHARDS = range(11, 31)
CODE = ["tools/repaint.py", "tools/ear_strands.py", "tools/eye_crop_styles.py", "tools/digital_face.py",
        "tools/review_export.py", "tools/mask_audit.py", "sellanything_ml/__init__.py", "sellanything_ml/sketch/__init__.py",
        "sellanything_ml/sketch/catbrush.py"]
DRIVE = "/content/drive/MyDrive/sell_anything_colab/repaint_cat"


def cell(kind: str, text: str) -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def notebook(n_cats: int) -> dict:
    cells = [
        cell("markdown", f"""
# 用 Colab 的免費 GPU 重畫 CAT 資料集的貓（數位畫法，shard 11–30）

共 {n_cats:,} 隻 CAT 貓。程式與 Kaggle 上重畫 AFHQ 時完全相同（`tools/repaint.py`）。

**第一次使用**
1. 把 `colab_repaint_pkg.zip` 放到 Google Drive 的 `我的雲端硬碟/sell_anything_colab/repaint_cat/`。
2. 「執行階段 → 變更執行階段類型」選 **T4 GPU**。
3. 由上往下執行每一格。最後一格會一直畫，每 100 隻印一行進度。

**中斷了怎麼辦**：重新連線後再從第一格執行一次，**等第一格印出「Mounted at /content/drive」再執行下一格**
（Drive 沒掛上就執行最後一格，會在執行環境裡建出 `out/`，之後就掛不上；那時要「執行階段 → 中斷連線並刪除執行階段」再重來）。已經畫好的貓存在 Drive 的 `out/` 裡，會自動跳過，最多只損失當時正在畫的那一隻。

每隻約 20 秒（T4），每隻的檔案約 0.36 MB，全部畫完約需 3.5 GB 的 Drive 空間。
"""),
        cell("code", """
from google.colab import drive
drive.mount('/content/drive')
"""),
        cell("code", f"""
import os, subprocess, torch
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), "沒有 GPU：請到「執行階段 → 變更執行階段類型」選 T4 GPU"
assert os.path.ismount("/content/drive"), "Google Drive 沒有掛上：先只執行第一格，等它印出 Mounted at 再往下"
DRIVE = "{DRIVE}"
assert os.path.exists(f"{{DRIVE}}/colab_repaint_pkg.zip"), "找不到 colab_repaint_pkg.zip，請確認放在 " + DRIVE
"""),
        cell("code", """
import zipfile
if not os.path.exists("/content/pkg/ml/tools/repaint.py"):
    with zipfile.ZipFile(f"{DRIVE}/colab_repaint_pkg.zip") as z:
        z.extractall("/content/pkg")
print(sorted(os.listdir("/content/pkg/inputs")))
"""),
        cell("code", """
import glob, json, sys, time
OUT = f"{DRIVE}/out"
os.makedirs(OUT, exist_ok=True)
env = {**os.environ, "PYTHONPATH": "/content/pkg/ml:/content/pkg/ml/tools"}

def done():
    return len([f for f in glob.glob(f"{OUT}/shard_*/*.npz") if not f.endswith(".tmp.npz")])

print("已經畫好：", done(), flush=True)
while True:                       # a fresh process every 100 cats, so memory cannot pile up (as on Kaggle and the Mac)
    r = subprocess.run([sys.executable, "tools/repaint.py", "--shards", "11-30", "--source", "cat", "--device", "cuda",
                        "--inputs", "/content/pkg/inputs", "--out", OUT, "--max-cats", "100"],
                       cwd="/content/pkg/ml", env=env, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    print(time.strftime("%H:%M"), last, "已經畫好：", done(), flush=True)
    if '"max_cats"' not in last:  # finished all shards, or an error
        print(r.stderr[-3000:])
        break
"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n_cats = 0
    for sh in SHARDS:
        z = np.load(INPUTS / f"shard_{sh:03d}.npz")
        n_cats += int((z["source"] == "cat").sum())
    with zipfile.ZipFile(OUT / "colab_repaint_pkg.zip", "w") as zf:
        for f in CODE:
            zf.write(ML / f, f"ml/{f}", compress_type=zipfile.ZIP_DEFLATED)
        for sh in SHARDS:                                           # already compressed npz: stored as they are
            zf.write(INPUTS / f"shard_{sh:03d}.npz", f"inputs/shard_{sh:03d}.npz", compress_type=zipfile.ZIP_STORED)
    (OUT / "repaint_cat_colab.ipynb").write_text(json.dumps(notebook(n_cats), ensure_ascii=False, indent=1))
    size = (OUT / "colab_repaint_pkg.zip").stat().st_size / 1e9
    print(json.dumps({"cats": n_cats, "zip_gb": round(size, 2), "folder": str(OUT)}))


if __name__ == "__main__":
    main()
