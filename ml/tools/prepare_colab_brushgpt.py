"""Package a Brush GPT ablation of tools/modal_brushgpt.py for Google Colab's free T4 (owner, 2026-10-17).

    PYTHONPATH=. python tools/prepare_colab_brushgpt.py --runs A_list,B_overlap

The package holds the training code (sellanything_ml: llm/model.py and sketch/brushgpt.py, brushtokens.py,
catbrush.py), the frozen token file the run was defined on (tokens_trial_1792.npz, the data of run C) and a notebook.
The flags are modal_brushgpt.py's MODEL + TRAIN + the run's own, so the run is the one that script defines. One flag
differs: the time limit. On the A100 the 35-minute limit never bound (3,000 steps take about 12 minutes there); on a
T4 it would cut the run short, so it is raised to 150 minutes and the step limit and patience decide, as they did on
the A100. Results and checkpoints go to Google Drive (--resume 1), so a disconnected session continues from its
last evaluation. Bring runs/<name>/ back into ml/runs/brushgpt/<name>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML / "tools"))
import modal_brushgpt as M  # noqa: E402

CODE = ["sellanything_ml/__init__.py", "sellanything_ml/llm/__init__.py", "sellanything_ml/llm/model.py",
        "sellanything_ml/sketch/__init__.py", "sellanything_ml/sketch/brushgpt.py",
        "sellanything_ml/sketch/brushtokens.py", "sellanything_ml/sketch/catbrush.py"]
TIME_LIMIT_MIN = 150
DRIVE = "/content/drive/MyDrive/brushgpt_colab"


def flags(name: str) -> list[str]:
    return M.MODEL + M.TRAIN + M.RUNS[name] + ["--stop-after-min", str(TIME_LIMIT_MIN), "--resume", "1"]


def notebook(names: list[str], tag: str, data_name: str, sha: str) -> dict:
    cells = [
        ("markdown", f"# Brush GPT {', '.join(names)} on a Colab T4\n\nRuntime → Change runtime type → **T4 GPU**. Upload "
                     f"`brushgpt_{tag}.zip` to Google Drive at `MyDrive/brushgpt_colab/`, then run the cells in order "
                     f"(about an hour per run). If the session disconnects, run all cells again: a finished run is "
                     f"skipped and an unfinished one resumes from its last evaluation."),
        ("code", "from google.colab import drive\ndrive.mount('/content/drive')"),
        ("code", f"!mkdir -p /content/pkg {DRIVE}/runs && cd /content/pkg && unzip -oq {DRIVE}/brushgpt_{tag}.zip\n"
                 f"!pip install -q pytorch-fid==0.3.0\n!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader\n"
                 f"!cd /content/pkg && sha256sum data/{data_name} | cut -c1-16   # expect {sha}"),
    ]
    for name in names:
        train = " ".join(["python -m sellanything_ml.sketch.brushgpt train --name", name, "--data", f"data/{data_name}",
                          "--runs", f"{DRIVE}/runs"] + flags(name))
        evaluate = " ".join(["python -m sellanything_ml.sketch.brushgpt eval --name", name, "--data",
                             f"data/{data_name}", "--runs", f"{DRIVE}/runs", "--n", str(M.SAMPLES)])
        done = f"{DRIVE}/runs/{name}/eval.json"
        cells.append(("code", f"# {name}: train, then evaluate (256 samples, FID, nearest training painting)\n"
                              f"%cd /content/pkg\n"
                              f"!test -f {done} && echo '{name} already finished' || "
                              f"(BRUSHGPT_DEVICE=cuda {train} 2>&1 | tail -30 && BRUSHGPT_DEVICE=cuda {evaluate} 2>&1 | tail -15)\n"
                              f"!cat {done}"))
    out = []
    for kind, src in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        out.append(cell)
    return {"cells": out, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="A_list,B_overlap", help="comma-separated run names of modal_brushgpt.RUNS")
    a = ap.parse_args()
    names = a.runs.split(",")
    for n_ in names:
        if n_ not in M.RUNS:
            raise SystemExit(f"unknown run {n_}")
    tag = "_".join(n_.split("_")[0] for n_ in names)
    data = ML / M.TRIAL
    sha = hashlib.sha256(data.read_bytes()).hexdigest()[:16]
    out = ML / "runs" / "colab_brushgpt"
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / f"brushgpt_{tag}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in CODE:
            z.write(ML / rel, rel)
        z.write(data, f"data/{data.name}")
    nb = out / f"colab_brushgpt_{tag}.ipynb"
    nb.write_text(json.dumps(notebook(names, tag, data.name, sha), indent=1))
    print(json.dumps({"zip": str(zpath), "zip_mb": round(zpath.stat().st_size / 1e6, 1), "notebook": str(nb),
                      "data_sha256_16": sha, "flags": {n_: flags(n_) for n_ in names}}, indent=1))
    shutil.copy(nb, Path.home() / "Downloads" / nb.name)
    shutil.copy(zpath, Path.home() / "Downloads" / zpath.name)
    print("copied both to ~/Downloads")


if __name__ == "__main__":
    main()
