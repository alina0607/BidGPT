"""A Kaggle job for one Brush GPT L run (owner, 2026-10-20), as data/kaggle/brush_l5 was for L5: a dataset with the
code and the run's data file (plus a starting checkpoint for --init-from), and a T4 kernel that trains, samples 64
paintings at temperature 1 and copies the run folder to /kaggle/working/out/<name>. See docs/kaggle.md.

    PYTHONPATH=. python tools/kaggle_brush_l.py --name L5m --data data/processed/brush_l/data_L5m.npz --layers 5
    PYTHONPATH=. python tools/kaggle_brush_l.py --name L8 --data data/processed/brush_l/data_L8m.npz --layers 8 \
        --init-from runs/brushgpt/L5m/best.pt --train-args --batch 2 --accum 8
then
    kaggle datasets create -p data/kaggle/brush_<name>/data -r zip      (new dataset; `version` to update one)
    kaggle kernels push -p data/kaggle/brush_<name>/kernel
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ML = Path(__file__).resolve().parents[1]

RUN_PY = '''"""Brush GPT {name} on a Kaggle T4 (tools/kaggle_brush_l.py; sellanything_ml/sketch/brushgpt_l.py): train on {data}
(dataset {dataset}, with the code), then sample 64 paintings at temperature 1. Stops cleanly before Kaggle's 12-hour
limit; the run folder is copied to /kaggle/working/out/{name}."""
import glob, json, os, shutil, subprocess, sys
data = glob.glob("/kaggle/input/**/{data}", recursive=True)[0]
code = glob.glob("/kaggle/input/**/sellanything_ml/sketch/brushgpt_l.py", recursive=True)[0].rsplit("/sellanything_ml/", 1)[0]
shutil.copytree(code, "/kaggle/working/ml", ignore=shutil.ignore_patterns("*.npz", "*.pt"))
env = {{**os.environ, "PYTHONPATH": "/kaggle/working/ml", "CUDA_VISIBLE_DEVICES": "0"}}
print(json.dumps({{"data": data, "gpus": subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout}}), flush=True)
run = lambda *a: subprocess.run([sys.executable, "-m", "sellanything_ml.sketch.brushgpt_l", *a], cwd="/kaggle/working/ml", env=env)
init = {init!r}
extra = []
if init:
    init_path = glob.glob("/kaggle/input/**/" + init, recursive=True)[0]
    extra = ["--init-from", init_path]
r = run("train", "--data", data, "--name", "{name}", "--layers", "{layers}", "--max-steps", "{max_steps}", "--eval-every", "100",
        "--patience", "8", "--stop-after-min", "{stop_min}", *extra, *{train_args!r})
print(json.dumps({{"train_returncode": r.returncode}}), flush=True)
r = run("sample", "--name", "{name}", "--n", "64", "--temperature", "1.0", "--seed", "0", "--max-len", "{sample_len}")
print(json.dumps({{"sample_returncode": r.returncode}}), flush=True)
os.makedirs("/kaggle/working/out", exist_ok=True)
shutil.copytree("/kaggle/working/ml/runs/brushgpt/{name}", "/kaggle/working/out/{name}", dirs_exist_ok=True)
shutil.rmtree("/kaggle/working/ml")
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--stop-min", type=float, default=620)
    ap.add_argument("--train-args", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()
    slug = a.name.lower()
    k = ML / "data" / "kaggle" / f"brush_{slug}"
    dataset, kernel = f"brush-{slug}-data", f"brush-gpt-{slug}-train"
    if (k / "data").exists():
        shutil.rmtree(k / "data")
    (k / "data").mkdir(parents=True)
    (k / "kernel").mkdir(parents=True, exist_ok=True)
    shutil.copytree(ML / "sellanything_ml", k / "data" / "ml" / "sellanything_ml",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(a.data, k / "data" / a.data.name)
    if a.init_from:
        shutil.copy(a.init_from, k / "data" / f"init_{a.init_from.parent.name}.pt")
    (k / "data" / "dataset-metadata.json").write_text(json.dumps(
        {"title": dataset.replace("-", " "), "id": f"alinachuang/{dataset}", "licenses": [{"name": "other"}]}))
    (k / "kernel" / "run.py").write_text(RUN_PY.format(
        name=a.name, data=a.data.name, dataset=dataset, layers=a.layers, max_steps=a.max_steps, stop_min=a.stop_min,
        init=f"init_{a.init_from.parent.name}.pt" if a.init_from else "", train_args=a.train_args,
        sample_len=3072 if a.layers == 5 else 9728))
    (k / "kernel" / "kernel-metadata.json").write_text(json.dumps({
        "id": f"alinachuang/{kernel}", "title": kernel.replace("-", " "), "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"alinachuang/{dataset}"], "competition_sources": [],
        "kernel_sources": []}, indent=1))
    print(json.dumps({"folder": str(k), "dataset": dataset, "kernel": kernel}))


if __name__ == "__main__":
    main()
