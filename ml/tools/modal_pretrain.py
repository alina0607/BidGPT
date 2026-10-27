"""Pre-train the buyer GPT on a Modal A100, streaming metrics and checkpoints back to this machine.

    cd ml && modal run tools/modal_pretrain.py --smoke        # 60 steps, checkpoint every 50: checks the plumbing
    cd ml && modal run --detach tools/modal_pretrain.py

The code, the tokenizer and the tokenized FineWeb-Edu shards travel inside the container image; nothing is kept in
a Modal Volume (volume storage is billed outside the workspace's usage limit). The container yields every new
metrics line, and the whole training state every CKPT_EVERY steps, which the local side writes to RUN_DIR, so a
failed run can be resumed from the last state that arrived.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ML = Path(__file__).resolve().parents[1]
DATA = ML / "data" / "processed" / "pretrain_20shards"
RUN_DIR = ML / "runs" / "pretrain_large"
ARGS = ["--preset", "large", "--tokens", "1.28e9", "--batch", "64", "--accum", "1", "--block", "512",
        "--lr", "6e-4", "--warmup", "700", "--eval-every", "1000", "--dtype", "bfloat16"]
CKPT_EVERY = 10_000

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "numpy", "sentencepiece", "pyarrow")
         .add_local_dir(ML / "sellanything_ml", "/ml/sellanything_ml", ignore=["**/__pycache__/**"])
         .add_local_file(ML / "checkpoints" / "tokenizer" / "sp8192.model", "/ml/checkpoints/tokenizer/sp8192.model")
         .add_local_file(DATA / "train.bin", "/data/train.bin")
         .add_local_file(DATA / "val.bin", "/data/val.bin"))
app = modal.App("buyer-gpt-pretrain", image=image)


@app.function(gpu="A100-40GB", timeout=6 * 3600)
def pretrain(args: list[str], ckpt_every: int):
    import subprocess
    import time
    run = Path("/tmp/run")
    run.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["python", "-m", "sellanything_ml.llm.pretrain", *args, "--data-dir", "/data",
                             "--run-dir", str(run)], cwd="/ml", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    sent, sent_ckpt = 0, 0
    metrics = run / "metrics.jsonl"
    for line in proc.stdout:
        if not line.startswith("{"):
            yield ("log", line.rstrip())
        if metrics.exists():
            rows = metrics.read_text().splitlines()
            for row in rows[sent:]:
                yield ("metrics", row)
                step = json.loads(row)["step"]
                if step // ckpt_every > sent_ckpt and (run / "last.pt").exists():
                    sent_ckpt = step // ckpt_every
                    time.sleep(2)                                      # let torch.save finish writing
                    yield ("last.pt", (run / "last.pt").read_bytes())
            sent = len(rows)
    proc.wait()
    if (run / "final.pt").exists():
        yield ("final.pt", (run / "final.pt").read_bytes())
    yield ("exit", str(proc.returncode))


@app.local_entrypoint()
def main(smoke: bool = False):
    args, ckpt_every, run_dir = ARGS, CKPT_EVERY, RUN_DIR
    if smoke:
        args, ckpt_every, run_dir = ARGS + ["--max-steps", "60", "--eval-every", "50"], 50, RUN_DIR.with_name("pretrain_smoke")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps({"args": args, "data": str(DATA), "gpu": "A100-40GB"}, indent=1))
    with open(run_dir / "train.log", "a") as log:
        for kind, payload in pretrain.remote_gen(args, ckpt_every):
            if kind in ("last.pt", "final.pt"):
                tmp = run_dir / (kind + ".tmp")
                tmp.write_bytes(payload)
                tmp.replace(run_dir / kind)
                print(f"saved {kind} ({len(payload) / 2 ** 20:.0f} MB)", flush=True)
            elif kind == "metrics":
                with open(run_dir / "metrics.jsonl", "a") as f:
                    f.write(payload + "\n")
                print(payload[:200], flush=True)
            else:
                log.write(payload + "\n")
                log.flush()
                if kind == "exit" or payload.startswith(("model ", "resumed", "Traceback")) or "Error" in payload:
                    print(payload, flush=True)
