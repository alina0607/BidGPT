"""Buyer SFT on base game data plus the reviewed Qwen cat-dialogue adapter, on a Modal A100.

    cd ml && modal run --detach tools/modal_sft.py

Stage 1 keeps v6's 2-epoch, batch-32, lr-3e-4 per-token recipe from the pre-trained model; stage 2 keeps v6b's
1-epoch, lr-1e-4 per-example recipe from stage 1's best. It returns the four regression evaluations, a cat-specific
evaluation, both stages' metrics and stage 2's checkpoint. No Modal Volume is used.
"""

from __future__ import annotations

from pathlib import Path

import modal

ML = Path(__file__).resolve().parents[1]
INIT = "runs/pretrain_large/final.pt"
STAGE1, STAGE2 = "runs/sft_cat_v1", "runs/sft_cat_v1b"

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "numpy", "sentencepiece", "pyarrow")
         .add_local_dir(ML / "sellanything_ml", "/ml/sellanything_ml", ignore=["**/__pycache__/**"])
         .add_local_file(ML / "checkpoints" / "tokenizer" / "sp8192.model", "/ml/checkpoints/tokenizer/sp8192.model")
         .add_local_dir(ML / "data" / "processed" / "buyers_v5", "/ml/data/processed/buyers_v5")
         .add_local_dir(ML / "data" / "processed" / "buyers_v6", "/ml/data/processed/buyers_v6")
         .add_local_dir(ML / "data" / "processed" / "buyers_cat_v1", "/ml/data/processed/buyers_cat_v1")
         .add_local_file(ML / INIT, f"/ml/{INIT}"))
for name in ["item_schema.json", "item_splits.json", "items.tsv", "props.json", "traits.json"]:
    image = image.add_local_file(ML / "data" / name, f"/ml/data/{name}")
app = modal.App("buyer-gpt-sft", image=image)

STEPS = [
    ("sft stage 1", ["-m", "sellanything_ml.llm.sft", "--init", INIT, "--data", "data/processed/buyers_v6",
                     "--extra-data", "data/processed/buyers_cat_v1",
                     "--run-dir", STAGE1, "--epochs", "2"], None),
    ("sft stage 2", ["-m", "sellanything_ml.llm.sft", "--init", f"{STAGE1}/best.pt", "--data", "data/processed/buyers_v6",
                     "--extra-data", "data/processed/buyers_cat_v1",
                     "--run-dir", STAGE2, "--epochs", "1", "--lr", "1e-4", "--loss", "per-example"], None),
]
for split in ("test_seen", "test_heldout"):
    STEPS += [(f"eval v5 tasks {split}", ["-m", "sellanything_ml.llm.evaluate", "--checkpoint", f"{STAGE2}/best.pt",
                                          "--split", split, "--n", "2500", "--data", "data/processed/buyers_v5"],
               f"{STAGE2}/eval_v5tasks_{split}.json"),
              (f"eval parse {split}", ["-m", "sellanything_ml.llm.evaluate", "--checkpoint", f"{STAGE2}/best.pt",
                                       "--split", split, "--data", "data/processed/buyers_v6", "--parse"],
               f"{STAGE2}/eval_parse_{split}.json")]
STEPS += [("eval Qwen cats test_seen",
           ["-m", "sellanything_ml.llm.evaluate", "--checkpoint", f"{STAGE2}/best.pt",
            "--split", "test_seen", "--data", "data/processed/buyers_cat_v1", "--n", "1000"],
           f"{STAGE2}/eval_cats_test_seen.json")]


@app.function(gpu="A100-40GB", timeout=4 * 3600)
def run():
    import subprocess
    for name, args, stdout_to in STEPS:
        yield ("log", f"== {name}")
        proc = subprocess.run(["python", *args], cwd="/ml", capture_output=True, text=True)
        if stdout_to:
            Path("/ml", stdout_to).write_text(proc.stdout)
        for line in (proc.stdout if not stdout_to else "").splitlines()[-5:] + proc.stderr.splitlines()[-15:]:
            yield ("log", line)
        if proc.returncode:
            yield ("log", f"!! {name} failed with exit {proc.returncode}")
            break
    for rel in [f"{STAGE1}/metrics.jsonl", f"{STAGE2}/metrics.jsonl", f"{STAGE2}/best.pt"] + \
               [s for _, _, s in STEPS if s]:
        path = Path("/ml", rel)
        if path.exists():
            yield ("file", (rel, path.read_bytes()))


@app.local_entrypoint()
def main():
    for kind, payload in run.remote_gen():
        if kind == "file":
            rel, data = payload
            out = ML / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.with_suffix(out.suffix + ".tmp").write_bytes(data)
            out.with_suffix(out.suffix + ".tmp").replace(out)
            print(f"saved {rel} ({len(data) / 2 ** 20:.1f} MB)", flush=True)
        else:
            print(payload, flush=True)
