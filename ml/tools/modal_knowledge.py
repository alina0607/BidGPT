"""The knowledge experiment on Modal: two copies of B2 that differ only in one extra stage of pre-training.

    cd ml && modal run tools/modal_knowledge.py

B2k:  pretrain_large → more pre-training on FineWeb-Edu with the cat knowledge texts mixed in (data/processed/
      pretrain_knowledge, tools/make_knowledge_mix.py) → SFT on B2's data with B2's recipe.
B2k0: the same, but the extra pre-training is on the same number of FineWeb-Edu tokens without the knowledge
      (data/processed/pretrain_knowledge_control): the control, so a difference comes from the knowledge, not from
      the extra steps.

The two run in parallel, one A100 each. Cost is capped by sft.py's --max-minutes and the functions' timeouts; no
Volume. Returns each run's pre-training final.pt and log, and the SFT best.pt, metrics and log, into
runs/knowledge/<name>/. The held-out evaluation and the probes run on the Mac afterwards (tools/eval_knowledge.sh).
"""

from __future__ import annotations

from pathlib import Path

import modal

ML = Path(__file__).resolve().parents[1]
INIT = "runs/pretrain_large/final.pt"
SFT_DATA = "data/processed/buyers_b2"
MIXES = {"b2k": "data/processed/pretrain_knowledge", "b2k0": "data/processed/pretrain_knowledge_control"}
PRETRAIN = ["--tokens", "5551500", "--batch", "32", "--accum", "1", "--block", "512", "--lr", "3e-4",
            "--warmup", "20", "--eval-every", "100"]          # one pass over the 5.55 M-token mix, 338 steps
SFT = ["--epochs", "3", "--batch", "16", "--lr", "3e-4", "--eval-every", "500"]   # B2's recipe
MAX_MINUTES = 25

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "numpy", "sentencepiece", "pyarrow")
         .add_local_dir(ML / "sellanything_ml", "/ml/sellanything_ml", ignore=["**/__pycache__/**"])
         .add_local_file(ML / "checkpoints" / "tokenizer" / "sp8192.model", "/ml/checkpoints/tokenizer/sp8192.model")
         .add_local_dir(ML / SFT_DATA, f"/ml/{SFT_DATA}")
         .add_local_file(ML / INIT, f"/ml/{INIT}"))
for mix in MIXES.values():                     # val.bin is a link to FineWeb-Edu's; copy the file itself
    image = (image.add_local_file(ML / mix / "train.bin", f"/ml/{mix}/train.bin")
             .add_local_file(ML / "data" / "processed" / "pretrain" / "val.bin", f"/ml/{mix}/val.bin"))
for name in ["item_schema.json", "item_splits.json", "items.tsv", "props.json", "traits.json"]:
    image = image.add_local_file(ML / "data" / name, f"/ml/data/{name}")
app = modal.App("buyer-gpt-knowledge", image=image)


@app.function(gpu="A100-40GB", timeout=45 * 60)
def run(name: str) -> dict[str, bytes]:
    """Extra pre-training on this run's mix, then SFT on B2's data; the files to bring back."""
    import subprocess
    pre, sft = f"runs/knowledge/{name}/pretrain", f"runs/knowledge/{name}/sft"
    files: dict[str, bytes] = {}
    steps = [("pretrain", ["python", "-m", "sellanything_ml.llm.pretrain", "--init", INIT,
                           "--data-dir", MIXES[name], "--run-dir", pre, *PRETRAIN]),
             ("sft", ["python", "-m", "sellanything_ml.llm.sft", "--init", f"{pre}/final.pt", "--data", SFT_DATA,
                      "--run-dir", sft, *SFT, "--max-minutes", str(MAX_MINUTES)])]
    for stage, args in steps:
        proc = subprocess.run(args, cwd="/ml", capture_output=True, text=True)
        files[f"{stage}/train.log"] = (proc.stdout + proc.stderr).encode()
        if proc.returncode != 0:
            break
    for rel in ("pretrain/final.pt", "pretrain/metrics.jsonl", "sft/best.pt", "sft/metrics.jsonl"):
        path = Path("/ml/runs/knowledge", name, rel)
        if path.exists():
            files[rel] = path.read_bytes()
    return files


@app.local_entrypoint()
def main() -> None:
    for name, files in zip(MIXES, run.map(list(MIXES)), strict=True):
        out = ML / "runs" / "knowledge" / name
        for rel, data in files.items():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_bytes(data)
        print(name, sorted(files), (out / "sft/train.log").read_text()[-600:] if (out / "sft/train.log").exists()
              else (out / "pretrain/train.log").read_text()[-1500:])
