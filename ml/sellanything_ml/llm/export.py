"""Put a trained buyer model where the game loads it, with the gate for reading typed answers.

    python -m sellanything_ml.llm.export runs/sft_v6b      # -> checkpoints/buyer_gpt.pt (old one kept)

A model trained on `parse` reads typed answers in the game only if its evaluation beat the rule matcher
on both test splits: more answers right and no more wrong options (runs/<run>/eval_parse_<split>.json,
written by run.sh). Otherwise `parse_ready` is False and the game keeps the rule matcher, while the
model still writes the buyers' lines.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch

from .tokenizer import ML_DIR

TARGET = ML_DIR / "checkpoints" / "buyer_gpt.pt"


def parse_ready(run: Path) -> tuple[bool, dict]:
    report = {}
    ok = True
    for split in ("test_seen", "test_heldout"):
        path = run / f"eval_parse_{split}.json"
        if not path.exists():
            return False, {"missing": str(path)}
        p = json.loads(path.read_text())["parse"]
        model, rules = p["model all"], p["rules all"]
        report[split] = {"model right": model.get("right", 0), "model wrong": model.get("wrong option", 0),
                         "rules right": rules.get("right", 0), "rules wrong": rules.get("wrong option", 0)}
        ok &= model.get("right", 0) > rules.get("right", 0) and model.get("wrong option", 0) <= rules.get("wrong option", 0)
    return ok, report


def export(run: Path) -> None:
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    ready, report = parse_ready(run) if "parse" in state.get("tasks", []) else (False, {})
    state.update({"parse_ready": ready, "parse_gate": report, "source": str(run / "best.pt")})
    if TARGET.exists():
        old = torch.load(TARGET, map_location="cpu", weights_only=False)
        keep = TARGET.with_name(f"buyer_gpt_{Path(old.get('data', 'old')).name}.pt")
        if not keep.exists():
            shutil.copy(TARGET, keep)
            print(f"previous model kept as {keep.name}")
    torch.save(state, TARGET)
    print(f"exported {run / 'best.pt'} -> {TARGET}; parse_ready={ready} {json.dumps(report)}")


if __name__ == "__main__":
    export(Path(sys.argv[1]))
