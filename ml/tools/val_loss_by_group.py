"""Validation loss of a buyer model per data source and task, in nats per token and bits per byte (paper-ideas B19, B21).

    PYTHONPATH=. python tools/val_loss_by_group.py runs/sft_b3_resume/best.pt --data data/processed/buyers_b4/val.jsonl \
        --out runs/measure_b1/b3_loss_by_source.json
    PYTHONPATH=. python tools/val_loss_by_group.py runs/sft_b2f6/best.pt --data runs/eval_sets_b2f6/val_old_format6.jsonl \
        --bytes-from runs/eval_sets_b2/val_old_format5.jsonl --out …

The loss is the same masked loss SFT trains on (the target's tokens, its <|end|> included), summed over every example
instead of sampled batches, so the numbers are exact for the whole file. Bits per byte divide the summed loss (in bits)
by the UTF-8 bytes of the target's text, so two tokenizers are compared on the same text (B21); --bytes-from takes
the bytes from a parallel file (same examples in the same order, e.g. the format-5 twin of a format-6 set).

Source of an example (B19): Qwen v9 (author qwen_v9_edited), the first Claude sales (claude, seeds 2048-2099), Muse
(muse_edited, 2100-2199), the format-5 sales (2200-2349), the scene-quota sales (2500-2649) and the human-style
seller sales (2650-2699).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

END = "<|end|>"


def source(ex: dict[str, Any]) -> str:
    seed, author = ex["seed"], ex.get("author", "")
    if author == "qwen_v9_edited":
        return "qwen_v9"
    if author == "claude":
        return "claude_b1"
    if author == "muse_edited":
        return "muse"
    if 2650 <= seed < 2700:
        return "human_seller"
    if 2500 <= seed < 2650:
        return "scene_quota"
    if 2200 <= seed < 2500:
        return "format5"
    return f"other:{author}"


def main() -> None:
    import torch

    from sellanything_ml.llm.evaluate import load_model
    from sellanything_ml.llm.sft import collate
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--bytes-from", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("cpu")
    model, tok = load_model(a.checkpoint, device)
    rows = [json.loads(line) for line in a.data.read_text().splitlines()]
    twins = [json.loads(line) for line in a.bytes_from.read_text().splitlines()] if a.bytes_from else rows
    assert len(twins) == len(rows) and all((r["seed"], r["name"], r["task"]) == (t["seed"], t["name"], t["task"])
                                           for r, t in zip(rows, twins, strict=True)), "--bytes-from is not parallel"
    sums: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0, 0, 0])   # nats, tokens, bytes, examples
    per_example = []
    with torch.no_grad():
        for ex, twin in zip(rows, twins, strict=True):
            p, t = tok.encode(ex["prompt"]), tok.encode(ex["target"])
            x, y, m = collate([(p + t, len(p))], tok.pad_id, device)
            mean = model(x, y, m, masked_only=True)[1].item()
            nats = mean * len(t)
            nbytes = len(twin["target"].replace(END, "").encode())
            per_example.append({"seed": ex["seed"], "task": ex["task"], "name": ex["name"], "source": source(ex),
                                "nats": round(nats, 4), "tokens": len(t), "bytes": nbytes})
            for key in ((source(ex), ex["task"]), (source(ex), "all"), ("all", ex["task"]), ("all", "all")):
                s = sums[key]
                s[0] += nats
                s[1] += len(t)
                s[2] += nbytes
                s[3] += 1
    table = {f"{src} / {task}": {"examples": s[3], "loss_per_token": round(s[0] / s[1], 4),
                                 "bits_per_byte": round(s[0] / math.log(2) / s[2], 4),
                                 "bytes_per_token": round(s[2] / s[1], 3)}
             for (src, task), s in sorted(sums.items())}
    report = {"checkpoint": str(a.checkpoint), "data": str(a.data), "groups": table}
    a.out.write_text(json.dumps(report, indent=1))
    a.out.with_suffix(".examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_example))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
