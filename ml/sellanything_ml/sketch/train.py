"""Train the sketch classifier on all Quick, Draw! classes.

    python -m sellanything_ml.sketch.train --per-class 6000 --epochs 15

Resumable: progress is checkpointed every epoch to <run-dir>/last.pt; rerunning the *same* command
(same arguments) continues where it stopped. Metrics are appended to <run-dir>/metrics.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import ML_DIR, load_categories, load_splits
from .model import make_model, random_affine


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_float(x_uint8: torch.Tensor) -> torch.Tensor:
    """(N, 784) uint8 on device -> (N, 1, 28, 28) float in [0, 1]."""
    return x_uint8.view(-1, 1, 28, 28).float().div_(255.0)


@torch.no_grad()
def evaluate(model, x, y, batch_size=4096):
    """Returns (loss, top1, top3) over a whole split kept on the device."""
    model.eval()
    total_loss, top1, top3 = 0.0, 0, 0
    for i in range(0, len(y), batch_size):
        logits = model(to_float(x[i:i + batch_size]))
        target = y[i:i + batch_size]
        total_loss += F.cross_entropy(logits, target, reduction="sum").item()
        best3 = logits.topk(3, dim=1).indices
        top1 += (best3[:, 0] == target).sum().item()
        top3 += (best3 == target[:, None]).any(dim=1).sum().item()
    n = len(y)
    return total_loss / n, top1 / n, top3 / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=6000)
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-lr", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--run-dir", type=Path, default=ML_DIR / "runs" / "sketch345")
    parser.add_argument("--out", type=Path, default=ML_DIR / "checkpoints" / "sketch_cnn_345.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classes", type=int, default=0, help="use only the first N classes (smoke tests)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    categories = load_categories()
    if args.classes:
        categories = categories[:args.classes]
    t0 = time.time()
    train, val, test = load_splits(categories, args.per_class, args.val, args.test, args.seed)
    print(f"data: {len(categories)} classes, train {len(train.y):,} val {len(val.y):,} "
          f"test {len(test.y):,} ({time.time() - t0:.0f}s)", flush=True)

    # Keep everything on the GPU as uint8 (1.6 GB for 345 x 6,000); convert per batch.
    xs = {k: torch.from_numpy(s.x).to(device) for k, s in (("train", train), ("val", val), ("test", test))}
    ys = {k: torch.from_numpy(s.y).to(device) for k, s in (("train", train), ("val", val), ("test", test))}

    model = make_model(len(categories), args.width, args.hidden).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.max_lr / 25, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    steps_per_epoch = len(train.y) // args.batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.max_lr, total_steps=args.epochs * steps_per_epoch)

    start_epoch, best_val = 0, float("inf")
    last = args.run_dir / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_val = state["epoch"], state["best_val"]
        torch.set_rng_state(state["rng"].cpu())  # map_location moved it to the GPU
        print(f"resumed after epoch {start_epoch}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters on {device}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t_epoch = time.time()
        # Seeded per epoch, so a resumed run sees exactly the order an uninterrupted run would.
        generator = torch.Generator(device="cpu").manual_seed(args.seed * 1000 + epoch)
        order = torch.randperm(len(train.y), generator=generator).to(device)
        running = 0.0
        for step in range(steps_per_epoch):
            idx = order[step * args.batch_size:(step + 1) * args.batch_size]
            x = random_affine(to_float(xs["train"][idx]))
            loss = F.cross_entropy(model(x), ys["train"][idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() if step % 50 == 0 else 0.0

        val_loss, val_top1, val_top3 = evaluate(model, xs["val"], ys["val"])
        record = {"epoch": epoch + 1, "train_loss_sampled": running / max(1, steps_per_epoch // 50),
                  "val_loss": val_loss, "val_top1": val_top1, "val_top3": val_top3,
                  "lr": scheduler.get_last_lr()[0], "seconds": time.time() - t_epoch}
        with open(args.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(), "categories": categories,
                        "width": args.width, "hidden": args.hidden, "epoch": epoch + 1,
                        "val_loss": val_loss, "val_top1": val_top1, "val_top3": val_top3},
                       args.out)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "best_val": best_val,
                    "rng": torch.get_rng_state()}, last)

    # Final test evaluation with the best checkpoint (by validation loss).
    best = torch.load(args.out, map_location=device)
    model.load_state_dict(best["state_dict"])
    test_loss, test_top1, test_top3 = evaluate(model, xs["test"], ys["test"])
    best.update(test_loss=test_loss, test_top1=test_top1, test_top3=test_top3)
    torch.save(best, args.out)
    summary = {"best_epoch": best["epoch"], "test_loss": test_loss, "test_top1": test_top1,
               "test_top3": test_top3, "params": n_params}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("TEST", json.dumps(summary), flush=True)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
