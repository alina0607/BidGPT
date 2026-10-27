"""Brush GPT L (owner, 2026-10-19): the whole-face model on the digital repaint, variable-length sequences with a
sketch plan first (tokens: brushtokens_l). L5 learns the first five layers; L8 the eight, starting from L5's weights.

Same network as runs D-K (width 512, 7 layers, 8 heads; llm.model.GPT) except positions: RoPE only, no learned
position embedding, so the maximum length is a buffer and L5's weights load into L8 unchanged. MAX_SEQ is fixed for
both. A position's allowed tokens come from the grammar (brushtokens_l.Grammar), the same in training and sampling:
the cross-entropy at every position is over the tokens its grammar state allows (as runs C-K over a field).
Strokes whose centre is in the uncertain region carry loss weight 0 (tools/prepare_l.py).

Splits: AFHQ's test (380) and validation (291) cats are those of runs D-K (data/processed/brushgpt/tokens_runs_DEFG.npz,
so photos can be compared with run I's); every other cat (including the 1,467 not in that split) is training.
Mirror images of the training cats (tools/mirror_l.py, plan mirrored too) are added by listing their folder in `data`.

    python -m sellanything_ml.sketch.brushgpt_l data --cats data/processed/brush_l/cats --layers 5 --out data/processed/brush_l/data_L5.npz
    python -m sellanything_ml.sketch.brushgpt_l train --name L5 --data data/processed/brush_l/data_L5.npz
    python -m sellanything_ml.sketch.brushgpt_l train --name L8 --data …/data_L8.npz --layers 8 --init-from runs/brushgpt/L5/best.pt
    python -m sellanything_ml.sketch.brushgpt_l sample --name L5 --n 16
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import brushtokens_l as L

ML_DIR = Path(__file__).resolve().parents[2]
RUN_DIR = ML_DIR / "runs" / "brushgpt"
SPLIT_FILE = ML_DIR / "data" / "processed" / "brushgpt" / "tokens_runs_DEFG.npz"
MAX_SEQ = 9728                      # L8's longest sequence is about 9,660 tokens; L5 uses the same RoPE table
ALL = "ALL"                         # the allowed set of padding positions (their loss weight is 0)


# ------------------------------------------------------------------ data
def build(cats_dirs: list[Path], n_layers: int, out: Path) -> dict:
    """Every prepared cat (tools/prepare_l.py; mirror images from tools/mirror_l.py) in the folders → one file:
    concatenated tokens, per-token loss weights and allowed-set ids, offsets, names and splits. A mirror image
    (<name>__mirror) takes its cat's split and must be a training cat."""
    z = np.load(SPLIT_FILE)
    old = {str(i).split("/")[-1].rsplit(".", 1)[0]: str(s) for i, s in zip(z["ids"], z["split"])}
    files = sorted(f for d in cats_dirs for f in d.glob("*.npz") if not f.name.endswith(".tmp.npz"))
    sets: dict[frozenset, int] = {frozenset([ALL]): 0}
    toks, wts, sids, offs, names, split = [], [], [], [0], [], []
    for f in files:
        d = np.load(f)
        t = d["tokens"].astype(np.int64)
        allowed = L.allowed_groups(t, n_layers)
        ids = [sets.setdefault(frozenset(a), len(sets)) for a in allowed]
        toks.append(t.astype(np.int16))
        wts.append(d["loss_w"].astype(np.uint8))
        sids.append(np.array(ids, np.uint8))
        offs.append(offs[-1] + len(t))
        name = f.stem.split("__", 1)[1].removesuffix("__mirror")
        names.append(f.stem)
        split.append(old.get(name, "train"))
        assert split[-1] == "train" or not f.stem.endswith("__mirror"), f"mirror of a {split[-1]} cat: {f.stem}"
    info = {"cats": len(files), "mirrors": sum(n.endswith("__mirror") for n in names), "split": {s: split.count(s) for s in ("train", "val", "test")},
            "tokens": [int(min(np.diff(offs))), int(max(np.diff(offs)))], "sets": len(sets), "n_layers": n_layers}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, tokens=np.concatenate(toks), loss_w=np.concatenate(wts), set_ids=np.concatenate(sids),
                        offsets=np.array(offs, np.int64), names=np.array(names), split=np.array(split),
                        sets=np.array(json.dumps([sorted(s) for s in sorted(sets, key=sets.get)])),
                        n_layers=np.array(n_layers))
    return info


class Data:
    def __init__(self, path: Path):
        z = np.load(path)
        self.tokens, self.loss_w, self.set_ids, self.offsets = z["tokens"], z["loss_w"], z["set_ids"], z["offsets"]
        self.names, self.split, self.n_layers = z["names"], z["split"], int(z["n_layers"])
        self.sets = [set(s) for s in json.loads(str(z["sets"]))]

    def rows(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)

    def batch(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(B, T) tokens, loss weights and set ids, padded with EOS / 0 / ALL to the longest row."""
        lens = self.offsets[idx + 1] - self.offsets[idx]
        T = int(lens.max())
        tok = np.full((len(idx), T), L.SP["EOS"], np.int64)
        w = np.zeros((len(idx), T), np.float32)
        sid = np.zeros((len(idx), T), np.int64)
        for r, (i, n) in enumerate(zip(idx, lens)):
            a = self.offsets[i]
            tok[r, :n] = self.tokens[a:a + n]
            w[r, :n] = self.loss_w[a:a + n]
            sid[r, :n] = self.set_ids[a:a + n]
        return tok, w, sid

    def mask_table(self, dev: str):
        import torch
        m = np.stack([np.ones(L.VOCAB, bool) if s == {ALL} else L.token_mask(s) for s in self.sets])
        return torch.from_numpy(m).to(dev)


# ------------------------------------------------------------------ model and training
@dataclass
class Config:
    name: str = "L5"
    n_paint_layers: int = 5
    d_model: int = 512
    n_layers: int = 7
    n_heads: int = 8
    n_kv_heads: int = 8
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch: int = 8
    accum: int = 2
    max_steps: int = 8000
    warmup: int = 100
    eval_every: int = 100
    patience: int = 8
    stop_after_min: float = 0.0
    seed: int = 0
    init_from: str = ""
    resume: bool = False


def model_config(cfg: Config):
    from ..llm.model import GPTConfig
    return GPTConfig(vocab_size=L.VOCAB, d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                     n_kv_heads=cfg.n_kv_heads, max_seq_len=MAX_SEQ, dropout=cfg.dropout, abs_pos=False)


def device() -> str:
    import os

    import torch
    if os.environ.get("BRUSHGPT_DEVICE"):
        return os.environ["BRUSHGPT_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def autocast(dev: str):
    import contextlib

    import torch
    return torch.autocast("cuda", dtype=torch.float16) if dev == "cuda" else contextlib.nullcontext()


def losses(model, tok, w, sid, table, dev: str) -> tuple:
    """Weighted mean cross-entropy over each position's allowed tokens, and the per-position losses."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(tok).to(dev)
    with autocast(dev):
        logits = model(t[:, :-1])[0]
    logits = logits.float().masked_fill(~table[torch.from_numpy(sid[:, 1:]).to(dev)], float("-inf"))
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1), reduction="none").view(len(t), -1)
    wt = torch.from_numpy(w[:, 1:]).to(dev)
    return (ce * wt).sum() / wt.sum().clamp(min=1), ce, wt


GROUP_REPORT = ("x", "y", "angle", "size", "paint", "strength", "plan", "end")


def report_group(g: str) -> str:
    if g in ("x", "y", "angle", "size", "paint", "strength"):
        return g
    if g in ("LAYER_END", "REGION_END", "EOS") or g.startswith("LAYER"):
        return "end"
    return "plan"


def evaluate_loss(model, data: Data, rows: np.ndarray, table, dev: str, batch: int = 8) -> dict:
    import torch
    model.eval()
    tot, cnt = {g: 0.0 for g in GROUP_REPORT}, {g: 0.0 for g in GROUP_REPORT}
    allt, allc = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            tok, w, sid = data.batch(rows[i:i + batch])
            _, ce, wt = losses(model, tok, w, sid, table, dev)
            ce, wt = ce.cpu().numpy(), wt.cpu().numpy()
            allt += float((ce * wt).sum())
            allc += float(wt.sum())
            groups = np.vectorize(lambda x: report_group(L.group_of(x)))(tok[:, 1:])
            for g in GROUP_REPORT:
                m = (groups == g) & (wt > 0)
                tot[g] += float(ce[m].sum())
                cnt[g] += float(m.sum())
    model.train()
    return {"all": allt / max(allc, 1), **{g: tot[g] / cnt[g] for g in GROUP_REPORT if cnt[g]}}


def train(cfg: Config, data_path: Path) -> Path:
    import torch

    from ..llm.model import GPT
    dev = device()
    torch.manual_seed(cfg.seed)
    data = Data(data_path)
    assert data.n_layers == cfg.n_paint_layers, f"data has {data.n_layers} layers, config {cfg.n_paint_layers}"
    run = RUN_DIR / cfg.name
    run.mkdir(parents=True, exist_ok=True)
    model = GPT(model_config(cfg)).to(dev)
    step0, best = 0, float("inf")
    if cfg.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        step0, best = ck["step"], ck["best"]
    elif cfg.init_from:
        ck = torch.load(cfg.init_from, map_location=dev)
        model.load_state_dict(ck["model"])                       # same shapes: L5 → L8 (see the module docstring)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": other, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.95))
    if cfg.resume and (run / "last.pt").exists() and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    table = data.mask_table(dev)
    train_rows, val_rows = data.rows("train"), data.rows("val")
    (run / "config.json").write_text(json.dumps({"config": asdict(cfg), "model": model_config(cfg).to_dict(),
                                                 "parameters": model.num_params(), "train_cats": len(train_rows),
                                                 "val_cats": len(val_rows), "device": dev, "data": str(data_path)}, indent=1))
    rng = np.random.default_rng(cfg.seed + step0)
    t_start, bad, log = time.time(), 0, open(run / "metrics.jsonl", "a")
    model.train()
    for step in range(step0, cfg.max_steps):
        lr = cfg.lr * (step + 1) / cfg.warmup if step < cfg.warmup else \
            cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - cfg.warmup) / max(1, cfg.max_steps - cfg.warmup))))
        for gr in opt.param_groups:
            gr["lr"] = lr
        tl = 0.0
        for _ in range(cfg.accum):
            tok, w, sid = data.batch(rng.choice(train_rows, cfg.batch, replace=False))
            loss, _, _ = losses(model, tok, w, sid, table, dev)
            scaler.scale(loss / cfg.accum).backward()
            tl += float(loss.detach()) / cfg.accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.max_steps:
            val = evaluate_loss(model, data, val_rows, table, dev)
            row = {"step": step + 1, "train_loss": round(tl, 4), "val": {k: round(v, 4) for k, v in val.items()},
                   "lr": lr, "minutes": round((time.time() - t_start) / 60, 1)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            if val["all"] < best:
                best, bad = val["all"], 0
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step + 1, "val": val}, run / "best.pt")
            else:
                bad += 1
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1, "best": best}, run / "last.pt")
            if bad >= cfg.patience:
                print(json.dumps({"stopped": "patience", "step": step + 1}), flush=True)
                break
        if cfg.stop_after_min and time.time() - t_start > 60 * cfg.stop_after_min:
            print(json.dumps({"stopped": "time", "step": step + 1}), flush=True)
            break
    return run


# ------------------------------------------------------------------ sampling
def load_model(path: Path, dev: str):
    import torch

    from ..llm.model import GPT
    ck = torch.load(path, map_location=dev)
    cfg = Config(**ck["config"])
    model = GPT(model_config(cfg)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    model.brush_cfg = cfg
    return model


def sample(model, n: int, temperature: float = 1.0, seed: int = 0, batch: int = 16, max_len: int = MAX_SEQ,
           prefix: list[np.ndarray] | None = None) -> list[np.ndarray]:
    """n paintings sampled from BOS one token at a time, every token among those its grammar state allows; a row
    stops at EOS (or at max_len, then it is cut there). prefix[i] (starting with BOS), if given, is written into row i
    instead of sampling (an evaluation that hands the model a given start, such as a real cat's plan); the model
    samples everything after it."""
    import torch
    dev = next(model.parameters()).device.type
    nl = model.brush_cfg.n_paint_layers
    torch.manual_seed(seed)
    masks: dict[frozenset, torch.Tensor] = {}
    out = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        rows = torch.full((b, max_len), L.SP["EOS"], dtype=torch.long, device=dev)
        rows[:, 0] = L.SP["BOS"]
        gram = [L.Grammar(nl) for _ in range(b)]
        for g in gram:
            g.step(L.SP["BOS"])
        done = np.zeros(b, bool)
        cache = [{"size": max_len} for _ in model.blocks]
        with torch.no_grad(), autocast(dev):
            logits, _ = model(rows[:, :1], cache=cache)
            for p in range(1, max_len):
                m = torch.stack([masks.setdefault(frozenset(g.allowed()), torch.from_numpy(L.token_mask(g.allowed())).to(dev))
                                 if not done[r] else masks.setdefault(frozenset(["EOS"]), torch.from_numpy(L.token_mask({"EOS"})).to(dev))
                                 for r, g in enumerate(gram)])
                last = logits[:, -1].float().masked_fill(~m, float("-inf")) / max(temperature, 1e-6)
                nxt = torch.multinomial(last.softmax(-1), 1)[:, 0]
                if prefix is not None:
                    for r in range(b):
                        if p < len(prefix[i + r]):
                            nxt[r] = int(prefix[i + r][p])
                rows[:, p] = nxt
                for r, tk in enumerate(nxt.tolist()):
                    if not done[r]:
                        gram[r].step(tk)
                        done[r] = tk == L.SP["EOS"]
                if done.all():
                    break
                logits, _ = model(rows[:, p:p + 1], cache=cache, start_pos=p)
        for r in range(b):
            row = rows[r].cpu().numpy()
            e = np.flatnonzero(row == L.SP["EOS"])
            out.append(row[:e[0] + 1] if len(e) else row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--cats", type=Path, nargs="+", required=True)
    d.add_argument("--layers", type=int, default=5)
    d.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--layers", type=int, default=5)
    for k, v in asdict(Config()).items():
        if k != "n_paint_layers":
            t.add_argument(f"--{k.replace('_', '-')}", type=type(v) if not isinstance(v, bool) else int, default=v)
    s = sub.add_parser("sample")
    s.add_argument("--name", required=True)
    s.add_argument("--n", type=int, default=16)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-len", type=int, default=MAX_SEQ)
    a = ap.parse_args()
    if a.cmd == "data":
        print(json.dumps(build(a.cats, a.layers, a.out)))
    elif a.cmd == "train":
        cfg = Config(**{k: (bool(getattr(a, k)) if isinstance(v, bool) else getattr(a, k)) for k, v in asdict(Config()).items()
                        if k != "n_paint_layers"}, n_paint_layers=a.layers)
        print(train(cfg, a.data))
    else:
        import torch
        dev = device()
        model = load_model(RUN_DIR / a.name / "best.pt", dev)
        t0 = time.time()
        rows = sample(model, a.n, a.temperature, a.seed, max_len=a.max_len)
        np.savez_compressed(RUN_DIR / a.name / f"samples_t{a.temperature}_s{a.seed}.npz",
                            tokens=np.array([np.pad(r, (0, MAX_SEQ - len(r)), constant_values=-1) for r in rows]),
                            lengths=np.array([len(r) for r in rows]))
        print(json.dumps({"n": len(rows), "lengths": [len(r) for r in rows], "seconds": round(time.time() - t0, 1),
                          "ended": int(sum(r[-1] == L.SP["EOS"] for r in rows)), "device": dev,
                          "torch": torch.__version__}))


if __name__ == "__main__":
    main()
