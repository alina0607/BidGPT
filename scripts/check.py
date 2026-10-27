"""Every check, cheapest first, in one command.

    python scripts/check.py --quick           # lint, unit, build, engine: after every edit (~10 s)
    python scripts/check.py                   # + e2e, data, model, quality: before a commit (~2 min)
    python scripts/check.py --only model      # one stage
    python scripts/check.py --require-all     # a skipped stage counts as a failure
    python scripts/check.py --update-baseline # re-record the CUPID and model baselines after review

This is the debugging loop: run it, read the failures, fix the cause, run it again. Each stage
narrows where a bug can be (docs/quality.md).

Rules, borrowed from a gate that was used for months before this one:

- Only `lint` and `build` stop the run when they fail, because nothing after them can be believed.
  Every other stage runs anyway and the failures are listed together at the end: stopping at the
  first one hides whether the rest would have passed.
- A stage that can't run here (no CUPID, no trained model, no dataset) reports SKIP, never PASS,
  and the last line says so. A green line that nothing looked at is worse than a red one.
- Baselines are ratchets: `quality` fails on warnings a change adds, `model` on any metric that
  drops. Both are deterministic, so any movement means something changed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "ml"
QUALITY = ROOT / "docs" / "quality"
CUPID_BASELINE = QUALITY / "cupid-baseline.jsonl"
MODEL_BASELINE = QUALITY / "model-baseline.json"
CUPID_SCOPE = ["ml/sellanything_ml", "engine", "game"]
CUPID_DEFAULT = Path.home() / "autar" / "CUPID" / ".venv" / "bin" / "python"
MODEL_SAMPLE = 300          # examples per split for the model stage; deterministic, ~30 s on CPU

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Result:
    status: str
    output: str = ""


def run(cmd: list[str], cwd: Path = ROOT) -> Result:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return Result(PASS if proc.returncode == 0 else FAIL, (proc.stdout + proc.stderr).strip())


# --- stages ------------------------------------------------------------------------------------

def lint() -> Result:
    return run(["ruff", "check", "ml", "scripts"])


def unit() -> Result:
    return run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ML)


def build() -> Result:
    if not (ROOT / "build" / "CMakeCache.txt").exists():
        configured = run(["cmake", "-S", ".", "-B", "build"])
        if configured.status == FAIL:
            return configured
    return run(["cmake", "--build", "build", "-j8"])


def dgl() -> Result:
    """The DGL backend's source, compiled against compat/ (it ships for Windows; this checks it here)."""
    if sys.platform == "win32":
        return Result(SKIP, "on Windows the real DGL backend is built by the build stage")
    if not (ROOT / "build-dglc" / "CMakeCache.txt").exists():
        configured = run(["cmake", "-S", ".", "-B", "build-dglc", "-DSA_BACKEND_DGL_COMPAT=ON"])
        if configured.status == FAIL:
            return configured
    return run(["cmake", "--build", "build-dglc", "-j8", "--target", "sell_anything"])


def engine() -> Result:
    return run(["ctest", "--test-dir", "build", "--output-on-failure"])


def e2e() -> Result:
    return run([sys.executable, "scripts/e2e_autoplay.py"])


def data() -> Result:
    """The default dataset (what evaluation and baselines read) must be what today's code writes."""
    script = ("from sellanything_ml.buyers.datagen import BUYERS_DIR, is_current\n"
              "import sys\n"
              "if not (BUYERS_DIR / 'train.jsonl').exists(): print('SKIP'); sys.exit(0)\n"
              "ok = is_current(BUYERS_DIR)\n"
              "print(BUYERS_DIR.name, 'matches today\\'s code' if ok else 'was written by older code: regenerate it '"
              "'(python -m sellanything_ml.buyers.datagen) or point BUYERS_DIR at a current set')\n"
              "sys.exit(0 if ok else 1)\n")
    result = run([sys.executable, "-c", script], cwd=ML)
    if result.output == "SKIP":
        return Result(SKIP, "no dataset in ml/data/processed (generate one with python -m sellanything_ml.buyers.datagen)")
    return result


def _model_report() -> tuple[Path, dict] | None:
    script = f"""
import json, sys, torch
from pathlib import Path
from sellanything_ml.buyers.voice import DEFAULT_CHECKPOINTS
from sellanything_ml.llm.evaluate import load_model, run
ckpt = next((p for p in DEFAULT_CHECKPOINTS if p.exists()), None)
if ckpt is None:
    print(json.dumps(None)); sys.exit(0)
state = torch.load(ckpt, map_location="cpu")
data = Path(state["data"]) if "data" in state else None
model, tok = load_model(ckpt, torch.device("cpu"))
report = {{s: run(model, tok, s, {MODEL_SAMPLE}, data_dir=data)["report"] for s in ("test_seen", "test_heldout")}}
print(json.dumps({{"checkpoint": str(ckpt.relative_to(Path.cwd())), "data": str(data), "report": report}}))
"""
    proc = subprocess.run([sys.executable, "-c", script], cwd=ML, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-2000:])
    found = json.loads(proc.stdout.strip().splitlines()[-1])
    return found


def _metrics(report: dict) -> dict[str, float]:
    keys = {"ask": ("valid", "on_topic"), "react": ("valid", "delta_exact", "delta_sign")}
    return {f"{split}/{task}/{k}": report[split][task][k]
            for split in report for task, ks in keys.items() for k in ks}


def model() -> Result:
    """The installed buyer model, scored on a fixed sample; no metric may drop below the baseline."""
    try:
        found = _model_report()
    except RuntimeError as e:
        return Result(FAIL, str(e))
    if found is None:
        return Result(SKIP, "no trained buyer model (ml/checkpoints/buyer_gpt.pt)")
    if not MODEL_BASELINE.exists():
        return Result(FAIL, "no model baseline yet; run with --update-baseline after reviewing the numbers")
    baseline = json.loads(MODEL_BASELINE.read_text())
    now, before = _metrics(found["report"]), baseline["metrics"]
    if found["checkpoint"] != baseline["checkpoint"] or found["data"] != baseline["data"]:
        return Result(FAIL, f"baseline is for {baseline['checkpoint']} on {baseline['data']}, installed is "
                            f"{found['checkpoint']} on {found['data']}; re-record after comparing them")
    lines = [f"{k:<32} {before.get(k, float('nan')):.4f} → {v:.4f}" for k, v in now.items()]
    dropped = [k for k, v in now.items() if v < before.get(k, 0)]
    lines.append(f"{len(dropped)} metric(s) dropped" + (f": {', '.join(dropped)}" if dropped else ""))
    return Result(FAIL if dropped else PASS, "\n".join(lines))


def cupid_python() -> Path | None:
    path = Path(os.environ.get("CUPID_PYTHON", CUPID_DEFAULT))
    return path if path.exists() else None


def cupid_scan(python: Path, out: Path, baseline: Path | None = None) -> Result:
    cmd = [str(python), "-m", "cupid", *CUPID_SCOPE, "--root", ".", "--findings-jsonl", str(out),
           "--compile-commands", "build/compile_commands.json"]
    if baseline is not None:
        cmd += ["--baseline", str(baseline)]
    return run(cmd)


def quality() -> Result:
    python = cupid_python()
    if python is None:
        return Result(SKIP, "CUPID not found (set CUPID_PYTHON to its interpreter)")
    if not CUPID_BASELINE.exists():
        return Result(FAIL, "no CUPID baseline yet; run with --update-baseline")
    new = ROOT / "build" / "cupid-new.jsonl"
    scanned = cupid_scan(python, new, CUPID_BASELINE)
    if scanned.status == FAIL:
        return scanned
    findings = [json.loads(line) for line in new.read_text().splitlines() if line.strip()]
    blocking = [f for f in findings if f["severity"] in ("warning", "error")]
    lines = [f"{f['severity']:>7} {f['rule']} {f['path']}:{f['line']} {f['function']}\n"
             f"        {f['message'].splitlines()[0]}" for f in findings]
    lines.append(f"{len(findings)} new finding(s), {len(blocking)} blocking (warning or error)")
    return Result(FAIL if blocking else PASS, "\n".join(lines))


@dataclass
class Stage:
    name: str
    check: Callable[[], Result]
    quick: bool = False        # part of --quick
    stops_run: bool = False    # nothing after it can be believed when it fails


STAGES = [
    Stage("lint", lint, quick=True, stops_run=True),
    Stage("unit", unit, quick=True),
    Stage("build", build, quick=True, stops_run=True),
    Stage("engine", engine, quick=True),
    Stage("dgl", dgl),
    Stage("e2e", e2e),
    Stage("data", data),
    Stage("model", model),
    Stage("quality", quality),
]


# --- baselines ---------------------------------------------------------------------------------

def update_baselines() -> int:
    QUALITY.mkdir(parents=True, exist_ok=True)
    python = cupid_python()
    if python is None:
        print("CUPID: not found, baseline left as it is (set CUPID_PYTHON)")
    else:
        scanned = cupid_scan(python, CUPID_BASELINE)
        print(f"CUPID: {len(CUPID_BASELINE.read_text().splitlines())} findings → {CUPID_BASELINE.relative_to(ROOT)}"
              if scanned.status == PASS else scanned.output)
    found = _model_report()
    if found is None:
        print("model: no trained model, baseline left as it is")
    else:
        record = {"checkpoint": found["checkpoint"], "data": found["data"], "sample": MODEL_SAMPLE,
                  "metrics": _metrics(found["report"])}
        MODEL_BASELINE.write_text(json.dumps(record, indent=2) + "\n")
        print(f"model: {found['checkpoint']} → {MODEL_BASELINE.relative_to(ROOT)}")
        for k, v in record["metrics"].items():
            print(f"  {k:<32} {v:.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="only the stages worth running after every edit")
    parser.add_argument("--only", choices=[s.name for s in STAGES], action="append", help="run just these")
    parser.add_argument("--require-all", action="store_true", help="treat a skipped stage as a failure")
    parser.add_argument("--update-baseline", action="store_true", help="re-record the CUPID and model baselines")
    args = parser.parse_args()
    if args.update_baseline:
        return update_baselines()

    stages = [s for s in STAGES if (s.name in args.only if args.only else (s.quick or not args.quick))]
    results: list[tuple[str, Result]] = []
    for stage in stages:
        started = time.time()
        result = stage.check()
        results.append((stage.name, result))
        print(f"{result.status}  {stage.name:<8} {time.time() - started:6.1f}s")
        if result.status != PASS:
            print("\n".join("        " + line for line in result.output.splitlines()[-40:]))
        if result.status == FAIL and stage.stops_run:
            print(f"stopped: nothing after {stage.name} can be believed until it passes")
            break

    failed = [n for n, r in results if r.status == FAIL]
    skipped = [n for n, r in results if r.status == SKIP]
    not_run = [s.name for s in stages][len(results):]
    if failed or (skipped and args.require_all):
        print(f"FAILED: {', '.join(failed + (skipped if args.require_all else []))}"
              + (f"; not run: {', '.join(not_run)}" if not_run else ""))
        return 1
    print("all green" if not skipped else f"green, but not checked here: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
