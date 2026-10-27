"""Cat sales written by the teacher (Qwen2.5-32B-Instruct, 4-bit AWQ) through vLLM on one Modal A100, for when the
free Kaggle quota is used up. Same model, vLLM version and sampling as tools/kaggle_dialogues/dialogues.py.

    cd ml && modal run tools/modal_dialogues.py --out dialogues_modal_v2

All sales run side by side (stories.run_many); the sales come back to this machine and are written to
data/kaggle_out/<out>/sales.jsonl. No Modal Volume is used (storage would be charged); the weights are
downloaded inside the container on each run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

ML = Path(__file__).resolve().parents[1]
BUYERS = ML / "sellanything_ml" / "buyers"
OUT_ROOT = ML.parent / "data" / "kaggle_out"
TEACHER = "Qwen/Qwen2.5-32B-Instruct-AWQ"
FIRST_SEED = 2000
SALES = 16                  # one batch: wall time is set by the turns of a sale, not by how many run side by side
ROUNDS = 10
MAX_TOKENS = 3000
TIMEOUT_H = 4                # the first run (one prompt for all buyers) took 175 min of the 3 h limit

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.8.5", "transformers==4.51.3", "huggingface_hub[hf_transfer]")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))
for name in ("stories.py", "cat_knowledge.py", "drawable.py"):
    image = image.add_local_file(BUYERS / name, f"/root/kit/{name}")
app = modal.App("buyer-dialogues", image=image)


@app.function(gpu="A100-40GB", timeout=TIMEOUT_H * 3600)
def generate(first_seed: int, sales: int) -> dict[str, object]:
    import random
    import sys

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    Path("/root/kit/__init__.py").write_text("")
    sys.path.insert(0, "/root")
    from kit import stories

    start = time.time()
    weights = snapshot_download(TEACHER, local_dir="/tmp/teacher")
    llm = LLM(model=weights, quantization="awq", dtype="float16", max_model_len=12288, gpu_memory_utilization=0.90,
              enable_prefix_caching=True, enable_chunked_prefill=True, max_num_batched_tokens=4096, max_num_seqs=32)
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=MAX_TOKENS)
    tokens = {"prompt": 0, "output": 0}
    setup_min = round((time.time() - start) / 60, 1)

    def ask_many(prompts: list[str]) -> list[str]:
        chats = [[{"role": "system", "content": stories.SETTING}, {"role": "user", "content": p}] for p in prompts]
        outs = llm.chat(chats, params, use_tqdm=False)
        tokens["prompt"] += sum(len(o.prompt_token_ids) for o in outs)
        tokens["output"] += sum(len(o.outputs[0].token_ids) for o in outs)
        return [o.outputs[0].text.strip() for o in outs]

    t0 = time.time()
    seeds = list(range(first_seed, first_seed + sales))
    sale_list = [stories.seed_sale(random.Random(seed)) for seed in seeds]
    markets = stories.run_many([stories.sale_steps(s, ROUNDS) for s in sale_list], ask_many)
    rows = [{"seed": seed, "sale": s, "cards": m.cards, "events": m.events, "check": stories.check(m), "review": [],
             "text": stories.render(m)} for seed, s, m in zip(seeds, sale_list, markets, strict=True)]
    return {"rows": rows, "setup_min": setup_min, "sales_min": round((time.time() - t0) / 60, 1), **tokens}


@app.local_entrypoint()
def main(first_seed: int = FIRST_SEED, sales: int = SALES, out: str = "dialogues_modal") -> None:
    """--out names the folder under data/kaggle_out/, so a new run never overwrites an earlier one."""
    OUT = OUT_ROOT / out
    if (OUT / "sales.jsonl").exists():
        raise SystemExit(f"{OUT}/sales.jsonl already exists; pick another --out")
    result = generate.remote(first_seed, sales)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "sales.jsonl", "w") as f:
        for row in result["rows"]:
            f.write(json.dumps(row) + "\n")
    summary = {k: v for k, v in result.items() if k != "rows"} | {"sales": len(result["rows"])}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
