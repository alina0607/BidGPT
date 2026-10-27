"""Cat sales and cat-knowledge texts written by the teacher (Qwen2.5-32B-Instruct, 4-bit AWQ) through vLLM on
Kaggle's two T4s, with the teacher's own review of each.

Attach the private dataset holding stories.py and cat_knowledge.py. Many sales run side by side
(stories.run_many), so every turn of every sale in a chunk goes to the GPUs as one batch; each chunk is appended
to /kaggle/working/sales.jsonl as soon as it is done, so a session that runs out of time keeps what it finished.

Settings are constants because Kaggle runs scripts without a command line.
"""

import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

REVIEW_ONLY = False         # only re-review the sales in an attached sales_to_review.jsonl, writing reviews.jsonl
KNOWLEDGE = False           # every topic in every form, before the sales
REVIEW = False              # the teacher also reviews each sale (audit only: too imprecise to filter with)
FIRST_SEED = 2000
SALES = 48                  # 2026-10-11 v10: example pools + rule rewrites, same seeds 2000-2047 as v9 for comparison
CHUNK = 16                  # sales in flight at once
ROUNDS = 10
STOP_AFTER_H = 10.5         # no new chunk after this (a Kaggle session ends at 12 h)
TEACHER = "Qwen/Qwen2.5-32B-Instruct-AWQ"
MAX_TOKENS = 3000

start = time.time()
VENV = Path("/tmp/venv")
if sys.prefix != str(VENV):
    # Kaggle's Python is 3.13 and vLLM 0.8.5 (the last release with the V0 engine, which T4s need) stops at 3.12,
    # so the rest of this script reruns under a Python 3.12 environment built with uv.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    subprocess.run([sys.executable, "-m", "uv", "venv", "-q", "-p", "3.12", str(VENV)], check=True)
    subprocess.run([sys.executable, "-m", "uv", "pip", "install", "-q", "-p", str(VENV / "bin" / "python"),
                    "vllm==0.8.5", "transformers==4.51.3", "huggingface_hub[hf_transfer]"], check=True)
    print(json.dumps({"environment_min": round((time.time() - start) / 60, 1)}), flush=True)
    sys.exit(subprocess.run([str(VENV / "bin" / "python"), os.path.abspath(sys.argv[0])]).returncode)
os.environ["VLLM_USE_V1"] = "0"                  # the V1 engine of vLLM 0.8.5 needs compute capability 8.0; T4 is 7.5
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"   # spawned workers would rerun this whole script
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

kit = Path("/kaggle/working/src/kit")
kit.mkdir(parents=True, exist_ok=True)
(kit / "__init__.py").write_text("")
for name in ("stories.py", "cat_knowledge.py", "drawable.py"):
    shutil.copy(glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0], kit / name)
sys.path.insert(0, str(kit.parent))

from huggingface_hub import snapshot_download  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from kit import cat_knowledge, stories  # noqa: E402

# nvidia-smi rather than torch: initialising CUDA in this process would stop vLLM from forking its workers
names = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True,
                       text=True, check=True).stdout.strip().splitlines()
gpus = len(names)
weights = snapshot_download(TEACHER, local_dir="/tmp/teacher")
print(json.dumps({"gpus": names,
                  "setup_min": round((time.time() - start) / 60, 1),
                  "free_disk_gb": round(shutil.disk_usage("/tmp").free / 2 ** 30)}), flush=True)
# Prefill in chunks of 2,048 tokens: AWQ on a T4 dequantises each weight for a long prefill, and a whole 12k-token
# prompt at once ran out of memory; smaller chunks also leave more room for the KV cache.
llm = LLM(model=weights, quantization="awq", dtype="float16", tensor_parallel_size=gpus, max_model_len=12288,
          gpu_memory_utilization=0.90, enable_prefix_caching=True, enable_chunked_prefill=True,
          max_num_batched_tokens=2048, max_num_seqs=32)
params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=MAX_TOKENS)
tokens = {"prompt": 0, "output": 0}


def ask_many(prompts: list[str]) -> list[str]:
    chats = [[{"role": "system", "content": stories.SETTING}, {"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(chats, params, use_tqdm=False)
    tokens["prompt"] += sum(len(o.prompt_token_ids) for o in outs)
    tokens["output"] += sum(len(o.outputs[0].token_ids) for o in outs)
    return [o.outputs[0].text.strip() for o in outs]


out = Path("/kaggle/working")
if REVIEW_ONLY:
    t0 = time.time()
    rows = [json.loads(line) for line in open(glob.glob("/kaggle/input/**/sales_to_review.jsonl", recursive=True)[0])]
    markets = []
    for row in rows:
        m = stories.Market(row["sale"])
        m.cards, m.events = row["cards"], row["events"]
        markets.append(m)
    reviews = stories.run_many([stories.judge_steps(m) for m in markets], ask_many)
    with open(out / "reviews.jsonl", "w") as f:
        for row, review in zip(rows, reviews, strict=True):
            f.write(json.dumps({"seed": row["seed"], "review": review}) + "\n")
    print(json.dumps({"sales": len(rows), "flagged": sum(map(len, reviews)),
                      "minutes": round((time.time() - t0) / 60, 1), **tokens}), flush=True)
    sys.exit(0)
if KNOWLEDGE:
    t0, before = time.time(), dict(tokens)
    jobs = [(t, f) for t in cat_knowledge.TOPICS for f in cat_knowledge.FORMS]
    texts = stories.run_many([stories.knowledge_steps(t, f) for t, f in jobs], ask_many)
    with open(out / "knowledge.jsonl", "w") as f:
        for row in texts:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({"knowledge_texts": len(texts), "with_problems": sum(bool(r["problems"]) for r in texts),
                      "minutes": round((time.time() - t0) / 60, 1),
                      "output_tokens": tokens["output"] - before["output"]}), flush=True)

for first in range(FIRST_SEED, FIRST_SEED + SALES, CHUNK):
    if time.time() - start > STOP_AFTER_H * 3600:
        print("time limit reached, no new chunk", flush=True)
        break
    t0, before = time.time(), dict(tokens)
    seeds = list(range(first, min(first + CHUNK, FIRST_SEED + SALES)))
    sales = [stories.seed_sale(random.Random(seed)) for seed in seeds]
    if REVIEW:
        done = stories.run_many([stories.sale_and_review(s, ROUNDS) for s in sales], ask_many)
    else:
        done = [(m, []) for m in stories.run_many([stories.sale_steps(s, ROUNDS) for s in sales], ask_many)]
    with open(out / "sales.jsonl", "a") as f:
        for seed, s, (m, review) in zip(seeds, sales, done, strict=True):
            f.write(json.dumps({"seed": seed, "sale": s, "cards": m.cards, "events": m.events,
                                "check": stories.check(m), "review": review, "text": stories.render(m)}) + "\n")
    print(json.dumps({"seeds": [seeds[0], seeds[-1]], "minutes": round((time.time() - t0) / 60, 1),
                      "prompt_tokens": tokens["prompt"] - before["prompt"],
                      "output_tokens": tokens["output"] - before["output"],
                      "problems": [len(r) for _, r in done]}), flush=True)
print(json.dumps({"total_hours": round((time.time() - start) / 3600, 2), **tokens}), flush=True)
