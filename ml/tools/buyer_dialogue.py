"""Cat sales from sellanything_ml.buyers.stories, written by the teacher on a Modal GPU through vLLM.

The same prompts and turn order as on the Mac (python -m sellanything_ml.buyers.stories); this is for volume.
Qwen2.5-32B-Instruct, 4-bit AWQ, on an L40S (48 GB: on a 24 GB A10 the weights left no room for the KV cache).

    cd ml && modal run tools/buyer_dialogue.py --seed 1 --rounds 1     # opening + one question
    cd ml && modal run tools/buyer_dialogue.py --seed 1                  # a whole sale
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import modal

ML = Path(__file__).resolve().parents[1]
TEACHER = "Qwen/Qwen2.5-32B-Instruct-AWQ"

image = (modal.Image.debian_slim(python_version="3.12").pip_install("vllm==0.8.5", "transformers==4.51.3")
         .add_local_file(ML / "sellanything_ml" / "buyers" / "stories.py", "/root/stories.py"))
app = modal.App("buyer-dialogue", image=image)


@app.cls(gpu="L40S", timeout=3600)
class Teacher:
    @modal.enter()
    def load(self) -> None:
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=TEACHER, quantization="awq_marlin", max_model_len=8192, gpu_memory_utilization=0.90)
        self.params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1500)

    def ask(self, prompt: str) -> str:
        import stories
        msgs = [{"role": "system", "content": stories.SETTING}, {"role": "user", "content": prompt}]
        return self.llm.chat(msgs, self.params)[0].outputs[0].text.strip()

    @modal.method()
    def sale(self, seed: int, rounds: int) -> dict[str, object]:
        import stories
        s = stories.seed_sale(random.Random(seed))
        m = stories.sale(self.ask, s, rounds)
        return {"seed": s, "cards": m.cards, "events": m.events, "check": stories.check(m), "text": stories.render(m)}


@app.local_entrypoint()
def main(seed: int = 1, rounds: int = 10) -> None:
    from sellanything_ml.buyers import stories
    out = Teacher().sale.remote(seed, rounds)
    stories.OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = stories.OUT_DIR / f"cat_seed{seed}_rounds{rounds}_cloud.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"cat: {stories.facts(out['seed'])}\nseller: {out['seed']['seller_style']}\n\n{out['text']}\n\n"
          f"check: {out['check']}\nsaved {path}")
