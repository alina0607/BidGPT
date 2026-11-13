# BidGPT

[![CI](https://github.com/alina0607/BidGPT/actions/workflows/ci.yml/badge.svg)](https://github.com/alina0607/BidGPT/actions/workflows/ci.yml)

> A GPT trained from scratch to play a room of buyers: each one asks about the cat for sale, hears every answer,
> moves their offer for reasons from their own life, and says why.

**BidGPT** is the buyer model of **Sell Anything**, the game below.

- **Trained from scratch:** a 71M-parameter Llama-style GPT with its own SentencePiece tokenizer, pre-training on FineWeb-Edu, then supervised
  fine-tuning on written cat sales. No pretrained weights, no API.
- **One model, many buyers:** every buyer has a persona card (allergic, two dogs at home, works long hours, breeds
  show cats, ...). The same seller answer, such as "barely sheds, but she isn't spayed yet", moves each buyer a
  different way, and each buyer explains the move in their own words. The model also plays the seller, who accepts,
  counters or refuses each final offer.
- **Improved only through data and training,** and measured on held-out sales. Example: when the training data for
  "sheds very little" went from 0 to 15 reactions, the allergic buyer went from lowering their offer 98% of the time
  to raising it 99% of the time.
- **Training data (version B7):** 929 sales, 133,564 examples over seven tasks (open, ask, react, imagine, answer,
  decide, reply).
- **Where the sales come from:** larger LLMs wrote them as teachers (Qwen2.5-32B-Instruct, Claude, ChatGPT and Muse),
  and every line was checked against the price engine's rules. BidGPT learns only from that text, never from a
  teacher's weights.

"GPT" here is the architecture (a decoder-only transformer, trained from scratch); it is not based on OpenAI's
models.

## Sell Anything: the game around it

> Draw anything. Try to sell it. A room of AI buyers decides what it's worth.

![A round played by --autoplay: draw a bicycle, confirm it, answer the buyers, sell](docs/images/round.gif)

**Sell Anything** is a game built end-to-end from scratch:

- a **custom C++ engine** with swappable rendering backends (OpenGL, DigiPen's DGL, headless),
- a **sketch classifier** trained on all 345 classes of Google's [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset),
- a **price engine and host** that run a 3–5 buyer negotiation for any of the 259 sellable items,
- a **small GPT trained from scratch** (tokenizer → pre-training → SFT) to speak and judge for the buyers.

| Draw | Confirm | Negotiate | Sell |
|---|---|---|---|
| ![Drawing a bicycle](docs/images/1-draw.png) | ![Top-3 guesses](docs/images/2-confirm.png) | ![Buyers asking questions](docs/images/3-negotiate.png) | ![Sold](docs/images/4-sold.png) |

> 🚧 **Status:** a full round is playable (draw → classify → negotiate → sell). Buyers speak through
> the trained model, with the price engine as a guardrail: the engine owns every number, and a model
> line that disagrees with it (or garbles the question) falls back to a phrase pool. The character
> art is placeholder, in the final format. See the [plan](docs/plan.md) and [roadmap](docs/roadmap.md).

## How a round plays

Character controls: **WASD / arrow keys** move, **Space** jumps, **Shift** slides in the facing direction,
**J** performs a one-shot spinning kick, and **K** casts a heart light wave from an open palm.
The wave travels in the facing direction and fades out (no enemy/damage system yet). Jump height is twice
the previous height. Click a text field to type; the answer field also accepts Enter to focus.
Gameplay keys are disabled while typing. Click outside or press Esc to return to movement.

1. **Draw** with the mouse. The game sends the strokes to the ML service.
2. **Confirm**: the classifier shows its top 3 guesses (on 172,500 drawings it never trained on: 72.4% top-1, 87.6% top-3 across 345 classes); the player picks one.
3. **Negotiate**: 3–5 buyers, each with their own traits and budget, take turns asking about the
   item. Click an answer or type your own ("shes like 12"); a typed answer is matched to the closest
   option by [explainable rules](docs/buyer-ai.md#10-typed-answers), and the game shows what it was
   taken as. *Everyone* hears every answer and moves their offer: a big dog is a plus for the hiker and
   a deal-breaker for the buyer in a small apartment. Buyers who lose interest walk away; the rest
   outbid each other.
4. **Sell** to whoever offers most, or keep it.

Try to sell a rainbow, the sun or your elbow and every buyer refuses. A tiger or a rifle does sell,
black-market style: buyers with kids, retirees and apartment dwellers walk away at once, and the
answer without papers is the cheap one.

## The first buyer model (v1 to v4, any drawn item)

Before BidGPT, a 12.9M-parameter GPT (pre-trained on 100M FineWeb-Edu tokens, then fine-tuned on dialogue the price
engine generates) writes what buyers say and judges how each answer moves their interest. Every
version was scored on the same test questions, including 33 items it never saw in training:

| Version | What changed | Held-out items: question on topic | Held-out items: interest change exact | Lines used in play (held-out items) |
|---|---|---|---|---|
| v1 | — | 29% | 70% | 45%\* |
| v2 | 30% of training items renamed to made-up words ([D16](docs/plan.md)) | 41% | 70% | 53%\* |
| v3 | the engine's question goes in the prompt ([D18](docs/plan.md)) | 77% | 70% | 67% |
| v4 | the model writes `{item}`, the engine fills in the name ([D19](docs/plan.md)) | **91%** | 71% | **76%** |

Test sets of about 700 questions and 1,750 reactions per column (±2–4 points); "used in play" is
30 autoplayed rounds. \*Measured before a guardrail fix that made short questions stricter.

For comparison, guessing the most common interest change scores 42% on held-out items, and on
items it has seen the model gets 97–98% of interest changes exactly right, against 64% for a baseline
that looks only at the answer and ignores who the buyer is. The rest of the story, including the
two hypotheses that turned out wrong, is in the [devlog](docs/devlog/2026-10-03.md).

## Architecture

```
┌──────────── C++ game ────────────┐                ┌──────────── Python ML service ─────────────┐
│ engine: loop, scenes, text,      │   NDJSON over  │ sketch: rasterize strokes like Quick, Draw!, │
│         OpenGL / DGL / null      │◀──────────────▶│         CNN (345 classes)                    │
│ game:   canvas, confirm, round UI│  127.0.0.1:5555│ buyers: item schemas, price engine, host     │
└──────────────────────────────────┘                │ llm:    tokenizer, GPT, pre-training, SFT    │
                                                    └──────────────────────────────────────────────┘
```

| Component | Path | Highlights |
|---|---|---|
| Engine | [`engine/`](engine) | Backend interface selectable at run time; batched OpenGL draws; bitmap-font text; textures, sprite sheets and animation on every backend |
| Game | [`game/`](game) | Async ML client, scenes, animated buyers that react to every answer, `--autoplay` for demos, screenshots and the end-to-end test |
| Sketch classifier | [`ml/sellanything_ml/sketch/`](ml/sellanything_ml/sketch) → moving to [sketch-recognition-from-scratch](https://github.com/alina0607/sketch-recognition-from-scratch) | Reproduces Quick, Draw!'s preprocessing; pixel correlation **0.996** with the official bitmaps |
| Buyers | [`ml/sellanything_ml/buyers/`](ml/sellanything_ml/buyers) | 259 item schemas, 11 buyer traits, valuation, bidding, host — [spec](docs/buyer-ai.md) |
| Buyer model | [`ml/sellanything_ml/llm/`](ml/sellanything_ml/llm) | SentencePiece (8k), Llama-style GPT (RMSNorm, RoPE, SwiGLU, GQA, KV cache) |
| DGL on OpenGL | [`compat/`](compat) | Runs programs written for DigiPen's DGL on macOS/Linux — including this game's own DGL backend, so it is built and run off Windows too |

Design decisions and their reasons are recorded in [`docs/plan.md`](docs/plan.md).

## Running

**Quickest:** demo mode needs no model download. It builds the game if needed, starts a lightweight
local service, and opens the game (closing the game stops the service again):

```bash
python3 scripts/play.py --demo
```

Use `python3 scripts/play.py` after installing the trained-model dependencies; add `--autoplay` to
either command to watch it play a round by itself.

On macOS, `python3 scripts/make_mac_app.py` makes `build/Sell Anything.app`: double-click it, or
drag it to the Dock. The parts it runs, separately:

**ML service** (Python ≥ 3.10):

```bash
cd ml
uv sync --extra train                   # environment from uv.lock, with the dev tools
uv run python -m sellanything_ml.server # 127.0.0.1:5555
```

The classifier checkpoint comes from `python -m sellanything_ml.sketch.train --per-class 15000 --epochs 12`
(about 4 hours on an Apple M-series GPU when plugged in, 20 minutes an epoch; the default `--per-class 6000 --epochs 15` takes about 1.5 hours and scores 0.8 points lower); model weights will be published on Hugging Face.

**Game** (CMake ≥ 3.24, C++20; GLFW, nlohmann/json, stb and font8x8 are fetched automatically):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/game/sell_anything                # --autoplay plays a round by itself
./build/game/sell_anything --buy          # buy a cat: the buyer GPT B6 is the seller and the other buyers
./build/game/sell_anything --compare b4f9,b5   # the same cat sale to two buyer models, side by side
```

"Buy a cat" (on the drawing screen, or `--buy`) turns the sale around: the player is the last buyer in the row,
opens with an offer, asks the seller one question and reacts to every answer; at the end the seller decides on each
final offer (accept, counter or refuse). It needs `ml/runs/sft_b6/best.pt` (prompt format 10) or
`ml/runs/sft_b5/best.pt` (format 8).

The renderer is chosen at run time (`--backend opengl | dgl | null`); Windows/DGL setup is in
[`docs/backends.md`](docs/backends.md).

**Without the game**, a round can be played in the terminal:

```bash
cd ml && python -m sellanything_ml.play dog
```

**Development** (in `ml/`, managed with [uv](https://docs.astral.sh/uv/); `uv sync` installs the
dev tools from `uv.lock`):

```bash
uv sync --extra train
uv run ruff check . ../scripts   # lint
uv run mypy                      # type check (the files listed in pyproject.toml, in batches)
uv run pytest                    # tests
uv run pre-commit install        # run ruff and mypy before every commit
```

GitHub Actions runs the engine build on three systems and the same three ml checks on every
push to `main` and `dev` and on every pull request ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

**Checks**: `python scripts/check.py` runs every stage, cheapest first: lint, the Python tests,
the C++ build, engine tests, an end-to-end round (the real game, headless, against the real server),
data freshness, a model regression check and a static-analysis gate on new findings. `--quick` runs
the first four. A stage that can't run on this machine says SKIP, not PASS. See
[docs/quality.md](docs/quality.md).

## Data

| Dataset | Licence | Used for |
|---|---|---|
| [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) | CC BY 4.0 | Sketch classifier, item list |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (via `karpathy/fineweb-edu-100b-shuffle`) | ODC-By | Tokenizer and GPT pre-training (201M tokens) |
| Buyer dialogues | generated here | Buyer SFT: 166k examples (112k for training) from simulated rounds, labels from the price engine |

## Acknowledgements

- [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- [MiniMind](https://github.com/jingyaogong/minimind) by jingyaogong
- [Quick, Draw!](https://quickdraw.withgoogle.com/data) by Google Creative Lab
- [font8x8](https://github.com/dhepper/font8x8) by Daniel Hepper (public domain)

## License

[MIT](LICENSE)
