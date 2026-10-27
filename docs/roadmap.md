# Roadmap

Detailed milestones and their acceptance checks live in [plan.md](plan.md); this is the summary.

## Milestone 0 — Project setup ✅
- [x] Repository layout, CMake, CI
- [x] Game ↔ ML protocol
- [x] Label all 345 Quick, Draw! categories as `buy` / `refuse`

## Milestone 1 — Engine ✅
- [x] Backend interface, selectable at run time (`--backend`)
- [x] OpenGL backend (GLFW + glad, Windows/macOS/Linux), one batched draw call per frame
- [x] DGL backend (Direct3D 11, Windows) — compiles against DGL 1.4.0; also built and run on macOS/Linux through `compat/` (screenshots match OpenGL); a run on real DGL is still the final check
- [x] Headless backend + CI smoke test
- [x] Mouse input and drawing canvas
- [x] Bitmap-font text, wrapping, buttons
- [x] Async TCP client for the ML service, with a reply timeout
- [x] Textures, sprite sheets and animation (OpenGL, DGL, headless)
- [ ] Free-text input box (backend already reports typed characters)
- [ ] *Stretch:* Vulkan backend

## Milestone 2 — Sketch classifier ✅
- [x] Quick, Draw! bitmaps, 345 classes × 6,000, then × 15,500 (range-request download)
- [x] Teaching notebook: random guess → linear → MLP → CNN → BatchNorm → dropout/augmentation → schedule
- [x] 345-class CNN: 71.7% top-1, 87.1% top-3 on the test set (5,000 drawings per class)
- [x] Retrained on 14,000 per class: 72.4% top-1, 87.6% top-3 on drawings 15,000–15,499 of every class, which neither model saw (the old one: 71.6% / 87.1% there). 2.8× the data bought 0.8 points, so data is not the bottleneck
- [x] Rasterizer matching the dataset (pixel correlation 0.996)
- [x] Served through `classify`

## Milestone 3 — Buyers ✅
- [x] Specification ([buyer-ai.md](buyer-ai.md)), 221 item schemas, 11 traits
- [x] Price engine, bidding and host, with hand-computed tests
- [x] Rounds served through `round_start` / `round_answer` / `round_sell`
- [x] Training data v1 from the engine (167k examples, 15% of items held out)
- [x] Tokenizer (SentencePiece 8k, chosen by measurement)
- [x] Llama-style GPT, pre-trained on 100M FineWeb-Edu tokens
- [x] Buyer SFT and evaluation on held-out items (v1 → v4, each compared on the same test questions)
- [x] Model-generated buyer lines in the game, behind a guardrail (the engine's numbers always win)
- [x] Prompt formats 2 and 3: the engine's question in the prompt, the item name filled in by the engine (D18, D19)
- [ ] Free-text answers, local-LLM paraphrases, rebalanced interest labels; DPO

## Milestone 4 — Playable demo 🚧
- [x] Full round: draw → confirm → negotiate → sell
- [x] Screenshots in the README (captured automatically with `--autoplay`)
- [x] Animated buyers (placeholder art in the format of [art.md](art.md); real art replaces PNGs only)
- [x] `scripts/check.py`: lint → tests → build → DGL build → end-to-end → data → model → static analysis
- [ ] Real character art (8 looks × 6 states)
- [x] Gameplay GIF in the README (`scripts/record_round.py`, real models)
- [ ] Gameplay video with sound and narration
- [ ] Publish model weights and data on Hugging Face

## Milestone 5 — v2 features 🚧
Design: [design-v2.md](design-v2.md).
- [x] Final guess from CNN v2 + stroke Transformer (82.1% top-1, 93.6% top-3 on 172,500 test drawings)
- [x] Live guessing while drawing (F1), with the full-drawing models
- [x] Drawing GPT: the computer draws a random or a named class stroke by stroke (F3)
- [x] Buyer scenes: each buyer's "how I picture it at my place", rules + drawing GPT (F4)
- [ ] Partial-drawing classifier for live guessing (fine-tuned on drawing prefixes)
- [ ] Dislikes (F5), buyer's seat (F6), understanding the player's own words (F7)
