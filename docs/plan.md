# Engineering plan

Living plan for getting Sell Anything from "two separate prototypes" to "a playable round with
trained models". Each milestone has a definition of done that is checked by running something,
not by reading code. Progress is logged in [`devlog/`](devlog).

## Ground rules

| Rule | Why |
|---|---|
| **Model quality comes only from the model: pre-training on a lot of text, then SFT on a lot of conversations.** No rules standing in for understanding, no best-of-n or reranking, no retrieval of training data, no patches for failures seen in play (owner, 2026-10-07, strict) | It is how GPT-2, ChatGPT, MiniMind and nanochat understand text; anything else hides how good the model really is |
| No commits or pushes until the owner reviews | History should read as finished milestones |
| No money spent on compute: free quotas (Kaggle, Colab) are fine, paid GPUs are not; every training run must still fit on this laptop | Training must be reproducible on one laptop. Clarified by the owner 2026-10-05: generating training text with Claude is allowed; renting compute is not. Clarified again 2026-10-18: the rule is about spending money, so a free Kaggle GPU can run a job the laptop could also run (run K) |
| Course folders (`CS529 GameEngine/`, …) are read-only | Course code must never change |
| Every number written in docs or notebooks comes from a run | Same rule as the lecture notes |
| Data and checkpoints stay out of git (`ml/data/raw`, `ml/data/processed`, `ml/checkpoints`) | Size; publish to Hugging Face later |
| Long jobs run in the background with logs in `ml/runs/<job>/` | One GPU (Apple MPS); jobs are serialized, never run two trainings at once |

## Decisions

| # | Decision | Alternatives considered | Reason |
|---|---|---|---|
| D1 | The whole game is in **English** | Traditional Chinese | Engine needs only ASCII text; English pre-training data and SFT data are abundant |
| D2 | **GPT judges, engine counts**: the buyer model outputs words and an interest change (−2…+2); a deterministic price engine turns that into money | GPT outputs prices directly | A 30–60M model is unreliable at arithmetic and at keeping five buyers' numbers consistent |
| D3 | The price engine is also the **labeler** for training data | LLM-written labels | Labels are exact and checkable; the model learns to reproduce the engine's judgment from text, then generalizes to free-text answers and unseen items where the engine can't help |
| D4 | One model plays every buyer; a **persona card** in the prompt makes it a different person | One model per buyer | Same weights, 3–5 buyers per round, no extra training |
| D5 | Buyer preferences are **generic traits** mapped to items through attribute **tags** | Per-item personas | The same buyer works for a dog and for a bicycle |
| D6 | **Host** (turn order, which attribute to ask) is game logic, not the model | Model decides who speaks | Prevents five buyers asking the same question |
| D7 | Tokenizer: **SentencePiece BPE**, vocabulary chosen by measurement (4k / 8k / 16k), with byte fallback and the chat special tokens built in | tiktoken GPT-2 (50k) | Measured: a 50k vocabulary costs 25.7M embedding parameters at d=512, too much for a 30–60M model; tiktoken can't train new vocabularies |
| D8 | Buyer data **v1 = engine skeleton + handwritten phrase pools** (no LLM); **v2** adds paraphrases from a local open-weights LLM | Claude API | Free, unlimited, no licence question about training on model outputs |
| D9 | Storage: JSON for schemas, JSONL for dialogues, Parquet for raw text, uint16 `.bin` for tokenized pre-training. **No SQL** | SQLite | Training reads data sequentially; SQL adds nothing until generation needs job tracking |
| D10 | Sketch classifier covers **all 345** Quick, Draw! classes (221 `buy` + 124 `refuse`) | Only `buy` classes | Buyers must recognise items they refuse to buy |
| D11 | Production code lives in `ml/sellanything_ml/`; teaching notebooks import it rather than duplicate it | Notebook-only | One source of truth; notebooks stay short and focused on explanation |
| D12 | **No general SFT in v1**: pre-training → buyer SFT directly | SmolTalk SFT first | v1 player answers are canonical options, so there is nothing for general chat data to teach yet; also sidesteps SmolTalk's Llama-generated content. v2 (free-text answers) adds general SFT from OpenAssistant (human-written, Apache 2.0) |
| D13 | Pre-training text: **FineWeb-Edu** (`karpathy/fineweb-edu-100b-shuffle`, ODC-By), 3 shards = 201M tokens, 4th shard = validation | ClimbMix (nanochat's current choice) | Clear licence and provenance for a model that will be published |
| D14 | Tokenizer **sp8192** (measured, see below) | 4k, 16k, GPT-2 50k | 16k saves only 2% tokens on buyer text (9% on FineWeb) for twice the embedding; GPT-2 saves 19% for 6× the embedding |
| D15 | Item schemas for 211 items are written as **Python with shared builders** (`buyers/catalog/`) and compiled to JSON | 221 hand-written JSON entries | Shared attributes (condition, brand, freshness…) stay consistent; validation runs on every build |
| D16 | Buyer data v2 renames the item to a **made-up word** in 30% of training examples | Swapping in other real items; more items | v1 memorized item names (a hedgehog became "the dog"); made-up words force copying from `<\|item\|>` without pairing a real item with another's attributes |
| D17 | SFT pads batches to a **multiple of 64** tokens | Pad to each batch's max | One shape per batch made MPS hold 20 GB and swap; 12 shapes fixed it and ran faster (0.9 → 0.55 s/step) |
| D18 | Ask prompts carry **the engine's question** (prompt format 2, data v3); the model says it in the buyer's voice | The model writes the question from the attribute name alone | For an item it never saw, the model cannot know the schema's wording (held-out on-topic stayed at 60–64% through v1 and v2); the engine already decides what to ask (D2), so the model should only phrase it |
| D19 | The model writes the item as **`{item}`** and the engine fills the name in (prompt format 3, data v4) | Teach the model to copy names (noun swaps alone, v3) | A 13M model garbles rare names it must copy (*saxophone* → *saxmalone*), with or without noun swaps (v3b without them was worse: 62.5% vs 77.5% on topic); the engine always knows what is for sale |
| D20 | Quality comes from the models, **not from selecting or patching their output** (owner, 2026-10-07): no best-of-n with a classifier, no retrieval of training drawings, no rule patches on top of a model | Best-of-6 drawings (97.7% top-1 by the v1 judge, measured and then removed); a breed-name rule for typed answers | Selection hides how well the model itself draws; a patch fixes one logged case and none of its neighbours |
| D21 | Typed answers are read by the buyer model (`parse`, prompt format 4): options lettered and **shuffled per example**, answer = one letter or `none`; `<\|parse\|>`, `<\|said\|>` and 12 reserved tokens get ids after SentencePiece's vocabulary and the embedding grows | Model writes the option id; rules extended with aliases (design-v2 §6.8) | A letter is the same output for every item, so it transfers to items never seen; shuffling removes any position habit; growing the vocabulary keeps every old logit (tested) |
| D22 | Claude-written data is accepted only through a **blind review that cannot look the answer up**: random keys, letters instead of option ids, key file outside the repo, and 8 honesty probes per chunk (a chunk needs 5/8 read as none) | Review lines carrying the request id (round 1) | Round 1 reviewers agreed with the writer on 99.7–100% in about a minute per 900 lines with templated reasons: not shown to be independent (devlog 2026-10-07) |
| D23 | Drawing GPT v2 learns `[no class]` (class dropout 10%) and samples with **classifier-free guidance** w = 2; trains only on drawings the Quick, Draw! game recognised; warm start from v1 | Bigger model from scratch (no time in one night); pixel diffusion (A9; the game draws strokes) | Measured: guidance brings stroke count and unfinished drawings to people's level and raises recognisability; unrecognised training drawings are recognised 20.8% by the v1 CNN |
| D24 | A condition reason's wording follows **the attribute answered** (health, freshness, an animal's age), not the item group | Per item group (design-v2 §6.6) | A cat's litter training also carries a condition signal and is not about health; risky items mix wild animals and objects |
| D25 | The colour cat doodle board offers **only colour families Brush GPT has learnt** from real cats (black, white, grey, orange, cream, brown, yellow, green, blue, pink); red and purple are off the board, and no recolouring data is made for now (owner, 2026-10-17). When Brush GPT gets condition inputs (coat, pattern, eyes), its colour classes **reserve all 12 original paints plus unspecified**, red and purple included, with no data yet | Add red and purple now, with photos recoloured and painted again by the oil painter; keep the 12-paint board | Measured on 3,000 of run J's training paintings: red and purple are the nearest paint for 0.40% and 0.17% of strokes (0.29% and 0.22% around the eyes), so Brush GPT could not paint them with natural shading. Reserving the classes means a later recolouring set is a fine-tune with the old data mixed in (measured before and after on held-out real cats), not a new model |

## Architecture of a round

```
draw ─▶ classifier top-3 ─▶ player confirms item ─▶ items.tsv: refuse? ─▶ every buyer refuses
                                                    └─ buy ─▶ item_schema.json attributes
host picks (buyer, attribute) ─▶ buyer asks ─▶ player answers ─▶ every buyer: interest Δ ─▶ price engine
        ▲                                                                                   │
        └──────────────────────────── until attributes or turns run out ◀───────────────────┘
final offers ─▶ player sells to one buyer
```

Specification: [`buyer-ai.md`](buyer-ai.md).

## Milestones

| | Milestone | Done when | Depends on |
|---|---|---|---|
| **M1** | Buyer AI spec + schema for 10 items + trait pool | `docs/buyer-ai.md` written; `item_schema.json` and `traits.json` pass schema tests | — |
| **M2** | Price engine + host + text-mode round | Unit tests for every formula; `python -m sellanything_ml.play bicycle` plays a full 3–5 buyer round in the terminal with option buttons | M1 |
| **M3** | Sketch classifier at 345 classes | Training script (not notebook) with a background run; test top-1/top-3 reported; stroke rasterizer reproduces Quick, Draw! bitmaps, checked by classifying real stroke data; `classify` in `server.py` uses the model | — (runs in background during M1/M2) |
| **M4** | Schema for all 221 `buy` items | Every `buy` row in `items.tsv` has a schema entry; tests pass; engine plays a round for each | M1, M2 |
| **M5** | Buyer dialogue data v1 | Generator produces JSONL scenes from the engine; held-out item list fixed; consistency checks pass on 100% of scenes | M2, M4 |
| **M6** | Tokenizer | SentencePiece trained on pre-training text + dialogue text; 4k/8k/16k compared on bytes-per-token and embedding cost; special tokens are single ids | M5 |
| **M7** | GPT model + pre-training | Llama-style decoder (RMSNorm, RoPE, SwiGLU) with tests; throughput measured on MPS; a pre-training run in the background with loss curve | M6 |
| **M8** | SFT + evaluation | General SFT (SmolTalk subset, after licence check) then buyer SFT; eval: format validity, interest-Δ accuracy vs engine, held-out items | M7 |
| M9 | Engine: text + dialogue UI | Bitmap font, text input, buyer panels; round playable in the C++ game against the Python server | M2 |
| M10 | DGL compat finishing | `compat/README.md`, demo built in CI | — |

Night 1 target: **M1–M3 done, M4 well under way, M6 started.** M7 and later depend on how long M3 trains.

## Background job schedule (single GPU)

| Order | Job | Estimate | Notes |
|---|---|---|---|
| 1 | Download Quick, Draw! bitmaps, 345 classes × 6,000 | ~1.6 GB | Range requests, first 6,000 drawings per class |
| 2 | Train 345-class CNN | ~1 h (to be measured) | Architecture from notebook stage 6 |
| 3 | Download stroke data for rasterizer check | small | `simplified/*.ndjson`, first few hundred lines per class |
| 4 | Tokenizer training | minutes | CPU |
| 5 | Pre-training | hours | Only after the CNN finishes |

## Risks

| Risk | Mitigation |
|---|---|
| Machine sleeps overnight | Keep-awake hold for the session; jobs write checkpoints every epoch and resume |
| 345-class accuracy is low | Expected; the product uses top-3 + player confirmation, so top-3 is the headline metric |
| Rasterizer mismatch with dataset bitmaps | Verify by classifying real Quick, Draw! strokes rendered by our rasterizer and comparing accuracy with the bitmap test set |
| Hand-written phrase pools make dialogue repetitive | v1 is for plumbing and first training; v2 adds LLM paraphrases |
| SmolTalk licence (Llama 3.1-generated) | Check before use; alternatives exist |
| MPS throughput too low for pre-training | Measure first; shrink model/corpus or move only pre-training elsewhere later |
