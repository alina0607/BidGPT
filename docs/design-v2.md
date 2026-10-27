# Design v2: live guessing, a drawing GPT, buyer scenes, dislikes, a buyer's seat, understanding the player

Status: **draft for review** (2026-10-05). Nothing here is implemented yet. Every number in this
document was measured on this machine or quoted from a paper with its table or section; anything not
yet measured is marked *to measure*. Papers are summarised in [`related-work.md`](related-work.md);
where a paper changed this design, its name is given next to the change.

## 0. Why one document first

Each feature below touches the training data or the prompt format of a model. Designed one at a
time, every feature would mean another data version and another training run. Designed together,
each model is retrained **once**:

| Model | Today | Retrained for v2 | Retrained how often |
|---|---|---|---|
| Sketch classifier v2 (CNN + stroke Transformer + stack) | training now (`ml/runs/sketch_v2/`) | no | — |
| Partial-drawing classifier | — | new, fine-tuned from the sketch v2 stroke Transformer | once |
| Drawing GPT | — | new | once (plus at most one size adjustment) |
| Buyer GPT | v5, prompt format 3 (`ml/runs/sft_v5/`) | SFT v6 on the existing pre-trained model | once |

The rule that makes this work: **freeze the data formats in §2 and the prompt format in §6 before
any v2 training starts.**

Priorities agreed with the owner: the buyer GPT is the core of the project; the sketch side is done
when F1–F3 exist as complete features in the game. All of F1–F7 will be built.

## 1. Features

| # | Feature | Player sees |
|---|---|---|
| F1 | **Live guessing** | While drawing, the guess updates after every stroke |
| F2 | **Final guess** | Exists: "Sell it!" sends the finished drawing; the player confirms one of the top 3 |
| F3 | **Drawing GPT** | The computer draws stroke by stroke: (a) a random category, (b) a category the player names, (c) scenes for F4 |
| F4 | **Buyer scenes** | During a round, each buyer's panel shows "how I picture it at my place" |
| F5 | **Dislikes** | Buyers (computer and player) can dislike things, e.g. "doesn't like being in the sun" |
| F6 | **Buyer's seat** | The computer sells; the player is one of the buyers, describes themselves and asks questions in English, and bids against computer buyers |
| F7 | **Understanding the player (L2)** | Buyers understand answers in the player's own words ("ragdoll", "a street cat", "she's 3 months old" when asked about breed) |

Decisions already made by the owner:

| Question | Decision |
|---|---|
| Language of player input | **English** (D1 unchanged) |
| Is there a hidden "true" answer in the buyer's seat? | **No** (owner, 2026-10-05, replacing an earlier "honest seller" reading): whatever the seller says becomes the fact. What matters is that the dialogue and every reaction stay **logically consistent** with what has been said |
| Can every buyer have dislikes? | **Yes**, computer buyers too |
| When are scenes drawn? | **During the round, one per buyer** |

Open decisions for the owner (§12): conversation level beyond L2 (§6.5), where play-time Δ comes
from (§3.2), learned relatedness instead of `base` (§2.3).

## 2. Shared language: the 8 tags

Everything in v2 talks through the 8 tags in `ml/data/traits.json`: `condition`, `space`,
`low_maintenance`, `family`, `status`, `performance`, `style`, `active`. No rule names a specific item
and a specific buyer together, so a new trait, item or prop works without touching any other table.

### 2.1 Buyer weights may be negative (F5)

- Trait weights may be −2…+2. `weights[tag]` = sum, clamped to [−2, +2].
- New traits that carry dislikes (draft, to review):

| Trait | Persona | Weights |
|---|---|---|
| `homebody` | prefers staying in to going out | `active −2`, `space +1` |
| `minimalist` | owns as little as possible | `space +2`, `status −1` |
| `sun_averse` | avoids being out in the sun | `active −1` |

PrefBench (Lei 2026) uses the same structure (hidden weights × item features → willingness to pay)
but non-negative weights that sum to 1, which can express neither dislikes nor "cares about nothing
much"; signed weights are needed for F5.

### 2.2 Item profile (computed, not written)

```
profile_item[t] = Σ over attributes a, Σ over options o of a: |signals(o)[t]|
```

Measured from `ml/data/item_schema.json` (259 items with a schema):

| Item | Profile |
|---|---|
| dog | low_maintenance 10, family 6, **active 6**, space 5, status 2, condition 1, performance 1 |
| cat | low_maintenance 8, family 7, condition 5, status 2, space 1, **active 0** |
| bicycle | condition 8, status 4, performance 4, space 4, low_maintenance 3, active 3, family 2, style 1 |
| laptop | performance 8, condition 7, status 2, family 1, low_maintenance 1, space 1, active 1 |

Once an attribute is answered, its term uses the chosen option's signed signals instead (a senior
dog, `active −1`, loses the grass). *To decide*: whether scenes are redrawn after answers (§5.4).

### 2.3 New data file: `ml/data/props.json`

The only new hand-written table. One entry per drawable prop (a Quick, Draw! class used as scenery):

```json
"sun":   {"tags": {"active": 2}, "zone": "sky", "size": "medium", "fits": "*"},
"steak": {"tags": {}, "base": 2, "zone": "ground_side", "size": "small", "fits": ["pets_livestock"]},
"house": {"tags": {"space": 1, "family": 1}, "zone": "background", "size": "large", "fits": "*"}
```

| Field | Meaning |
|---|---|
| `tags` | −2…+2 per tag, same scale as option signals |
| `base` | relevance on its own for the items it fits (default 0); food for a pet matters whoever the buyer is |
| `zone` | `sky`, `background`, `ground`, `ground_side`, `beside`, `wall` (layout in §5.4) |
| `size` | `small`, `medium`, `large` — real-world size class, so a dog is never drawn bigger than a house (Sketchforme/Scones keep real proportions; Scones encodes 3 sizes) |
| `fits` | item groups from `items.tsv` it may appear with (`*` = any) |

Validation (a test): every key is one of the 345 classes; every tag is one of the 8; every zone,
size and group is known.

Draft candidates, every name checked against the 345 classes on 2026-10-05:

| Tag | Drawable props |
|---|---|
| active | sun, grass, tree, mountain, bicycle, soccer ball, cloud, fence, tent, river |
| space | house, couch, bed, door, stairs, ladder |
| family | teddy-bear, house, birthday cake, hat, smiley face, toothbrush |
| status | crown, diamond, castle, necklace, star |
| style | flower, vase, lipstick, paint can, eyeglasses, bowtie |
| low_maintenance | clock, alarm clock, calendar, couch, pillow |
| condition (weak) | star, screwdriver, hammer, paintbrush |
| performance (weak) | lightning, speedboat |
| pets (via `fits`) | steak, fish, mouse, bed, pillow, bird |

Not drawable (not among the 345): bowl, bone, leash, dog house, cat tree, window, toolbox, race car,
stopwatch, yarn.

*Open (§12)*: replace `base` and `fits` with item–prop relatedness learned from our own FineWeb-Edu
text (PPMI/SVD or word2vec with a wide window), compared against `base` on the owner's judgements
for ~20 items. Tags stay either way: they are what links scenes to buyers and to prices.

## 3. Engine changes (rules, no model)

| Where | Today | v2 | Why |
|---|---|---|---|
| Interest Δ | `clamp(Σ w·s, −2, 2)` | unchanged formula | negative weights already work in it |
| Host concern | `Σ weights_i[t]` | `Σ |weights_i[t]|` | a buyer who dislikes sun still cares about `active` |
| Traits | positive weights | §2.1 | F5 |
| Walk-away (interest ≤ −4) | unchanged | unchanged | *to measure*: walk-away rate per round before and after F5, engine only; if it rises a lot, tune the dislike traits, not the rule |

### 3.1 Buyer's seat (F6)

| Step | Who | What |
|---|---|---|
| 1 | Engine | Pick the item, 2–4 computer buyers and the **player's budget**. No answers are decided in advance |
| 2 | Player | Types a self-description (optional, any time) → GPT `profile` (§6) → the player's weights |
| 3 | Player or a computer buyer | Asks about an attribute. Player text → GPT `which` → attribute, or "can't tell" |
| 4 | Seller | GPT `answer` replies in the seller's voice and may add colour. The engine `parse`s the reply into an option and **records it as a committed fact**; from then on nobody may contradict it, and Δ and offers follow from it. If the reply cannot be parsed, the engine picks an option and the seller rephrases |
| 5 | Everyone | Interest and offers update as today; the player's interest comes from the player's weights, for display and scenes |
| 6 | Player | Bids with buttons (+1 step, +2 steps, pass). Money is never parsed from text (D2; SIBP: 100% price accuracy with code-filled prices vs ~80% without) |
| 7 | Seller | When attributes run out or nobody raises, the highest bid wins |

**Player utility** (He et al. 2018's normalisation): +1 when winning at the player's own valuation
target, 0 at the midpoint between target and budget, −1 for losing an item worth more to them than
the winning bid. **Before any player sees F6**, simulate scripted policies against the engine (always
+1 step; last-moment snipe; always pass; target 75–90% of value as in MART's simulated player) and
report win rate and utility, to find dominant strategies. He et al. found a simulated partner hid
behaviour that frustrated real people.

### 3.2 Where play-time Δ comes from (open decision)

Today the buyer GPT's Δ moves the money. On held-out items its sign is right 76.18% and exact 70.66%
(`ml/runs/sft_v5/final_v5.json`). Every answer in the game ends up as a schema option (buttons, the
typed-answer matcher, the seller's parsed reply), so the engine could compute the exact Δ instead. He et al.'s
best bot used the same split: the model chose the act, rules set the numbers.

Measure first (CPU, minutes): replay N rounds on held-out items twice from the same seed, once with
GPT Δ and once with engine Δ; report MAPE of final offers (comparable with MART Table 1), how often
the winning buyer changes, walk-away mismatches and sale rate. Then the owner decides: engine Δ at
play time (GPT Δ logged for evaluation), or GPT Δ kept as a deliberate choice. **This matters for the
thesis**: scenes are computed from the tags, so if bids come from a wrong Δ, a scene can "explain" a
bid the buyer did not make (related-work §6).

### 3.3 Fixable problems become a discount, not a dislike

Owner's example (2026-10-05): a cat without its shots is not a dealbreaker; the buyer just has to pay
the vet, so they should **ask for that much off** instead of losing interest. Today every downside is
one multiplicative `factor` plus taste `signals` (cat `health = unknown`: factor 0.85, `condition −1`),
so the buyer can only sound unhappy.

New optional field on an option:

```json
{"id": "unknown", "answer": "I'm not sure about the shots.", "factor": 0.85,
 "signals": {"condition": -1},
 "fix": {"what": "vaccinations", "cost": null}}
```

| Part | Rule |
|---|---|
| Money | The discount is what the engine already computes: valuation before minus after the answer (from `factor`). If `cost` is filled in, the engine uses `valuation − cost` instead, so the discount equals that fee. **Either way the engine computes the number** (D2) |
| Interest Δ | For a fixable option, taste signals apply only to tags about the hassle (`low_maintenance`): a busy buyer still minds the vet trip; a bargain hunter does not lose interest, they negotiate |
| Speech | The buyer names the fix and the amount through a slot the engine fills, like `{item}` (D19) and SIBP's `__PRICE__`: "No shots yet? That's fine, I'll take her to the vet, so I'd knock `{amount}` off." The model never writes digits (§6.7 checks this) |
| Buyer's seat | The player, as a buyer, can be offered the same: the seller's answer includes the fix, and the player's valuation drops by the same amount |

Scope: 2,721 options in `item_schema.json`; 181 have `factor < 1` and repair-like wording (regex,
2026-10-05), but that count includes plain wear ("Pretty worn, it shows its age.") that cannot be
fixed. Which options are fixable (e.g. "It needs sharpening.", "the gear needs work", "not sure about
the shots") is a **hand-review list**, and any real-world `cost` is written by a person and reviewed,
not estimated by a model.

## 4. Live guessing (F1)

| Part | Design |
|---|---|
| When to guess | On every pen-up, the game sends the drawing so far with `"partial": true`; at most one request in flight |
| Which model | `partial: true` → partial-drawing classifier; "Sell it!" → the full-drawing model, unchanged |
| Partial-drawing classifier | **Fine-tuned (all layers) from the sketch v2 stroke Transformer** (1,872,345 parameters), not trained from scratch: DeepSketch 2 found whole-network fine-tuning clearly better at low completeness (33.14 vs 24.53 top-1 at 20%, Tables III, XII) |
| Cuts | Each training drawing cut at a stroke boundary with the completeness fraction drawn uniformly, **including the full drawing**; label unchanged (DeepSketch 2's "merged" model: 4.66 → 28.33 at 20%, full-sketch accuracy kept at 75.01 vs 75.02) |
| Optional | A completeness head (one linear layer + loss), +3.79 at 20% in DeepSketch 2 (Table VII); decide after the baseline curve |
| Why two models | The final guess must not get worse (owner's priority). Two models make that true by construction |
| Known issue | Live and final guesses can disagree. The confirm screen shows the final top 3, so the player has the last word |

**Decided (owner, 2026-10-06, after the final report):** the final guess ("Sell it!") uses
**CNN v2 + stroke Transformer, one weight (0.4 / 0.6)** — 82.11% top-1 / 93.56% top-3 on the
172,500 test drawings, against 78.29% for CNN v2 alone and 72.44% for the CNN now in the game
(`ml/runs/sketch_v2/final_report.json`). Wired into the game server on 2026-10-06 (devlog).

**Which model does what** (owner's question, 2026-10-05): a candidate is *live guess = stroke
Transformer fine-tuned on prefixes, final guess = CNN alone*. It is the right choice if the final
report shows the stack is not clearly better than the CNN alone. A CNN can also guess partial
drawings when the strokes so far are rendered to an image and it is fine-tuned on them — DeepSketch 2
is exactly that (a CNN) — so the live-guess model is chosen by measurement, not assumed to be the
Transformer.

Measurements (validation split only; the test split is measured once, by `final`):

1. Before training anything: the current sketch v2 models (**CNN and stroke Transformer separately**)
   at 25 / 50 / 75 / 100% of the strokes,
   rounding as `max(1, ceil(f × n_strokes))`, **top-1 and top-3** (top-3 is what the confirm screen
   shows), with bootstrap 95% CIs.
2. The same as a curve by **absolute stroke index** (after stroke 1, 2, 3…), which is what the player
   experiences.
3. After training: both curves for the partial-drawing classifier.
4. *Optional, evaluation only*: completeness-weighted fusion of the prefix predictions for the final
   guess (DeepSketch 2 Table IX, +2.08 on complete sketches); adopt only if final validation top-1
   improves.
5. *To measure*: latency per guess on this Mac.

## 5. Drawing GPT (F3) and buyer scenes (F4)

### 5.1 Tokens

Following Sketchformer §3.1 (the paper, not its released code, which samples 5M moves and skips the
pen-up resampling):

- A dictionary of **K = 1,000** pen moves (Δx, Δy) found by K-means on 100k moves from the training
  split, 20% of them pen-up moves.
- Special tokens: start, end of drawing, pen lift, padding, plus one token per class (345).
- A drawing becomes `[class] [start] move move … [lift] move … [end]`.
- **Error-feedback encoding**: each move is assigned the code nearest to (true absolute point −
  current decoded point), so quantisation error cannot accumulate along a drawing. Sketchformer
  states that error accumulates on long sketches; this removes it at no model cost.

Measure before training (CPU minutes), for K = 500 and 1,000, with and without error feedback,
separately for drawings at the 10th / 50th / 90th length centile: mean and max absolute-position
error (canvas px, 0–255), drift of the last point, and the sketch v2 classifier's top-1 on decoded vs
original drawings. Sketchformer reports no number for this (its Fig. 2 is visual only).

### 5.2 Model

Decoder-only Transformer with a causal mask, next-token cross-entropy, conditioned on the class
token. Sketch-RNN Supp §4 shows that one model over many classes without class information mixes
them; the class token is the fix it proposes. Data: the stroke data already downloaded (345 classes ×
15,500), **training split only**. Size *to measure*: start small and grow only if drawings are
unrecognisable; more drawings per class is the other lever (Sketch-RNN used 70K per class, we have
15,500).

- **Every training sequence ends in `[end]`.** Choose the context length from the 99th percentile of
  token lengths and **drop** longer drawings instead of truncating them; report the share dropped per
  class. (Sketch-RNN calls stopping "a key challenge"; Sketchformer's code cuts at 200 tokens and
  forces an end token, which would teach early stopping. Our classifier packing cuts at 128 points,
  so it cannot be reused as is.)
- Augmentation: multiply Δx, Δy by independent U(0.9, 1.1) before tokenising (Sketch-RNN Supp §3).
- Sampling: sweep temperature τ ∈ {0.2, 0.4, 0.6, 0.8, 1.0} (Sketch-RNN used 0.2–0.9), possibly a
  separate τ for `[lift]`/`[end]`; choose τ separately for F3 drawings and for F4 props.
- **Checkpoint and τ are chosen by recognisability (§5.5), not validation loss**: Sketch-RNN Supp §6
  shows lower loss does not mean better drawings.

### 5.3 Modes

| Mode | Input | Behaviour |
|---|---|---|
| Random | — | Pick a class uniformly, draw it |
| Named | English text | Match to one of the 345 class names (exact or close spelling). No match → "I don't know how to draw that", **plus the closest drawable names** (Scones: unsupported requests that left the canvas unchanged caused most low enjoyment scores) |
| Scene | item + one buyer | §5.4 |

### 5.4 Buyer scenes

For buyer *b* and item *i*, every prop *p* that fits the item's group gets

```
score(p) = base_p  +  Σ_t profile_i[t] × tags_p[t]  +  λ × Σ_t [profile_i[t] > 0] × weights_b[t] × tags_p[t]
```

`profile_i` is scaled so its largest tag is 2, putting the three terms on the −2…+2 scale. The
bracket lets a buyer's taste show only through tags the item touches: an outdoorsy buyer pulls in
grass for a dog (`active` 6) but not for a cat (`active` 0).

Then: drop props with score ≤ 0 (a dislike removes a prop: `sun_averse` loses the sun), take the top
3 with at most one per zone, and lay out:

- The item is centred and **most prominent**; `large` props (house, castle, mountain, tree) go in a
  `background` band behind it or are cropped at the edge.
- Each object is **scaled uniformly** into a box shaped by its class's median aspect ratio (measured
  on the training split), **never stretched** (Sketchforme Fig. 14b: an imposed square ratio drew
  only an elephant's face).
- **No overlap**: a test checks that zone boxes are disjoint from each other and from the item box
  for every review pair (§9.3). Overlap is Sketchforme's main stated limitation.
- Draw the item first, then each prop, stroke by stroke.

λ is *to tune* on the review pairs (§9). Each prop keeps its reason ("active: buyer +2 × sun +2").

**Implemented 2026-10-07** (devlog): λ = 1; `props.json` differs from the draft below in four places, each listed with its reason in the devlog.

Not adopted: a learned layout from Visual Genome or CoDraw (Sketchforme, Scones). Those learn where
things physically sit in photos; our props are symbolic (a crown for status), and the mapping to the
345 classes, the data and the GPU time are not worth it.

### 5.5 How drawings and scenes are judged

| Check | How |
|---|---|
| Recognisable | Draw N per class; sketch v2 classifier top-1 / top-3 per class; worst classes listed |
| Class mixing | Most-confused class pairs for generated drawings (Sketch-RNN Supp §4: multi-class models blend classes) |
| Not copied | Nearest training drawing (rasterised, same renderer); distance distribution and closest cases |
| Varied | Pairwise distance between drawings of the same class |
| Scene conveys the buyer | Rule proxy: does the props' top tag match the buyer's top tag? People: show a scene, ask which of the round's buyers it belongs to (Sketchforme's "conveys" question, §11) |
| Fast enough | Milliseconds per drawing on MPS; a round needs 3–5 scenes of up to 4 objects |

## 6. Buyer GPT: understanding the player (L2) and prompt format 4

### 6.1 Why L2

Play-testing on 2026-10-05 (`ml/logs/2026-10-05.jsonl`): four typed answers to "What breed is the
cat?" were all rejected by the rule matcher (`buyers/answers.py`) — "A tiny kitty is just 3 months
old" (an *age* answer; the matcher maps it correctly to `kitten` when run against age), "british
longhair", "pursh", "ragdoll". The matcher only compares words with the option texts, so it cannot
know that a ragdoll is a purebred. The 98.0% / 96.7% in `buyer-ai.md` §10 were measured on answers
written by the rules' author, which the doc itself says flatters it. These four lines become the first
**real-player test set**, and every logged typed answer is added to it.

The same session also produced a **wrong pick**, which is worse than "can't tell". Priya asked "is the
cat healthy?"; the player typed "it has its shots"; the matcher picked `unknown` ("I'm not sure about
the shots.") because "shots" appears only in that option's text — the opposite of what was meant
(reproduced: "it has had its shots" and "all shots done" also map to `unknown`). Priya's reply, written
by the v5 model (`sources` in the log), then showed two more problems:

- **Intensity**: it opened with "Oh no, that's a dealbreaker-level issue.", a phrase from the Δ = −2
  pool, while the engine's Δ for Priya is −1 (bargain hunter, `condition` weight 1 × signal −1).
  This is the speech–Δ mismatch of §6.3.
- **Wording that does not fit the item**: "That's more wear than I'd like." is the `(condition, −1)`
  reason phrase, written for used goods; for a cat's health it reads wrong. Reason phrases are per tag,
  not per kind of item.

### 6.2 What L2 covers

| Player does | Understood as | Task |
|---|---|---|
| Answers in own words ("ragdoll", "just a street cat", "she's like 12") | the matching option, or "can't tell" | `parse` |
| Answers a different question than asked ("she's 3 months old" to the breed question) | recorded for that attribute; the buyer acknowledges it and asks again | `parse` over all attributes |
| Volunteers several facts ("vaccinated and litter trained") | each recorded | `parse` |
| Describes themselves in the buyer's seat ("two kids, I hate the sun") | signed tag weights | `profile` |
| Asks the seller ("is she good with kids?") | the attribute | `which` |
| — | the seller's answer in its own words, which then becomes a committed fact | `answer` + `parse` |

Not in L2: small talk, questions to buyers about themselves, haggling in text (bids stay buttons),
facts outside the schema ("one blue eye" → nearest option or "can't tell"). Facts and money always
come from the engine.

**Misspellings** ("pursh"). `parse` returns candidates, not one guess. Measured with plain fuzzy
matching against 33 breed names (difflib): "pursh" → persian 0.50, burmese 0.50, purebred 0.46 — the
breed is ambiguous, **but every candidate maps to the same option, `pedigree`**, so the answer is
taken as `pedigree` and the buyer can say "A Persian? Lovely." Other typos were clear: "persn" →
persian 0.83, "ragdol" → ragdoll 0.92, "siamse" → siamese 0.92, "brittish longhair" → british
longhair 0.97. When the top candidates map to **different** options, the buyer asks to confirm
("Persian, you mean?") instead of guessing. v6 `parse` data includes typo-augmented answers.

**Reading an evasive answer** (owner's example, 2026-10-05). Elena asked about the breed and added
"My place is pretty small"; the player answered "A tiny kitty is just 3 months old", which speaks to
her space worry and avoids the breed. A real buyer would take the age, and suspect that a seller who
had a purebred would have said so. Rule (engine, so it stays consistent with prices and scenes):

1. `parse` records the answered attribute (`age = kitten`) and marks the asked one as **evaded**.
2. If the evaded attribute has an option that clearly raises value (factor ≥ 1.5, e.g. `pedigree`
   4.0), the buyer **assumes the plainer option** (`mixed`), flagged as an assumption, and says so:
   "No papers then, I take it?"
3. Δ and offers use the assumed option; if the player later states the real answer, it replaces the
   assumption and everything is recomputed.
4. A buyer re-asks once before assuming, because the player may simply have misread the question.

This is pragmatic inference ("a seller would mention a pedigree"), so it is a heuristic: it can be
wrong, which is why the assumption is spoken aloud and correctable. *To decide*: the factor threshold
and whether assumptions are drawn in the buyer's scene.

### 6.3 Tasks (format 4)

| Task | Input | Output | New? |
|---|---|---|---|
| ask | item, attrs, persona, log, engine's question | the question in the buyer's voice | as format 3 |
| react | … latest answer | speech + Δ | as format 3; labels change with negative weights; **Δ position decided at the freeze** (below) |
| refuse | item, persona | refusal | as format 3 |
| `parse` | attrs with option ids, the asked attribute, the player's text | `attribute=option` pairs, or `none` | **new** |
| `profile` | the player's self-description | tag weights, e.g. `family +2, active -1` | **new** |
| `which` | attrs, the player's question | one attribute, or `none` | **new** |
| `answer` | item slot, attribute, committed facts so far, the player's question | the seller's words (parsed afterwards into an option) | **new** |

**Δ before or after speech.** Format 3 writes speech, then `<|interest|>Δ`, so a buyer can say "Oh
good!" with Δ = −2. He et al. generate words *from* the decision, so they cannot contradict it. At the
format freeze, measure on v5 how often speech sentiment and Δ sign disagree; move Δ first only if
that rate is material and Δ accuracy does not drop (speech currently acts like reasoning before Δ).

**Why v5 sounds like a template** (measured 2026-10-05). v5's training targets
(`data/processed/buyers_v5/train.jsonl`) are 128,242 examples but only **2,119 distinct sentences**,
all from the hand-written phrase pools; the most frequent ("I'm curious.") appears 4,621 times. In the
owner's three logged play sessions, 12 lines were written by the model and 21 came straight from
phrase pools (`sources` in the log). Of the model's 22 sentences, **17 appear word for word in the
training targets** and the other 5 differ only in the item name ("A sun?", "Before I decide, is the
cat litter trained?"). The model does generate token by token; it has simply learned to reproduce
the only sentences it was ever shown. D8's local-LLM paraphrasing was planned but never run, so
L3 data (§6.5) is what changes this, and the copy rate in §6.7 measures it.

### 6.3b Our own bargaining dataset, modelled on CraigslistBargain (owner's idea, 2026-10-05)

Build a dataset **as large as CraigslistBargain (≈ 6,700 dialogues)** whose content fits our game:
our items, our schema, our buyers, our two modes. CraigslistBargain is used for its **patterns**, not
its sentences (its licence is not stated; using statistics instead of text keeps us away from its
wording, though this is not a legal judgement).

Measured from the official files (`ml/data/raw/craigslist_bargains/`, train + validation):

| Pattern | Value |
|---|---|
| Dialogue-act counts | unknown 9,479; counter-price 8,066; inquiry 5,205; init-price 4,517; offer 4,195; intro 4,162; accept 3,843; inform 2,448; disagree 2,126; agree 1,929; insist 447; vague-price 412; quit 221; reject 203 |
| Housing: buyer questions (2,371 in 1,188 dialogues), by topic (keyword counts, a question can hit several) | price/negotiation 855, move-in date 362, location/transit 225, size/rooms 212, utilities 159, pets 133, condition/repairs 85, deposit 74, laundry 58, parking 57, appliances 57, furnished 39, yard/outdoor 27, noise/safety 26 |
| Postings that map to our classes (per-category rules) | ≈ 1,135 of 1,402, onto 18 of our `buy` classes; none for animals, instruments, sports, food, tools |

How it is built:

1. From CraigslistBargain take: act-sequence shapes (intro → inquiry → inform → counter-price …),
   turn counts, question-topic mix, how often people insist, quit or accept.
2. For each of our 259 sellable items (≈ 26 dialogues each), sample a scene from the engine (buyers,
   traits, which options get said) and an act shape, and have Claude write the dialogue in our
   format: slots for `{item}` and `{amount}`, committed facts recorded, free details allowed.
3. Check every dialogue (writer → checker → engine): no contradictions, no digits, reactions follow
   from preferences, Δ wording matches the engine. Several Claude roles review each other and
   disagreements are logged (the owner's thesis idea).
4. Hold out items and phrasings exactly as for v5/v6 so generalisation can be measured.

Ideas taken from the real listings:

- **House** (owner decisions, 2026-10-05): add **all** of parking, laundry, appliances, yard,
  noise/safety, utilities and pets allowed as new `house` attributes (options, factors and signals
  drafted in the catalog builders and reviewed before use). That makes 11 attributes, more than the
  3–5 that `buyer-ai.md` §2.3 assumes; the host already asks at most 5 per round, chosen by concern,
  so each round asks the 5 that matter most to the buyers present; the schema validation limit must
  be raised for this. "Is it for rent or for sale?" is always answered **"for sale"** (option A);
  it is conversation only and does not change the price.
- **Phones and accessories**: phone listings bundle chargers, cases, headphones, iPads. Co-occurrence
  in listings is direct evidence of "what goes next to what" for scenes (§2.3). Drawable among them:
  headphones, camera, power outlet, laptop, keyboard, remote control (charger and phone case are not
  among the 345 classes). With only 1,402 postings this complements, not replaces, the text-based
  relatedness in §2.3.

### 6.3c Is 6,700 dialogues enough? (data mix and a learning-curve rule)

The 6,700 dialogues are one of four sources, each teaching something different:

| Source | Teaches | Size |
|---|---|---|
| Pre-training text (FineWeb-Edu) | English itself, which is what lets the model read phrasings it never saw | ≈ 1–1.35B tokens (§6.5) |
| Engine-generated examples (as v5) | ask / react with exact Δ labels | unlimited; v5 used 128,242 |
| **Short task examples** written by Claude + programmatic typos | `parse`, `profile`, `which`: many phrasings per option | e.g. 20 phrasings × 2,721 options ≈ 54k (estimate), plus typo variants generated by code |
| **Full dialogues** (§6.3b) | conversation flow, consistency, free details, tone | ≈ 6,700, roughly 9 turns each (estimate ≈ 60k turns) |

So understanding the player does not rest on the 6,700 dialogues: short `parse` examples cover each
option directly, and they are cheap to make in large numbers.

How much is enough is **not known in advance** for a model this small. Rule: in the pilot (§6.5),
train on 25%, 50% and 100% of each source and plot held-out metrics (parse accuracy on held-out
phrasings and the real-player set, contradiction rate, copy rate). If a metric is still rising
steeply from 50% to 100%, generate more of that source; if it has flattened, more data of that kind
will not help and the model size or pre-training is the limit.

### 6.3d Storing Claude-generated data

Owner's decision (2026-10-05): the data is generated in Claude Code sessions (subagents), not via
the API. Generation is slow and not exactly repeatable, so what Claude wrote is treated as **raw,
immutable source data**, and everything after it is rebuilt by one command.

```
ml/data/raw/claude_gen/v6/                 (out of git, like all raw data)
  requests/<task>.jsonl                    what was asked: request_id, task, item, attribute, option,
                                           variant, held-out flag, prompt template id; made by the
                                           engine with a fixed seed before any generation
  batches/<task>/batch_0001.jsonl          what Claude wrote, append-only, one file per batch:
                                           request_id, outputs, generator (model id, role, date),
                                           prompt_sha256
  reviews/<task>/batch_0001.jsonl          reviewer/checker verdicts per output, with reasons;
                                           rejected outputs are kept, not deleted (they show where
                                           the writer and reviewer disagree)
  MANIFEST.json                            per file: rows, SHA-256
ml/sellanything_ml/datagen/                (in git)
  prompts/<task>_v1.md                     prompt templates, versioned; their hash is stored per output
  requests.py, validate.py, build.py       make requests; check outputs; build the processed set
ml/data/processed/buyers_v6/               rebuilt by `python -m sellanything_ml.datagen.build`
  train / val / test_seen / test_heldout   labels (Δ, option ids) recomputed by the engine at build
  stats.json, MANIFEST.json, DATASHEET.md  counts, checksums, a datasheet (purpose, sources and
                                           licences, generation process, known issues)
```

Rules:

1. **Text and labels are separate.** Claude writes text; the engine computes labels when the
   processed set is built. An engine change then means rebuilding, not regenerating — except where
   wording depends on the label (reaction strength), which is why dialogues wait for the engine and
   trait freeze (§8), while `parse` / `profile` / `which` examples, which depend only on the schema,
   can start first.
2. **Stable ids and resumability.** `request_id` = hash of (task, item, attribute, option, variant);
   a batch skips ids that already have an accepted output, so an interrupted session resumes.
3. **Splits fixed before generation.** Held-out items come from `ml/data/item_splits.json`
   (40 `buy` held out of 259 sellable) and a held-out list of phrasings and instances (e.g. some breed
   names) is drawn by the request sampler, so nothing is tuned on test data.
4. **Checks before acceptance**: schema of the record, no digits in speech, the parse target is a
   real option, exact and near-duplicate removal, no held-out item in train.
5. **Provenance per record** (which model, which role, which prompt version), so later analysis can
   compare writers and reviewers — the evidence the owner's thesis idea needs.

### 6.4 Special tokens without re-pre-training

The tokenizer has 9 special tokens and no spares. New tokens (`<|parse|>`, `<|profile|>`, `<|which|>`,
`<|answer|>`, `<|weights|>`) get ids after the SentencePiece vocabulary in our tokenizer wrapper; the
embedding and output layers grow and SFT learns them. **Add 8 spare tokens at the same time.**
*To verify first*: load the pre-trained checkpoint, grow the vocabulary, check identical logits for
the old tokens.

### 6.5 L3: grounded conversation (target, within a 5-day training budget)

Owner's decision (2026-10-05): aim for L3, with **all v2 training finishing in 5 days on this Mac**.

**What L3 adds to L2**: buyers keep a conversation in character — follow-up questions ("Is she okay
alone all day? I work long hours."), short comments ("Ragdolls are so cuddly!"), polite deflection of
off-topic chat, and answers to the player's questions about themselves that fit their persona card.
**The principle is logical consistency, not a hidden truth** (owner, 2026-10-05; an earlier draft
stated a stricter "all item facts only from the engine" rule and an "honest seller with hidden
answers" rule, neither of which the owner asked for):

| Rule | Status |
|---|---|
| Money is computed by the engine and never written by the model | owner decision (D2) |
| Whatever is said about the item becomes a **committed fact** (the engine records parsed facts; free details are kept in the dialogue context) | owner decision |
| Nobody contradicts a committed fact later ("brand new" … "a few scratches") | **main consistency metric** |
| Reactions follow from what was said: Δ and the bid move the way the buyer's preferences say they should | owner decision; checked against the engine |
| A buyer misstating what the player said (player: "she has her shots"; buyer: "so no shots?") | error, because it changes Δ and the bid |
| Opinions, enthusiasm and sales talk ("She's a sweetheart, you'll love her") | allowed for everyone |
| **Free details the engine does not know**: a buyer "I had a ragdoll as a kid", a seller "She loves the spot by the window" | **allowed** (owner, 2026-10-05). They do not move the price; once said they are committed and must not be contradicted |

**Measured on this Mac** (M5, 32 GB):

| Quantity | Value | Source |
|---|---|---|
| Pre-training throughput, 12.9M (d384 × 6) | 5,078 tok/s at the end of its run (part of the run shared the GPU with the CNN) | `ml/runs/pretrain/metrics.jsonl` |
| Pre-training throughput, 27.8M (d512 × 8), bf16 | 3,582 tok/s, short run, shared with the CNN | devlog 2026-10-03 M7 |
| SFT v5 | 8,015 steps × 0.88 s ≈ 2.0 h | `ml/runs/sft_v5/train.log` |
| Game-side latency on CPU, 40 new tokens after a 300-token prompt, KV cache, while sketch training runs | 12.9M **0.098 s**; 27.8M **0.166 s**; 50.7M (d640 × 10) **0.283 s** | measured 2026-10-05 |

So CPU latency does not limit the model size up to ~50M; GPU training time does.

**Two paths, chosen by a pilot**:

| Path | Model | Pre-training | Tokens in 74 h (from the throughputs above) |
|---|---|---|---|
| A | keep 12.9M, **continue** from `runs/pretrain/final.pt` (100M tokens so far) | +74 h | ≈ 1.35B more |
| B | 27.8M from scratch | 74 h | ≈ 0.95B (≈ 34 tokens per parameter) |

Pilot (≈ 4 h GPU): SFT the current 12.9M on a small L3 set. Go to path B only if the 12.9M model fails
the L3 bar on held-out items: hallucination rate, persona consistency, and a read-through of 50
dialogues by the owner. All throughputs above were measured while sharing the GPU, so a 30-minute
uncontended benchmark per size comes first.

**Data for L3** (the largest unknown):

- Pre-training text: **20 FineWeb-Edu shards are on disk** (`shard_00000`–`00019`, 1,068,032
  documents, ≈ 1.34B sp8192 tokens; downloaded 2026-10-05 with the owner's OK; SHA256SUMS in the
  folder). Path A needs ≈ 1.35B tokens, path B ≈ 0.95B, so no repetition is needed.
  ⚠️ **Pin the validation shard**: `llm/prepare.py` uses the *last* shard as validation, so re-running
  it with 20 shards would make shard 19 the validation set and put the old one (shard 3) into
  training — old and new validation losses would no longer be comparable. Keep shard 3 as
  validation explicitly before path A or B starts.
- General dialogue: OpenAssistant (Apache 2.0, human-written), as D12 planned (**download**).
- In-character, grounded buyer dialogues: **written by Claude from engine scenes** (owner,
  2026-10-05: generating text with Claude is allowed; only rented compute is not; plan.md updated).
  Claude gets the scene (item, the options chosen in that dialogue, buyers' personas and weights, engine Δ and offers) and
  writes the dialogue with slots (`{item}`, `{amount}`); the engine then checks every dialogue —
  facts only from the scene, no digits, Δ wording matches the engine's Δ in sign and strength — and
  drops the ones that fail. Several Claude passes with different roles (writer, fact-checker, a
  "difficult player") can review each other, as in the owner's thesis idea; disagreements are logged.
  This takes no GPU time, which moves 24 h of the budget to pre-training (§8).
- **CraigslistBargain** (He et al. 2018; HF `stanfordnlp/craigslist_bargains`): 6,682 human–human
  bargaining dialogues (5,247 / 597 / 838), avg 9.2 turns of 15.5 tokens (He et al. Tables 2–3), so
  roughly 0.75M words (estimate). Useful for natural haggling style and seller/buyer turns in the
  buyer's seat. Problems: **licence not stated** (HF card: "unknown"; the CoCoA code repo is MIT but
  says nothing about the data); prices are written as digits in text (would need He et al.'s
  delexicalisation to `{amount}`); facts are free text that the engine cannot check, and He et al.
  Table 4 lists embellishment and cheap talk as common. Sales talk is fine; the concern is the
  model learning to **contradict itself** ("brand new" now, "a few scratches" later), which breaks the
  owner's consistency requirement (F6). The HF loader runs a remote script, so the raw `parsed.json` files would be fetched
  from CodaLab instead. Use in a style stage before our own SFT, and measure the contradiction rate
  with and without it. Owner decides on the licence risk before download.

### 6.6 Data v6

Generated by the engine as before (D3), so labels are exact:

- Rounds re-run with the new traits; `react` labels recomputed with negative weights.
- `parse`: for every option, player-style answers from phrase pools plus local-LLM paraphrases (D8),
  **including world-knowledge instances** (named breeds → `pedigree`; "mutt", "stray", "rescue" →
  `mixed`; brand names; relative times). Some instances and phrasings are **held out** to test
  generalisation, not memory.
- `profile`: English self-descriptions per tag and sign; held-out phrasings.
- `which`: paraphrased questions, and questions phrased by tag ("good with kids?" → the attribute
  whose options carry `family`).
- `answer`: canonical answers paraphrased, item written as `{item}` (D19).
- `react` reasons written **per tag, sign and item group** (`pets_livestock`: health and vet words,
  not "wear"; `electronics`: works / battery; …), so the reason fits the item; the generator checks
  that each reason's group matches the item's group in `items.tsv`.
- Same held-out items as v5.

### 6.7 Evaluation

| Task | Metric | Baseline it must beat |
|---|---|---|
| ask, react, refuse | as v5 (`llm/evaluate.py`) | v5 |
| `parse` | exact option, can't-tell rate and **wrong-pick rate separately** (as §10 does); real-player set; held-out instances | the rule matcher (`answers.py`) |
| `profile` | per tag: sign and exact value; **downstream**: Δ and top-3 scene props computed with predicted vs true weights (PrefBench: surface metrics can look fine while outcomes are poor); a negation/dislike set ("I don't have kids", "no kids", "I hate the sun") | all-zero; keyword lookup over the phrase pools |
| `which` | accuracy; `none` precision/recall | keyword rule matcher |
| `answer` | parses to exactly one option; **contradiction rate** against facts already committed in the dialogue; hand audit of ~100 dialogues for contradictions and reactions that do not follow (MART §5.6 found self-contradiction and invented items in every small model), 2 raters with agreement | canonical text |
| all speech | **degeneracy**: share of the 3 most frequent sentences per task (He et al. §4.4), distinct-1/2; **copy rate**: share of model sentences that appear word for word in the training targets | v5 (below) |
| react | speech–Δ agreement in **sign and strength** (a −2 phrase for a −1 change counts as a mismatch, §6.3); reason fits the item group | — |
| `parse` safety | wrong picks that **reverse** the meaning ("it has its shots" → "not sure about the shots") counted separately | the rule matcher |

**Statistics.** Every metric with a bootstrap 95% CI, resampled by round, fixed seed (PrefBench).
**No-regression rule**: v6 vs v5 compared on the same test examples with paired tests (McNemar for
exact Δ, or a paired bootstrap), not unpaired ANOVA over nested utterances (MART). If v6 is worse on
the old tasks, the new tasks get a smaller share of the data, not a separate model.

v5 reference (`ml/runs/sft_v5/final_v5.json`): react Δ exact **98.33%** seen / **70.66%** held-out,
ask on-topic **100%** seen / **93.24%** held-out.

**Fallback at play time.** If a model line fails validation or the on-topic check, show the engine's
canonical text and log it; report the fallback rate (He et al.'s retrieval generator is always
well-formed; our held-out ask validity is 98.48%).

### 6.8 Interim rule fixes (pending owner approval)

Until v6 ships, two rule changes in `answers.py` / `rounds.py` would fix the four logged failures:
(A) if the text matches another attribute's option, record it and re-ask, then apply the evasion
rule (§6.2); (B) per-option `aliases` in the schema (breed names, "mutt", "stray") plus fuzzy
matching that accepts when all close candidates map to one option and confirms otherwise. v6 `parse`
replaces the matching; the evasion rule stays in the engine.

## 7. Protocol and logging

| Message | Change |
|---|---|
| `classify` | optional `"partial": true` |
| `draw` | new: `{"class": "dog"}` or `{"random": true}` → strokes |
| `scene` | new: item + buyer id → objects, placement, strokes, reasons |
| `seat_start`, `seat_profile`, `seat_ask`, `seat_bid` | new (§3.1) |

**Structured event log per round** (He et al.: logging acts made strategy inspectable): who asked
which attribute, the player's text and how it was parsed, engine Δ and GPT Δ, offers, bids, winner,
scenes shown, timings. Needed for the player study and for measuring parsing in the wild.

## 8. Order of work

| # | Work | Needs GPU | Why this order |
|---|---|---|---|
| 1 | Finish sketch v2, report | yes (running) | already running |
| 2 | Measurements that decide the format: Δ-source replay (§3.2), speech–Δ agreement on v5 (§6.3), partial-stroke curve of current models (§4) | no | they change the frozen formats |
| 3 | Engine: negative weights, new traits, host concern, props.json + tests, buyer's seat, player utility, simulated players | no | everything else reads these |
| 4 | Data v6 generator + checks; tokenizer growth check; real-player test set; baselines | no | freezes the buyer GPT format |
| 5 | **Buyer GPT SFT v6** | yes | core of the project, first in the GPU queue |
| 6 | Stroke dictionary measurements (CPU) → drawing GPT training | yes | needed by F3 and F4 |
| 7 | Partial-drawing classifier (fine-tune) | yes | smallest risk, last |
| 8 | Game: live guessing, buyer scenes, draw modes, buyer's seat, typed-answer UI | no | each when its model is ready; scenes can first use dataset drawings as placeholders |

**Working method (owner, 2026-10-05): drawing models first, buyer GPT grown a few categories at a
time.**

- **Drawing side first**: after sketch v2, the GPU goes to the drawing GPT (named items, related-prop
  scenes) and the partial-drawing classifier, each trained until its §4 / §5.5 targets are met.
- **Buyer GPT in small steps**: start with **2 item categories** (e.g. cat + bicycle, which already
  have pilot `parse` data; house after its new attributes exist), play them until the dialogue is
  right, then add 1–2 categories per step.
  - Each step **retrains SFT from the existing pre-trained checkpoint on the full mixture so far**
    (not fine-tuning the last step's model on the new categories only), so earlier categories are
    not forgotten. SFT on a few categories takes well under the 2 h of v5.
  - A wrong dialogue means fixing the prompt (new version) and regenerating **only the affected
    request ids**; raw outputs of the old version stay on disk (§6.3d).
  - Every step keeps a probe: held-out phrasings of the trained categories plus one category that is
    never trained, to see whether the model generalises or only memorises.
  - The big pre-training run (path A/B) happens once, when the GPU is free of drawing work; the
    early steps use the current 12.9M checkpoint, and their results inform the A/B choice.

**GPU budget: 5 days = 120 h**, after sketch v2 finishes. Estimates come from the measurements in
§6.5; items marked *to measure* are caps, and if a cap is hit we cut tokens or samples, not stages.

| Job | Hours |
|---|---|
| Uncontended throughput benchmarks (3 GPT sizes) | 2 |
| L3 dialogue generation (Claude, no GPU) | 0 |
| L3 pilot SFT on 12.9M | 4 |
| Pre-training, path A or B | 74 |
| SFT v6 (all tasks, L2 + L3) + evaluation | 8 |
| Drawing GPT (*to measure*, capped) | 16 |
| Partial-drawing classifier fine-tune | 3 |
| Slack for sleep, restarts, a rerun | 13 |
| **Total** | **120** |

Every model swap, game change, commit and push is approved by the owner first.

## 9. Before anything is trained: review list

1. The new traits with dislikes (§2.1).
2. `props.json`: tags, zones, sizes, `fits` (§2.3).
3. A printed table of scenes for about 20 item × buyer pairs, from 1 and 2 with the engine alone
   (dog × outdoorsy → sun, grass; cat × outdoorsy → no grass), with the no-overlap test passing.
4. Example prompts for the four new GPT tasks (§6.3).
5. Results of the §8 step-2 measurements and the owner's decisions in §12.
6. The list of fixable options and any explicit costs (§3.3).

## 10. Risks

| Risk | Mitigation |
|---|---|
| Dislikes make many buyers walk away | Measure with the engine before generating data (§3) |
| Growing the vocabulary breaks the pre-trained model | Identical-logits check (§6.4) |
| New tasks dilute the old ones | No-regression rule with paired tests (§6.7) |
| 12.9M model lacks world knowledge for `parse` | Teach it through v6 data; measure on held-out instances; keep the rule matcher as a fallback |
| Drawings unrecognisable, copied or mixed | §5.5 measures all three |
| 3–5 scenes per round too slow, or overload the player | Measure; draw scenes on demand or as each buyer speaks (Anderson et al.: "Everything" caused overload) |
| Scenes do not beat simpler explanations | Plausible (COViz: outcome views alone not better than bars; Sketchforme: faster but not more accurate); pre-register the primary metric (§11) |
| Weak props for `condition` / `performance` | Accepted |

## 11. Player study (thesis)

Claim and prior work: [`related-work.md`](related-work.md) §6.

| Part | Design | Source |
|---|---|---|
| Conditions | dialogue only / + scenes / + tag bars / + both; plus a **control with random props that fit the item**, to show it is preference-driven props that help, not pictures | Anderson et al.; COViz; Sketchforme |
| Assignment | within subjects with counterbalanced order and different buyer sets per block if N is small; between subjects if ~30 per condition is reachable (Anderson ~31, COViz ~27–31) | COViz Study 2 |
| Primary measure (pre-registered) | **forward simulation**: before the final bids, "which buyer will pay most?" on held-out item × buyer pairs, explained pairs kept apart from test pairs, balanced answers, forced choice | Hase & Bansal |
| Secondary | counterfactual: "if this dog were senior, does Maya's bid go up or down?" (exact answers from the engine); free-text "why", coded against the buyer's top tags (2 coders on 20%) | Hase & Bansal; Anderson et al. |
| Item difficulty | include counter-intuitive cases; easy cases hit a ceiling (control ≥ 75% in Anderson et al.) | Anderson et al. |
| Subjective | Explanation Satisfaction Scale (8 items, 5-point), confidence per prediction; reported, but not the main outcome (ratings did not track correctness in Hase & Bansal) | Hoffman et al.; COViz |
| Manipulation check | players name the props and the trait they suggest; tutorial quiz | COViz |
| Overload | scenes on demand; log time per decision | Anderson et al. |
| Reporting | N, means, SDs, effect sizes, CIs, excluded participants (missing in Sketchforme, He et al., Sketchformer) | — |
| Before running | power estimate; **ask the school whether review/consent is required** | — |

## 12. Open decisions for the owner

1. ~~Conversation level~~: L3 is the target within 5 days of training (§6.5). Still open: OK to
   download more FineWeb-Edu shards, OpenAssistant and a local LLM's weights.
2. **Play-time Δ source** (§3.2), after the replay measurement.
3. **Learned relatedness** instead of `base`/`fits` (§2.3).
4. **Interim rule fixes** A and B (§6.8).

## 13. Changes from the second reading batch (2026-10-05)

Twelve tier-1 papers read in full (reading-list E1–E10, D1, D4, B6, C4; summaries in
`related-work.md` §7). Each change names its source. Items marked *experiment* are tested before they
are adopted.

### 13.1 Pre-training data and model size

| Change | Source | Affects |
|---|---|---|
| Select item-related text with **DSIR** (hashed unigram+bigram importance resampling over 128–256-word chunks), not keywords. Keyword matching is too noisy: on 2 shards, 67.1% of documents "mention an item", led by verbs (*saw*, *watch*, *bear*, *train*) | Xie et al. 2023; measured 2026-10-05 | §6.5 |
| **Importance resampling, not top-k**, so the selection does not collapse onto one kind of text | Xie et al. §3, App. A | §6.5 |
| Build the selection target from **train categories only**; held-out categories then also test whether the *pre-training* generalises | Gururangan et al. Table 6; Xie et al. Fig. 2 (mismatched targets cost 6% on average) | §6.5, §8 probe |
| Check every arm with **KL reduction** before any GPU time (r = 0.82 with downstream F1 in DSIR) | Xie et al. §6 | §6.5 |
| *Experiment*: pilot arms at equal tokens — random / keyword / DSIR 30% + general 70% / DSIR 100% | Gururangan et al.; Xie et al. | §6.5 pilot |
| FineWeb-Edu **down-samples commerce topics** (cars −1.0%, furniture −0.8%, shopping −0.8%) and fits conversational text worse; an item-relevance selection over plain FineWeb (new download, owner's OK) may add what Edu removed | Penedo et al. Fig. 18, Table 3 | §6.5 |
| Repetition is cheap up to ~4 epochs (+0.5% loss; no significant downstream change); with 20 shards on disk none is needed; do not stop on loss spikes at epoch starts | Muennighoff et al. §6–7, App. J | §6.5 |
| The data-constrained scaling fit favours **path B (27.8M)** over path A by ≈ 0.24 nats (extrapolated from C4 coefficients; a hint, not a result) | Muennighoff et al. App. A–B | §6.5 A/B choice |
| *Experiment*: short Claude-written **"textbook" passages per item** (what a ragdoll is, what kitten shots are), with and without passages for held-out categories — the cleanest test of "enough structured context" | Gunasekar et al. §2.2; Eldan & Li | §6.5, §11 thesis B |

### 13.2 Generated data

| Change | Source | Affects |
|---|---|---|
| Each request draws **1–3 random constraints** (register, a feature such as negation or a named breed, 1–2 everyday words), stored in the request; without them LLM-written data is "very repetitive" | Eldan & Li §2; Gunasekar et al. §2.2 | §6.3c, §6.3d requests |
| **Diversity report before training**: distinct-1/2, distinct-sentence ratio, near-duplicates, top-3 sentence share, against v5's 2,119 distinct sentences | Gunasekar et al. §6 (names this as missing) | §6.3d stats |
| **Review tiers**: T0 engine checks → T1 two independent blind reviewers (different Claude model where possible; options shown as shuffled A/B/C) → T2 one debate exchange only on disagreement → T3 a person | Du et al. §3, Fig. 10–12 | §6.3d |
| Same-model reviewers share confident errors, so agreement ≠ correctness: a **human gold set** (~300 items, owner + one more person) measures real error; the thesis compares writer only / +1 reviewer / +2 different reviewers / +debate / majority-of-3 | Du et al. §3.2, §5, Fig. 23; our pilot (7 of 12 flagged items are world-knowledge or date judgements) | §6.3d, §6.7, §11 |
| World-knowledge mappings (Schwinn → known brand) become **person-reviewed alias facts**; date-dependent options carry an "as of" date in every prompt | Du et al. (shared knowledge errors); our pilot | §6.6, §6.8 |
| Review records store `prompt` and `prompt_sha256` (done 2026-10-05; the pilot's reviews are covered by `reviews/parse/PROVENANCE.json`) | Gebru et al.; our rule 5 | §6.3d |
| `DATASHEET.md` follows Gebru's 7 sections plus a review-agreement report; numbers generated, narrative written by a person; a new datasheet per build | Gebru et al. §3, §4 | §6.3d |
| `profile` output is **sparse**: an unlisted tag means "not mentioned", distinct from a stated 0; add metrics for hallucinated and missed tags; train on cumulative prefixes of multi-message self-descriptions | Chan et al. §3.3, §5.6 | §6.3, §6.7 |

### 13.3 Buyer GPT targets and evaluation

| Change | Source | Affects |
|---|---|---|
| *Experiment*: a short **engine-computed trace before Δ**, e.g. `<|tags|>condition-2 low_maintenance-1`; held-out errors are mostly sign errors (sign 76.18% vs exact 70.66%), i.e. misreading the option; keep it short (long traces made small models repeat in R1 §G.1) | Hinton et al. (richer targets than a hard label); DeepSeek-R1 | §6.3, §6.7 |
| No RL for Δ: labels are exact, and RL from small bases failed in R1 | DeepSeek-R1 §F.1, §G.1 | §6 |
| Report held-out results split into **near / far** items (similarity of item profiles and option texts to the nearest training item), plus one retrain with near neighbours removed | Gunasekar et al. Table 3 (gains mostly on near-duplicate problems) | §6.7, §8 probe |
| Add a stricter **held-out group** split (a whole `items.tsv` group never trained) next to the held-out-item split | Hinton et al. §3 (omitted class); DeepSeek-R1 (same-domain tests only) | §6.3d rule 3, §6.7 |
| Novelty metrics: share of generated 4/5-grams absent from training targets; Rouge-2 to the nearest training target; n-gram overlap report between `test_heldout` and train | Eldan & Li §4.4; DeepSeek-R1 §D.1 | §6.7 |
| A small **human-written test set** (owner + 1–2 friends, ~50 items for held-out categories, never seen the Claude data; no 6-gram overlap) | Gunasekar et al. §4; Eldan & Li fn. 4 | §6.1, §6.7 |
| Held-out Δ confusion matrix to detect falling back to a default; any correction tuned on validation, never test | Hinton et al. §3 | §6.7 |
| Committed-fact log in full in the prompt (one round fits); every affirmative claim checked against it, giving a hallucination rate comparable with Generative Agents (6/453 = 1.3%) | Park et al. §4.1, §7.1 | §6.5, §6.7 |
| **Interview-style evaluation** in 5 categories (self-knowledge, memory, reaction, reflection, deflection); reaction and reflection items have engine ground truth | Park et al. §6.1, App. B | §6.7 |
| Track **refusal / deflection rate**: instruction-tuned models are "overly cooperative", and our L3 data is written by an instruction-tuned model | Park et al. §7.2; Gallotta et al. §5 | §6.5, §6.7 |

### 13.4 Scenes and the player study

| Change | Source | Affects |
|---|---|---|
| **Render, then verify**: after drawing a prop, if the sketch classifier does not put it in its top 3, resample (up to N), then fall back to the next prop; recompute "conveys the buyer" from the recognised classes | PEARL reflection (idea only; its ablation does not isolate it) | §5.4, §5.5 |
| Score "which buyer is this scene?" separately for easy rounds (buyers' top tags differ) and hard rounds (shared top tag), with chance levels | PEARL App. D.3 | §5.5, §11 |
| Code free-text "why" answers as citing a **desire** (weights) or a **belief** (committed facts), each right or wrong; scenes should raise desire-based reasons; report κ per category; no intention coding | Chan et al. Tables 1–3 | §11 |
| Pilot check that dialogue alone does not already reveal the answer (ceiling) | Chan et al. (humans 91% when preferences are explicit) | §11 |
| Call our model a **small Transformer language model**, not an LLM (the survey's cut-off is a few hundred million parameters); cite CICERO and the survey for "LM speaks, other code decides" | Gallotta et al. §2, §3.1, §4 | thesis wording |

## 14. Cats: drawn by the model, adopted, and played (owner, 2026-10-09)

Status: **design**, the drawing model is in training. Data and code: `ml/sellanything_ml/sketch/catface.py`,
`catshape.py`, `centerline.py`, `infdraw.py`, `catsprite.py`; results in `docs/devlog/2026-10-08.md`.

### 14.1 Why sell: to adopt cats
Selling earns money; money adopts cats. This gives the bargaining rounds a goal the player cares about.

### 14.2 The angel draws, the player adopts
At the top of the screen an angel draws cats, **up to 7**. The player picks one and pays to adopt it.

- Each cat is drawn by the shape GPT (`catshape.py`), and the game paints its tokens **as they arrive**: lines
  part by part (outline, ears, eyes, nose and mouth, then the rest by place), then colour part by part at 85%
  opacity over everything before (face base, inner ears, iris, pupil, nose and mouth, lighter fur, shadows,
  darkest last). What the player watches is the model's own order of work, not an animation played afterwards.
- Every cat differs: they are samples, not picks from a set.
- *To measure*: the time to generate one cat (about 1,300 tokens, 2,688 at most on 16 faces) on the CPU, and
  for 7 at once in a batch. If it is slow, the angel draws at the model's pace.

### 14.3 One cat sprite for every cat
All cats share **one colourless sprite** (each state a row of frames, like `game/assets/characters`), and
`cat.json` gives, for every frame, where the face's **9 points** are: left eye, right eye, mouth, and the outer
base, tip and inner base of each ear, the points the CAT dataset annotates and the model draws first.

- **The face looks at the player in every state** (chibi, body turned as the state needs). The model draws
  faces from the front; a side view could not show them.
- The head's shape and the points follow the dataset's mean face (`game/assets/cat/head_template.png`): the
  generated face then needs only a small warp.
- **v0 placeholder**: `game/assets/cat/cat_idle.png` (4 frames of 128×128, idle) and `cat.json`, drawn by
  `catsprite.py`. It works for testing the mapping but is not good art: the head is boxy and the motion barely
  shows. To replace (§14.6).

### 14.4 Colouring the sprite from the drawn cat
- **Face**: the generated face is warped onto every frame's head by a **piecewise affine map from its own 9
  points to the frame's 9 points** (`skimage.transform.PiecewiseAffineTransform`). This is geometry only: the
  face shown is the model's face, its markings exactly where the model put them. Head pixels outside the cat's
  own face get its base colour.
- **Body**: the model draws faces only (the CAT dataset has no bodies), so the body is coloured by **game code
  from that cat's own colours** (its base, lighter and darker fur). The game and the portfolio must say so:
  *the face is drawn by the model; the body pattern is generated by code from the colours the model chose.*
  This does not make the model look better than it is (§0's rule), but a reader must not think the model drew it.
  v0 uses round patches, which read as a spotted dog, not a cat; a better generator (a dark back and a light
  belly, tabby bands from the face's stripe colour) is to design.
- Demo with real faces from the colour data: `ml/runs/catface/sprite_demo.png`.

### 14.5 Many cats
Each adopted cat is **baked once** at adoption: its face and body colours painted into its own copy of every
sprite sheet. After that a cat is an ordinary character, and many can move about at once with no model cost.
Engine work: `LoadTexture` reads a file today; it needs a variant that takes pixels from memory.

### 14.6 Open decisions and next steps
1. **Sprite art**: drawn by hand or by an image model from the brief below; whoever draws it keeps the head inside
   `head_template.png`'s outline and the face blank, and marks the 9 points on every frame.
2. **Body pattern generator** (§14.4).
3. Order: the shape GPT (in training) → idle art → the face mapping on generated faces → baking and the
   engine's texture-from-memory → walk, run, jump, lie down, sit.

**Brief for the idle sprite**: one cat, chibi, sitting, face toward the viewer with the eyes level and both ears
up; 4 frames of 128×128 in one row (512×128 PNG, transparent background); flat light grey fill (#CDCDCD) and a
dark grey outline (#464646, about 2 px) with no shading or texture; **the face left blank** (it is painted in
later), the head at most the size and shape of `head_template.png` scaled to 72 px wide, centred near the top;
the head moves at most 2 px between frames; the idle motion is breathing (body rises) and a slow tail swing.
A second PNG of the same frames with flat colours per part (head red, body green, tail blue, paws yellow) tells
the game which pixels to colour.
