# Game ↔ ML protocol (v1)

- Transport: TCP, `127.0.0.1:5555` by default
- Framing: one JSON object per line (UTF-8, `\n`-terminated)
- Every request gets exactly one response, in order

## `ping`

```json
{"type": "ping"}
```
```json
{"type": "pong", "protocol": 1}
```

## `classify`

`strokes` is a list of strokes; each stroke is `[xs, ys]` in canvas pixels (any scale or
offset). The server reproduces Quick, Draw!'s preprocessing (align, scale to 0–255, resample,
RDP ε = 2) and rendering (centered, 16-px round strokes, anti-aliased); on real dataset strokes
this matches the official 28×28 bitmaps with a pixel correlation of 0.996
([`raster.py`](../ml/sellanything_ml/sketch/raster.py)). The classifier is the v2 ensemble
([`infer.py`](../ml/sellanything_ml/sketch/infer.py)): a CNN on the 48×48 bitmap and a stroke
Transformer on the simplified points, log-probabilities mixed 0.4 / 0.6 (82.11% top-1 on the
172,500-drawing test set). Without the v2 checkpoints it falls back to the v1 CNN.

`"partial": true` (optional) marks a live guess sent while the player is still drawing. The reply
is the same, plus `"partial": true` (errors too), so the game never takes a live guess for the
final answer.

```json
{"type": "classify", "strokes": [[[10, 40, 80], [12, 60, 15]]]}
```
```json
{"type": "classify_result", "top_k": [{"label": "bicycle", "prob": 0.82}, {"label": "car", "prob": 0.07}]}
```
```json
{"type": "classify", "strokes": [[[10, 40, 80], [12, 60, 15]]], "partial": true}
```
```json
{"type": "classify_result", "top_k": [{"label": "bicycle", "prob": 0.41}, {"label": "car", "prob": 0.18}], "partial": true}
```

## `draw`

The drawing GPT ([`drawer.py`](../ml/sellanything_ml/sketch/drawer.py)) draws one of the 345
classes stroke by stroke. `text` is what the player typed: matched to a class name exactly, without
an article ("a", "the"), as a plural, or as a close spelling ("bicyle" → bicycle). Empty `text` (or
`"random": true`) draws a random class. `seed` (optional) makes the drawing reproducible.

```json
{"type": "draw", "text": "a bicyle"}
```
```json
{"type": "draw_result", "ok": true, "label": "bicycle", "strokes": [[[12, 40, 80], [130, 60, 132]]]}
```

Strokes are in Quick, Draw!'s simplified form (0–255, the same space `classify` uses). When nothing
matches, the reply says so and offers up to three close names:

```json
{"type": "draw_result", "ok": false, "suggestions": ["onion"], "message": "I don't know how to draw \"unicorn\". Try: onion."}
```

## Selling rounds

The server owns the round (engine + host, [buyer-ai.md](buyer-ai.md)); the game draws `state` and
sends the player's choices. Every `round_state` carries the lines added since the last message
(`new_lines`) and the full `state`.

### `round_start`

```json
{"type": "round_start", "session": "r42", "item": "bicycle", "buyers": 4, "seed": 7}
```
`buyers` (1–5) and `seed` are optional.

```json
{"type": "round_state",
 "new_lines": ["Maya: What shape is it in? Any rust or damage? I'm pretty picky about condition."],
 "state": {
   "item": "bicycle", "label": "buy", "noun": "bicycle", "finished": false,
   "buyers": [{"name": "Maya", "persona": "Maya, who wants things in excellent shape.",
               "offer": 125.0, "present": true, "mood": "talk"}],
   "log": ["…every line so far…"],
   "question": {"asker": "Maya", "attribute": "condition",
                "options": [{"id": "like_new", "answer": "Like new, no scratches."}]},
   "sold": null}}
```

A `refuse` item (`"label": "refuse"`) comes back already `finished`, with every buyer's refusal in
`new_lines` and no offers.

`round_result` carries the final `state` as well, so the winner's `won` mood reaches the game.

`mood` tells the game which sprite animation to play (docs/art.md): `talk` for the buyer asking
the open question, `happy` / `unhappy` for a buyer whose interest went up / down on the last
answer, `left` once they walk away, `won` for the buyer the item was sold to, `idle` otherwise.
On a refused item every buyer is `unhappy`. A reaction outranks `talk` when the next asker has
just reacted.

### `round_answer`

```json
{"type": "round_answer", "session": "r42", "option": "like_new"}
{"type": "round_answer", "session": "r42", "text": "basically new, no scratches"}
```
→ `round_state`. When `state.finished` is true there is no `question`; the player picks a buyer.

A clicked option sends `option`; a typed answer sends `text` (cleaned to one line of printable ASCII,
at most 120 characters). The server maps typed text to one option (docs/buyer-ai.md §10) and adds
`heard` to the reply. `rule` says who read it: `"model"` when the buyer model was trained on the
`parse` task (buyer GPT v6 and later), otherwise the rule matcher's `"yes/no"`, `"number"` or `"words"`:

```json
{"type": "round_state", "heard": {"option": "like_new", "answer": "Like new, barely used.", "rule": "words"}, ...}
{"type": "round_state", "heard": null, ...}
```

`heard: null` means it matched no option: the log gets the player's line and the asker repeating the
question, and the same question stays open. The log always shows what the player typed; the game
shows `heard.answer` so a wrong guess is visible. Empty text is an `error`.

### Buying a cat: `round_start` with `"role": "buyer"`, then `round_buy`

With a model that learned the seller's side too (prompt format 8, `runs/sft_b5`), the player can be one of the
buyers of a cat while the model is the seller and every other buyer (`ml/sellanything_ml/buyers/cat_market.py`;
the game's screen is `game/src/BuyScene.cpp`, reached by "Buy a cat" or `--buy`):

```json
{"type": "round_start", "session": "b7", "item": "cat", "buyers": 3, "seed": 7, "role": "buyer", "name": "Alina"}
```

The reply's `state` adds `asking` (the price tag on the stall), `player`, `waiting` (what the player must do next:
`open`, `ask`, `react`, `reply`, or null once the sale is over) and the seller's `decisions` (per buyer: `offer`,
`action` accept / counter / refuse, `price` for a counter, `says`, `problem` when it breaks the written decisions'
rules, and the buyer's `reply`). The model buyers and the seller act on their own between the player's steps:

```json
{"type": "round_buy", "session": "b7", "step": "open", "text": "Hi! I'd love a calm cat.", "offer": 50}
{"type": "round_buy", "session": "b7", "step": "ask", "text": "Has she seen a vet?"}
{"type": "round_buy", "session": "b7", "step": "react", "text": "Good to know.", "offer": 55}
{"type": "round_buy", "session": "b7", "step": "reply", "text": "Deal.", "accept": true}
```
→ `round_state`. `react` with `offer` 0 walks away. A step that is not the one `waiting` names is an `error`. When
the seller has decided on every final offer `state.finished` is true and `sold` names the buyer with the highest
deal (an accepted offer, or a counter-offer the buyer accepted), with `price`, `asking` and every `deals` amount;
`buyer` is null when there was no deal. There is no `round_sell` in these sales: the seller decides.

### Comparing two buyer models: `round_start` with `"compare"`

While the data is tested (owner, 2026-10-28), one cat sale can be played by two buyer models at once
(`ml/sellanything_ml/buyers/paired.py`; in the game: `--compare b4f9,b5`, `game/src/CompareScene.cpp`):

```json
{"type": "round_start", "session": "c1", "item": "cat", "buyers": 4, "compare": ["b4f9", "b5"]}
```

Each name is a checkpoint, `ml/runs/sft_<name>/best.pt`. Both models' sales have the same buyers (same seed and cards);
every question is put to both (the models take turns writing it, in the asking buyer's name, `question.by`), and the
player's `round_answer` text goes to both. Each model writes its own buyers' openings and reactions and keeps its own
offers. The `state` has `compare`, `seed`, `buyers` (each with `offers` per model), `openings` per model and buyer,
`turns` (question, `by`, the answer, and `reactions` per model and buyer: `says`, `old`, `offer`, `readable`, or
`left_before`), the waiting `question`, and `saved`: after the openings and every answer the sale is written to
`ml/logs/compare/<started>_seed<seed>.json` and `.md`. A comparison is not sold (`round_sell` is an error).

**One model, several ways of picking words** (`ml/sellanything_ml/buyers/variants.py`; in the game:
`--compare b5:greedy,b5:topp,b5:topk,b5:topk2`). Names of the form `<model>:<way>` play one model in each way:
`greedy` (always the likeliest word), `topp` (top-p 0.9), `topk` (top-k 50, the game's way) and `topk2` (top-k 50 on
the seed plus 1), all at temperature 0.7 except greedy; `topk` must be among them. Every buyer's story follows `topk`,
so in every turn each way answers the very same prompt. The `state` has the same fields as above with one column per
way, plus `lead` (the way the stories follow), `labels` (each way's name for the screen) and, in every turn and the
waiting `question`, `ways`: the question each way would have written in that place. Saved to
`ml/logs/compare/<started>_seed<seed>_<model>_ways.json` and `.md`.

### `round_sell`

```json
{"type": "round_sell", "session": "r42", "buyer": "Maya"}
```
```json
{"type": "round_result", "buyer": "Maya", "price": 185.0, "typical": 150.0}
```

### `scene`

How one buyer pictures the item at their place (design-v2 §5.4): the item and up to three props,
each drawn by the drawing GPT and fitted into its own box on a `width` × `height` canvas. The boxes
never overlap. Props are chosen by rules from `ml/data/props.json`, the item's tag profile (using the
answers given so far) and the buyer's trait weights; `reason` says why each prop is there. Each
request draws anew, so a different `seed` gives different drawings of the same props.

```json
{"type": "scene", "session": "r42", "buyer": "Maya", "seed": 1}
```
```json
{"type": "scene_result", "ok": true, "buyer": "Maya", "width": 512, "height": 288,
 "objects": [
   {"label": "dog", "role": "item", "zone": "item", "box": [184, 104, 144, 176], "share": 1.0,
    "reason": "", "strokes": [[[x0, x1, ...], [y0, y1, ...]], ...]},
   {"label": "grass", "role": "prop", "zone": "ground", "box": [8, 160, 168, 120], "share": 1.0,
    "reason": "Outdoorsy: good outdoors", "score": 6.4, "strokes": [...]}
 ]}
```

An unknown session or buyer, a refused item (nobody pictures what they won't buy) or a missing
drawing model is an `error`.

## `chat`

`session` groups the turns of one negotiation. `item` is sent on the first turn.

```json
{"type": "chat", "session": "r42", "item": "bicycle", "text": "I want to sell this bicycle"}
```
```json
{"type": "chat_reply", "text": "How many years have you had it?", "offer": null}
```

On the final turn `offer` is filled in, or `refused` is set:

```json
{"type": "chat_reply", "text": "...", "offer": {"price": 2800, "currency": "TWD"}}
```
```json
{"type": "chat_reply", "text": "Sorry, I can't buy a protected animal.", "offer": null, "refused": true}
```

## `error`

```json
{"type": "error", "message": "unknown message type: 'foo'"}
```
