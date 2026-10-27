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
