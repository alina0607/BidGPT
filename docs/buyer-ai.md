# Buyer AI

How the buyers in a round think, talk and bid. This is the contract between the game, the
price engine, the training data and the buyer model. The engine in
[`ml/sellanything_ml/buyers/`](../ml/sellanything_ml/buyers) implements every rule here exactly.

## 1. Who does what

| Part | Kind | Responsibility |
|---|---|---|
| Sketch classifier | CNN | Drawing → top-3 Quick, Draw! classes |
| Player | Human | Confirms the item; answers questions; picks a buyer at the end |
| **Host** | Rules | Opens and closes the round; decides which buyer asks about which attribute next |
| **Price engine** | Rules | Turns answers into each buyer's valuation and offer; runs the bidding |
| **Buyer model** | Our GPT | Speaks for one buyer at a time: asks, reacts, refuses; judges how much an answer matters *to that buyer* |

The model never does arithmetic. It outputs words plus one small integer, the **interest change**
Δ ∈ {−2, −1, 0, +1, +2}. Everything with a dollar sign is the engine's job.

The engine can also compute Δ itself from the item schema. That is how training labels are made
(§7). At play time the model's Δ is used, which is what lets buyers handle free-text answers and
items that have no schema.

## 2. Data files

### 2.1 `ml/data/items.tsv`

Every Quick, Draw! class with `label` = `buy` or `refuse`. A `refuse` item ends the round at once:
every buyer declines, in character.

### 2.2 `ml/data/traits.json`

**Tags** are the concerns a buyer can have. There are eight, and every attribute in every item
is described in terms of them:

| Tag | Meaning |
|---|---|
| `condition` | wear, damage, freshness, whether it works |
| `space` | how much room it takes up |
| `low_maintenance` | how little care, training or upkeep it needs |
| `family` | safe and suitable for a household with kids |
| `status` | brand, rarity, pedigree, prestige |
| `performance` | how well it does its job |
| `style` | looks, materials, design |
| `active` | suits an active, outdoor lifestyle |

**Traits** are buyer archetypes. Each has a one-line `persona` and `weights` over tags (0–2), and
optionally a `budget` level. Traits are item-independent: an `apartment_dweller` cares about
`space` whether the item is a dog (size) or a bicycle (frame size).

**Budgets** are multipliers on the item's typical price: `tight` 0.6–0.95, `normal` 0.95–1.5,
`generous` 1.5–2.5.

### 2.3 `ml/data/item_schema.json`

```json
"dog": {
  "noun": "dog",
  "price": {"low": 20, "typical": 200, "high": 2500},
  "attributes": {
    "potty": {
      "question": "Is the dog potty trained?",
      "options": [
        {"id": "yes", "answer": "Yes, fully potty trained.", "factor": 1.15, "signals": {"low_maintenance": 2}},
        {"id": "no",  "answer": "Not yet.",                  "factor": 0.8,  "signals": {"low_maintenance": -2, "condition": -1}}
      ]
    }
  }
}
```

| Field | Meaning |
|---|---|
| `price.typical` | What an average buyer pays for an average example |
| `price.low` / `high` | Hard floor / ceiling for any valuation |
| `question` | Canonical wording; the model paraphrases it |
| `options[].answer` | Canonical answer; used as a button label before free-text input exists |
| `options[].factor` | **Objective** effect on value, the same for every buyer (a cracked screen hurts everyone) |
| `options[].signals` | **Subjective** effect per tag, −2…+2 (a big dog is bad for `space`, good for `active`) |

An item has 3–5 attributes; an attribute has 2–4 options.

## 3. A buyer

```
Buyer = name, traits (1–2), weights, budget_level, budget ($),
        eagerness ∈ [0.75, 0.9], interest (integer, starts 0), present (bool)
```

- `weights[tag]` = sum of the buyer's trait weights for that tag, capped at 2.
- `budget` = typical × U(budget range of `budget_level`). The level comes from a trait that sets
  one, otherwise it is drawn from `default_budget_odds`.
- `eagerness` is how low the buyer opens, as a fraction of what they would really pay.

A round has 3–5 buyers with distinct names. Traits are drawn without replacement where possible,
so buyers differ.

## 4. Valuation and offers

For buyer *i* after a set *A* of answered attributes:

```
objective   = Π factor(answer)                     over answered attributes
valuation_i = clamp( typical × objective × (1 + 0.12 × interest_i),  low, high )
willing_i   = min(valuation_i, budget_i)            ← the most buyer i will ever pay
progress    = |A| / number_of_attributes
base_offer_i = willing_i × (eagerness_i + (1 − eagerness_i) × progress)
offer_i      = min(willing_i, max(base_offer_i, standing_bid_i))     ← standing bid: §5
```

So a buyer opens at `eagerness × willing` and moves toward `willing` as they learn more. Interest is
clamped to [−5, +5].

**Interest change from one answer** (this is the label the model learns):

```
raw_i = Σ_tag weights_i[tag] × signals(answer)[tag]
Δ_i   = clamp(raw_i, −2, +2)
```

**Walking away.** A buyer whose interest reaches −4 leaves the round (`present = false`); their
offer is withdrawn.

**Rounding** (applied only when a price is shown or stored):

| Price | Rounded to |
|---|---|
| < $10 | $0.25 |
| < $100 | $1 |
| < $1,000 | $5 |
| ≥ $1,000 | $25 |

## 5. Bidding

After every answer, once all offers are recomputed:

1. Let `best` be the highest offer among present buyers and `step` = 5% of typical.
2. Every other present buyer with interest ≥ 1 and `willing_i ≥ best + step` raises to `best + step`
   (each buyer raises at most once per answer). Raises are processed from highest interest down,
   and `best` is updated after each raise.
3. A raise is a **standing bid**: later recomputations keep it, so the full rule is
   `offer_i = min(willing_i, max(base_offer_i, standing_bid_i))`. Bidding never lowers an offer;
   an answer can, by lowering `willing_i` below the standing bid.

## 6. Host

A round has as many questions as the item has attributes (max 5). For each question:

1. **Concern** of buyer *i* for attribute *a* = Σ over tags *t* that appear in any of *a*'s option
   signals of `weights_i[t]`.
2. Pick the (buyer, unasked attribute) pair with the highest concern. Ties: the buyer who has
   asked fewest questions, then the earlier buyer, then the earlier attribute in the schema. The buyer who asked last
   is skipped if any other buyer has a concern > 0.
3. If every concern is 0, the buyer who has asked fewest questions asks the first unasked attribute.
4. The player answers. **Every present buyer** updates interest and offer, not only the asker.
5. **Reactions**: buyers with |Δ| ≥ 1 may speak, at most two per answer, largest |Δ| first;
   the asker always reacts. Others update silently. When Δ = 0 but the answer's factor is
   ≥ 1.2 or ≤ 0.8, the reaction comments on value ("that makes it worth more") instead of taste,
   because every buyer's price just moved.

The round ends when all attributes are asked or every buyer has left. The player then sells to
one present buyer at that buyer's offer, or keeps the item.

## 7. Model tasks and prompt format

One model, four tasks. Special tokens are single vocabulary entries.

| Token | Meaning |
|---|---|
| `<\|item\|>` | Item noun follows |
| `<\|attrs\|>` | Comma-separated attribute names follow (omitted when the item has no schema) |
| `<\|persona\|>` | Buyer card follows |
| `<\|log\|>` | Public transcript follows, one line per utterance |
| `<\|ask\|>` / `<\|react\|>` / `<\|refuse\|>` | Task to perform |
| `<\|interest\|>` | Δ follows, written `-2`…`+2` |
| `<\|end\|>` | End of a field / of the model's turn |

**ask** — produce one question about one attribute, in this buyer's voice.

```
<|item|>dog<|end|><|attrs|>age, breed, potty, temperament, size<|end|>
<|persona|>Maya, who lives in a small apartment.<|end|>
<|log|>Leo: How old is the dog?
You: Two years old.<|end|>
<|ask|>size<|end|>
Maya: How big is he going to get? My place is pretty small.<|end|>
```

**react** — given the latest answer, speak (optional) and output Δ.

```
… <|log|>…
Sam: Is the dog potty trained?
You: Yes, fully potty trained.<|end|>
<|react|>
Maya: Oh good, that matters a lot in an apartment.<|interest|>+2<|end|>
```

**refuse** — the item is in the `refuse` list.

```
<|item|>tiger<|end|><|persona|>Leo, who spends weekends hiking.<|end|><|refuse|>
Leo: A tiger? I'm not buying a wild animal, that's not even legal.<|end|>
```

The fourth task, **parse** (free text → option id), is added when free-text input exists in the
game; until then the player picks option buttons.

### Prompt format versions

The examples above are format 1. Every checkpoint records the format its data used
(`prompt_format`), and `ModelVoice` builds prompts to match, so older models keep working.

| Format | Data | `<|ask|>` carries | The model writes the item as | Why it changed |
|---|---|---|---|---|
| 1 | v1, v2 | the attribute: `size` | its name | — |
| 2 | v3 | the attribute and the engine's question: `size: How big is the dog?` | its name | On items it never saw, the model can't know the schema's wording; only 41% of its held-out questions were on topic (D18) |
| 3 | v4 | the same, with the name replaced: `size: How big is the {item}?` | `{item}`, filled in by the engine (`format.fill_slot`, which also fixes *a*/*an* and capitals) | With the question given, 77% were on topic; most of the rest garbled the name (*saxophone* → *saxmalone*) (D19) |

The log keeps real names in every format: it is what the player sees on screen.

## 8. Training data

Generated by the engine (§3–6) so every label is exact:

1. Sample an item (from the training split), 3–5 buyers, and hidden true answers.
2. Run the host and engine for a full round, recording every ask, answer and reaction with Δ.
3. Turn each event into one training example in the format of §7.
4. Fill the text with phrase pools (v1) or local-LLM paraphrases (v2). The text never changes a
   label: Δ and the chosen attribute come only from the engine.

Stored as JSONL, one example per line, plus the full scene for debugging.

**Held-out items.** About 15% of `buy` items never appear in training. They measure whether the
model generalizes to items it has never seen.

## 9. Evaluation

| Metric | How |
|---|---|
| Format validity | Output parses: speaker prefix, `<\|interest\|>` with a value in −2…+2, `<\|end\|>` |
| Δ accuracy | Exact match and mean absolute error vs. engine label, on seen and held-out items |
| Δ sign accuracy | Did the buyer get happier/unhappier in the right direction |
| Attribute adherence | An `ask` example mentions the requested attribute (keyword check) |
| Repetition | Same question asked twice in a round |
| Persona consistency | Δ agrees with the persona's weights (follows from Δ accuracy) |

## 10. Typed answers

**Who reads them.** A buyer model trained on the `parse` task (design-v2 §6.3, prompt format 4) reads
typed answers when its checkpoint says `parse_ready`; `llm/export.py` sets that only if the model beat
the rules below on both test splits (more answers right, no more wrong options). Buyer GPT v6b did not
(test_seen 69.76% right but 27.26% wrong option, against the rules' 67.78% / 8.76%; devlog 2026-10-07), so
the game uses these rules.


The player can type an answer instead of clicking one (`round_answer` with `text`, docs/protocol.md).
Buyers only understand options, so `buyers/answers.py` maps the text to exactly one option, or to
none, and then the asker asks again. It is rules on the option texts, no model, so every pick can be
explained; three rules, each narrowing the choice for the next:

| Rule | Example |
|---|---|
| Yes or no: the first word ("yeah", "nope", "not") keeps the options that say the same | "yep" → "Yes, never misses." |
| Numbers: when 2+ options carry a quantity of one kind (an age, kilos, people, a decade), the nearest | "shes like 12" → "She's eleven." |
| Words: shared words weighted by how few options use them; a tie is "can't tell" | "used, good shape" → "Used, but in good shape." |

The buyer model is shown the matched option's text, not what was typed (`RoundSession.heard`): it was
trained on option texts. The conversation on screen shows what was typed.

Measured on hand-written answers (`ml/tests/fixtures/typed_answers_*.tsv`, written by the developer
who also wrote the rules, which flatters it; real players will be harder):

| Set | Attributes | First run, before any change | Wrong option picked |
|---|---|---|---|
| dev (rules tuned on it) | 15 | 80/81 | 0 |
| test, frozen before the matcher existed | 30, sampled at random | **97/99 (98.0%)** | 0 |
| test2, frozen after tuning on test | 20, sampled at random | **59/61 (96.7%)** | 1 |

Every option's own text, typed back: 2,720/2,721 map to themselves, 0 to another option.
A miss that says "can't tell" costs the player one more try; a wrong pick is worse, and the game shows
what the answer was taken as so the player sees it. The known wrong pick: "bike helmet" → "A kids'
bike helmet." (the bike option's text is just "Cycling."). Not handled: relative time ("bought it last
year"), and answers that only make sense with world knowledge ("works fine" when one option is
"Works perfectly." and another "One of the sockets doesn't work").

