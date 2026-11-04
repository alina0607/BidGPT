# Prompt format 10: older rounds keep the seller's answers

Notebook 10 item 12 D (owner, 2026-10-28). Code: `ml/sellanything_ml/buyers/format.py` (`ANSWER_FORMATS`),
`ml/sellanything_ml/buyers/sales_sft.py` (`Story`, `_earlier`, `STORY_WORDS`), `ml/sellanything_ml/buyers/cat_market.py`
(`CatMarket._story`). No model has been trained on it yet.

## The problem

A buyer's prompt (format 8, B5) keeps the last `K = 2` rounds in full. Older rounds keep only their question
("Asked earlier: q / q"). `data/written_sales/review_tools/check_facts.py` flags 632 training lines (rule D) that
state a fact the seller gave in a dropped round: breed 378, FIV/FeLV 111, condition 75, age 54, vaccines 7,
microchip 5, spaying 2. The model learns from these lines to state facts that its prompt does not hold.

## The design

Format 10 is format 8 (the same tokens, the same tasks, seller tasks included) with one change. The "Asked earlier" line
keeps each older round's question **with the seller's answer**:

    Asked earlier: What breed is she? -> She's a Siamese, from a breeder. / Has she been tested? -> FIV and FeLV negative.

- The buyer's own old reactions stay out. They are the buyer's lines, not facts, and they cost the most tokens.
- The answers are shortened only as much as the 768-token limit needs. The whole story (opening, older line and
  the last two rounds) may hold `STORY_WORDS = 350` words. The words the rest of the story leaves are shared evenly
  by the older answers. Each answer is cut to its first `share` words, with "..." where it was cut. When the share
  would be under `MIN_ANSWER_WORDS = 1`, the oldest answers go first and their rounds keep the question alone, as
  in format 8. Facts sit near the start of an answer: of the D facts, half are at word 2 or earlier, 95% by word
  13, and the latest is at word 23.
- The rule reads only the story itself, so the game builds exactly the training prompt. `CatMarket` makes its
  stories with `answers=True` for a format-10 model, and both places record the answer with `Story.answered`.
  A long answer typed by a player only makes the older answers shorter.
- Formats 1-9 are unchanged. The full suite passes (300 tests), including new tests in `test_sales_sft.py` and
  `test_cat_market.py`.

## Numbers (B5 sources, 803 sales, 114,712 examples, B5 tokenizer, prompt + target tokens)

| | mean | p50 | p90 | p99 | max | over 768 |
|---|---|---|---|---|---|---|
| format 8 (B5) | 405.6 | 429 | 595 | 698 | 820 | 45 (all seller `decide`/`answer`; buyer 0) |
| format 10, every answer verbatim | 479.3 | 506 | 719 | 940 | 1218 | 6,959 (6.07%) |
| format 10, budget 380 words | 463.5 | 506 | 670 | 720 | 820 | 59 (buyer 14) |
| format 10, budget 370 | 461.4 | 506 | 660 | 712 | 820 | 50 (buyer 5) |
| format 10, budget 360 | 459.0 | 506 | 650 | 705 | 820 | 46 (buyer 1) |
| **format 10, budget 350 (chosen)** | **456.4** | **506** | **641** | **701** | **820** | **45 (buyer 0)** |
| format 10, budget 340 | 453.6 | 506 | 632 | 699 | 820 | 45 (buyer 0) |

The 45 examples over 768 are the seller's `decide`/`answer` prompts (the last 12 public lines). They are the same in
format 8 and format 10, and `llm/sft.py` already skips them. Every buyer example fits.

Shortenings that did worse in the measurements:
- A cap per answer alone: 15 words still left 1,678 examples over 768.
- Keeping only the 3 newest older answers: 997 over 768, and only 208 of the 632 D lines covered.
- A 20- or 15-word cap on top of the 400-word budget: more over 768, and fewer facts kept, than the budget alone.

## D coverage

Each D flag (`check_facts.check`, window 2) is matched to its training example (same seed, buyer, task, and the line
in the target). The flag counts as covered when `check_facts.covered` finds the fact in the example's real prompt.

| | covered | missed |
|---|---|---|
| format 8 | 105 / 632 | 527 |
| format 10, every answer verbatim | 632 / 632 | 0 |
| **format 10, chosen** | **570 / 632** | **62** |

Missed by kind (chosen): FIV/FeLV 33, condition 15, breed 11, vaccines 2, microchip 1, age 0, spaying 0. These are
the longest sales, where the older answers had to be cut short or dropped.

105 lines count as dropped by `check_facts` but are already in format 8's prompt. Its "prompt" text leaves out
things the real prompt has, such as the buyer's own kept reactions. I did not check which part covers them. So
format 8's true count of dropped facts is 527, and format 10 takes it to 62.

## When B6's data is built

1. Build with the new format into a new folder (this writes nothing else; `write_cards` runs only for format 5):
   `python -m sellanything_ml.buyers.sales_sft --prompt-format 10 --sources b5 --out data/processed/buyers_b6`.
   Use a B6 sources tuple once the rewritten specs are rebuilt.
2. Measure again on the new sources: `measure_format10.py` in the shared notes folder (it takes
   `[STORY_WORDS, ANSWER_WORDS, MIN_ANSWER_WORDS]` lists). Keep `STORY_WORDS` at the largest value with no buyer
   example over 768. If it changes, retrain; the game must use the same value as the training data.
3. Run `check_facts.py --window 0` to look at the remaining L lines, and the measuring script for the D lines that
   are still missed. The remaining D lines can then be fixed in the data, or the budget raised if the context grows.
4. `tools/self_play.py` still requires `prompt_format == 8`. Allow 10 there (`in SELLER_FORMATS`) before self-play
   with a format-10 model. That file was outside this task.
5. A format-10 model must be judged on held-out sales against B5, like every change (the ground rules in docs/plan.md).
