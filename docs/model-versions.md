# Model versions

Every trained version, with what it was trained on, how, and how well it did, so each one can be run again and
compared when the work is committed (owner, 2026-10-12). Files that a version needs are kept read-only (`chmod a-w`);
hashes are the first 16 hex digits of SHA-256. All numbers come from runs; none is estimated.

⚠️ The code these versions were made with is not committed yet (commits wait for the owner's review). The
"code" line of each version lists the files it depends on, so the commits can be grouped by version.

---

## Buyer GPT (B plan: each buyer is one prompt; cat sales)

Common to all versions: tokenizer `ml/checkpoints/tokenizer/sp8192.model` (c0abef10af2e2f76); start from
`ml/runs/pretrain_large/final.pt` (0f8e501b446e2804; 71.0 M parameters, d768 × 9 layers, pre-trained on 1.28 B
FineWeb-Edu tokens in 512-token blocks); torch 2.14.0 on an Apple Silicon Mac (MPS).

Sales (inputs of every version so far):

| file | sales | sha256 |
|---|---|---|
| `ml/data/processed/buyer_sales_v9_official.jsonl` | 48 (Qwen v9, every line read and corrected by Alina Chuang) | 16b9d429130d1872 |
| `data/written_sales/sales.jsonl` | 52 (written by Alina Chuang, seeds 2048–2099) | c5870d2559bc5cf3 |
| `data/written_sales/sales_b2_150.jsonl` | 150 (written by Alina Chuang, seeds 2200–2349; B2 and B2f6) | 78084aa4366674f7 |

### B1 (2026-10-12) — in the game until 2026-10-13 (since then the game loads B3, `ml/runs/sft_b3_resume/best.pt`; B3 and B4 have no section here yet)

- **Data**: `ml/data/processed/buyers_b1/` — prompt format 4 (the card shows a budget), K = 2 full rounds plus the
  earlier questions; 90 sales train / 10 val (by seed hash); train 9,956 examples (900 open, 824 ask, 7,470 react,
  762 imagine), val 1,195. train.jsonl a6d9ec1fb5dec112, val.jsonl 42fa95f461a50442. Rebuilt byte for byte by
  `PYTHONPATH=. python -m sellanything_ml.buyers.sales_sft --prompt-format 4 --out <dir>`.
- **Training**:
  ```
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 PYTHONPATH=. \
  python -m sellanything_ml.llm.sft --init runs/pretrain_large/final.pt --data data/processed/buyers_b1 \
      --epochs 3 --batch 8 --lr 3e-4 --eval-every 400 --run-dir runs/sft_b1
  ```
  3,733 steps, about 60 min, 0.83–1.07 s a step; per-token loss on the target only; cosine lr with 100 warm-up steps.
- **Weights**: `ml/runs/sft_b1/best.pt` (6c4517b868ccb2b9, step 3,733, val loss 0.906), `last.pt`
  (347b421bc62fae23, with optimizer state). Logs: `train.log`, `train_attempt1_watchdog_stop.log`, `metrics.jsonl`.
- **Validation loss**: 1.236 (step 400) → 0.906 (step 3,733), still falling; train loss about 0.41 at the end.
- **Held-out evaluation** (`ml/tools/eval_buyer_b1.py` as it was on 2026-10-12 03:57, with budget checks;
  `runs/sft_b1/eval_b1.json`, samples in `eval_b1_samples.md`; sampling temperature 0.7, top-k 50):

  | task | readable | rules | other |
  |---|---|---|---|
  | open (100) | 100% | within budget 100% | |
  | ask (98) | 100% | not an exact repeat 100% | |
  | react (908) | 99.6% | 89.7% | same direction as the sale's buyer 78.8%; offer off by $2.52; 14 withdrawals (sale 11) |
  | imagine (89) | 100% | 70.8% | |

  Before SFT (`pretrain_large` alone, 30 per task, `eval_before_sft.json`): open 0% readable, react 0%, imagine 3%.
  Current evaluator (no budget; `runs/sft_b1/eval_v2_no_budget.json`, run 2026-10-12): open readable 100%; ask new in
  meaning **89.8%** (the sales: 96.9%); react rules 90.2%, same direction 78.8%, said amount = offer 99.9%, reasonable
  move 99.9%; imagine rules 70.8%.
- **Code**: `ml/sellanything_ml/buyers/sales_sft.py`, `buyers/format.py` (`<|open|>`, `<|imagine|>`),
  `llm/sft.py` (format 4, slowdown watchdog, MPS cache, val batch 16), `buyers/cat_market.py`, `server.py`,
  `tools/eval_buyer_b1.py`, tests `test_sales_sft.py`, `test_cat_market.py`.
- **Known problems** (devlog 2026-10-12 §6): forgets facts older than two rounds; stitches phrases from the
  training sales instead of reading the answer (including reading "never lived with dogs" as "fine with the dog");
  raises above its own budget; asks again about what was answered; 30% of scenes break the rules.

### B2 (2026-10-12) — trained, not in the game

- **Data**: `ml/data/processed/buyers_b2/` — prompt format 5: the card has no budget (owner, 2026-10-12: buyers have
  no spending limit, their moves only have to be reasonable); everything else as B1 (K = 2). Sources: the 100 sales
  of B1 plus the 150 sales of seeds 2200–2349, kept as they were at 16:01 in `data/written_sales/sales_b2_150.jsonl`
  (78084aa4366674f7; `sales_b2.jsonl` has grown since, and its first 150 sales are still identical to this snapshot).
  223 sales train / 27 val (by seed hash: B1's 10 val sales plus 17
  new ones); train 25,836 examples (2,230 open, 2,106 ask, 19,513 react, 1,987 imagine), val 3,193.
  train.jsonl 6043ee1a47a79cc6, val.jsonl 16f7326c10afd42a (both written 2026-10-12 16:01, a minute before the
  upload). Rebuilt byte for byte (2026-10-13) by `PYTHONPATH=. python -m sellanything_ml.buyers.sales_sft`, whose
  sources now point to the snapshot.
- **Training**: one Modal A100-40GB, `cd ml && modal run tools/modal_sft_b2.py` — same recipe as B1 (`--init
  runs/pretrain_large/final.pt --epochs 3 --lr 3e-4`) with `--batch 16 --eval-every 500 --max-minutes 25`; 4,844
  steps at about 0.09 s a step. Log `runs/sft_b2/train.log`, Modal log `runs/sft_b2_modal.log`.
- **Weights**: `ml/runs/sft_b2/best.pt` (a8df01c605a09522, step 3,000, val loss 1.316). No `last.pt` was copied back.
- **Validation loss**: 1.687 (step 500) → 1.316 (step 3,000) → 1.343 (3,500) → 1.322 (4,844): no gain after about
  1.9 epochs. Not comparable with B1's 0.906 (a different, harder validation set).
- **Held-out evaluation** (`ml/tools/eval_buyer_b1.py`, no budget; temperature 0.7, top-k 50, a fixed seed per
  prompt; run on the Mac on 2026-10-12, reports and samples in `ml/runs/sft_b2/eval/`). Both models get the same
  prompts: the old set `runs/eval_sets_b2/val_old_format5.jsonl` (3e52ea5848e29038, B1's 10 val sales) and the new
  set `val_new_format5.jsonl` (c319c04a9beb8736, 17 new held-out sales). B1's old-set numbers are
  `runs/sft_b1/eval_v2_no_budget.json`.

  | | old set B1 | old set B2 | new set B1 | new set B2 |
  |---|---|---|---|---|
  | open readable | 100% | 100% | 97.6% | 98.2% |
  | ask new in meaning | 89.8% | 90.8% | 96.9% | 98.2% |
  | react passes the rules | 90.2% | 90.4% | 86.7% | **92.3%** |
  | react same direction as the sale | 78.8% | 77.5% | 67.4% | **74.2%** |
  | react offer off by | $2.52 | $2.58 | $4.62 | **$2.93** |
  | react walk-aways (sale) | 14 (11) | 10 (11) | 39 (18) | 14 (18) |
  | imagine passes the rules | 70.8% | **38.2%** | 59.9% | **11.9%** |

  Side by side, line by line: `ml/notebooks/06_buyer_gpt_b2.ipynb` (and `runs/sft_b2/eval/compare_val_*_b1_b2.md`).
- **Code**: as B1, plus `tools/modal_sft_b2.py`.
- **Known problems**: scenes break the rules far more often than B1's (things picked but never used, broken
  sentences); both models misread "fully vaccinated" as "no vaccinations" on the old set; not yet played as a live
  sale.

### B2f6 (2026-10-12/13) — trained, not in the game

B2 again on exactly the same sales, in prompt format 6, to test two of notebook 6's proposed changes: one token per
drawable thing, and the buyer's persona and the kind of animal as tokens.

- **Data**: `ml/data/processed/buyers_b2_f6/` — prompt format 6: as format 5, but the item is `<|cat|>`, the persona on
  the card is one token (`Chloe, <|buyer:is a student on a tight budget|> | cares about: …`) and a scene's list is three
  thing tokens (`<|thing:laptop|><|thing:book|><|thing:coffee cup|> | line`). 363 new tokens (2 animals, 17 personas,
  344 things) after the 14 older added tokens; vocabulary 8,569; no older id moves. Same 250 sales and split as B2
  (sources as B2, with `sales_b2_150.jsonl`): train 25,836 examples, val 3,193. train.jsonl 71225ea1eb715d38, val.jsonl
  ab063c6923c719db; rebuilt byte for byte (2026-10-13) by
  `PYTHONPATH=. python -m sellanything_ml.buyers.sales_sft --prompt-format 6 --out <dir>`.
- **Training** (this Mac's GPU, 2026-10-12 21:45 → 2026-10-13 00:29):
  ```
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 PYTHONPATH=. \
  python -m sellanything_ml.llm.sft --init runs/pretrain_large/final.pt --data data/processed/buyers_b2_f6 \
      --epochs 3 --batch 16 --lr 3e-4 --eval-every 500 --run-dir runs/sft_b2f6
  ```
  4,844 steps, 2.0 s a step on average. Each new token's embedding (and, tied, its output row) starts as the mean of
  the rows of its words' tokens (`llm/model.py start_rows_from_words`). A first attempt with watermarks 0.6/0.5 ran out
  of MPS memory at once (`train_attempt1_oom.log`).
- **Weights**: `ml/runs/sft_b2f6/best.pt` (0d65462db2b31dc5, step 3,000, val loss 1.334), `last.pt`
  (69ba4556d5eb6d2f, step 4,844, with optimizer state). Logs: `train.log`, `metrics.jsonl`.
- **Validation loss**: 1.710 (500) → 1.570 → 1.475 → 1.444 → 1.394 → 1.334 (3,000) → 1.363 → 1.357 → 1.345 → 1.342
  (4,844); train loss about 0.6 at the end. Not comparable with B2's 1.316: a thing is now one token, so the scene
  targets have fewer tokens.
- **Held-out evaluation** (same tool, sampling and prompts as B2, written in format 6:
  `runs/eval_sets_b2f6/val_old_format6.jsonl` 386d33dc98981e43, `val_new_format6.jsonl` 55724b6ffc720ffc; all 3,193
  prompts and answers equal B2's once the tokens are written back as words). Reports and samples in
  `ml/runs/sft_b2f6/eval/` (run by `eval/after_training.sh` on the Mac's GPU right after training).

  | | old B1 | old B2 | old B2f6 | new B1 | new B2 | new B2f6 |
  |---|---|---|---|---|---|---|
  | open readable | 100% | 100% | 100% | 97.6% | 98.2% | 100% |
  | ask new in meaning | 89.8% | 90.8% | 92.9% | 96.9% | 98.2% | 96.3% |
  | react passes the rules | 90.2% | 90.4% | 90.0% | 86.7% | 92.3% | 93.1% |
  | react same direction as the sale | 78.8% | 77.5% | 76.1% | 67.4% | 74.2% | 72.8% |
  | react offer off by | $2.52 | $2.58 | $2.54 | $4.62 | $2.93 | $3.38 |
  | react walk-aways (sale) | 14 (11) | 10 (11) | 14 (11) | 39 (18) | 14 (18) | 22 (18) |
  | imagine passes the rules | 70.8% | 38.2% | 50.6% | 59.9% | 11.9% | 14.5% |

- **Experiments** (CPU, `ml/runs/sft_b2f6/`, read in notebook 6 §10): scenes picking their own things pass 27.4% (B2
  27.8%); 13% of B2f6's lists are not three thing tokens; given the written things, a thing seen 1–9 times in training
  is used 6% of the time (B2 24%), and its word gets a median probability of 0.012 when listed (B2 0.183). Persona and
  misread results: notebook 6 §10.3–10.4.
- **Code**: `buyers/format.py` (format-6 tokens), `buyers/sales_sft.py` (`--prompt-format 6`, `SOURCES_B2`),
  `buyers/cat_market.py` (reads thing tokens), `llm/model.py` (`start_rows_from_words`), `llm/sft.py` (vocabulary
  growth for format 6), tests `test_sales_sft.py`, `test_cat_market.py`; experiments `tools/probe_imagine.py`,
  `probe_scene_list.py` (`--sets`), `probe_list_tokens.py`, `probe_persona_swap.py`, `probe_misread.py`.
- **Known problems**: rare things are worse than in B2; lists of other words between thing tokens; not played as a
  live sale.

---

## Brush GPT (cat painting, `ml/sellanything_ml/sketch/brushgpt.py`)

**In the game since 2026-10-27 (the cat shelter, `ml/sellanything_ml/sketch/shelter.py`):** Brush GPT L8
(`brushgpt_l.py`, `ml/runs/brushgpt/L8/best.pt`) paints the face, then the eye model E5 (`eyegpt_l8.py`,
`ml/runs/brushgpt/E5/best.pt`, brushes in `brushes.json` from `ml/tools/eye_brushes.py`) paints the eyes from what L8
produced. Both at temperature 1, every cat sampled once, nothing picked. On the Mac's CPU a cat takes about 75–90 s
(the face about 1,850–2,000 strokes, the eyes 192). Before that the shelter used run I below.

All on one Modal A100 (`ml/tools/modal_brushgpt.py --runs <name>`), d512 × 7 layers, 8 heads, 27,885,568
parameters, batch 8 × accumulation 2, lr 3e-4, early stop on validation loss; 256 samples evaluated per run. Each run
is tied in `DATA_OF` to the frozen token file it used:

| token file (read-only) | contents | sha256 |
|---|---|---|
| `tokens_trial_1792.npz` | trial: 1,639 train cats | 00457d1caaf486d7 |
| `tokens_run_D.npz` | 4,085 AFHQ cats (train 3,413 / val 292 / test 380) | eef2c0ecdb6cdaf3 |
| `tokens_runs_DEFG.npz` | the same tokens plus face points (= `tokens.npz` on 2026-10-12) | 61ea76e9ca4d8ce6 |
| `tokens_runs_HI.npz` | the same cats aligned by the eyes (`brushgpt.py align`) | 45486e9dc86ed364 |

| run | data | flags beyond D | best val | eye-stroke loss | FID | eye judge (gen ÷ real) | best.pt |
|---|---|---|---|---|---|---|---|
| C | trial | (overlap order, per-field heads) | 4.0417 | — | — | — | 674f4821ad20572d |
| D | DEFG | — | 3.9331 | 4.036 | 209 | 0.420 / 0.501 = 0.84 | c42c6aa9f2cd4761 |
| E | DEFG | `--points 1` | 3.9166 | 3.979 | 209 | 0.438 / 0.501 = 0.87 | 542e57c7607a31dd |
| F | DEFG | `--eye-weight 3` | 3.9609 | 3.990 | 232 | 0.434 / 0.501 = 0.87 | 4120cf923fdff45b |
| G | DEFG | both | 3.9471 | 3.932 | 229 | 0.451 / 0.501 = 0.90 | 59dd0aeb357b387b |
| H | HI | aligned | 3.886* | 3.937 | 194 | 0.479 / 0.537 = 0.89 | 5eef3372f296172b |
| I | HI | aligned + points + eye weight 3 | 3.9152* | 3.869 | 182 | 0.488 / 0.537 = 0.91 | 6f783ff675b632a3 |

\* validation on aligned tokens: not the same task as D–G. FID is against the run's own training renders (validation
renders score 39.5 for D–G, 41.8 for H–I). Eye judge: `ml/tools/eye_judge.py --run <name> --n 128` (H, I with
`--data` pointing to the aligned tokens), ± about 0.006.

Next data: CAT dataset paintings (`ml/data/processed/cat_brush_face/`, painting since 2026-10-12 03:58). When they are
tokenized, `tokens.npz` changes; the frozen files above keep C–I reproducible.

Run J (2026-10-16, on the Mac, `ml/tools/run_brushgpt_J_mac.sh`, log `runs/brushgpt/J_driver.log`): run I's flags on
AFHQ + CAT-dataset cats. Data: `tokens_J.npz` (sha256 4fe5536b3e7be2ed; 10,933 cats: train 3,413 + 6,327, val 292 + 521,
test 380 + 0, after leaving out 135 of the 6,983 painted CAT cats for badly painted eyes, `brushgpt.CAT_EYE_*`), aligned
with H/I's eye midpoint and distance (`--align-ref tokens_runs_HI.npz`) and the stroke order worked out again on the
aligned faces (`--reorder-after 1`) → `tokens_J_aligned_reordered.npz` (sha256 0e064371cee2644d). Why re-order: the CAT
dataset has upside-down and sideways cats (906 turned more than 20°, 325 more than 45°); aligned without re-ordering
(`tokens_J_aligned.npz`, 2fae31d37fbe08b0, kept but unused) their bands run with the turned face, and 92 cats were
painted bottom to top (last-stage order vs y: CAT 5th percentile 0.32 → 0.50 after re-ordering, AFHQ 0.56; no row
below 0). The AFHQ strokes are H/I's exactly (checked), in the re-worked order. Run I on the two orders: AFHQ val 3.9152
(own order) vs 3.9202 (re-ordered), CAT val 3.9491 vs 3.9407: re-ordering helps the CAT rows even for a model that
never saw them. No AFHQ val/test cat is a copy of a CAT training cat (32 × 32 face crops, mirror too: best correlation
0.93, the closest five pairs looked at: different cats).

J trains on the Mac (MPS), not on an A100: batch 1 × accumulation 16 (I's 8 × 2 ran out of MPS memory: 18.8 GiB at
batch 8, over 20 GiB at batch 4), the same 16 rows a step; max 8,000 steps, evaluation every 100, patience 8 as I;
about 14 s a step. `brushgpt.py train --resume 1` continues from `last.pt`, written at every evaluation (checked
2026-10-16: a run killed with `kill -9` after step 20 and resumed gave the same train and validation losses at steps
30 and 40, to 4 decimals, as an uninterrupted run); weights kept every 1,000 steps. Earlier starts, kept for the
record: 02:35 out of memory at step 0; 13:22 stopped after a few minutes to settle the setup; 14:03 stopped after step
100 (val 4.7283, `runs/brushgpt/J_attempt_before_reorder/`) to re-order. J started again at 14:47. Scored by
`tools/brushgpt_by_source.py` (each model on its own file: AFHQ val, AFHQ test, CAT val) and by the eye judge at
temperatures 1.0, 0.8, 0.6. Results to be filled in from the run.
