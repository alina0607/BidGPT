# Quality and the debugging loop

## The loop

```bash
python scripts/check.py --quick           # after every edit: lint, unit, build, engine (~10 s)
python scripts/check.py                   # before a commit: + e2e, data, model, quality (~2 min)
python scripts/check.py --require-all     # a skipped stage counts as a failure
python scripts/check.py --update-baseline # re-record the CUPID and model baselines, after review
```

| Stage | What runs | A failure here (with the stages above green) means |
|---|---|---|
| `lint` | `ruff` on `ml/` and `scripts/` | style or an obvious bug. **Stops the run** |
| `unit` | `pytest`: protocol, buyer engine, data generation and freshness, augmentation, model, rasterizer, voice, and the hand-labeled on-topic set | Python logic |
| `build` | `cmake --build build` | C++ doesn't compile. **Stops the run** |
| `engine` | `ctest`: text layout, sprite sheets and animation, texture loading, null-backend smoke, network client against fake servers | engine basics, or the client's handling of a server that answers / goes quiet |
| `dgl` | the DGL backend compiled against `compat/` (macOS/Linux) | the Windows-only backend no longer compiles |
| `e2e` | `scripts/e2e_autoplay.py`: the real game, headless, loads its art and plays a whole round against the real server | the seam between game and server: protocol, scene flow, sessions, or missing art |
| `data` | re-simulates the first item of the default dataset with today's code | the dataset was written by older engine or prompt code |
| `model` | the installed buyer model on a fixed sample (greedy, seed 0, CPU: deterministic) | a metric dropped below `docs/quality/model-baseline.json` |
| `quality` | CUPID static analysis against `docs/quality/cupid-baseline.jsonl` | this change introduced a warning (resource leak, complexity, …) |

Each stage narrows where a bug can be: when `e2e` fails but `unit` passes, the Python rules are
fine and the problem is in how the game and the server talk to each other.

### How the gate behaves, and why

The rules come from two loops that were run for months on another codebase (a gate script and a
static-analysis verification loop); each one matches something that went wrong here first.

| Rule | What happened here before it |
|---|---|
| Only `lint` and `build` stop the run; every other failure is listed at the end | The first version stopped at the first failure, so a red `unit` hid whether `e2e` and `quality` would pass |
| A stage that can't run reports **SKIP**, never PASS, and the last line names it | Without CUPID, `quality` printed PASS: in CI it was always green and never looked at anything |
| A measurement that disagrees with the rule it checks is a hypothesis, not a result | The evaluation's "on topic" and the in-game guardrail's were two different rules; v2 was reported at 62% when the shared rule gives 41% |
| Rules are checked against labels they didn't produce, including look-alikes that must be rejected | `tests/fixtures/on_topic_gold.jsonl`: 24 real model lines read one by one. The rule from the morning gets 12 of them wrong; a test keeps it that way, so the set can't quietly lose its hard cases |
| A suite must be able to fail | The gold set has to contain both labels; `e2e` was shown to fail on a refused item, a missing server and a bad port; the CUPID gate on a planted leak; the timeout test with the timeout removed |
| Freshness is checked by behavior, not by file hashes | The v1 dataset was written by the morning's engine and nothing noticed until a regeneration didn't match. A hash of the sources would also fire on an edited comment; re-simulating the first item fires only when the data would change, and skips rows `augment.py` renamed on purpose |
| Numbers that change belong to a command, not to prose | The README's test count was edited by hand three times in one day (58 → 64 → 68); it is no longer written down |

### The end-to-end test

`e2e_autoplay.py` starts the server on a free port with `--fixed-sketch bicycle --phrases-only`
(no trained models, no torch — it runs in CI) and runs the game with
`--backend null --autoplay`: draw → confirm → answer every question → sell. It passes only if the
game prints a sale and exits 0. On failure it prints the game's transcript and the server's log.

Autoplay was made into a real test for this:

| Before | After |
|---|---|
| a server error left autoplay waiting until the frame limit, then exiting **0** | it prints the error and exits at once (`ReplyError`) |
| the run "passed" whether or not anything was sold | exit 0 only after a sale; 3 otherwise; 2 for bad arguments |
| the null backend ran 120 frames as fast as it could, so frame-counted waits meant nothing | ~1 ms per frame and `SA_EXIT_AFTER=N`, like the OpenGL backend |
| nothing showed what happened | the conversation is printed line by line |
| a server-side exception closed the connection silently | it is logged with a traceback and returned as an error reply |

It was checked that the test can fail: `--item tiger` (buyers refuse) fails with exit 3, a server
that isn't running fails with "Can't reach the ML server", `--server host:99999999999` exits 2
(it used to crash in `std::stoi`).

## CUPID review

CUPID is a rule-based static analyzer (not part of this repo). Every finding was checked against
the code and marked **confirmed** (fixed), **false positive**, or **not actionable** (true but not
worth changing, with the reason).

| Finding | Verdict | What changed |
|---|---|---|
| `open()` without close in `evaluate.run`, `baselines.load`, `compare_tokenizers.samples` | confirmed | one `load_split()` with a context manager |
| (found while fixing it) the buyer data path `buyers_v1` defined in three modules | confirmed | defined once in `buyers/datagen.py` |
| unused `rng` parameter in `iter_fineweb_paragraphs`; unused `seed` in `write_corpus` | confirmed | removed |
| `handle_message` cognitive complexity 28 | confirmed | a handler table: one small function per message type |
| (found while fixing it) round sessions kept forever by a long-running server | confirmed | `MAX_ROUNDS = 64`, oldest dropped; test added |
| `main` (game) cognitive complexity 30, 64 lines | confirmed | `ParseArgs` / `ParseServer` |
| dead store in `DGL_System_FrameControl` | confirmed | removed |
| `RoundScene::Update` / `Render`, `CanvasScene::Update` (cyclomatic 22 / 19 / 25) | confirmed | split by phase: `PollReply`, `UpdateQuestion`, `UpdateSale`, `RenderBuyers`, … |
| (found while fixing it) the session id built in two scenes from the same counter | confirmed | `GameContext::session`, set once by `NewSession()` |
| sockets acquired but not released in `MlClient::Connect` / `RequestBlocking` | false positive | a failed socket is closed on the spot; the connected one is owned by `socket_` and closed in `Disconnect()` / the destructor |
| `DGL_Graphics_AddTriangle` takes 9 parameters | not actionable | it is DGL's own API, which the compat layer must match |
| drawing functions with 6 parameters (`DrawWrapped`, `DrawStrokesInBox`) | not actionable | position, size, thickness, color: the usual shape of a draw call |
| `_sign` and `pick_device` duplicated | not actionable | two copies of a few lines; CUPID's own advice is to keep them in sync until there are more |
| `unused-parameter` on message handlers | not actionable | every handler has the same signature so the table can call them alike |
| `broad-except` in `server._reply_to` | not actionable | deliberate: a bug in one handler must not drop the game's connection |
| quadratic loops in `compare_tokenizers`, `play.py` | not actionable | bounded inputs (a few tokenizers × samples; a handful of buyers) |
| `schema.validate` cognitive complexity 62 | deferred | it is a list of independent data checks; worth splitting per item when it next changes |

Warnings before → after (CUPID, same scope): Python 17 → 13, C++ 9 → 7, the one C++ `error` → 0.
The loop's baseline (`docs/quality/cupid-baseline.jsonl`, `ml/sellanything_ml` + `engine` + `game`) is
226 findings, 18 of them warnings (each reviewed above); `check.py` fails only on warnings a change adds on top of that.
To accept new findings after reviewing them: `python scripts/check.py --update-baseline`.

### Fixed after the review

- `MlClient` had no receive timeout: a server that hangs (not crashes) left the game waiting
  forever. Now 30 s, reported as "didn't answer within 30 s". `game/tests/mlclient_test.cpp` runs
  the client against an in-process server that accepts and never answers (timeout 1 s): it gives
  up after 1.01 s. With the timeout line commented out the same test fails.

- The gate caught two warnings introduced by prompt format 2 (`ask_prompt` and `evaluate.run` at
  6 parameters). Fixed at the cause: what a buyer knows at a moment (item, attributes, persona,
  conversation) always travelled as four parameters, so it is now one value, `format.BuyerView`.
  That also cleared the older warnings on `ModelVoice.ask` / `react`. The training data generated
  before and after the change was compared byte for byte: identical.
- The evaluation and the in-game guardrail judged "is this question on topic" with two different
  rules; the evaluation's never checked the item's name. Both now call `format.on_topic`. Measured
  with the shared rule, v2's held-out on-topic rate is 41%, not the 62% the old rule reported.

### Still open

- `RoundSession.start` takes 7 parameters (the optional ones inject test data).
- `sft.main` and `pretrain.main` are 80-line scripts; their phases (setup, loop, final eval) could be
  named functions.
