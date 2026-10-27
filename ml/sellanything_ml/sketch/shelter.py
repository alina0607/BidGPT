"""The cat shelter's eight paintings, painted live for the game: Brush GPT L8 paints the whole face, then the eye
model paints the eyes over it (owner, 2026-10-27: L8 + E5 in the game; E8L replaces E5 if it does better).

When the player opens the shelter, eight new cats are painted one after another (left easel first). For each cat:

  face  L8 samples a painting from BOS one token at a time, as `brushgpt_l.sample` does (temperature 1, every token
        among those its grammar allows); each stroke is laid on the canvas, with its layer's brush, the moment its
        five tokens are out. The plan comes first and is not drawn.
  eyes  the eye model (`eyegpt_l8`) gets only what L8 produced, as in tools/eye_eval_e1.py: L8's painting on the eye
        canvas placed by the eye outlines in L8's plan, and those outlines and pupils. It samples its strokes the
        same way (temperature 1, its own grammar) and each is laid straight onto the face canvas with its layer's
        brush as soon as its tokens are out. A plan without two eye outlines gets no eye strokes.

Nothing is picked or filtered afterwards, and every cat is sampled once (sell-anything/CLAUDE.md: part models). The
game shows the painting while the models make it: with each canvas, which layer is being painted and where the last
RECENT strokes went (L8's last three layers are fine hairs of 12-32 px a stroke, which no one sees land without a
marker), and while the eyes are painted, the eye canvas itself (the eye model's own 256 px view of the eyes, L8's
painting zoomed in and its strokes on it), at least EYE_STROKE_SECONDS a stroke so they can be followed.

The painting runs on a worker thread; the game asks for a canvas (`state`) as often as it likes. A new `start` (the
game's Reset) stops the old run after its current token and begins eight new cats. The models run on the CPU unless
SHELTER_DEVICE says otherwise, so they never share the GPU with a training or painting job.

The eye run's brushes are read from brushes.json next to its checkpoint (tools/eye_brushes.py writes it from the
material the run learned from): the eye strokes carry no brush, the material's layers do.

Pause and the record (owner, 2026-10-27: pause any time and say from which second to which second it looks wrong).
`pause(True)` holds the painting before its next token; a cat's clock (`cat_seconds`, what the game shows) leaves the
paused time out. Every run is recorded in ml/logs/shelter/<date-time>_run<N>/: run.json, one line per stroke in
cat<k>.jsonl (the cat's clock t, the phase, the layer and the stroke's own tokens: L8's five, or the eye stroke's
bins, after an "eyes" line with the eye canvas's box), each finished cat's tokens and painting, and on every pause
paused.json with paused.png (the canvas) and, while the eyes are painted, paused_eyes.png. tools/shelter_replay.py
paints any stretch of a cat again from that record.
"""

from __future__ import annotations

import base64
import datetime
import json
import os
import sys
import threading
import time

import numpy as np

from . import brushgpt_l as G
from . import brushtokens_l as L
from . import catbrush as B
from . import eyegpt_l8 as EG

FACE_RUN, EYE_RUN = "L8", "E5"
CATS = 8
FACE_LAYERS = ("washes", "blocks", "blend", "small blocks", "blend", "fur", "dark hairs", "light hairs")  # digital_face
RECORD_ROOT = G.ML_DIR / "logs" / "shelter"
RECENT = 24                          # stroke positions sent with a canvas
EYE_STROKE_SECONDS = 0.04            # the eye model makes 192 strokes in about 4 s on the CPU: about 8 s to watch
TOOLS = G.ML_DIR / "tools"           # the digital painter's brushes and the eye canvas


def _u8(canvas) -> np.ndarray:
    if isinstance(canvas, np.ndarray):
        return (np.clip(canvas, 0, 1) * 255).round().astype(np.uint8)
    return (canvas.clamp(0, 1) * 255).round().byte().cpu().numpy()


def _tools():
    """The tools/ modules that paint L8's layers and the eye canvas (brushtokens_l.render imports them the same way)."""
    if str(TOOLS) not in sys.path:
        sys.path.append(str(TOOLS))
    import eye_crop_styles as E
    import eye_layers_l as ELL
    import eye_render as ER
    return E, ELL, ER


class FaceStrokes:
    """Follows an L8 sequence token by token: `step` returns (layer, (1, 5) stroke row) when a stroke is complete."""

    def __init__(self) -> None:
        self.layer_tok = {L.SP[f"LAYER{j}"]: j for j in range(L.MAX_LAYERS)}
        self.layer, self.buf = -1, []

    def step(self, tok: int):
        if tok in self.layer_tok:
            self.layer, self.buf = self.layer_tok[tok], []
        elif tok == L.SP["LAYER_END"]:
            self.layer = -1
        elif self.layer >= 0 and tok != L.SP["REGION_END"]:
            self.buf.append(tok)
            if len(self.buf) == L.TOKENS_PER_STROKE:
                row, self.buf = np.array([self.buf]), []
                return self.layer, row
        return None


class EyeStrokes:
    """Follows an eye sequence after its plan part: `step` returns (layer, (1, 9) bins) when a stroke is complete,
    as eyegpt_l8.decode_layers reads them (EYE_NEXT starts the plan's second eye)."""

    def __init__(self, vocab, plan: list[int]) -> None:
        self.vocab, sp = vocab, vocab.sp
        self.layer_tok = {sp[f"LAYER{j}"]: j for j in range(vocab.n_layers)}
        self.next, self.end = sp["EYE_NEXT"], sp["LAYER_END"]
        self.centres = EG.eye_centres(plan, vocab) if vocab.coords == "polar" else None
        self.layer, self.eye, self.buf = -1, 0, []

    def step(self, tok: int):
        if tok in self.layer_tok:
            self.layer, self.eye, self.buf = self.layer_tok[tok], 0, []
        elif tok == self.next:
            self.eye = 1
        elif tok == self.end:
            self.layer = -1
        elif self.layer >= 0:
            self.buf.append(tok)
            if len(self.buf) == len(self.vocab.layer_fields[self.layer]):
                at = None if self.centres is None else self.centres[[self.eye]]
                bins = EG.tokens_bins(np.array([self.buf]), self.vocab, at, self.layer)
                self.buf = []
                return self.layer, bins
        return None


class Layers:
    """A canvas painted one stroke at a time that ends as the layer-at-once render does. A paint stroke goes on top
    of what is there. Blending strokes of one layer act together on the canvas as it was before the layer
    (eye_crop_styles.blend_composite: their pulls are averaged), so each new one repaints the layer's blending
    strokes so far from that canvas: the last one gives exactly the layer-at-once result.

    paint(canvas, layer, items) paints a list of strokes of one layer; blend(layer) says whether it smudges."""

    def __init__(self, canvas, paint, blend) -> None:
        self.canvas, self.paint, self.blend = canvas, paint, blend
        self.layer, self.before, self.items = None, canvas, []

    def add(self, layer: int, item):
        if layer != self.layer:
            self.layer, self.before, self.items = layer, self.canvas, []
        if self.blend(layer):
            self.items.append(item)
            self.canvas = self.paint(self.before, layer, self.items)
        else:
            self.canvas = self.paint(self.canvas, layer, [item])
        return self.canvas


def face_strokes(E, canvas, layer: int, rows: list[np.ndarray], grid):
    """L8 strokes of one layer on the face canvas (torch) with the layer's brush, as brushtokens_l.render paints it."""
    br = L.brush(layer)
    with E.brush_settings(br):
        s = B.Strokes.from_stage(L.stroke_values(np.concatenate(rows), layer), (br["length"], br["width"])).to(canvas.device)
        return E.blend_composite(canvas, s) if br["blend"] else B.composite(canvas, s, grid)


class ShelterPainter:
    """Paints CATS cats in turn on a worker thread; `state` reads where it is without waiting for it."""

    def __init__(self, face_run: str = FACE_RUN, eye_run: str | None = EYE_RUN, device: str | None = None) -> None:
        self.face_checkpoint = G.RUN_DIR / face_run / "best.pt"
        self.eye_dir = None if eye_run is None else EG.RUN_DIR / eye_run
        self.face_run, self.eye_run = face_run, eye_run
        self.device = device or os.environ.get("SHELTER_DEVICE", "cpu")
        self._face = self._eye = self._brushes = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._go = threading.Event()         # set while painting may go on (cleared by pause)
        self._go.set()
        self.record_dir = None
        self.run = 0
        self._clear()

    def _clear(self) -> None:
        self.frames = [None] * CATS          # (SIDE, SIDE, 3) uint8: each cat as painted so far
        self.strokes = [0] * CATS            # strokes laid on each canvas (face and eyes)
        self.eye_strokes = [0] * CATS        # of which the eye model's
        self.phases = ["waiting"] * CATS     # waiting, face, eyes, done
        self.layers = [None] * CATS          # (name, 1-based index, number of layers) being painted
        self.recent = [[] for _ in range(CATS)]   # the last RECENT strokes' centres, on the canvas being shown
        self.eye_frames = [None] * CATS      # (256, 256, 3) uint8: the eye canvas, once the eyes are painted
        self.eye_boxes = [None] * CATS       # where the eye canvas is on the face canvas (face px x0, y0, x1, y1)
        self.versions = [0] * CATS           # bumped on every change, so an unchanged canvas is not sent again
        self.seconds = [None] * CATS         # how long each finished cat took
        self.tokens = [None] * CATS          # each finished cat's (face tokens, eye tokens or None)
        self.current = 0                     # the cat being painted (CATS once all are done)
        self.cat_start = [None] * CATS       # each cat's start and end on the run's clock (pauses left out)
        self.cat_end = [None] * CATS
        self.paused_total, self.paused_at = 0.0, None
        self._files: dict = {}
        self.started = time.time()
        self.total_seconds = None
        self.error = None

    def models(self):
        if self._face is None:
            self._face = G.load_model(self.face_checkpoint, self.device)
            if self.eye_dir is not None:
                self._eye = EG.load_model(self.eye_dir / "best.pt", self.device)
                self._brushes = json.loads((self.eye_dir / "brushes.json").read_text())
                if len(self._brushes) != self._eye.vocab.n_layers:
                    raise ValueError(f"{self.eye_dir}: {len(self._brushes)} brushes for {self._eye.vocab.n_layers} layers")
        return self._face, self._eye

    def clock(self) -> float:
        """Seconds the run has been painting, the paused time left out."""
        now = time.time()
        return now - self.started - self.paused_total - (now - self.paused_at if self.paused_at is not None else 0.0)

    def cat_seconds(self, k: int) -> float | None:
        if self.cat_start[k] is None:
            return None
        return (self.cat_end[k] if self.cat_end[k] is not None else self.clock()) - self.cat_start[k]

    def pause(self, on: bool) -> None:
        """Holds the painting before its next token (on) or lets it go on; a pause also saves what is shown."""
        with self._lock:
            if on and self.paused_at is None:
                self.paused_at = time.time()
                self._go.clear()
            elif not on and self.paused_at is not None:
                self.paused_total += time.time() - self.paused_at
                self.paused_at = None
                self._go.set()
            else:
                return
            if on:
                self._snapshot()

    def _snapshot(self) -> None:
        """paused.json, paused.png and paused_eyes.png: what the game is showing (call with the lock held)."""
        if self.record_dir is None:
            return
        from PIL import Image
        k = min(self.current, CATS - 1)
        info = {"run": self.run, "cat": k + 1, "cat_seconds": None if self.cat_seconds(k) is None else round(self.cat_seconds(k), 2),
                "phase": self.phases[k], "layer": self.layers[k], "strokes": self.strokes[k],
                "eye_strokes": self.eye_strokes[k], "at": datetime.datetime.now().isoformat(timespec="seconds")}
        (self.record_dir / "paused.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n")
        if self.frames[k] is not None:
            Image.fromarray(self.frames[k]).save(self.record_dir / "paused.png")
        eyes = self.record_dir / "paused_eyes.png"
        if self.phases[k] == "eyes" and self.eye_frames[k] is not None:
            Image.fromarray(self.eye_frames[k]).save(eyes)
        elif eyes.exists():
            eyes.unlink()

    def _record(self, k: int, line: dict) -> None:
        """One line of cat k's record (from the painting thread only)."""
        if self.record_dir is None:
            return
        f = self._files.get(k)
        if f is None:
            f = self._files[k] = open(self.record_dir / f"cat{k + 1}.jsonl", "a")
        f.write(json.dumps({"t": round(self.cat_seconds(k) or 0.0, 3), **line}) + "\n")
        f.flush()

    def _wait(self, stop: threading.Event) -> bool:
        """Waits while paused; False when the run is stopped meanwhile."""
        while not self._go.wait(0.1):
            if stop.is_set():
                return False
        return not stop.is_set()

    def start(self, seed: int | None = None) -> int:
        """Stops a run in progress and starts painting eight new cats; returns the new run's id."""
        if self._thread is not None:
            self._stop.set()
            self._go.set()                   # a paused run must reach its stop
            self._thread.join()
        self.models()                        # loaded before the clock starts
        self._stop = threading.Event()
        seed = time.time_ns() % 2**31 if seed is None else seed
        with self._lock:
            for f in self._files.values():
                f.close()
            self.run += 1
            self._clear()
            self._go.set()
            run = self.run
            self.record_dir = RECORD_ROOT / f"{datetime.datetime.now():%Y%m%d-%H%M%S}_run{run}"
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / "run.json").write_text(json.dumps(
                {"run": run, "seed": seed, "face_run": self.face_run, "eye_run": self.eye_run, "device": self.device,
                 "eye_stroke_seconds": EYE_STROKE_SECONDS}, indent=1) + "\n")
        self._thread = threading.Thread(target=self._paint_all, args=(run, seed, self._stop), daemon=True)
        self._thread.start()
        return run

    def state(self, cat: int, have_version: int = -1) -> dict:
        """Where the run is, and cat `cat`'s canvas (base64 RGB bytes, rows top to bottom) unless the game already
        has this version of it. The number of strokes is the models' choice, so there is no total (0). With the
        canvas come the eye canvas and its box once the eyes are being painted; `recent` is in the pixels of the
        canvas being painted (the face, or the eye canvas while the eyes are)."""
        with self._lock:
            cat = min(max(cat, 0), CATS - 1)
            out = {"run": self.run, "cats": CATS, "current": self.current, "cat": cat,
                   "strokes": self.strokes[cat], "eye_strokes": self.eye_strokes[cat], "total": 0,
                   "phase": self.phases[cat], "finished": self.seconds[cat] is not None,
                   "recent": self.recent[cat],
                   "version": self.versions[cat], "seconds": [s for s in self.seconds if s is not None],
                   "elapsed": round(self.clock(), 2), "total_seconds": self.total_seconds,
                   "paused": self.paused_at is not None,
                   "cat_seconds": None if self.cat_seconds(cat) is None else round(self.cat_seconds(cat), 2),
                   "record": None if self.record_dir is None else str(self.record_dir)}
            if self.layers[cat]:
                out.update(layer=self.layers[cat][0], layer_index=self.layers[cat][1], layers=self.layers[cat][2])
            if self.error:
                out["error"] = self.error
            frame = self.frames[cat]
            if frame is not None and self.versions[cat] != have_version:
                out.update(width=B.SIDE, height=B.SIDE, image=base64.b64encode(frame.tobytes()).decode())
                if self.eye_frames[cat] is not None:
                    out.update(eye_image=base64.b64encode(self.eye_frames[cat].tobytes()).decode(),
                               eye_box=[round(float(v), 2) for v in self.eye_boxes[cat]])
        return out

    def _paint_all(self, run: int, seed: int, stop: threading.Event) -> None:
        import torch
        try:
            torch.manual_seed(seed)
            for k in range(CATS):
                with self._lock:
                    self.cat_start[k] = self.clock()
                face = self._paint_face(k, run, stop)
                if face is None:
                    return
                eyes = self._paint_eyes(k, run, stop, *face)
                if eyes is False:
                    return
                with self._lock:
                    if self.run != run:
                        return
                    self.tokens[k] = (face[0], eyes)
                    self.phases[k] = "done"
                    self.cat_end[k] = self.clock()
                    self.seconds[k] = round(self.cat_end[k] - self.cat_start[k], 2)
                    self.current = k + 1
                    self.versions[k] += 1
                self._finish(k, face[0], eyes)
            with self._lock:
                if self.run != run:
                    return
                self.total_seconds = round(self.clock(), 2)
                seconds = ", ".join(str(s) for s in self.seconds)
            print(f"shelter: run {run} painted {CATS} cats in {self.total_seconds} s ({seconds})", flush=True)
        except Exception as e:                 # the game shows it instead of waiting forever
            with self._lock:
                self.error = f"{type(e).__name__}: {e}"
            raise

    def _paint_face(self, k: int, run: int, stop: threading.Event):
        """Samples cat k's L8 painting token by token, laying each stroke on the canvas as soon as its five tokens are
        out. Returns (tokens, canvas 0-1 numpy), or None when stopped."""
        import torch
        E, _, _ = _tools()
        model, _ = self.models()
        dev = next(model.parameters()).device.type
        ys, xs = torch.meshgrid(torch.arange(B.SIDE) + 0.5, torch.arange(B.SIDE) + 0.5, indexing="ij")
        grid = torch.stack([xs, ys], -1).to(dev)
        canvas = torch.ones(B.SIDE, B.SIDE, 3, device=dev)
        self._show(k, run, canvas, 0, "face", recent=[])
        max_len = G.MAX_SEQ
        rows = torch.full((1, max_len), L.SP["EOS"], dtype=torch.long, device=dev)
        rows[0, 0] = L.SP["BOS"]
        gram = L.Grammar(model.brush_cfg.n_paint_layers)
        gram.step(L.SP["BOS"])
        masks: dict[frozenset, torch.Tensor] = {}
        cache = [{"size": max_len} for _ in model.blocks]
        track, laid, end = FaceStrokes(), 0, max_len
        paint = Layers(canvas, lambda c, j, rows: face_strokes(E, c, j, rows, grid), lambda j: L.brush(j)["blend"])
        with torch.no_grad(), G.autocast(dev):
            logits, _ = model(rows[:, :1], cache=cache)
            for p in range(1, max_len):
                if not self._wait(stop):
                    return None
                allowed = frozenset(gram.allowed())
                m = masks.setdefault(allowed, torch.from_numpy(L.token_mask(allowed)).to(dev))
                last = logits[0, -1].float().masked_fill(~m, float("-inf"))
                tok = int(torch.multinomial(last.softmax(-1), 1)[0])
                rows[0, p] = tok
                gram.step(tok)
                if tok == L.SP["EOS"]:
                    end = p + 1
                    break
                stroke = track.step(tok)
                if stroke is not None:
                    canvas = paint.add(*stroke)
                    laid += 1
                    layer = stroke[0]
                    xy = L.stroke_values(stroke[1], layer)["xy"][0]
                    self._record(k, {"n": laid, "phase": "face", "layer": layer, "row": stroke[1][0].tolist()})
                    self._show(k, run, canvas, laid, "face", (FACE_LAYERS[layer], layer + 1, len(FACE_LAYERS)), xy)
                logits, _ = model(rows[:, p:p + 1], cache=cache, start_pos=p)
        return rows[0, :end].cpu().numpy(), canvas.clamp(0, 1).float().cpu().numpy()

    def _paint_eyes(self, k: int, run: int, stop: threading.Event, tokens: np.ndarray, face: np.ndarray):
        """Samples the eye model for cat k from what L8 produced, laying each eye stroke onto the face canvas as soon
        as its tokens are out. Returns the eye tokens, None when there is no eye model or the plan has no two eye
        outlines, or False when stopped."""
        import torch
        _, ELL, ER = _tools()
        _, model = self.models()
        if model is None or len(L.decode_plan(tokens)["eyes"]) != 2:
            return None
        vocab, brushes = model.vocab, self._brushes
        box = EG.plan_box(tokens)
        base = ELL.l8_on_crop(tokens, EG.FACE_BOX, tuple(box))
        pix = EG.patches((base * 255).round().astype(np.uint8))[None]
        plan = EG.plan_tokens(tokens, EG.FACE_BOX, tuple(box), vocab)
        dev = next(model.parameters()).device.type
        max_len = vocab.max_tok
        rows = torch.full((1, max_len), EG.SP["EOS"], dtype=torch.long, device=dev)
        rows[0, :len(plan)] = torch.tensor(plan, device=dev)
        gram = EG.Grammar(vocab)
        for t in plan:
            gram.step(t)
        masks: dict[frozenset, torch.Tensor] = {}
        cache = [{"size": EG.NPATCH + max_len} for _ in model.blocks]
        track, laid, end = EyeStrokes(vocab, plan), 0, max_len
        blend = lambda j: bool(brushes[j].get("blend"))      # noqa: E731
        paint = Layers(face, lambda c, j, bins: ER.layers_on_face(c, [np.concatenate(bins)], [brushes[j]], tuple(box), EG.FACE_BOX),
                       blend)
        zoomed = Layers(base, lambda c, j, bins: ER.layers_on_crop(c, [np.concatenate(bins)], [brushes[j]]), blend)
        with self._lock:
            done_face = self.strokes[k]
        self._show(k, run, face, done_face, "eyes", recent=[], eye=(base, box))
        self._record(k, {"eyes": {"box": [float(v) for v in box], "run": self.eye_run}})
        shown = time.time()
        with torch.no_grad():
            logits = model(rows[:, :len(plan)], torch.from_numpy(pix).to(dev), cache=cache)
            for p in range(len(plan), max_len):
                if not self._wait(stop):
                    return False
                allowed = frozenset(gram.allowed())
                m = masks.setdefault(allowed, torch.from_numpy(EG.token_mask(allowed, vocab)).to(dev))
                last = logits[0, -1].float().masked_fill(~m, float("-inf"))
                tok = int(torch.multinomial(last.softmax(-1), 1)[0])
                rows[0, p] = tok
                gram.step(tok)
                if tok == EG.SP["EOS"]:
                    end = p + 1
                    break
                stroke = track.step(tok)
                if stroke is not None:
                    face = paint.add(*stroke)
                    crop = zoomed.add(*stroke)
                    laid += 1
                    time.sleep(max(0.0, EYE_STROKE_SECONDS - (time.time() - shown)))
                    shown = time.time()
                    layer, bins = stroke
                    self._record(k, {"n": done_face + laid, "phase": "eyes", "layer": layer, "bins": bins[0].tolist()})
                    self._show(k, run, face, done_face + laid, "eyes", (vocab.layer_names[layer], layer + 1, vocab.n_layers),
                               bins[0, :2] + 0.5, laid, eye=(crop, box))
                logits = model(rows[:, p:p + 1], cache=cache, start_pos=EG.NPATCH + p)
        return rows[0, :end].cpu().numpy()

    def _finish(self, k: int, face: np.ndarray, eyes) -> None:
        """Cat k's tokens and finished painting, next to its record."""
        if self.record_dir is None:
            return
        from PIL import Image
        np.savez_compressed(self.record_dir / f"cat{k + 1}_tokens.npz", face=face,
                            eyes=np.zeros(0, np.int64) if eyes is None else eyes)
        with self._lock:
            Image.fromarray(self.frames[k]).save(self.record_dir / f"cat{k + 1}.png")

    def _show(self, k: int, run: int, canvas, laid: int, phase: str, layer: tuple | None = None, xy=None,
              eye_laid: int = 0, recent: list | None = None, eye: tuple | None = None) -> None:
        """Cat k's canvas as painted so far, and what the game shows with it: the layer, the newest stroke's centre
        xy (added to the recent ones; `recent` replaces them), and the eye canvas with its box."""
        frame = _u8(canvas)
        crop = None if eye is None else _u8(eye[0])
        with self._lock:
            if self.run != run:
                return
            self.frames[k] = frame
            self.strokes[k] = laid
            self.eye_strokes[k] = eye_laid
            if layer is not None:
                self.layers[k] = layer
            if recent is not None:
                self.recent[k] = recent
            if xy is not None:
                self.recent[k] = (self.recent[k] + [[round(float(xy[0]), 1), round(float(xy[1]), 1)]])[-RECENT:]
            if crop is not None:
                self.eye_frames[k], self.eye_boxes[k] = crop, eye[1]
            self.phases[k] = phase
            self.versions[k] += 1
