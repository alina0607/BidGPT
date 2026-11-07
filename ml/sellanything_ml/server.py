"""Local inference server the game connects to.

Protocol: newline-delimited JSON over TCP (see docs/protocol.md).
`classify` runs the sketch classifier; `round_*` runs a selling round (engine + v1 phrases);
`read_doodle` reads a coloured cat doodle (the classifier, plus the cat-doodle-attributes reader when installed);
`save_doodle_exam` keeps a doodle with its true answer as an exam for that reader.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import difflib
import functools
import json
import random
import re
import traceback
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
MAX_ROUNDS = 64        # sessions kept in memory; the oldest is dropped (a game only plays one at a time)
MAX_LINE = 8 * 2**20   # bytes in one request line: read_doodle carries a 256x256 RGBA image (~350 KB of base64)
DOODLE_REGIONS = ("crown", "ear_left", "ear_right", "bridge", "muzzle_chin", "cheek_left", "cheek_right")
# The reader's names (cat-doodle-attributes catdoodle/schema.py); None anywhere is "unspecified".
DOODLE_COATS = ("white", "black", "grey", "orange", "cream", "brown", "mixed")
DOODLE_PATTERNS = ("solid", "tabby", "spots", "bicolour", "tortie_calico")
DOODLE_EYES = ("yellow", "green", "blue", "orange", "copper")
DOODLE_PATCHES = ("none", *DOODLE_COATS[:-1])       # "none": looked, and there is no patch there
ROOT = Path(__file__).resolve().parents[2]
EXAM_DIR = ROOT.parent / "cat-doodle-attributes" / "data" / "exam"     # CATDOODLE_EXAM_DIR overrides it


class FixedSketch:
    """Stands in for the classifier: every drawing is `label` (for end-to-end tests and demos)."""

    def __init__(self, label: str) -> None:
        self.label = label

    def predict(self, strokes, k: int = 3) -> list[dict]:
        return [{"label": self.label, "prob": 1.0}]


class FixedDrawer:
    """Stands in for the drawing GPT in tests: real name matching, the same small drawing every time."""

    def __init__(self) -> None:
        # Keep demo/test mode genuinely dependency-free. Importing sketch.data just to read this TSV
        # also imports NumPy, which defeats the purpose of --fixed-sketch on a fresh checkout.
        items = Path(__file__).resolve().parents[1] / "data" / "items.tsv"
        with open(items, encoding="utf-8") as f:
            self.categories = [row["en"] for row in csv.DictReader(f, delimiter="\t")]

    def match(self, text: str):
        names = {category.lower(): category for category in self.categories}
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()
        cleaned = re.sub(r"\s+", " ", re.sub(r"^(a|an|the|some|my)\s+", "", cleaned))
        for candidate in (cleaned, cleaned[:-2] if cleaned.endswith("es") else "",
                          cleaned[:-1] if cleaned.endswith("s") else ""):
            if candidate in names:
                return names[candidate], []
        close = difflib.get_close_matches(cleaned, list(names), n=3, cutoff=0.6)
        if close and difflib.SequenceMatcher(None, cleaned, close[0]).ratio() >= 0.85:
            return names[close[0]], []
        if not close:
            close = [name for name in names if cleaned and (cleaned in name or name in cleaned)][:3]
        return None, [names[name] for name in close]

    def random_label(self, rng) -> str:
        return rng.choice(self.categories)

    def draw(self, label: str, seed: int):
        return [([0, 80, 160, 80, 0], [80, 0, 80, 160, 80]), ([200, 255], [200, 255])]


class Models:
    """Loads models on first use, so `ping` works without torch installed.

    sketch_label: answer every classify with this label instead of loading the classifier.
    model_voice: False keeps the phrase pools and never loads the buyer GPT.
    """

    def __init__(self, sketch_label: str | None = None, model_voice: bool = True) -> None:
        self._classifier = FixedSketch(sketch_label) if sketch_label else None
        self._tried = sketch_label is not None
        self.rounds: dict[str, object] = {}     # session id -> RoundSession
        self._voice = None
        self._voice_tried = not model_voice
        self._drawer = FixedDrawer() if sketch_label else None
        self._drawer_tried = sketch_label is not None
        self._partial = None
        self._partial_tried = sketch_label is not None
        self._cat_buyers = None
        self._cat_models: dict[str, Any] = {}
        self._cat_buyers_tried = not model_voice
        self._cat_seller = None
        self._cat_seller_tried = not model_voice
        self._shelter = None
        self._doodle_reader = None
        self._doodle_reader_tried = False

    @property
    def voice(self):
        """The trained buyer model, if a checkpoint exists (runs on CPU, leaving the GPU free)."""
        if not self._voice_tried:
            self._voice_tried = True
            from .buyers.voice import load_default
            self._voice = load_default()
        return self._voice

    @property
    def cat_buyers(self):
        """Buyer GPT for cat sales, the newest checkpoint in `cat_market.CHECKPOINTS` (now B5); CPU, leaving the GPU free."""
        if not self._cat_buyers_tried:
            self._cat_buyers_tried = True
            from .buyers.cat_market import load_default
            self._cat_buyers = load_default()
        return self._cat_buyers

    def cat_model(self, name: str | None):
        """The cat buyers' model: the default one (`cat_buyers`), or runs/sft_<name>/best.pt by name ("b5"), for a
        sale that compares two models (`"compare"`); loaded once, CPU."""
        if not name:
            return self.cat_buyers
        if name not in self._cat_models:
            from .buyers.cat_market import load_named
            self._cat_models[name] = load_named(name)
        return self._cat_models[name]

    @property
    def cat_seller(self):
        """A model that learned the seller's side too (prompt format 8, `cat_market.SELLER_CHECKPOINTS`), for cat sales
        where the player buys; CPU, leaving the GPU free."""
        if not self._cat_seller_tried:
            self._cat_seller_tried = True
            from .buyers.cat_market import load_seller
            self._cat_seller = load_seller()
        return self._cat_seller

    @property
    def drawer(self):
        """The drawing GPT, if exported (CPU: a drawing takes well under a second)."""
        if not self._drawer_tried:
            self._drawer_tried = True
            from .sketch.drawer import load_default
            self._drawer = load_default()
        return self._drawer

    @property
    def partial(self):
        """The classifier fine-tuned on drawing prefixes, for live guesses (None: use `classifier`)."""
        if not self._partial_tried:
            self._partial_tried = True
            from .sketch.infer import load_partial
            self._partial = load_partial()
        return self._partial

    @property
    def shelter(self):
        """Brush GPT painting the cat shelter's eight easels (CPU by default, see sketch/shelter.py)."""
        if self._shelter is None:
            from .sketch.shelter import ShelterPainter
            self._shelter = ShelterPainter()
        return self._shelter

    @property
    def doodle_reader(self):
        """(read_attributes, version) from the cat-doodle-attributes package, or None while it is not there.

        read_attributes takes an H x W x 3 uint8 RGB image and returns the attributes dict of read_doodle_result.
        The package is imported as installed, else from its checkout: CATDOODLE_PATH, or ../cat-doodle-attributes
        beside this repo (it is not pip-installed into the game's environment)."""
        if not self._doodle_reader_tried:
            self._doodle_reader_tried = True
            import importlib
            import os
            import sys
            checkout = Path(os.environ.get("CATDOODLE_PATH") or ROOT.parent / "cat-doodle-attributes")
            try:
                try:
                    reader = importlib.import_module("catdoodle.reader")
                except ModuleNotFoundError as e:
                    if e.name != "catdoodle" or not (checkout / "catdoodle").is_dir():
                        raise
                    sys.path.append(str(checkout))
                    reader = importlib.import_module("catdoodle.reader")
            except Exception as e:     # not there, or it cannot load: every attribute stays unspecified
                print(f"read_doodle: no attribute reader ({type(e).__name__}: {e})", flush=True)
                return None
            version = next((getattr(reader, name) for name in ("READER_VERSION", "VERSION", "__version__")
                            if getattr(reader, name, None)), "catdoodle.reader")
            self._doodle_reader = (reader.read_attributes, str(version))
            print(f"read_doodle: attribute reader {version} from {Path(reader.__file__).parents[1]}", flush=True)
        return self._doodle_reader

    @property
    def classifier(self):
        if not self._tried:
            self._tried = True
            from .sketch.infer import load_default
            self._classifier = load_default()
        return self._classifier


DEFAULT_MODELS = Models()


def _valid_strokes(strokes: Any) -> bool:
    if not isinstance(strokes, list) or not strokes:
        return False
    for stroke in strokes:
        if not (isinstance(stroke, list) and len(stroke) == 2 and len(stroke[0]) == len(stroke[1])):
            return False
        if not all(isinstance(v, (int, float)) for axis in stroke for v in axis):
            return False
    return True


def handle_message(msg: dict[str, Any], models: Models | None = None) -> dict[str, Any]:
    """Map one request to one response. `models` can be swapped out in tests."""
    handler = HANDLERS.get(msg.get("type"))
    if handler is None:
        return _error(f"unknown message type: {msg.get('type')!r}")
    return handler(msg, models or DEFAULT_MODELS)


def _ping(msg: dict[str, Any], models: Models) -> dict[str, Any]:
    return {"type": "pong", "protocol": PROTOCOL_VERSION}


def _classify(msg: dict[str, Any], models: Models) -> dict[str, Any]:
    live = msg.get("partial") is True
    reply = _classify_reply(msg.get("strokes"), models, live)
    if msg.get("partial") is True:      # a live guess while drawing; echoed so the game can tell
        reply["partial"] = True
    return reply


def _classify_reply(strokes: Any, models: Models, live: bool = False) -> dict[str, Any]:
    if not _valid_strokes(strokes):
        return _error("classify needs 'strokes': a non-empty list of [xs, ys] number lists")
    # A live guess uses the partial-drawing model when one is exported; the final guess never does.
    classifier = (live and getattr(models, "partial", None)) or models.classifier
    if classifier is None:
        return _error("no sketch model found; train one with python -m sellanything_ml.sketch.train")
    return {"type": "classify_result", "top_k": classifier.predict(strokes, k=3)}


def _doodle_rgb(image: Any):
    """The read_doodle image's RGBA bytes (checked against width x height x 4), or an error message."""
    import base64
    import binascii
    if not isinstance(image, dict):
        return "read_doodle needs 'image': {width, height, rgba_b64}"
    w, h, data = image.get("width"), image.get("height"), image.get("rgba_b64")
    if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0 and isinstance(data, str)):
        return "read_doodle 'image' needs positive integer width and height and an rgba_b64 string"
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return "read_doodle 'image.rgba_b64' is not base64"
    if len(raw) != w * h * 4:
        return f"read_doodle image is {len(raw)} bytes, expected {w}x{h}x4 = {w * h * 4}"
    return raw


def _doodle_attributes(found: Any) -> dict[str, Any]:
    """The reader's answer in the protocol's shape: every field present, each a name or None."""
    def name(v):
        v = getattr(v, "value", v)           # the reader may hand back its str enums
        return v if isinstance(v, str) else None
    found = found if isinstance(found, dict) else {}
    patches = found.get("patches") if isinstance(found.get("patches"), dict) else {}
    return {"coat": name(found.get("coat")), "pattern": name(found.get("pattern")),
            "left_eye": name(found.get("left_eye")), "right_eye": name(found.get("right_eye")),
            "patches": {r: name(patches.get(r)) for r in DOODLE_REGIONS}}


def _read_doodle(msg: dict[str, Any], models: Models) -> dict[str, Any]:
    """A coloured cat doodle: is it a cat (the sketch classifier, as a live guess), and what the attribute
    reader sees in the picture. Without a reader every attribute is None (unspecified) and reader is "none".

    `strokes` are only the pencil outline: the classifier learnt from Quick, Draw!'s black lines and never sees
    colour, so the painted fill (short thick brush strokes) would read as stripes. No pencil strokes: is_cat is
    false and top_k empty, the classifier is not asked; the reader still reads the picture."""
    strokes = msg.get("strokes")
    if not (strokes == [] or _valid_strokes(strokes)):
        return _error("read_doodle needs 'strokes': a list (empty without a pencil outline) of [xs, ys] number lists")
    raw = _doodle_rgb(msg.get("image"))
    if isinstance(raw, str):
        return _error(raw)
    top_k = []
    if strokes:
        classifier = getattr(models, "partial", None) or models.classifier
        if classifier is None:
            return _error("no sketch model found; train one with python -m sellanything_ml.sketch.train")
        top_k = [{"label": c["label"], "p": c["prob"]} for c in classifier.predict(strokes, k=3)]
    reader = getattr(models, "doodle_reader", None)
    attributes = _doodle_attributes(None)
    if reader is not None:
        import numpy as np
        read, version = reader
        image = msg["image"]
        rgba = np.frombuffer(raw, np.uint8).reshape(image["height"], image["width"], 4)
        attributes = _doodle_attributes(read(np.ascontiguousarray(rgba[:, :, :3])))
    return {"type": "read_doodle_result", "is_cat": bool(top_k) and top_k[0]["label"] == "cat", "top_k": top_k,
            "attributes": attributes, "reader": version if reader is not None else "none"}


def _doodle_answer(answer: Any) -> dict[str, Any] | str:
    """The exam's true answer in read_doodle's attributes shape, or what is wrong with it."""
    if not isinstance(answer, dict) or not isinstance(answer.get("patches", {}), dict):
        return "save_doodle_exam needs 'answer': read_doodle's attributes (coat, pattern, left_eye, right_eye, patches)"
    allowed = {"coat": DOODLE_COATS, "pattern": DOODLE_PATTERNS, "left_eye": DOODLE_EYES, "right_eye": DOODLE_EYES}
    for key, names in allowed.items():
        if answer.get(key) is not None and answer[key] not in names:
            return f"answer {key} {answer[key]!r} is not one of {', '.join(names)} or null"
    for region, colour in answer.get("patches", {}).items():
        if region not in DOODLE_REGIONS or (colour is not None and colour not in DOODLE_PATCHES):
            return f"answer patch {region}: {colour!r} is not a region with one of {', '.join(DOODLE_PATCHES)} or null"
    return _doodle_attributes(answer)


def _doodle_edited(edited: Any) -> dict[str, Any]:
    """Which answers the owner changed from the reader's prefill (missing: not changed)."""
    edited = edited if isinstance(edited, dict) else {}
    patches = edited.get("patches") if isinstance(edited.get("patches"), dict) else {}
    return {**{key: edited.get(key) is True for key in ("coat", "pattern", "left_eye", "right_eye")},
            "patches": {r: patches.get(r) is True for r in DOODLE_REGIONS}}


def _png(width: int, height: int, rgba: bytes) -> bytes:
    """RGBA rows, top to bottom, as a PNG file (written by hand: the server runs without Pillow)."""
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\0" + rgba[y * width * 4:(y + 1) * width * 4] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def _git_commit() -> str | None:
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _save_doodle_exam(msg: dict[str, Any], models: Models) -> dict[str, Any]:
    """A doodle and its true answer, kept as an exam for the cat-doodle-attributes reader, in its own folder
    under CATDOODLE_EXAM_DIR (default ../cat-doodle-attributes/data/exam, which that repo does not commit):
    board.png, strokes.json (every stroke with its tool, and the pencil strokes the classifier reads), answer.json,
    edited.json (which answers the owner changed from the reader's prefill), prediction.json (read_doodle on the
    same picture, now) and meta.json."""
    import os
    raw = _doodle_rgb(msg.get("image"))
    if isinstance(raw, str):
        return _error(raw.replace("read_doodle", "save_doodle_exam"))
    answer = _doodle_answer(msg.get("answer"))
    if isinstance(answer, str):
        return _error(answer)
    strokes, pencil = msg.get("strokes"), msg.get("pencil_strokes", [])
    if not isinstance(strokes, list) or not (pencil == [] or _valid_strokes(pencil)):
        return _error("save_doodle_exam needs 'strokes' (a list) and 'pencil_strokes' (read_doodle's strokes)")
    prediction = _read_doodle({"strokes": pencil, "image": msg["image"]}, models)
    prediction = ({key: prediction[key] for key in ("is_cat", "top_k", "attributes", "reader")}
                  if prediction["type"] == "read_doodle_result" else {"error": prediction["message"]})
    now = datetime.datetime.now()
    base = Path(os.environ.get("CATDOODLE_EXAM_DIR") or EXAM_DIR)
    folder = base / now.strftime("%Y%m%d-%H%M%S-%f")[:-3]
    n = 1
    while folder.exists():
        n += 1
        folder = base / f"{now.strftime('%Y%m%d-%H%M%S-%f')[:-3]}-{n}"
    folder.mkdir(parents=True)
    image = msg["image"]
    (folder / "board.png").write_bytes(_png(image["width"], image["height"], raw))
    files = {
        "strokes.json": {"strokes": strokes, "pencil_strokes": pencil},
        "answer.json": answer,
        "edited.json": _doodle_edited(msg.get("edited")),
        "prediction.json": prediction,
        "meta.json": {"saved_at": now.isoformat(timespec="seconds"), "game_version": msg.get("game_version"),
                      "sell_anything_commit": _git_commit(), "protocol": PROTOCOL_VERSION,
                      "width": image["width"], "height": image["height"]},
    }
    for name, data in files.items():
        (folder / name).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"type": "save_doodle_exam_result", "ok": True, "path": str(folder)}


def _draw(msg: dict[str, Any], models: Models) -> dict[str, Any]:
    """The drawing GPT draws what the player typed (`text`), or a random class (`random`, or empty text)."""
    import random
    drawer = models.drawer
    if drawer is None:
        return _error("no drawing model found; export one with python -m sellanything_ml.sketch.drawer export")
    text = msg.get("text", "")
    if not isinstance(text, str):
        return _error("draw needs 'text': a string (empty for a random drawing)")
    seed = msg.get("seed")
    rng = random.Random(seed)
    if msg.get("random") or not text.strip():
        label = drawer.random_label(rng)
    else:
        label, suggestions = drawer.match(text)
        if label is None:
            hint = f" Try: {', '.join(suggestions)}." if suggestions else ""
            return {"type": "draw_result", "ok": False, "suggestions": suggestions,
                    "message": f"I don't know how to draw \"{text.strip()}\".{hint}"}
    strokes = drawer.draw(label, seed=rng.randrange(2**31) if seed is None else int(seed))
    return {"type": "draw_result", "ok": True, "label": label, "strokes": [[xs, ys] for xs, ys in strokes]}


def _scene(msg: dict[str, Any], models: Models) -> dict[str, Any]:
    """One buyer's scene: the item and the props they picture with it, each drawn by the drawing GPT
    and fitted into its box on a SCENE_W x SCENE_H canvas (design-v2 §5.4)."""
    import random
    from .buyers import scenes
    if not isinstance(msg.get("buyer"), str):
        return _error("scene needs 'session' and 'buyer' strings")
    drawer = models.drawer
    if drawer is None:
        return _error("no drawing model found; export one with python -m sellanything_ml.sketch.drawer export")
    rng = random.Random(msg.get("seed"))

    def act(rnd):
        objects = scenes.render(rnd.scene(msg["buyer"]), lambda label: drawer.draw(label, seed=rng.randrange(2**31)))
        return {"type": "scene_result", "ok": True, "buyer": msg["buyer"], "width": scenes.SCENE_W,
                "height": scenes.SCENE_H, "objects": objects}
    return _in_round(msg, models, act)


def _shelter_paint(msg: dict[str, Any], models: Models) -> dict[str, Any]:
    """Brush GPT starts painting the shelter's eight cats (again, from blank canvases, on Reset)."""
    painter = models.shelter
    painter.start(msg.get("seed") if isinstance(msg.get("seed"), int) else None)
    return {"type": "shelter_state", **painter.state(0, painter.versions[0])}


def _shelter_poll(msg: dict[str, Any], models: Models) -> dict[str, Any]:
    """Where the painting is, and cat `cat`'s canvas unless the game already has version `have`."""
    cat, have = msg.get("cat", 0), msg.get("have", -1)
    if not isinstance(cat, int) or not isinstance(have, int):
        return _error("shelter_poll needs integer cat and have")
    return {"type": "shelter_state", **models.shelter.state(cat, have)}


def _shelter_pause(msg: dict[str, Any], models: Models) -> dict[str, Any]:
    """Holds the shelter's painting (`paused` true) or lets it go on (false); replies with cat `cat`'s state."""
    paused, cat = msg.get("paused"), msg.get("cat", 0)
    if not isinstance(paused, bool) or not isinstance(cat, int):
        return _error("shelter_pause needs a boolean paused and an integer cat")
    models.shelter.pause(paused)
    return {"type": "shelter_state", **models.shelter.state(cat, -1)}


def _round_start(msg: dict[str, Any], models: Models) -> dict[str, Any]:
    from .buyers.rounds import RoundSession
    session, item = msg.get("session"), msg.get("item")
    if not isinstance(session, str) or not isinstance(item, str):
        return _error("round_start needs 'session' and 'item' strings")
    n = msg.get("buyers", 4)
    if not isinstance(n, int) or not 1 <= n <= 5:
        return _error("'buyers' must be an integer 1..5")
    role = msg.get("role", "seller")
    if role not in ("seller", "buyer"):
        return _error("'role' must be seller or buyer")
    cat_gpt = getattr(models, "cat_buyers", None) if item == "cat" and msg.get("voice") != "phrases" else None
    voice = None if msg.get("voice") == "phrases" else getattr(models, "voice", None)
    try:
        if item == "cat" and msg.get("compare") is not None:     # models or ways side by side (owner, 2026-10-28)
            rnd = _paired(msg, models, n)
            if isinstance(rnd, dict):
                return rnd
        elif role == "buyer":       # a cat sale where the model is the seller and the player one of the buyers
            rnd = _cat_buying(msg, models, n)
            if isinstance(rnd, dict):
                return rnd
        elif cat_gpt is not None:   # a cat sale run by the buyer GPT: the player types every answer
            from .buyers.cat_market import CatMarket
            rnd = CatMarket.start(cat_gpt, n, seed=msg.get("seed"))
        else:
            rnd = RoundSession.start(item, n, seed=msg.get("seed"), voice=voice)
    except ValueError as e:
        return _error(str(e))
    models.rounds.pop(session, None)          # re-inserting moves it to the newest end
    models.rounds[session] = rnd
    while len(models.rounds) > MAX_ROUNDS:
        models.rounds.pop(next(iter(models.rounds)))
    return {"type": "round_state", "new_lines": list(rnd.log), "state": rnd.state()}


def _paired(msg: dict[str, Any], models: Models, n: int) -> Any:
    """A cat sale played by two buyer models at once (`"compare": ["b3_resume", "b5"]`, runs/sft_<name>): the same
    buyers, question and answers for both, each model's own lines and offers (buyers/paired.py). Or one model in
    several ways of picking words (`"compare": ["b5:greedy", "b5:topp", "b5:topk", "b5:topk2"]`): every way answers
    the same prompts, the stories follow b5:topk (buyers/variants.py). An error reply when the names do not fit."""
    names = msg["compare"]
    if not (isinstance(names, list) and all(isinstance(x, str) for x in names)):
        return _error("'compare' must be a list of model names")
    if not hasattr(models, "cat_model"):
        return _error("this server cannot load models by name")
    if any(":" in x for x in names):
        bases = {x.partition(":")[0] for x in names}
        if len(bases) != 1 or not all(":" in x for x in names):
            return _error("ways of picking words are compared on one model, like [\"b5:greedy\", \"b5:topk\"]")
        base = bases.pop()
        gpt = models.cat_model(base)
        if gpt is None:
            return _error(f"no buyer model named {base!r} (runs/sft_{base}/best.pt)")
        from .buyers.variants import WaysSale
        return WaysSale.start(gpt, [x.partition(":")[2] for x in names], n, seed=msg.get("seed"))
    if len(names) != 2:
        return _error("'compare' must name two models, like [\"b3_resume\", \"b5\"]")
    gpts = []
    for name in names:
        gpt = models.cat_model(name)
        if gpt is None:
            return _error(f"no buyer model named {name!r} (runs/sft_{name}/best.pt)")
        gpts.append(gpt)
    from .buyers.paired import PairedSale
    return PairedSale.start(gpts, n, seed=msg.get("seed"))


def _cat_buying(msg: dict[str, Any], models: Models, n: int) -> Any:
    """A cat sale with the model as the seller and the player as the last buyer, or an error reply."""
    seller_gpt = getattr(models, "cat_seller", None) if msg.get("item") == "cat" else None
    if seller_gpt is None:
        return _error("buying needs item 'cat' and a model that learned the seller's side (runs/sft_b5)")
    from .buyers.cat_market import CatMarket, seller_sheet
    seed = msg.get("seed") if isinstance(msg.get("seed"), int) else random.randrange(2**31)
    name = msg.get("name") if isinstance(msg.get("name"), str) and msg["name"].strip() else "You"
    return CatMarket.start(seller_gpt, n, seed=seed, seller=seller_sheet(seed), player=name.strip()[:20])


def _round_answer(msg: dict[str, Any], models: Models) -> dict[str, Any]:
    """An option the player clicked ("option"), or what they typed ("text"). A typed answer's reply
    says which option it was taken as ("heard"), or null when the buyer asked again."""
    if isinstance(msg.get("text"), str):
        def typed(rnd):
            lines, heard = rnd.answer_text(msg["text"])
            return {"type": "round_state", "new_lines": lines, "heard": heard, "state": rnd.state()}
        return _in_round(msg, models, typed)
    return _in_round(msg, models, lambda rnd: {
        "type": "round_state", "new_lines": rnd.answer(str(msg.get("option"))), "state": rnd.state()})


def _round_buy(msg: dict[str, Any], models: Models) -> dict[str, Any]:
    """The player-buyer's step in a cat sale with a model seller (`state.waiting` says which): open (text, offer),
    ask (text), react (text, offer; 0 walks away) or reply (accept, text) to the seller's counter-offer."""
    step, text, offer = msg.get("step"), msg.get("text"), msg.get("offer")

    def act(rnd):
        if getattr(rnd, "player", None) is None:
            raise ValueError("this round has no player-buyer; start it with role 'buyer'")
        if not isinstance(text, str):
            raise ValueError("round_buy needs 'text'")
        if step == "open":
            lines = rnd.player_open(text, offer)
        elif step == "ask":
            lines = rnd.player_ask(text)
        elif step == "react":
            lines = rnd.player_react(text, offer)
        elif step == "reply":
            lines = rnd.player_reply(msg.get("accept") is True, text)
        else:
            raise ValueError("'step' must be open, ask, react or reply")
        return {"type": "round_state", "new_lines": lines, "state": rnd.state()}
    return _in_round(msg, models, act)


def _round_sell(msg: dict[str, Any], models: Models) -> dict[str, Any]:
    return _in_round(msg, models, lambda rnd: {"type": "round_result", **rnd.sell(str(msg.get("buyer"))),
                                               "state": rnd.state()})


def _in_round(msg: dict[str, Any], models: Models, act) -> dict[str, Any]:
    """Run `act` on the session's round, turning the round's rule errors into error replies."""
    rnd = models.rounds.get(msg.get("session"))
    if rnd is None:
        return _error("no round for this session; send round_start first")
    try:
        return act(rnd)
    except (ValueError, KeyError) as e:
        return _error(str(e).strip("'\""))


def _chat(msg: dict[str, Any], models: Models) -> dict[str, Any]:
    if not isinstance(msg.get("text"), str):
        return _error("chat needs a 'text' string")
    # Free-form chat is not built yet (rounds cover the selling flow); this canned reply keeps
    # the message type reserved so old clients still get a well-formed answer.
    return {"type": "chat_reply", "text": "How old is it?", "offer": None}


HANDLERS = {
    "ping": _ping,
    "draw": _draw,
    "classify": _classify,
    "read_doodle": _read_doodle,
    "save_doodle_exam": _save_doodle_exam,
    "round_start": _round_start,
    "round_answer": _round_answer,
    "round_sell": _round_sell,
    "round_buy": _round_buy,
    "scene": _scene,
    "shelter_paint": _shelter_paint,
    "shelter_poll": _shelter_poll,
    "shelter_pause": _shelter_pause,
    "chat": _chat,
}


def _error(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}


class PlayLog:
    """Every request and its reply, one JSON line each, in <directory>/<date>.jsonl: the drawing's
    strokes and the guesses, every round line with who wrote it (state.sources), offers and sales.
    So a strange round can be read back afterwards instead of reconstructed from memory."""

    def __init__(self, directory: Path | None) -> None:
        self.directory = directory

    def write(self, line: bytes, reply: dict[str, Any]) -> None:
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now()
        try:
            request: Any = json.loads(line)
        except json.JSONDecodeError:
            request = line.decode(errors="replace")
        if isinstance(request, dict) and request.get("type") == "shelter_poll":
            return          # several a second while the shelter is painted; the run's times are printed instead
        if isinstance(request, dict) and isinstance(request.get("image"), dict):
            # read_doodle's picture is ~350 KB of base64 per stroke; the strokes and the reply are what to read back
            image = request["image"]
            request = {**request, "image": {**image, "rgba_b64": f"<{len(str(image.get('rgba_b64', '')))} chars>"}}
        record = {"t": now.isoformat(timespec="milliseconds"), "request": request, "reply": reply}
        with open(self.directory / f"{now.date()}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _reply_to(line: bytes, models: Models) -> dict[str, Any]:
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as e:
        return _error(f"invalid JSON: {e.msg}")
    if not isinstance(msg, dict):
        return _error("expected an object")
    try:
        return handle_message(msg, models)
    except Exception as e:     # a bug must not drop the game's connection; log it and say so
        traceback.print_exc()
        return _error(f"server bug while handling {msg.get('type')!r}: {type(e).__name__}: {e}")


async def _serve_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, models: Models,
                        play_log: PlayLog) -> None:
    try:
        while line := await reader.readline():
            reply = _reply_to(line, models)
            play_log.write(line, reply)
            writer.write((json.dumps(reply, ensure_ascii=False) + "\n").encode())
            await writer.drain()
    finally:
        writer.close()


async def serve(host: str, port: int, models: Models = DEFAULT_MODELS, play_log: PlayLog | None = None) -> None:
    play_log = play_log or PlayLog(None)
    server = await asyncio.start_server(functools.partial(_serve_client, models=models, play_log=play_log),
                                        host, port, limit=MAX_LINE)
    where = f", logging play to {play_log.directory}" if play_log.directory else ""
    print(f"sellanything-ml listening on {host}:{port}{where}", flush=True)
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--fixed-sketch", metavar="LABEL",
                        help="classify every drawing as LABEL (no sketch model needed; for tests)")
    parser.add_argument("--phrases-only", action="store_true",
                        help="buyers speak from the phrase pools; never load the buyer GPT")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR, help="where play is recorded (default ml/logs)")
    parser.add_argument("--no-log", action="store_true", help="don't record play")
    args = parser.parse_args()
    models = Models(sketch_label=args.fixed_sketch, model_voice=not args.phrases_only)
    asyncio.run(serve(args.host, args.port, models, PlayLog(None if args.no_log else args.log_dir)))


if __name__ == "__main__":
    main()
