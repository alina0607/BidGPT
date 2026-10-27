"""read_doodle: the colour cat doodle board's live reading (docs/protocol.md)."""

import asyncio
import base64
import json
import socket
from enum import Enum
from pathlib import Path

import pytest

from sellanything_ml import server
from sellanything_ml.server import DOODLE_REGIONS, handle_message

BEIGE = (0xED, 0xE6, 0xD6, 0xFF)


class Fixed:
    def __init__(self, *labels):
        self.labels = labels

    def predict(self, strokes, k=3):
        return [{"label": label, "prob": round(0.8 / (i + 1), 4)} for i, label in enumerate(self.labels)][:k]


class NoReader:
    classifier = Fixed("cat", "dog")


def doodle(width=256, height=256, pixel=BEIGE, strokes=None):
    rgba = bytes(pixel) * (width * height)
    return {"type": "read_doodle", "strokes": [[[10, 200], [10, 200]]] if strokes is None else strokes,
            "image": {"width": width, "height": height, "rgba_b64": base64.b64encode(rgba).decode()}}


def test_without_a_reader_every_attribute_is_unspecified():
    reply = handle_message(doodle(), NoReader())
    assert reply == {
        "type": "read_doodle_result", "is_cat": True,
        "top_k": [{"label": "cat", "p": 0.8}, {"label": "dog", "p": 0.4}],
        "attributes": {"coat": None, "pattern": None, "left_eye": None, "right_eye": None,
                       "patches": {r: None for r in DOODLE_REGIONS}},
        "reader": "none"}


def test_is_cat_is_the_classifier_first_guess():
    class Dog:
        classifier = Fixed("dog", "cat")

    reply = handle_message(doodle(), Dog())
    assert reply["is_cat"] is False and reply["top_k"][1] == {"label": "cat", "p": 0.4}


def test_the_live_partial_model_is_used_when_exported():
    class Both:
        classifier, partial = Fixed("final"), Fixed("cat")

    assert handle_message(doodle(), Both())["top_k"][0]["label"] == "cat"


class Coat(str, Enum):
    ORANGE = "orange"


def test_the_reader_gets_the_rgb_picture_and_its_answer_is_passed_on():
    seen = {}

    def read(image):
        seen["shape"], seen["dtype"], seen["pixel"] = image.shape, str(image.dtype), tuple(image[3, 5])
        return {"coat": Coat.ORANGE, "pattern": "tabby", "left_eye": "green",
                "patches": {"crown": "white", "cheek_left": "none", "nonsense": "black"}}

    class WithReader:
        classifier = Fixed("cat")
        doodle_reader = (read, "v0-test")

    reply = handle_message(doodle(width=8, height=6, pixel=(200, 100, 50, 255)), WithReader())
    assert seen == {"shape": (6, 8, 3), "dtype": "uint8", "pixel": (200, 100, 50)}
    assert reply["reader"] == "v0-test"
    assert reply["attributes"] == {
        "coat": "orange", "pattern": "tabby", "left_eye": "green", "right_eye": None,
        "patches": {"crown": "white", "ear_left": None, "ear_right": None, "bridge": None, "muzzle_chin": None,
                    "cheek_left": "none", "cheek_right": None}}


@pytest.mark.parametrize("change", [
    {"strokes": "nope"},
    {"strokes": [[[0, 1], [0]]]},
    {"image": None},
    {"image": {"width": 4, "height": 4, "rgba_b64": "not base64!"}},
    {"image": {"width": 4, "height": 4, "rgba_b64": base64.b64encode(b"\0" * 63).decode()}},
    {"image": {"width": 0, "height": 4, "rgba_b64": ""}},
])
def test_bad_requests_are_error_replies(change):
    assert handle_message({**doodle(4, 4), **change}, NoReader())["type"] == "error"


def test_without_pencil_strokes_the_classifier_is_not_asked_but_the_reader_still_reads():
    """Only the pencil outline goes to the classifier (it never learnt colour); the picture still goes to the reader."""
    class ReaderOnly:
        classifier = None           # would be an error if it were asked
        doodle_reader = (lambda image: {"coat": "black"}, "v0")

    reply = handle_message(doodle(strokes=[]), ReaderOnly())
    assert reply["type"] == "read_doodle_result"
    assert reply["is_cat"] is False and reply["top_k"] == []
    assert reply["attributes"]["coat"] == "black" and reply["reader"] == "v0"


def test_no_classifier_is_an_error():
    class Nothing:
        classifier = None

    assert handle_message(doodle(), Nothing())["type"] == "error"


def test_models_report_no_reader_when_catdoodle_is_missing(monkeypatch, capsys):
    import sys
    monkeypatch.setitem(sys.modules, "catdoodle", None)        # import catdoodle now raises ImportError
    models = server.Models(sketch_label="cat")
    assert models.doodle_reader is None
    reply = handle_message(doodle(), models)
    assert reply["reader"] == "none" and reply["is_cat"] is True
    assert "no attribute reader" in capsys.readouterr().out


def test_models_find_the_reader_in_its_checkout(tmp_path, monkeypatch):
    """cat-doodle-attributes is not pip-installed into the game's environment: the server finds its checkout."""
    import sys
    package = tmp_path / "catdoodle"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "reader.py").write_text('VERSION = "v9"\n\ndef read_attributes(image):\n    return {"coat": "grey"}\n')
    monkeypatch.setenv("CATDOODLE_PATH", str(tmp_path))
    for name in ("catdoodle", "catdoodle.reader"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))          # the server appends the checkout; undo it after
    models = server.Models(sketch_label="cat")
    reply = handle_message(doodle(), models)
    assert reply["reader"] == "v9" and reply["attributes"]["coat"] == "grey"
    for name in ("catdoodle", "catdoodle.reader"):
        sys.modules.pop(name, None)


def test_play_log_keeps_the_strokes_but_not_the_picture(tmp_path):
    log = server.PlayLog(tmp_path)
    request = doodle()
    log.write(json.dumps(request).encode(), {"type": "read_doodle_result"})
    (day,) = tmp_path.glob("*.jsonl")
    row = json.loads(day.read_text())
    assert row["request"]["strokes"] == request["strokes"]
    assert row["request"]["image"]["width"] == 256
    assert row["request"]["image"]["rgba_b64"] == f"<{len(request['image']['rgba_b64'])} chars>"


def test_the_server_reads_a_full_size_picture_in_one_line():
    """A 256x256 picture is ~350 KB of base64, far over asyncio's default 64 KB line limit."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    async def run():
        task = asyncio.create_task(server.serve("127.0.0.1", port, NoReader()))
        for _ in range(100):
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                break
            except OSError:
                await asyncio.sleep(0.02)
        line = json.dumps(doodle()).encode() + b"\n"
        assert len(line) > 300_000
        writer.write(line)
        await writer.drain()
        reply = json.loads(await reader.readline())
        writer.close()
        task.cancel()
        return reply

    assert asyncio.run(run())["type"] == "read_doodle_result"


# --- save_doodle_exam ---------------------------------------------------------------------------------------------

ANSWER = {"coat": "orange", "pattern": "tabby", "left_eye": "green", "right_eye": None,
          "patches": {"crown": "white", "cheek_left": "none"}}


def exam(**change):
    msg = doodle(width=4, height=3, pixel=(200, 100, 50, 255))
    msg.update(type="save_doodle_exam", answer=ANSWER, game_version="0.1.0",
               strokes=[{"tool": "pencil", "colour": "black", "thick": False, "xs": [0, 3], "ys": [0, 2]},
                        {"tool": "flat", "colour": "orange", "thick": True, "xs": [1], "ys": [1]}],
               pencil_strokes=[[[0, 3], [0, 2]]],
               edited={"coat": True, "patches": {"crown": True, "nonsense": True}})
    msg.update(change)
    return msg


def test_an_exam_is_saved_with_its_answer_and_the_prediction_of_that_moment(tmp_path, monkeypatch):
    monkeypatch.setenv("CATDOODLE_EXAM_DIR", str(tmp_path))

    class WithReader:
        classifier = Fixed("cat", "tiger")
        doodle_reader = (lambda image: {"coat": "brown", "pattern": "tabby"}, "v0-test")

    reply = handle_message(exam(), WithReader())
    assert reply["type"] == "save_doodle_exam_result" and reply["ok"] is True
    folder = Path(reply["path"])
    assert folder.parent == tmp_path
    assert sorted(f.name for f in folder.iterdir()) == ["answer.json", "board.png", "edited.json", "meta.json",
                                                        "prediction.json", "strokes.json"]
    read = lambda name: json.loads((folder / name).read_text())      # noqa: E731
    assert read("answer.json") == {
        "coat": "orange", "pattern": "tabby", "left_eye": "green", "right_eye": None,
        "patches": {"crown": "white", "ear_left": None, "ear_right": None, "bridge": None, "muzzle_chin": None,
                    "cheek_left": "none", "cheek_right": None}}
    assert read("edited.json") == {"coat": True, "pattern": False, "left_eye": False, "right_eye": False,
                                   "patches": {r: r == "crown" for r in DOODLE_REGIONS}}
    prediction = read("prediction.json")
    assert prediction["reader"] == "v0-test" and prediction["attributes"]["coat"] == "brown"
    assert prediction["is_cat"] is True and prediction["top_k"][0] == {"label": "cat", "p": 0.8}
    strokes = read("strokes.json")
    assert [s["tool"] for s in strokes["strokes"]] == ["pencil", "flat"]
    assert strokes["pencil_strokes"] == [[[0, 3], [0, 2]]]
    meta = read("meta.json")
    assert meta["game_version"] == "0.1.0" and meta["width"] == 4 and meta["height"] == 3 and meta["saved_at"]
    Image = pytest.importorskip("PIL.Image")
    with Image.open(folder / "board.png") as png:
        assert png.size == (4, 3) and png.mode == "RGBA" and png.getpixel((3, 2)) == (200, 100, 50, 255)


def test_without_a_reader_the_saved_prediction_is_unspecified(tmp_path, monkeypatch):
    monkeypatch.setenv("CATDOODLE_EXAM_DIR", str(tmp_path))
    first = Path(handle_message(exam(), NoReader())["path"])
    second = Path(handle_message(exam(), NoReader())["path"])
    assert first != second                                    # never overwritten, even within one millisecond
    prediction = json.loads((first / "prediction.json").read_text())
    assert prediction["reader"] == "none"
    assert prediction["attributes"] == {"coat": None, "pattern": None, "left_eye": None, "right_eye": None,
                                        "patches": {r: None for r in DOODLE_REGIONS}}


@pytest.mark.parametrize("change", [
    {"answer": None},
    {"answer": {**ANSWER, "coat": "purple"}},
    {"answer": {**ANSWER, "left_eye": "red"}},
    {"answer": {**ANSWER, "patches": {"tail": "white"}}},
    {"answer": {**ANSWER, "patches": {"crown": "mixed"}}},
    {"image": None},
    {"strokes": None},
])
def test_bad_exams_are_errors_and_save_nothing(tmp_path, monkeypatch, change):
    monkeypatch.setenv("CATDOODLE_EXAM_DIR", str(tmp_path))
    assert handle_message(exam(**change), NoReader())["type"] == "error"
    assert not any(tmp_path.iterdir())
