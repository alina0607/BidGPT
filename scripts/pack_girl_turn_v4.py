"""Whole-frame sprite packing only. No limb warping, colour edits or crossfades."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "art/stage/girl_turn_v4"
STAGE = ROOT / "game/assets/stage"


def cell(name, index):
    meta = json.loads((STAGE / f"{name}.json").read_text())
    w, h = meta["cell"]
    im = Image.open(STAGE / f"{name}.png").convert("RGBA")
    return im.crop((index*w, 0, (index+1)*w, h)), meta["anchor"]


def main():
    source = Image.open(ART / "source.png").convert("RGBA")
    assert source.size == (1536, 1024)
    raw = [source.crop((c*512, r*512, (c+1)*512, (r+1)*512)) for r in range(2) for c in range(3)]
    boxes = [im.getchannel("A").point(lambda a: 255 if a > 128 else 0).getbbox() for im in raw]
    factor = 320/(boxes[0][3]-boxes[0][1])
    frames = [cell("girl_idle", 0)]
    for i in range(1, 5):
        im = raw[i]
        a = np.asarray(im)[:, :, 3]
        bottom = boxes[i][3]
        xs = np.nonzero(a[bottom-25:bottom] > 128)[1]
        root = (int(xs.min())+int(xs.max()))/2
        size = round(512*factor)
        # ONE scale for the entire clip, whole-frame translation for the floor/root registration.
        frames.append((im.resize((size,size), Image.Resampling.LANCZOS),
                       (round(root*factor), round(bottom*factor))))
    frames.append(cell("girl_run", 4))
    atlas = Image.new("RGBA", (400*len(frames), 360))
    contact = Image.new("RGB", (400*3, 390*2), "#d5e8f0")
    draw = ImageDraw.Draw(contact)
    for i, (im, anchor) in enumerate(frames):
        canvas = Image.new("RGBA", (400,360))
        canvas.paste(im,(220-anchor[0],350-anchor[1]))
        atlas.paste(canvas,(i*400,0))
        contact.paste(canvas,(i%3*400,i//3*390),canvas)
        draw.text((i%3*400+10,i//3*390+365),f"Frame {i}: complete sprite",fill="#332222")
    atlas.save(ART / "girl_turn_v4.png")
    contact.save(ART / "contact.png")
    meta = {"cell":[400,360], "anchor":[220,350], "cols":6, "rows":1, "frames":6,
            "standing":320, "frame_ms":[75]*6, "source":"art/stage/girl_turn_v4/source.png",
            "whole_frame_only":True, "common_source_scale":factor}
    (ART / "girl_turn_v4.json").write_text(json.dumps(meta,indent=2)+"\n")


if __name__ == "__main__":
    main()
