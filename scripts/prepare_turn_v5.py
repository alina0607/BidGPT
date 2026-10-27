"""Reference/preview assembly only; character pixels are never warped or recoloured."""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "art/stage/girl_turn_v5"
STAGE = ROOT / "game/assets/stage"


def reference(name, index, output):
    meta = json.loads((STAGE / f"{name}.json").read_text())
    w, h = meta["cell"]
    source = Image.open(STAGE / f"{name}.png").convert("RGBA")
    im = source.crop((index*w, 0, (index+1)*w, h))
    canvas = Image.new("RGBA", (400,360))
    canvas.paste(im,(220-meta["anchor"][0],350-meta["anchor"][1]))
    canvas.resize((800,720), Image.Resampling.NEAREST).save(OUT / output)


def bounds(im):
    a = np.asarray(im)[:, :, 3]
    ys, xs = np.nonzero(a > 128)
    bottom = int(ys.max())+1
    foot_x = np.nonzero(a[max(0,bottom-50):bottom] > 128)[1]
    return int(ys.min()), bottom, (int(foot_x.min())+int(foot_x.max()))/2


def register(name, progress):
    im = Image.open(OUT / f"{name}-raw.png").convert("RGBA")
    ends = [Image.open(OUT / f"endpoint-{n}.png").convert("RGBA") for n in ("idle","run-ready")]
    start, end = [bounds(e) for e in ends]
    top, bottom, foot = bounds(im)
    height = (start[1]-start[0])*(1-progress)+(end[1]-end[0])*progress
    scale = height/(bottom-top)
    target_floor = start[1]*(1-progress)+end[1]*progress
    target_foot = start[2]*(1-progress)+end[2]*progress
    scaled = im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA",(800,720))
    x, y = round(target_foot-foot*scale), round(target_floor-bottom*scale)
    canvas.paste(scaled,(x,y))
    canvas.save(OUT/f"{name}.png")
    return {"name":name,"progress":progress,"whole_frame_scale":scale,"translation":[x,y]}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    reference("girl_idle",0,"endpoint-idle.png")
    reference("girl_run",4,"endpoint-run-ready.png")
    if len(sys.argv)>1:
        print(json.dumps(register(sys.argv[1],float(sys.argv[2]))))
