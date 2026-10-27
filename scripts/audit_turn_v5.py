"""Read-only supporting measurements; visual review remains mandatory."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1] / "art/stage/girl_turn_v5"


def measure(path):
    a = np.asarray(Image.open(path).convert("RGBA"))
    r, g, b = a[:,:,:3].astype(float).transpose(2,0,1)
    yy, xx = np.indices(r.shape)
    mask = (a[:,:,3]>220)&(r<160)&(g>r+5)&(g>b+15)&(yy>130)&(yy<340)&(xx>240)
    labels, _ = ndimage.label(mask)
    eyes = []
    for i, box in enumerate(ndimage.find_objects(labels),1):
        if box is None:
            continue
        y, x = box
        pixels = labels[box]==i
        if 80 < pixels.sum() < 2500 and 8 < x.stop-x.start < 70 and 10 < y.stop-y.start < 70:
            ys, xs = np.nonzero(labels==i)
            eyes.append({"x":round(float(xs.mean()),1),"y":round(float(ys.mean()),1),
                         "w":x.stop-x.start,"h":y.stop-y.start,"area":len(xs)})
    skin = (a[:,:,3]>240)&(yy<400)&(r-g>9)&(r-g<40)&(g-b>9)&(g-b<35)&(g>150)
    return {"file":path.name,"eyes":eyes,
            "skin_rgb":np.round(a[:,:,:3][skin].mean(axis=0),1).tolist(),
            "alpha_bbox":Image.fromarray((a[:,:,3]>128).astype("uint8")*255).getbbox()}


if __name__ == "__main__":
    files = [ROOT/"endpoint-idle.png",*sorted(ROOT.glob("pose-??.png")),ROOT/"endpoint-run-ready.png"]
    print(json.dumps([measure(path) for path in files],indent=2))
