"""Whole-frame registration and comparison only. No recolouring or anatomical warping."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'art/stage/girl_turn_v6'


def bounds(im):
    a=np.asarray(im)[:,:,3]
    y,x=np.nonzero(a>128)
    floor=int(y.max())+1
    # Scale-relative sole band: a fixed 50px band can include the braid tips only
    # after resizing and falsely report different foot centres at different resolutions.
    band=max(2,round((floor-int(y.min()))*.012))
    fx=np.nonzero(a[floor-band:floor]>128)[1]
    return int(y.min()),floor,(int(fx.min())+int(fx.max()))/2


def main():
    name=sys.argv[1] if len(sys.argv)>1 else 'pose-01'
    ref=Image.open(OUT/'idle-reference.png').convert('RGBA')
    # Verify the reference has not drifted from current runtime idle.
    stage=ROOT/'game/assets/stage'
    meta=json.loads((stage/'girl_idle.json').read_text())
    w,h=meta['cell']
    exact=Image.new('RGBA',(400,360))
    exact.paste(Image.open(stage/'girl_idle.png').convert('RGBA').crop((0,0,w,h)),
                (220-meta['anchor'][0],350-meta['anchor'][1]))
    assert exact.resize((800,720),Image.Resampling.NEAREST).tobytes()==ref.tobytes()
    raw=Image.open(OUT/f'{name}-raw.png').convert('RGBA')
    rt,rb,rx=bounds(ref)
    top,bottom,x=bounds(raw)
    scale=(rb-rt)/(bottom-top)
    resized=raw.resize((round(raw.width*scale),round(raw.height*scale)),Image.Resampling.LANCZOS)
    offset=(round(rx-x*scale),round(rb-bottom*scale))
    fitted=Image.new('RGBA',ref.size)
    fitted.paste(resized,offset)
    fitted.save(OUT/f'{name}.png')
    board=Image.new('RGB',(800,420),'#fff3e9')
    draw=ImageDraw.Draw(board)
    font=ImageFont.truetype(str(ROOT/'art/stage/Nunito.ttf'),18)
    for i,(im,label) in enumerate([(ref,'Original idle'),(fitted,'New small turn — candidate')]):
        cell=Image.new('RGBA',ref.size,'#fff3e9')
        cell.alpha_composite(im)
        board.paste(cell.resize((400,360),Image.Resampling.LANCZOS).convert('RGB'),(400*i,0))
        draw.text((400*i+20,375),label,font=font,fill='#66483a')
    board.save(OUT/f'{name}-comparison.png')
    report={'source_idle_sha256':hashlib.sha256((stage/'girl_idle.png').read_bytes()).hexdigest(),
            'reference_matches_runtime':True,'whole_frame_scale':scale,'translation':offset,
            'reference_bounds':bounds(ref),'candidate_bounds':bounds(fitted),
            'warning':'Height alignment is not a style/anatomy approval. Inspect the comparison.'}
    (OUT/f'{name}-registration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
