"""Atlas assembly + internal review only; never repaints, warps or dissolves sprite parts."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'art/stage/girl_turn_v7'
STAGE=ROOT/'game/assets/stage'


def source_frame(name,index):
    meta=json.loads((STAGE/f'{name}.json').read_text())
    w,h=meta['cell']
    source=Image.open(STAGE/f'{name}.png').convert('RGBA')
    im=source.crop((index*w,0,(index+1)*w,h))
    placed=Image.new('RGBA',(400,360))
    placed.paste(im,(220-meta['anchor'][0],350-meta['anchor'][1]))
    return placed


def measure(im):
    a=np.asarray(im)[:,:,3]
    y,x=np.nonzero(a>128)
    top,bottom=int(y.min()),int(y.max())+1
    band=max(2,round((bottom-top)*.012))
    xs=np.nonzero(a[bottom-band:bottom]>128)[1]
    return {'top':top,'bottom':bottom,'height':bottom-top,'foot':(int(xs.min())+int(xs.max()))/2,
            'bbox':[int(x.min()),top,int(x.max())+1,bottom]}


def opaque(im):
    back=Image.new('RGBA',im.size,'#fff3e9')
    back.alpha_composite(im)
    return back.convert('RGB')


def main():
    source=Image.open(OUT/'source.png').convert('RGBA')
    w,h=source.width/4,source.height/2
    raw=[source.crop((round(c*w),round(r*h),round((c+1)*w),round((r+1)*h)))
         for r in range(2) for c in range(4)]
    measurements=[measure(im) for im in raw]
    idle,run=source_frame('girl_idle',0),source_frame('girl_run',4)
    im,rm=measure(idle),measure(run)
    # Same scale for every exposure, rather than hiding inconsistent anatomy with frame fitting.
    scale=im['height']/measurements[0]['height']
    registered=[]
    for i,(sprite,m) in enumerate(zip(raw,measurements)):
        p=i/8
        target_floor=im['bottom']*(1-p)+rm['bottom']*p
        target_root=im['foot']*(1-p)+rm['foot']*p
        scaled=sprite.resize((round(sprite.width*scale),round(sprite.height*scale)),Image.Resampling.LANCZOS)
        frame=Image.new('RGBA',(400,360))
        frame.paste(scaled,(round(target_root-m['foot']*scale),round(target_floor-m['bottom']*scale)))
        sprite.save(OUT/f'cell-{i:02}-raw.png')
        frame.save(OUT/f'cell-{i:02}.png')
        registered.append(frame)
    frames=[idle,*registered[1:],run]
    atlas=Image.new('RGBA',(len(frames)*400,360))
    contact=Image.new('RGB',(1200,1200),'#fff3e9')
    draw=ImageDraw.Draw(contact)
    font=ImageFont.truetype(str(ROOT/'art/stage/Nunito.ttf'),18)
    for i,frame in enumerate(frames):
        atlas.paste(frame,(i*400,0))
        contact.paste(opaque(frame),((i%3)*400,(i//3)*400))
        draw.text(((i%3)*400+15,(i//3)*400+365),f'{i}: '+('original idle' if i==0 else 'original run-ready' if i==8 else 'candidate'),font=font,fill='#634535')
    contact.save(OUT/'contact.png')
    calibration=Image.new('RGB',(800,400),'#fff3e9')
    calibration.paste(opaque(idle),(0,0))
    calibration.paste(opaque(registered[0]),(400,0))
    draw=ImageDraw.Draw(calibration)
    draw.text((15,365),'Original idle',font=font,fill='#634535')
    draw.text((415,365),'Generated calibration — must match',font=font,fill='#634535')
    calibration.save(OUT/'calibration.png')
    atlas.save(OUT/'girl_turn_v7.png')
    (OUT/'girl_turn_v7.json').write_text(json.dumps({'cell':[400,360],'anchor':[220,350],
        'cols':9,'rows':1,'frames':9,'standing':320,'frame_ms':[50]+[60]*7+[50]},indent=2)+'\n')
    assert atlas.crop((0,0,400,360)).tobytes()==idle.tobytes()
    assert atlas.crop((3200,0,3600,360)).tobytes()==run.tobytes()
    max_height_deviation=max(abs(m['height']/measurements[0]['height']-1) for m in measurements)
    margins_clear=all(m['bbox'][0]>2 and m['bbox'][1]>2 and m['bbox'][2]<sprite.width-2
                      and m['bbox'][3]<sprite.height-2 for sprite,m in zip(raw,measurements))
    report={'source_size':source.size,'grid_cell':[w,h],'common_scale':scale,'source_measurements':measurements,
            'max_height_deviation':max_height_deviation,'clear_cell_margins':margins_clear,
            'height_consistency_pass':max_height_deviation<=.025,
            'registered_measurements':[measure(f) for f in registered],
            'exact_endpoints':True,'idle_sha256':hashlib.sha256((STAGE/'girl_idle.png').read_bytes()).hexdigest(),
            'run_sha256':hashlib.sha256((STAGE/'girl_run.png').read_bytes()).hexdigest(),
            'warning':'These numerical checks do not establish visual approval.'}
    (OUT/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
    if '--gif' in sys.argv:
        approval=json.loads((OUT/'visual-review.json').read_text())
        assert approval.get('passed') is True,'No GIF delivery before explicit visual acceptance'
        assert approval.get('source_sha256')==hashlib.sha256((OUT/'source.png').read_bytes()).hexdigest()
        palette=contact.quantize(colors=256)
        cells=[opaque(f).quantize(palette=palette,dither=Image.Dither.NONE) for f in frames]
        order=list(range(9))+list(range(7,0,-1))
        for name,ms in [('turn',60),('turn-slow',180)]:
            images=[cells[i] for i in order]
            images[0].save(OUT/f'{name}.gif',save_all=True,append_images=images[1:],
                          duration=[600 if i in (0,8) else ms for i in order],loop=0,disposal=2)
    print(json.dumps(report))


if __name__=='__main__':
    main()
