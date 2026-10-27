"""Assemble complete sprite exposures; no interpolation, recolouring or body-part edits."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'art/stage/girl_turn_v5'
NAMES = ['endpoint-idle'] + [f'pose-{i:02}' for i in range(1, 8)] + ['endpoint-run-ready']


def opaque(im):
    bg = Image.new('RGBA', im.size, '#fff2e6')
    bg.alpha_composite(im)
    return bg.convert('RGB')


def main():
    frames = [Image.open(OUT / f'{n}.png').convert('RGBA') for n in NAMES]
    # Pixel-identical endpoints at game resolution; only whole-frame downsampling for new art.
    cells = [im.resize((400,360), Image.Resampling.NEAREST if i in (0,8)
                      else Image.Resampling.LANCZOS) for i,im in enumerate(frames)]
    atlas = Image.new('RGBA', (3600,360))
    contact = Image.new('RGB', (1200,1140), '#fff2e6')
    font = ImageFont.truetype(str(ROOT/'art/stage/Nunito.ttf'), 18)
    draw = ImageDraw.Draw(contact)
    for i,im in enumerate(cells):
        atlas.paste(im, (i*400,0))
        x,y = (i%3)*400,(i//3)*380
        contact.paste(opaque(im), (x,y))
        draw.text((x+16,y+356), f'{i}: {i*12.5:g}%', font=font, fill='#694a41')
    atlas.save(OUT/'girl_turn_v5.png')
    contact.save(OUT/'contact.png')
    meta = {'cell':[400,360], 'anchor':[220,350], 'cols':9, 'rows':1, 'frames':9,
            'frame_ms':[50,60,60,60,60,60,60,60,50], 'standing':320}
    (OUT/'girl_turn_v5.json').write_text(json.dumps(meta,indent=2)+'\n')
    # One shared GIF palette prevents independent frame quantization from adding colour flicker.
    palette = contact.quantize(colors=256)
    opaque_cells = [opaque(im).quantize(palette=palette,dither=Image.Dither.NONE) for im in cells]
    order = list(range(9))+list(range(7,0,-1))
    for name,delay in [('turn-normal',60),('turn-slow',220)]:
        durations = [650 if i in (0,8) else delay for i in order]
        ims = [opaque_cells[i] for i in order]
        ims[0].save(OUT/f'{name}.gif',save_all=True,append_images=ims[1:],
                    duration=durations,disposal=2,loop=0,optimize=False)
    # Endpoint integrity is a hard acceptance gate, not a visual guess.
    stage = ROOT/'game/assets/stage'
    for index,source,frame in [(0,'girl_idle',0),(8,'girl_run',4)]:
        m = json.loads((stage/f'{source}.json').read_text())
        w,h = m['cell']
        sprite = Image.open(stage/f'{source}.png').convert('RGBA').crop((frame*w,0,(frame+1)*w,h))
        exact = Image.new('RGBA',(400,360))
        exact.paste(sprite,(220-m['anchor'][0],350-m['anchor'][1]))
        assert cells[index].tobytes()==exact.tobytes(), source+' endpoint changed'
    print('9 exposures packed; both original endpoints pixel-identical; no blend or warp.')


if __name__=='__main__':
    main()
