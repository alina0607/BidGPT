"""Run real OpenGL with isolated candidate art, never overwriting production assets."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'art/stage/girl_turn_v5'
OUT = ROOT/'docs/quality/girl-turn-v5'
FRAMES = list(range(232,565,2))


def review():
    images = [Image.open(OUT/f'frame-{i:03}.png').convert('RGB') for i in FRAMES]
    crops = [im.crop((400,280,1220,580)) for im in images]
    palette = crops[0].quantize(colors=256)
    gif = [im.quantize(palette=palette,dither=Image.Dither.NONE) for im in crops]
    gif[0].save(OUT/'game-turn.gif',save_all=True,append_images=gif[1:],
                duration=[30,30,40]*(len(gif)//3)+[30]*(len(gif)%3),loop=0,disposal=2)
    for name,indices,left in [('start-right',range(240,274,4),800),('stop-right',range(366,400,4),800),
                              ('start-left',range(400,434,4),800),('stop-left',range(526,560,4),450)]:
        sheet = Image.new('RGB',(1260,960),'#fff4eb')
        draw = ImageDraw.Draw(sheet)
        for k,i in enumerate(indices):
            im = Image.open(OUT/f'frame-{i:03}.png').convert('RGB').crop((left,280,left+420,580))
            x,y=(k%3)*420,(k//3)*320
            sheet.paste(im,(x,y))
            draw.text((x+8,y+302),f'actual game frame {i}',fill='#442d2b')
        sheet.save(OUT/f'{name}.jpg')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if '--review-only' not in sys.argv:
        # Temp tree contains symlinks to immutable originals plus the versioned candidate.
        with tempfile.TemporaryDirectory(prefix='girl-turn-v5-') as temp:
            assets=Path(temp)
            for entry in (ROOT/'game/assets').iterdir():
                if entry.name!='stage':
                    (assets/entry.name).symlink_to(entry,target_is_directory=entry.is_dir())
            (assets/'stage').mkdir()
            for entry in (ROOT/'game/assets/stage').iterdir():
                if entry.name not in ('girl_turn_approved.png','girl_turn_approved.json'):
                    (assets/'stage'/entry.name).symlink_to(entry)
            for ext in ('png','json'):
                (assets/'stage'/f'girl_turn_approved.{ext}').symlink_to(ART/f'girl_turn_v5.{ext}')
            env={**os.environ,'SA_ASSETS':str(assets),'SA_KEYS':'d@240:330,a@400:490',
                 'SA_CAPTURE_DT':str(1/60),'SA_EXIT_AFTER':'568',
                 'SA_CAPTURE':','.join(f'{OUT}/frame-{i:03}.png@{i}' for i in FRAMES)}
            subprocess.run([str(ROOT/'build/game/sell_anything'),'--backend','opengl'],
                           cwd=ROOT,env=env,check=True,timeout=240)
    review()
    print(f'Recorded {len(FRAMES)} actual OpenGL frames, using isolated candidate assets.')


if __name__=='__main__':
    main()
