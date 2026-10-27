"""Actual engine captures of the construction probe, NOT final-turn acceptance.

Swaps only idle in a temporary asset tree. No movement, no approved-turn alias,
no production writes and no GIF export. This intentionally cannot test the
not-yet-authored run handoff.
"""
from pathlib import Path
import os
import subprocess
import tempfile
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'art/stage/girl_turn_rig_v1'
OUT=ROOT/'docs/quality/girl-turn-rig-v1'
FRAMES=list(range(240,336,3))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='girl-rig-probe-') as temp:
        assets=Path(temp)
        for entry in (ROOT/'game/assets').iterdir():
            if entry.name!='stage':
                (assets/entry.name).symlink_to(entry,target_is_directory=entry.is_dir())
        (assets/'stage').mkdir()
        for entry in (ROOT/'game/assets/stage').iterdir():
            if entry.name not in ('girl_idle.png','girl_idle.json',
                                   'girl_turn_approved.png','girl_turn_approved.json'):
                (assets/'stage'/entry.name).symlink_to(entry)
        for ext in ('png','json'):
            (assets/'stage'/f'girl_idle.{ext}').symlink_to(ART/f'construction-probe.{ext}')
        env={**os.environ,'SA_ASSETS':str(assets),'SA_KEYS':'',
             'SA_CAPTURE_DT':str(1/60),'SA_EXIT_AFTER':'337',
             'SA_CAPTURE':','.join(f'{OUT}/frame-{i:03}.png@{i}' for i in FRAMES)}
        subprocess.run([str(ROOT/'build/game/sell_anything'),'--backend','opengl'],
                       cwd=ROOT,env=env,check=True,timeout=90)
    # Overview only. Individual full-size actual renders are retained for inspection.
    overview=Image.new('RGB',(1600,1000),'#fff3e9')
    draw=ImageDraw.Draw(overview)
    for n,i in enumerate(FRAMES[::4]):
        frame=Image.open(OUT/f'frame-{i:03}.png').convert('RGB')
        frame.thumbnail((400,450))
        x,y=n%4*400,n//4*500
        overview.paste(frame,(x,y))
        draw.text((x+10,y+455),f'engine frame {i} / construction only',fill='#49362b')
    overview.save(OUT/'engine-overview.jpg')
    detail=Image.new('RGB',(1920,1120),'#fff3e9')
    draw=ImageDraw.Draw(detail)
    for n,i in enumerate(FRAMES):
        frame=Image.open(OUT/f'frame-{i:03}.png').convert('RGB')
        crop=frame.crop((875,340,1065,550)).resize((228,252))
        x,y=n%8*240,n//8*280
        detail.paste(crop,(x,y))
        draw.text((x+5,y+258),f'frame-{i}',fill='#49382b')
    detail.save(OUT/'engine-all-frames.jpg')
    print(f'Captured {len(FRAMES)} real OpenGL construction frames. No GIF exported.')


if __name__=='__main__':
    main()
