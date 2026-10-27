"""Original-pixel authoring rig. Experimental assets NEVER go to game/assets.

This is a small-angle construction test, not an approved full turn. All layers
sample the original idle; no generated character, colour transfer, or rescaling.
Connected landmarks deform one UV mesh, so separating layers cannot open a seam.
The layer PNGs and editable landmarks are exported for further art authoring.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import Delaunay
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'game/assets/stage'
OUT = ROOT / 'art/stage/girl_turn_rig_v1'
SIZE = (400, 360)
EXPECTED_SOURCE = {
    'girl_idle.png':'f4f25bd5f18146973c9bdb840b28c070f5e61954dfceeb2683a9cacb42f90d32',
    'girl_run.png':'969369f846edf9fa61637085ed6266fc21aef394041a034021b8778b47069567',
}


def source_frame(name, index):
    meta = json.loads((STAGE / f'{name}.json').read_text())
    w, h = meta['cell']
    col, row = index % meta['cols'], index // meta['cols']
    sprite = Image.open(STAGE / f'{name}.png').convert('RGBA')
    sprite = sprite.crop((col*w, row*h, (col+1)*w, (row+1)*h))
    placed = Image.new('RGBA', SIZE)
    placed.paste(sprite, (220-meta['anchor'][0], 350-meta['anchor'][1]))
    return placed


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contact(images, names, cols=3, grid=False):
    out = Image.new('RGB', (cols*400, ((len(images)+cols-1)//cols)*390), '#fff3e9')
    font = ImageFont.truetype(str(ROOT/'art/stage/Nunito.ttf'), 17)
    draw = ImageDraw.Draw(out)
    for i, (im, name) in enumerate(zip(images, names)):
        x,y = i%cols*400, i//cols*390
        bg = Image.new('RGBA', SIZE, '#fff3e9')
        bg.alpha_composite(im)
        out.paste(bg.convert('RGB'), (x,y))
        if grid:
            for xx in range(0,400,40):
                draw.line((x+xx,y,x+xx,y+360), fill='#a8c6c8')
            for yy in range(0,360,40):
                draw.line((x,y+yy,x+400,y+yy), fill='#a8c6c8')
        draw.text((x+12,y+363), name, font=font, fill='#65483c')
    return out


# Semantic regions are an editable authoring partition, NOT separate independently
# stretched images. Boundaries share a common UV mesh and all original pixels.
POLYGONS = {
    'braid_near': [(70,110),(147,110),(165,145),(182,165),(171,199),
                   (160,230),(155,255),(151,278),(165,301),(155,343),(70,343)],
    'braid_far': [(282,111),(360,110),(360,345),(290,345),(283,306),
                  (295,280),(286,254),(278,228),(265,201),(258,180),(276,152)],
    'leg_near': [(175,284),(218,284),(218,359),(175,359)],
    'leg_far': [(219,284),(260,284),(260,359),(219,359)],
    'arm_near': [(172,181),(186,191),(179,219),(172,235),(153,251),
                 (142,247),(147,234),(160,218),(162,198)],
    'arm_far': [(251,181),(268,190),(274,218),(283,235),(292,245),
                (280,250),(268,239),(258,221),(249,206)],
    'head': [(80,0),(360,0),(360,112),(315,122),(290,143),(276,162),
             (252,176),(227,182),(203,181),(180,174),(163,161),(148,144),
             (124,134),(80,128)],
}

# Hand-authored *small angle* projected key pose. Source coordinates are in the
# fixed 400x360 root space. This does not invent surfaces absent from the original.
# Values are deltas; cap and planted feet are deliberately fixed.
HANDLES = [
    ('crown',(220,31),(0,0)), ('cap_l',(121,92),(0,0)),
    ('cap_r',(321,94),(0,0)), ('cap_top_l',(160,47),(0,0)),
    ('cap_top_r',(279,46),(0,0)), ('fringe_l',(154,130),(4,0)),
    ('fringe_mid',(217,108),(8,0)), ('fringe_r',(287,125),(3,0)),
    ('eye_near_outer',(171,144),(6,-1)), ('eye_near',(188,145),(8,-1)),
    ('eye_near_inner',(201,142),(9,-1)), ('eye_far_inner',(233,138),(5,0)),
    ('eye_far',(246,139),(4,0)), ('eye_far_outer',(258,137),(2,0)),
    ('nose',(221,159),(10,0)), ('mouth',(217,166),(9,0)),
    ('jaw_l',(178,166),(5,0)), ('jaw_r',(267,161),(3,0)),
    ('chin',(219,178),(7,0)), ('neck',(222,186),(6,0)),
    ('shoulder_near',(178,194),(6,0)), ('shoulder_far',(260,194),(0,0)),
    ('chest',(219,215),(5,0)), ('waist',(219,242),(3,0)),
    ('hip_near',(198,280),(1,0)), ('hip_far',(237,280),(-1,0)),
    ('elbow_near',(164,224),(5,-2)), ('wrist_near',(152,244),(7,-4)),
    ('elbow_far',(272,224),(-2,-2)), ('wrist_far',(286,244),(-5,-3)),
    ('skirt_near',(147,268),(3,0)), ('skirt_far',(287,269),(-3,0)),
    ('hem_near',(184,290),(0,0)), ('hem_mid',(220,297),(0,0)),
    ('hem_far',(259,290),(0,0)),
    ('knee_near',(202,307),(0,0)), ('knee_far',(234,307),(0,0)),
    ('ankle_near',(203,324),(0,0)), ('ankle_far',(232,324),(0,0)),
    ('sole_near',(204,345),(0,0)), ('sole_far',(231,345),(0,0)),
    ('hair_near_root',(132,163),(-1,0)), ('hair_near_1',(128,202),(-3,-1)),
    ('hair_near_2',(127,241),(-6,-2)), ('hair_near_tie',(123,279),(-10,-4)),
    ('hair_near_tip',(122,313),(-14,-5)),
    ('hair_far_root',(307,161),(-1,0)), ('hair_far_1',(313,200),(-3,0)),
    ('hair_far_2',(315,240),(-5,-1)), ('hair_far_tie',(316,279),(-6,-2)),
    ('hair_far_tip',(318,315),(-8,-3)),
    ('extent_near',(96,246),(-6,-2)), ('extent_far',(342,247),(-5,-1)),
    ('extent_near_tip',(98,315),(-14,-5)), ('extent_far_tip',(343,315),(-8,-3)),
]


def export_layers(idle):
    labels = np.zeros((SIZE[1],SIZE[0]), dtype=np.uint8)
    names = ['torso'] + list(POLYGONS)
    for i,name in enumerate(names[1:], 1):
        mask = Image.new('L',SIZE)
        ImageDraw.Draw(mask).polygon(POLYGONS[name], fill=255)
        labels[np.asarray(mask)>0] = i
    rgba = np.asarray(idle)
    result = []
    rebuilt = np.zeros_like(rgba)
    for i,name in enumerate(names):
        pixels = np.zeros_like(rgba)
        pixels[labels==i] = rgba[labels==i]
        rebuilt[labels==i] = pixels[labels==i]
        layer = Image.fromarray(pixels)
        layer.save(OUT/'layers'/f'{name}.png')
        result.append(layer)
    assert np.array_equal(rebuilt,rgba), 'Layer extraction lost source pixels'
    Image.fromarray(labels).save(OUT/'layer-labels.png')
    contact(result,names).save(OUT/'layers-contact.png')
    return labels


def mesh_data():
    # Fixed transparent canvas guards plus manually named model landmarks.
    border = [(0,0),(399,0),(399,359),(0,359),(200,0),(399,180),(200,359),(0,180)]
    source = np.array(border+[p for _,p,_ in HANDLES], dtype=float)
    delta = np.array([(0,0)]*len(border)+[d for _,_,d in HANDLES], dtype=float)
    topology = Delaunay(source).simplices
    return source, delta, topology


def signed_area(points, triangles):
    p=points[triangles]
    a,b=p[:,1]-p[:,0],p[:,2]-p[:,0]
    return (a[:,0]*b[:,1]-a[:,1]*b[:,0])/2


def render(source, delta, triangles, image, weight):
    if weight == 0:
        return image.copy(), {'minimum_area_ratio':1.,'uncovered_pixels':0}
    target=source+delta*weight
    ratio=signed_area(target,triangles)/signed_area(source,triangles)
    assert ratio.min()>.2, 'Folded/collapsed triangle: rejected before rendering'
    uv=np.zeros((SIZE[1],SIZE[0],2), dtype=float)
    coverage=np.zeros((SIZE[1],SIZE[0]), dtype=bool)
    for tri in triangles:
        p=target[tri]
        x0,y0=np.maximum(np.floor(p.min(axis=0)).astype(int),0)
        x1,y1=np.minimum(np.ceil(p.max(axis=0)).astype(int),np.array(SIZE)-1)
        yy,xx=np.mgrid[y0:y1+1,x0:x1+1]
        matrix=np.column_stack((p[1]-p[0],p[2]-p[0]))
        ab=(np.stack((xx,yy),-1)-p[0])@np.linalg.inv(matrix).T
        bary=np.concatenate((1-ab.sum(-1,keepdims=True),ab),-1)
        inside=(bary>=-1e-7).all(-1)
        uv[y0:y1+1,x0:x1+1][inside]=(bary@source[tri])[inside]
        coverage[y0:y1+1,x0:x1+1][inside]=True
    assert coverage.all(), 'UV mesh left a crack'
    # Premultiplied sampling avoids dark/white outlines from transparent pixels.
    original=np.asarray(image).astype(float)/255
    original[:,:,:3]*=original[:,:,3:4]
    samples=np.stack([map_coordinates(original[:,:,c],[uv[:,:,1],uv[:,:,0]],
                                      order=1,mode='nearest') for c in range(4)],-1)
    samples[:,:,:3]=np.divide(samples[:,:,:3],samples[:,:,3:4],
                             out=np.zeros_like(samples[:,:,:3]),where=samples[:,:,3:4]>0)
    out=Image.fromarray(np.clip(np.rint(samples*255),0,255).astype(np.uint8))
    return out, {'minimum_area_ratio':float(ratio.min()),'uncovered_pixels':int((~coverage).sum())}


def main():
    for name,expected in EXPECTED_SOURCE.items():
        if digest(STAGE/name)!=expected:
            raise RuntimeError(f'{name}: original changed; re-author landmarks before building')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'layers').mkdir(exist_ok=True)
    (OUT/'frames').mkdir(exist_ok=True)
    idle=source_frame('girl_idle',0)
    idle.save(OUT/'original-idle.png')
    run=source_frame('girl_run',4)
    run.save(OUT/'original-run-ready.png')
    audit=[]
    angles=[]
    angle_names=[]
    for name,index in [('girl_idle',0),('girl_walk',17),('girl_walk',16),
                       ('girl_walk',15),('girl_run',4)]:
        frame=source_frame(name,index)
        alpha=np.asarray(frame)[:,:,3]
        ys,_=np.nonzero(alpha>128)
        floor=int(ys.max())
        band=max(2,round((floor-int(ys.min()))*.012))
        xs=np.nonzero(alpha[floor-band+1:floor+1]>128)[1]
        sole=(int(xs.min())+int(xs.max()))/2
        offset=(round(219.5-sole),347-floor)
        placed=Image.new('RGBA',SIZE)
        placed.paste(frame,offset)
        angles.append(placed)
        angle_names.append(f'{name}[{index}]  offset {offset}')
        audit.append({'source':name,'frame':index,'sole_band_midpoint':sole,
                      'floor':floor,'comparison_translation':offset,
                      'note':'Sole midpoint is a visual registration diagnostic, not a gait pivot.'})
    contact(angles,angle_names).save(OUT/'original-angles-comparison.png')
    (OUT/'source-registration.json').write_text(json.dumps(audit,indent=2)+'\n')
    export_layers(idle)
    source,delta,triangles=mesh_data()
    rig={'source':'game/assets/stage/girl_idle.png','source_frame':0,
         'cell':SIZE,'anchor':[220,350],'source_sha256':digest(STAGE/'girl_idle.png'),
         'polygons':POLYGONS,'landmarks':[{'name':n,'rest':p,'small_turn_delta':d} for n,p,d in HANDLES],
         'mesh_vertices':source.tolist(),'mesh_triangles':triangles.tolist(),
         'scope':'Small-angle construction test only. No hidden-angle reconstruction yet.',
         'keyframes':[{'time_ms':0,'weight':0},{'time_ms':240,'weight':1}]}
    (OUT/'rig.json').write_text(json.dumps(rig,indent=2)+'\n')
    frames, checks=[],[]
    for i in range(13):
        t=i/12
        weight=t*t*(3-2*t)
        frame,check=render(source,delta,triangles,idle,weight)
        frame.save(OUT/'frames'/f'pose-{i:02}.png')
        frames.append(frame)
        checks.append(check)
    assert frames[0].tobytes()==idle.tobytes()
    # Isolated engine probe only: play the small angle out/back while stationary.
    # This must NOT be installed under girl_turn_approved: it has no run endpoint.
    sequence=list(range(13))+list(range(11,0,-1))
    atlas=Image.new('RGBA',(SIZE[0]*len(sequence),SIZE[1]))
    for i,k in enumerate(sequence):
        atlas.paste(frames[k],(SIZE[0]*i,0))
    atlas.save(OUT/'construction-probe.png')
    (OUT/'construction-probe.json').write_text(json.dumps({
        'cell':SIZE,'anchor':[220,350],'cols':len(sequence),'rows':1,
        'frames':len(sequence),'standing':320,
        'frame_ms':[400 if k in (0,12) else 30 for k in sequence],
        'warning':'ISOLATED CONSTRUCTION PROBE ONLY; NOT A COMPLETE TURN'
    },indent=2)+'\n')
    contact([frames[i] for i in (0,2,4,6,8,10,12)],
            [f'construction {i}/12' for i in (0,2,4,6,8,10,12)]).save(OUT/'construction-contact.png')
    contact(frames,[f'construction {i}/12' for i in range(13)],cols=4).save(OUT/'all-exposures.png')
    wire=idle.copy()
    draw=ImageDraw.Draw(wire)
    for tri in triangles:
        pts=[tuple(source[j]) for j in tri]
        draw.line(pts+[pts[0]],fill='#52a4b7',width=1)
    wire.save(OUT/'mesh.png')
    report={'passed':False,'status':'construction-only; full turn and run handoff not implemented',
            'exact_original_rest':True,'layer_reconstruction_exact':True,
            'original_idle_sha256':digest(STAGE/'girl_idle.png'),
            'original_run_sha256':digest(STAGE/'girl_run.png'),
            'colour_operations':[], 'geometry_checks':checks,
            'gif_exported':False,'production_assets_changed':False}
    (OUT/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
