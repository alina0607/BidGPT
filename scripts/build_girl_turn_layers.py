"""Offline layered UV turn authoring, from the two existing painted endpoints.

Every patch is warped to a common keyed surface before texture interpolation;
this is not a dissolve of two differently posed complete characters. The far
braid has its own depth layer and tucks behind the head. No generated head used.
Outputs stay isolated until visual acceptance.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy.spatial import Delaunay
from scipy.ndimage import map_coordinates, distance_transform_edt, gaussian_filter, binary_fill_holes, binary_closing
from skimage.registration import optical_flow_tvl1
from skimage.color import rgb2hsv
from build_girl_turn_rig import source_frame, contact, digest, STAGE

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'art/stage/girl_turn_layers_v1'
SIZE=(400,360)


def match_run_palette(image):
    # Measured source skin medians: idle [243,210,191], run [249,218,198].
    # Soft material masks avoid hard colour boundaries. No change to geometry,
    # eye shape, outlines, gait, or transparent-pixel alpha.
    a=np.array(image)
    rgb=a[:,:,:3].astype(float)
    hsv=rgb2hsv(rgb/255)
    h,s,v=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    clamp=lambda x:np.clip(x,0,1)
    skin=clamp((h-.025)/.025)*clamp((.16-h)/.04)*clamp((v-.55)/.2)*clamp((.55-s)/.15)
    pink=(clamp((h-.88)/.06)+clamp((.04-h)/.03))*clamp((s-.15)/.15)*clamp((v-.55)/.2)
    rgb-=skin[:,:,None]*np.array([6,8,7])
    rgb-=np.clip(pink,0,1)[:,:,None]*np.array([5,2,5])
    a[:,:,:3]=np.clip(np.rint(rgb),0,255).astype(np.uint8)
    return Image.fromarray(a)


def padded(image):
    result=Image.new('RGBA',(440,360))
    result.paste(image,(0,0))
    return result


def mirrored(image):
    return ImageOps.mirror(padded(image))


def pack(name,frames,timings,parts=None):
    atlas=Image.new('RGBA',(440*len(frames),360))
    for i,f in enumerate(frames):atlas.paste(f,(440*i,0))
    atlas.save(OUT/f'{name}.png')
    meta={'cell':[440,360],'anchor':[220,350],'cols':len(frames),'rows':1,
          'frames':len(frames),'frame_ms':timings,'standing':320,
          'purpose':'isolated candidate; requires visual acceptance before production'}
    if parts:meta['parts']=parts
    (OUT/f'{name}.json').write_text(json.dumps(meta,indent=2)+'\n')


def polygon(points):
    im=Image.new('L',SIZE)
    ImageDraw.Draw(im).polygon(points,fill=255)
    return np.asarray(im)>0


def split(im, end=False):
    # Each source pixel has exactly one core owner. Only opaque internal seams
    # receive a two-pixel underlap, never duplicated silhouette alpha.
    if not end:
        regions={
            'near_hair':[(50,95),(155,95),(184,165),(170,206),(153,243),(151,280),(170,350),(50,350)],
            'far_hair':[(285,95),(370,95),(370,350),(282,350),(292,278),(277,239),(260,191),(260,165)],
            'near_leg':[(175,293),(218,293),(218,359),(175,359)],
            'far_leg':[(219,293),(260,293),(260,359),(219,359)],
            'near_arm':[(177,181),(192,191),(185,211),(173,232),(154,252),(140,247),(158,222),(163,195)],
            'far_arm':[(250,180),(265,186),(274,210),(280,229),(294,246),(282,253),(266,238),(254,215)],
            'head':[(70,0),(370,0),(370,112),(320,127),(294,147),(280,163),(250,178),(224,183),
                    (201,181),(179,174),(162,162),(142,146),(110,132),(70,130)],
        }
    else:
        regions={
            'near_hair':[(0,55),(153,55),(204,132),(196,165),(170,187),(128,202),(0,247)],
            # No visible far braid at the endpoint: its projected surface is
            # wholly behind the original near hair/head, not faded to nothing.
            'far_hair':[],
            'near_leg':[(157,277),(214,277),(224,359),(150,359)],
            'far_leg':[(215,276),(269,276),(282,359),(225,359)],
            'near_arm':[(196,169),(214,177),(210,193),(197,204),(181,208),(188,219),
                        (198,229),(192,240),(178,240),(163,222),(158,206),(168,190)],
            'far_arm':[(237,177),(253,185),(259,198),(279,202),(294,196),(304,201),(307,215),
                       (297,223),(278,224),(258,226),(241,215),(232,199)],
            'head':[(90,0),(350,0),(350,130),(308,152),(287,170),(257,178),(228,174),
                    (203,165),(184,160),(157,159),(133,147),(111,133),(90,125)],
        }
    labels=np.zeros((360,400),np.uint8)
    names=['body']+list(regions)
    for i,name in enumerate(names[1:],1):
        if regions[name]:
            labels[polygon(regions[name])]=i
    a=np.asarray(im)
    layers={}
    for i,name in enumerate(names):
        core=labels==i
        dilated=np.asarray(Image.fromarray(core.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(5)))>0
        # Padded seams use existing painted neighbouring pixels, not flat fills.
        mask=core | (dilated & (a[:,:,3]==255))
        if 'arm' in name:
            rgb=a[:,:,:3].astype(float)
            r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
            skin=(r>90)&(g>65)&(b>40)&(r-g>3)&(g-b>.20*(r-g))
            cream=(r>190)&(g>180)&(b>155)&(np.abs(r-g)<25)&(g>=b-10)
            # Do not let garment, green braid or berry/flower pixels grow into
            # a hand when the arm changes angle. Underlap remains skin/sleeve.
            material=(skin|cream)&mask
            mask &= binary_fill_holes(binary_closing(material,iterations=1))|material
        layer=a.copy()
        layer[~mask]=0
        layers[name]=Image.fromarray(layer)
    return layers, labels


def affine(points, pivot, target, angle, scale=1):
    a=np.deg2rad(angle)
    m=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])*scale
    return (np.asarray(points)-pivot)@m.T+target


def keyed_surfaces():
    surfaces={}
    # Complete head silhouette controls plus eye corners, pupils, jaw, fringe.
    head_a=[(70,0),(350,0),(350,190),(70,190),
            (220,31),(160,47),(279,46),(121,92),(321,94),
            (154,130),(217,108),(287,125),
            (171,144),(188,145),(201,142),(233,138),(246,139),(258,137),
            (221,159),(217,166),(178,166),(267,161),(219,178),(222,183)]
    head_b=[(70,0),(350,0),(350,190),(70,190),
            (220,29),(165,42),(277,49),(121,91),(319,94),
            (179,125),(242,110),(297,126),
            (203,138),(222,140),(236,135),(259,132),(271,133),(283,127),
            (253,153),(250,157),(200,159),(284,157),(246,170),(230,175)]
    surfaces['head']=(head_a,head_b)
    # Scalp-to-tip braid surface rotates as one continuous volume.
    hair_a=[(65,85),(195,85),(195,350),(65,350),
            (132,163),(128,202),(127,241),(123,279),(122,313),
            (101,203),(158,202),(99,311),(152,311)]
    hair_b=affine(hair_a,np.array([132,163]),np.array([142,126]),47,.79)
    surfaces['near_hair']=(hair_a,hair_b)
    far_a=[(245,95),(370,95),(370,350),(245,350),
           (307,161),(313,200),(315,240),(316,279),(318,315)]
    far_b=affine(far_a,np.array([307,161]),np.array([218,86]),70,.28)
    surfaces['far_hair']=(far_a,far_b)
    surfaces['body']=(
        [(130,170),(305,170),(305,310),(130,310),(222,186),(219,215),(219,242),
         (178,194),(260,194),(147,268),(287,269),(184,290),(220,297),(259,290)],
        [(110,165),(300,165),(300,300),(110,300),(230,175),(221,205),(220,227),
         (194,187),(246,192),(128,244),(269,249),(175,273),(214,279),(250,273)])
    surfaces['near_arm']=(
        [(135,175),(198,175),(198,260),(135,260),
         (178,194),(164,224),(152,244),(185,207),(173,235)],
        [(150,195),(223,160),(223,247),(169,250),
         (194,187),(172,209),(186,230),(206,194),(178,226)])
    surfaces['far_arm']=(
        [(240,170),(305,170),(305,265),(240,265),
         (260,194),(272,224),(286,244),(256,214),(279,235)],
        [(223,162),(320,162),(320,239),(223,239),
         (245,191),(261,215),(297,209),(243,208),(280,214)])
    surfaces['near_leg']=(
        [(170,285),(221,285),(221,359),(170,359),(200,299),(203,320),(204,342)],
        [(154,269),(221,269),(229,359),(154,359),(186,284),(184,316),(195,338)])
    surfaces['far_leg']=(
        [(216,285),(264,285),(264,359),(216,359),(235,299),(233,320),(232,342)],
        [(210,269),(273,269),(282,359),(220,359),(233,284),(234,316),(248,336)])
    return {name:(np.asarray(a,dtype=float),np.asarray(b,dtype=float)) for name,(a,b) in surfaces.items()}


def warp_layer(image, points, destination, triangles):
    rgba=np.asarray(image,dtype=float)/255
    rgba[:,:,:3]*=rgba[:,:,3:4]
    out=np.zeros((360,400,4),dtype=float)
    for tri in triangles:
        p=destination[tri]
        x0,y0=np.maximum(np.floor(p.min(0)).astype(int),0)
        x1,y1=np.minimum(np.ceil(p.max(0)).astype(int),[399,359])
        if x1<x0 or y1<y0:
            continue
        yy,xx=np.mgrid[y0:y1+1,x0:x1+1]
        mat=np.column_stack((p[1]-p[0],p[2]-p[0]))
        if abs(np.linalg.det(mat))<1e-5:
            continue
        ab=(np.stack((xx,yy),-1)-p[0])@np.linalg.inv(mat).T
        bary=np.concatenate((1-ab.sum(-1,keepdims=True),ab),-1)
        inside=(bary>=-1e-6).all(-1)
        uv=bary@points[tri]
        sample=np.stack([map_coordinates(rgba[:,:,c],[uv[:,:,1],uv[:,:,0]],order=1,mode='constant')
                         for c in range(4)],-1)
        out[y0:y1+1,x0:x1+1][inside]=sample[inside]
    return out


def rgba(premult):
    p=premult.copy()
    p[:,:,:3]=np.divide(p[:,:,:3],p[:,:,3:4],out=np.zeros_like(p[:,:,:3]),where=p[:,:,3:4]>1e-7)
    return Image.fromarray(np.clip(np.rint(p*255),0,255).astype(np.uint8))


def morph_surface(one,two,p):
    # A single signed-distance silhouette, not two translucent outlines.
    a=one[:,:,3]>.5
    b=two[:,:,3]>.5
    da=distance_transform_edt(a)-distance_transform_edt(~a)
    db=distance_transform_edt(b)-distance_transform_edt(~b)
    alpha=np.clip(((1-p)*da+p*db)*.5+.5,0,1)
    c1=np.divide(one[:,:,:3],one[:,:,3:4],out=np.zeros_like(one[:,:,:3]),where=one[:,:,3:4]>1e-6)
    c2=np.divide(two[:,:,:3],two[:,:,3:4],out=np.zeros_like(two[:,:,:3]),where=two[:,:,3:4]>1e-6)
    # Extend the nearest painted edge into the correspondence band; transparent
    # RGB must never enter colour interpolation.
    for c,mask in [(c1,a),(c2,b)]:
        indices=distance_transform_edt(~mask,return_distances=False,return_indices=True)
        c[~mask]=c[indices[0][~mask],indices[1][~mask]]
    colour=(1-p)*c1+p*c2
    return np.concatenate((colour*alpha[:,:,None],alpha[:,:,None]),-1)


def detail_registration(one,two):
    """Small UV texture correction AFTER hand-authored geometry registration.

    Cache at one common pose, then transport with the keyed surface. Never solve
    flow independently per output frame (that would introduce temporal jitter).
    The hard 6 px bound cannot substitute for anatomical key poses.
    """
    weights=np.array([.2126,.7152,.0722])
    g1=(one[:,:,:3]+1-one[:,:,3:4])@weights
    g2=(two[:,:,:3]+1-two[:,:,3:4])@weights
    flow=optical_flow_tvl1(g1.astype(np.float32),g2.astype(np.float32),
                         attachment=12,tightness=.25,num_warp=5,num_iter=12)
    flow=np.stack([gaussian_filter(f,1.5) for f in flow])
    flow=np.clip(flow,-6,6)
    pixels=np.ones((360,400,4),float)
    pixels[:,:,:2]=np.moveaxis(flow,0,-1)/32+.5
    pixels[:,:,2]=.5
    return Image.fromarray(np.rint(pixels*255).astype(np.uint8))


def align_details(one,two,flow,p):
    encoded=flow[:,:,:2]
    fy=(encoded[:,:,0]-.5)*32
    fx=(encoded[:,:,1]-.5)*32
    valid=flow[:,:,3]>.5
    fy[~valid]=0;fx[~valid]=0
    yy,xx=np.mgrid[:360,:400]
    def sample(im,x,y):
        return np.stack([map_coordinates(im[:,:,c],[y,x],order=1,mode='constant') for c in range(4)],-1)
    return sample(one,xx-p*fx,yy-p*fy),sample(two,xx+(1-p)*fx,yy+(1-p)*fy)


def main_surface(surfaces):
    a=[(0,0),(399,0),(399,359),(0,359)]
    b=a.copy()
    for name,skip in [('head',4),('body',4),('near_leg',4),('far_leg',4),('near_arm',4),('far_arm',4)]:
        aa,bb=surfaces[name]
        for x,y in zip(aa[skip:],bb[skip:]):
            if tuple(x) in ((222,183),(222,186)):
                continue
            if x.tolist() in a:
                continue
            a.append(x.tolist());b.append(y.tolist())
    extra=[
        ((132,163),(142,126)),((128,202),(113,146)),((127,241),(100,165)),
        ((123,279),(80,176)),((122,313),(48,198)),
        ((101,203),(79,142)),((158,202),(143,160)),
        ((99,311),(20,199)),((152,311),(77,196)),
        ((122,332),(48,228)),((96,284),(31,176)),((149,284),(82,188)),
        ((106,151),(115,112)),((166,159),(183,142)),
        ((190,329),(164,328)),((217,330),(216,332)),((191,345),(180,344)),
        ((219,330),(219,333)),((246,330),(263,329)),((245,343),(260,340)),
        ((188,296),(170,280)),((248,296),(238,280)),
    ]
    for x,y in extra:a.append(x);b.append(y)
    return np.array(a,float),np.array(b,float)


def compatible_topology(a,b):
    """Flip mesh diagonals only when both endpoint embeddings improve.

    Rasterising inverted triangles would hide a broken UV cage. Inspect and
    report any remaining bad triangles instead of counting frames as approval.
    """
    tris=Delaunay(a).simplices.tolist()
    def area(p,t):
        q=p[t]
        u,v=q[1]-q[0],q[2]-q[0]
        return (u[0]*v[1]-u[1]*v[0])/2
    def orient(t):
        return t if area(a,t)>0 else [t[0],t[2],t[1]]
    tris=[orient(t) for t in tris]
    def penalty(ts):
        return sum(max(0,-area(b,t)) for t in ts)
    for _ in range(100):
        edges={}
        for n,t in enumerate(tris):
            for i in range(3):edges.setdefault(tuple(sorted((t[i],t[(i+1)%3]))),[]).append(n)
        best=None
        for (i,j),adj in edges.items():
            if len(adj)!=2:continue
            old=[tris[k] for k in adj]
            k=next(x for x in old[0] if x not in (i,j))
            l=next(x for x in old[1] if x not in (i,j))
            new=[orient([k,l,i]),orient([l,k,j])]
            if min(area(a,t) for t in new)<1e-5:continue
            if abs(sum(area(a,t) for t in new)-sum(area(a,t) for t in old))>.001:continue
            improvement=penalty(old)-penalty(new)
            if improvement>1e-5 and (best is None or improvement>best[0]):
                best=(improvement,adj,new)
        if best is None:break
        _,adj,new=best
        for n,t in zip(adj,new):tris[n]=t
    return np.array(tris)


def build_direction(left=False):
    direction='left' if left else 'right'
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['source-layers','target-layers',direction]:
        (OUT/name).mkdir(exist_ok=True)
    idle=source_frame('girl_idle',0)
    if left:
        idle=mirrored(idle).crop((0,0,400,360))
    ready=Image.new('RGBA',SIZE)
    ready.paste(match_run_palette(source_frame('girl_run',4)),(-48,-2))
    start,ls=split(idle)
    end,le=split(ready,True)
    if not left:
        for name,im in start.items(): im.save(OUT/'source-layers'/f'{name}.png')
        for name,im in end.items(): im.save(OUT/'target-layers'/f'{name}.png')
    surfaces=keyed_surfaces()
    surfaces['main']=main_surface(surfaces)
    # Shared surface owns head, torso, near hair and legs. Only occluding braid
    # and arms are separate. Remove the previous artificial neck/hair cut lines.
    for collection,im,labels in [(start,idle,ls),(end,ready,le)]:
        names=list(collection)
        keep=np.ones((360,400),bool)
        for name in ['far_hair']:
            keep[labels==names.index(name)]=False
        expanded=np.asarray(Image.fromarray(keep.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(5)))>0
        pixels=np.array(im)
        use=keep | (expanded & (pixels[:,:,3]==255))
        pixels[~use]=0
        collection['main']=Image.fromarray(pixels)
    topologies={name:compatible_topology(a,b) for name,(a,b) in surfaces.items()}
    order=['far_hair','main','far_arm','near_arm']
    registration={}
    for name in order:
        if name=='far_hair':continue
        a,b=surfaces[name]
        middle=(a+b)/2
        one=warp_layer(start[name],a,middle,topologies[name])
        two=warp_layer(end[name],b,middle,topologies[name])
        registration[name]=detail_registration(one,two)
    frames=[]
    for i in range(17):
        t=i/16
        t=t*t*(3-2*t)
        out=np.zeros((360,400,4),float)
        for name in order:
            a,b=surfaces[name]
            # Hair follows the body, with zero lag at both endpoints.
            p=np.clip(t-.10*np.sin(np.pi*t),0,1) if 'hair' in name else t
            pos=a+(b-a)*p
            one=warp_layer(start[name],a,pos,topologies[name])
            if name=='far_hair':
                patch=one
            else:
                two=warp_layer(end[name],b,pos,topologies[name])
                flow=warp_layer(registration[name],(a+b)/2,pos,topologies[name])
                one,two=align_details(one,two,flow,p)
                patch=morph_surface(one,two,p)
            out=patch+out*(1-patch[:,:,3:4])
        frame=rgba(out)
        if i==0: frame=idle.copy()
        if i==16: frame=ready.copy()
        frame=mirrored(frame) if left else padded(frame)
        frame.save(OUT/direction/f'{i:02}.png')
        frames.append(frame)
    shown=[f.crop((40 if left else 0,0,440 if left else 400,360)) for f in frames[::2]]
    contact(shown,[f'{direction} key {i}/16' for i in range(0,17,2)],cols=3).save(OUT/f'{direction}-contact.png')
    if not left:
        contact(list(start.values()),list(start),cols=3).save(OUT/'source-layers.png')
        contact(list(end.values()),list(end),cols=3).save(OUT/'target-layers.png')
    pack('girl_turn_'+direction,frames,[20]*17)
    (OUT/f'authoring-{direction}.json').write_text(json.dumps({
        'source_idle_sha256':digest(STAGE/'girl_idle.png'),
        'source_run_sha256':digest(STAGE/'girl_run.png'),
        'surfaces':{n:{'rest':a.tolist(),'ready':b.tolist(),'triangles':topologies[n].tolist()}
                    for n,(a,b) in surfaces.items()},'depth_order':order,
        'run_root_translation':[-48,-2],'accepted':False,
    },indent=2)+'\n')
    print(f'Built {direction} candidate; no GIF or production changes.',flush=True)
    return frames


def build():
    right=build_direction(False)
    left=build_direction(True)
    assert right[0].tobytes()==left[0].tobytes()==padded(source_frame('girl_idle',0)).tobytes()
    meta=json.loads((STAGE/'girl_run.json').read_text())
    run=[]
    for i in range(meta['frames']):
        frame=Image.new('RGBA',(440,360))
        frame.paste(match_run_palette(source_frame('girl_run',i)),(-48,-2))
        run.append(frame)
    pack('girl_run',run,meta['frame_ms'],meta['parts'])
    assert right[-1].tobytes()==run[4].tobytes()
    assert left[-1].tobytes()==ImageOps.mirror(run[4]).tobytes()


if __name__=='__main__': build()
