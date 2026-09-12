"""One ID-based editable scene for the complete retrofit environment."""
from copy import deepcopy
import hashlib
import math
from gui.structure_canvas import viewport, canvas_objects
from gui.boundary_editor import validate_boundary
from core.envs.structure_geometry import column, wall, normalize_structures


def entrance_points(building):
    if 'door_positions' in building: return building['door_positions']
    xs,ys=zip(*building['boundary']); span=max(xs)-min(xs)
    return [[min(xs)+span*.46,min(ys)],[min(xs)+span*.54,min(ys)]]


def controls(config):
    result={}
    def add(group,key,role,point): result[(group,key,role)]=list(point)
    def rectangle(group,key,r):
        x1,y1,x2,y2=r
        for role,p in zip(('sw','se','ne','nw','center'),
                          ((x1,y1),(x2,y1),(x2,y2),(x1,y2),((x1+x2)/2,(y1+y2)/2))): add(group,key,role,p)
    b=config['ExistingBuilding']
    for i,p in enumerate(b['boundary']): add('boundary',str(i),'point',p)
    for item in b.get('original_spaces',[]): rectangle('original',item['id'],item['rect'])
    for item in config.get('TargetSpaces',[]):
        rectangle('agent',item['id'],item['initial_rect'])
        r=item['initial_rect']; add('seed',item['id'],'point',item.get('seed',[(r[0]+r[2])/2,(r[1]+r[3])/2]))
    for item in b.get('fixed_objects',[]):
        if item.get('derived'): continue
        key=item['id']
        if item['type'] in ('shear_wall','load_bearing_wall'):
            s,e=item['start'],item['end']; length=math.dist(s,e)
            n=[-(e[1]-s[1])/length,(e[0]-s[0])/length]; m=[(s[0]+e[0])/2,(s[1]+e[1])/2]
            for role,p in [('start',s),('end',e),('center',m),
                           ('left',[m[j]+n[j]*item['left_thickness'] for j in (0,1)]),
                           ('right',[m[j]-n[j]*item['right_thickness'] for j in (0,1)])]: add('wall',key,role,p)
        elif item['type']=='column': rectangle('column',key,item['rect'])
        elif 'polygon' in item:
            for i,p in enumerate(item['polygon']): add('fixed',key,str(i),p)
        else: rectangle('fixed_rect',key,item['rect'])
    for i,p in enumerate(entrance_points(b)): add('door',str(i),'point',p)
    return result


def color(key):
    return '#'+hashlib.sha256(repr(key).encode()).hexdigest()[:6]


DISPLAY_LAYERS = {'boundary':'场地边界','original':'原有房间','agent':'智能体初始区域','seed':'智能体种子点','column':'柱','shear_wall':'剪力墙','load_bearing_wall':'承重墙','core':'核心筒','retained_circulation':'保留交通空间','fixed':'其他固定构件'}


def control_layer(key, config):
    if key[0] in ('wall','fixed','fixed_rect','column'):
        item=next(i for i in config['ExistingBuilding'].get('fixed_objects',[]) if i['id']==key[1])
        return item['type'] if item['type'] in DISPLAY_LAYERS else 'fixed'
    return 'boundary' if key[0]=='door' else key[0]


def scene(config,width,height,styles,layer='all',selected=None,display=None):
    b=config['ExistingBuilding']; scale,ox,oy=viewport(b['boundary'],width,height)
    objects=canvas_objects(b.get('fixed_objects',[]),b['boundary'],width,height,styles)
    display=display or {}
    def appearance(category):
        visible, transparency=display.get(category,(True,0))
        return dict(visible=bool(visible),opacity=1-float(transparency)/100)
    for obj,item in zip(objects,b.get('fixed_objects',[])):
        obj.update(selectable=False,evented=False,**appearance(item.get('type','fixed')))
    points=[dict(x=ox+x*scale,y=oy-y*scale) for x,y in b['boundary']]
    objects.append(dict(type='polyline',points=points+[points[0]],left=min(p['x'] for p in points),top=min(p['y'] for p in points),fill='',stroke='#182c3d',strokeWidth=4,selectable=False,evented=False,**appearance('boundary')))
    door=[dict(x=ox+x*scale,y=oy-y*scale) for x,y in entrance_points(b)]
    if len(door)>=2:
        objects.append(dict(type='polyline',points=door,left=min(p['x'] for p in door),top=min(p['y'] for p in door),fill='',stroke='#2d6f9f',strokeWidth=7,selectable=False,evented=False,**appearance('boundary')))
    for group,items,rect_key,fill in [('original',b.get('original_spaces',[]),'rect','rgba(244,180,170,0.55)'),
                                     ('agent',config.get('TargetSpaces',[]),'initial_rect','rgba(66,165,245,0.30)')]:
        for item in items:
            x1,y1,x2,y2=item[rect_key]
            objects.append(dict(type='rect',left=ox+x1*scale,top=oy-y2*scale,width=(x2-x1)*scale,height=(y2-y1)*scale,
                                fill=fill,stroke='#1565c0' if group=='agent' else '#8d6e63',strokeWidth=1,selectable=False,evented=False,**appearance(group)))
            objects.append(dict(type='text',text=('初始·' if group=='agent' else '')+item.get('name',item['id']),
                                left=ox+x1*scale+4,top=oy-y2*scale+4,fontSize=12,fill='#123456',selectable=False,evented=False,visible=appearance(group)['visible'] and display.get('labels',(True,0))[0],opacity=appearance(group)['opacity']))
    palette=dict(boundary='#ef6c00',original='#a1887f',agent='#1976d2',seed='#8e24aa',column='#455a64',wall='#c62828',fixed='#546e7a',fixed_rect='#546e7a',door='#00897b')
    for key,p in controls(config).items():
        shown=appearance(control_layer(key,config))
        active=shown['visible'] and (layer=='all' or key[0]==layer) and (selected is None or key[1]==selected)
        objects.append(dict(type='circle',left=ox+p[0]*scale,top=oy-p[1]*scale,radius=5 if key[2]!='center' else 7,
                            originX='center',originY='center',fill=palette[key[0]],stroke=color(key),strokeWidth=1,
                            selectable=active,evented=active,visible=shown['visible'],opacity=shown['opacity']*(1 if active else .35),
                            hasControls=False,hasBorders=False,lockScalingX=True,lockScalingY=True,lockRotation=True))
    return objects


def parse_scene(objects,config,width,height):
    """Apply only moved control points; no list-position-based reassignment."""
    result=deepcopy(config); before=controls(config)
    scale,ox,oy=viewport(config['ExistingBuilding']['boundary'],width,height)
    by_color={o.get('stroke'):o for o in objects if o.get('type')=='circle'}
    changes={}
    for key,p in before.items():
        if color(key) not in by_color: raise ValueError('控制点尚未完整加载或被删除；请通过右侧表格增删对象')
        obj=by_color[color(key)]
        if obj.get('visible') is False: continue
        base=[ox+p[0]*scale,oy-p[1]*scale]; delta=[]
        for field,exact in zip(('left','top'),base):
            v=float(obj[field])
            if not math.isfinite(v): raise ValueError('控制点坐标无效')
            delta.append(0 if min(abs(v-exact),abs(v-round(exact,2)))<1e-7 else v-exact)
        if any(delta): changes[key]=[p[0]+delta[0]/scale,p[1]-delta[1]/scale]
    def rect(group,item,field):
        key=item['id']; r=list(item[field]); moved=False
        for role,xi,yi in [('sw',0,1),('se',2,1),('ne',2,3),('nw',0,3)]:
            p=changes.get((group,key,role))
            if p is not None: r[xi],r[yi]=p; moved=True
        center=changes.get((group,key,'center'))
        dx=dy=0
        if center is not None:
            old=before[(group,key,'center')]; dx,dy=center[0]-old[0],center[1]-old[1]
            r=[r[0]+dx,r[1]+dy,r[2]+dx,r[3]+dy]; moved=True
        if r[2]<=r[0] or r[3]<=r[1]: raise ValueError(f'{key} 的角点不能交叉；宽高必须为正')
        if moved: item[field]=r
        return moved,dx,dy
    b=result['ExistingBuilding']
    for i,p in enumerate(b['boundary']): b['boundary'][i]=changes.get(('boundary',str(i),'point'),p)
    b['boundary']=validate_boundary(b['boundary'])
    for item in b.get('original_spaces',[]): rect('original',item,'rect')
    for item in result.get('TargetSpaces',[]):
        moved,dx,dy=rect('agent',item,'initial_rect')
        if 'seed' in item and (dx or dy): item['seed']=[item['seed'][0]+dx,item['seed'][1]+dy]
        if ('seed',item['id'],'point') in changes: item['seed']=changes[('seed',item['id'],'point')]
    updated=[]
    for item in b.get('fixed_objects',[]):
        if item.get('derived'): continue
        key=item['id']
        if item['type']=='column':
            moved,_,_=rect('column',item,'rect')
            if moved:
                x1,y1,x2,y2=item['rect']; item.update(column(key,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1))
        elif item['type'] in ('shear_wall','load_bearing_wall'):
            if any(k[0]=='wall' and k[1]==key for k in changes):
                s=changes.get(('wall',key,'start'),item['start']); e=changes.get(('wall',key,'end'),item['end'])
                m=before[('wall',key,'center')]; center=changes.get(('wall',key,'center'),m)
                s=[s[j]+center[j]-m[j] for j in (0,1)]; e=[e[j]+center[j]-m[j] for j in (0,1)]
                length=math.dist(s,e)
                if length<1e-9: raise ValueError('墙线起终点不能重合')
                n=[-(e[1]-s[1])/length,(e[0]-s[0])/length]; mid=[(s[j]+e[j])/2 for j in (0,1)]
                widths=[]
                for side,sign in [('left',1),('right',-1)]:
                    p=changes.get(('wall',key,side))
                    widths.append(item[side+'_thickness'] if p is None else max(0,sign*sum((p[j]-mid[j])*n[j] for j in (0,1))))
                item.update(wall(key,item['type'],*s,*e,*widths))
        elif 'polygon' in item:
            if any(k[0]=='fixed' and k[1]==key for k in changes):
                item['polygon']=[changes.get(('fixed',key,str(i)),p) for i,p in enumerate(item['polygon'])]
                validate_boundary(item['polygon'])
        else: rect('fixed_rect',item,'rect')
        updated.append(item)
    b['fixed_objects']=normalize_structures(updated)
    if any(k[0]=='door' for k in changes):
        b['door_positions']=[changes.get(('door',str(i),'point'),p) for i,p in enumerate(entrance_points(config['ExistingBuilding']))]
    return result
