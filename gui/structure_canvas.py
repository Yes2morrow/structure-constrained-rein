"""Bidirectional Fabric rectangles ↔ canonical structure parameters.

Fabric's standard JSON drops custom properties. A stable stroke colour identifies
each existing object without relying on non-serialized custom IDs.
"""
import hashlib
import json
import math
import uuid
from core.envs.structure_geometry import column, wall, normalize_structures, fixed_polygon


def signature(items):
    def rounded(value):
        if isinstance(value, float):
            return round(value, 6) or 0.0
        if isinstance(value, list):
            return [rounded(v) for v in value]
        if isinstance(value, dict):
            return {k: rounded(v) for k,v in value.items()}
        return value
    return json.dumps(rounded(sorted(items, key=lambda i: i['id'])), sort_keys=True)


def viewport(boundary, width, height):
    xs, ys = zip(*boundary)
    scale = min((width-32)/(max(xs)-min(xs)), (height-32)/(max(ys)-min(ys)))
    ox = (width-(max(xs)-min(xs))*scale)/2-min(xs)*scale
    oy = (height-(max(ys)-min(ys))*scale)/2+max(ys)*scale
    return scale, ox, oy


def identity_colors(items):
    colors, occupied = {}, set()
    for item in sorted(items, key=lambda i: i['id']):
        seed = item['id']
        color = '#' + hashlib.sha256(seed.encode()).hexdigest()[:6]
        while color in occupied:
            seed += '_'
            color = '#' + hashlib.sha256(seed.encode()).hexdigest()[:6]
        occupied.add(color)
        colors[item['id']] = color
    return colors


def canvas_objects(items, boundary, width, height, styles):
    scale, ox, oy = viewport(boundary, width, height)
    colors = identity_colors(items)
    result = []
    for item in items:
        kind = item.get('type', 'fixed')
        common = dict(fill=styles.get(kind, styles['fixed'])[1], stroke=colors[item['id']],
                      strokeWidth=0, originX='left', originY='top',
                      lockSkewingX=True, lockSkewingY=True, lockScalingFlip=True)
        if item.get('derived') or ('polygon' in item and kind not in ('shear_wall','load_bearing_wall')):
            points = [dict(x=ox+x*scale,y=oy-y*scale) for x,y in fixed_polygon(item).exterior.coords]
            result.append(dict(common, type='polygon', points=points,
                               left=min(p['x'] for p in points), top=min(p['y'] for p in points),
                               selectable=False,evented=False))
        elif kind in ('shear_wall','load_bearing_wall'):
            sx, sy = item['start']; ex, ey = item['end']
            length = math.hypot(ex-sx, ey-sy)
            nx, ny = -(ey-sy)/length, (ex-sx)/length
            left = item['left_thickness']
            result.append(dict(common, type='rect',left=ox+(sx+nx*left)*scale,
                               top=oy-(sy+ny*left)*scale,width=length*scale,
                               height=(left+item['right_thickness'])*scale,
                               angle=-math.degrees(math.atan2(ey-sy,ex-sx))))
        else:
            x1,y1,x2,y2 = item['rect']
            result.append(dict(common,type='rect',left=ox+x1*scale,top=oy-y2*scale,
                               width=(x2-x1)*scale,height=(y2-y1)*scale,lockRotation=True))
    return result


def parse_canvas(objects, previous, boundary, width, height, selected_type, left_thickness=.12, right_thickness=.12):
    scale, ox, oy = viewport(boundary, width, height)
    colors = identity_colors(previous)
    lookup = {colors[i['id']].lower(): i for i in previous}
    baselines = {o['stroke']: o for o in canvas_objects(previous,boundary,width,height,{'fixed':('', '', '')})}
    result = []
    # Non-editable legacy polygons remain data, while derived cores are recomputed.
    result.extend(i for i in previous if 'polygon' in i and i['type'] not in ('shear_wall','load_bearing_wall') and not i.get('derived'))
    for obj in objects:
        if obj.get('type') not in ('rect','line'):
            continue
        old = lookup.get(str(obj.get('stroke','')).lower())
        if old:
            baseline = baselines[colors[old['id']]]
            fields = dict(left=0,top=0,width=0,height=0,angle=0,scaleX=1,scaleY=1)
            # Fabric 4 serializes coordinates/angles to two decimal places. Do not
            # round-trip those losses into precise YAML parameters on initial load.
            if all(min(abs(float(obj.get(k,d))-round(float(baseline.get(k,d)),2)),
                       abs(float(obj.get(k,d))-float(baseline.get(k,d)))) < 1e-8 for k,d in fields.items()):
                result.append(old)
                continue
            obj = dict(obj)
            if all(float(obj.get(k,0)) == round(float(obj.get(k,0)),2) for k in ('width','height')):
                for key in ('left','top','width','height','angle'):
                    exact = float(baseline.get(key,0))
                    obj[key] = exact + float(obj.get(key,0)) - round(exact,2)
        kind = old['type'] if old else selected_type
        identifier = old['id'] if old else f'{kind}_{uuid.uuid4().hex[:12]}'
        angle = math.radians(float(obj.get('angle',0)))
        sx = float(obj.get('scaleX',1)); sy = float(obj.get('scaleY',1))
        w = float(obj.get('width',0))*sx/scale
        h = float(obj.get('height',0))*sy/scale
        x = (float(obj.get('left',0))-ox)/scale
        y = (oy-float(obj.get('top',0)))/scale
        if not all(math.isfinite(v) for v in [w,h,x,y,angle]) or min(sx,sy) <= 0:
            raise ValueError('画布构件坐标无效')
        if obj['type'] == 'line':
            # New walls are drawn as centre lines, with separate left/right thickness.
            x1,y1,x2,y2 = [float(obj.get(k,0)) for k in ('x1','y1','x2','y2')]
            start = [x + (0 if x1 <= x2 else w), y - (0 if y1 <= y2 else h)]
            end = [x + (w if x1 <= x2 else 0), y - (h if y1 <= y2 else 0)]
            if math.dist(start,end) < 1e-8:
                continue
            result.append(wall(identifier,kind,*start,*end,left_thickness,right_thickness))
            continue
        if min(w,h) < 1e-8:
            continue
        if kind in ('shear_wall','load_bearing_wall'):
            ratio = old['left_thickness']/(old['left_thickness']+old['right_thickness']) if old else .5
            left, right = h*ratio, h*(1-ratio)
            # Screen local +Y is world right normal; undo left-side offset.
            start = [x-math.sin(angle)*left, y-math.cos(angle)*left]
            end = [start[0]+math.cos(angle)*w, start[1]-math.sin(angle)*w]
            result.append(wall(identifier,kind,*start,*end,left,right))
        elif kind == 'column':
            result.append(column(identifier,x+w/2,y-h/2,w,h))
        else:
            result.append(dict(id=identifier,type=kind,rect=[x,y-h,x+w,y]))
    return normalize_structures(result)
