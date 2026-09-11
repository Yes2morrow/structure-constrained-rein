"""Precise structural footprints shared by the editor and environment (metres)."""
import math
from shapely.geometry import Polygon, LineString, box
from shapely.ops import polygonize, unary_union


def fixed_polygon(item):
    if item.get('type') == 'column' and 'center' in item and 'size' in item:
        return box(*column(item['id'], *item['center'], *item['size'])['rect'])
    if item.get('type') in ('shear_wall', 'load_bearing_wall') and 'start' in item:
        return Polygon(wall(item['id'], item['type'], *item['start'], *item['end'],
                            item['left_thickness'], item['right_thickness'])['polygon'])
    return Polygon(item['polygon']) if 'polygon' in item else box(*item['rect'])


def _numbers(values):
    result = [round(float(v), 9) for v in values]
    if not all(math.isfinite(v) for v in result):
        raise ValueError('坐标和尺寸必须是有限数值')
    return result


def column(identifier, cx, cy, width, depth):
    cx, cy, width, depth = _numbers([cx, cy, width, depth])
    if width <= 0 or depth <= 0:
        raise ValueError('柱的 X、Y 尺寸必须大于零')
    return dict(id=identifier, type='column', center=[cx, cy], size=[width, depth],
                rect=[cx-width/2, cy-depth/2, cx+width/2, cy+depth/2], parametric=True)


def wall(identifier, kind, sx, sy, ex, ey, left, right):
    sx, sy, ex, ey, left, right = _numbers([sx, sy, ex, ey, left, right])
    length = math.hypot(ex-sx, ey-sy)
    if length <= 1e-9 or min(left, right) < 0 or left+right <= 0:
        raise ValueError('墙线起终点不可重合，左右厚度须非负且总厚度大于零')
    if kind not in ('shear_wall', 'load_bearing_wall'):
        raise ValueError('墙属性只能是剪力墙或承重墙')
    nx, ny = -(ey-sy)/length, (ex-sx)/length
    points = [[sx+nx*left, sy+ny*left], [ex+nx*left, ey+ny*left],
              [ex-nx*right, ey-ny*right], [sx-nx*right, sy-ny*right]]
    return dict(id=identifier, type=kind, start=[sx, sy], end=[ex, ey],
                left_thickness=left, right_thickness=right, polygon=points,
                rect=list(Polygon(points).bounds), parametric=True)


def detect_cores(walls):
    """Closed shear-wall centreline faces; gaps are deliberately not bridged."""
    shear = [w for w in walls if w['type'] == 'shear_wall']
    if not shear:
        return []
    faces = sorted(polygonize(unary_union([LineString([w['start'], w['end']]) for w in shear])),
                   key=lambda p: p.bounds)
    return [dict(id=f'auto_core_{i+1}', type='core', polygon=[list(p) for p in face.exterior.coords],
                 rect=list(face.bounds), parametric=True, derived=True)
            for i, face in enumerate(faces)]


def parameter_rows(items):
    columns, walls, other = [], [], []
    for item in items:
        if item.get('derived'):
            continue
        kind = item.get('type')
        x1, y1, x2, y2 = item.get('rect') or fixed_polygon(item).bounds
        if kind == 'column':
            cx, cy = item.get('center', [(x1+x2)/2, (y1+y2)/2])
            width, depth = item.get('size', [x2-x1, y2-y1])
            columns.append(dict(id=item['id'], cx=cx, cy=cy, width=width, depth=depth))
        elif kind in ('shear_wall', 'load_bearing_wall'):
            horizontal = x2-x1 >= y2-y1
            start = item.get('start', [x1, (y1+y2)/2] if horizontal else [(x1+x2)/2, y1])
            end = item.get('end', [x2, (y1+y2)/2] if horizontal else [(x1+x2)/2, y2])
            half = (y2-y1 if horizontal else x2-x1)/2
            walls.append(dict(id=item['id'], type=kind, sx=start[0], sy=start[1], ex=end[0], ey=end[1],
                              left=item.get('left_thickness', half), right=item.get('right_thickness', half)))
        else:
            other.append(item)
    return columns, walls, other


def build_structures(columns, walls, other):
    result = list(other)
    result += [column(r['id'], r['cx'], r['cy'], r['width'], r['depth']) for r in columns]
    wall_items = [wall(r['id'], r['type'], r['sx'], r['sy'], r['ex'], r['ey'], r['left'], r['right']) for r in walls]
    result += wall_items + detect_cores(wall_items)
    if any(not isinstance(i['id'], str) for i in result):
        raise ValueError('每行必须填写文本形式的构件 ID')
    identifiers = [i['id'].strip() for i in result]
    if any(not i for i in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError('构件 ID 必须填写且不能重复（auto_core_ 前缀供自动识别使用）')
    return result


def normalize_structures(items):
    """Migrate legacy rectangles and rebuild derived footprints from parameters."""
    return build_structures(*parameter_rows(items))
