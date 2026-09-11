"""Scale-normalized contour limits and conservative clearance-core checks."""
import math
from shapely.geometry import Point
from shapely.geometry.polygon import orient


def polygon_parts(geometry):
    if geometry.geom_type == 'Polygon':
        return [] if geometry.is_empty else [geometry]
    return [p for g in getattr(geometry,'geoms',[]) for p in polygon_parts(g)]


def clearance_width_ok(polygon, width, tolerance=1e-7):
    """Require a connected clearance core and recoverable branches.

    Mitred inward/outward offsets retain straight corners. A neck narrower than
    width disconnects the core; a thin arm disappears and fails reconstruction.
    This is a conservative geometric criterion, not a code-compliance proof.
    """
    if polygon.equals(polygon.envelope):
        x0,y0,x1,y1 = polygon.bounds
        return min(x1-x0,y1-y0) >= width-tolerance
    radius = max(0.,width-tolerance)/2
    core = polygon.buffer(-radius,join_style=2,mitre_limit=5)
    if core.is_empty or core.geom_type != 'Polygon' or not core.is_valid:
        return False
    restored = core.buffer(radius,join_style=2,mitre_limit=5)
    return polygon.difference(restored).area <= tolerance*max(1.,polygon.length)


def shape_metrics(polygon):
    p = orient(polygon.simplify(0),sign=1)
    points = list(p.exterior.coords)[:-1]
    reflex = []
    lengths = []
    for i,b in enumerate(points):
        a,c = points[i-1],points[(i+1)%len(points)]
        ab,bc = (b[0]-a[0],b[1]-a[1]),(c[0]-b[0],c[1]-b[1])
        if ab[0]*bc[1]-ab[1]*bc[0] < -1e-10:
            reflex.append(b)
        lengths.append(math.dist(b,c))
    scale = math.sqrt(p.area)
    hull = p.convex_hull
    return dict(reflex_count=len(reflex), rectangularity=p.area/p.envelope.area,
                notch_depth_ratio=max((Point(v).distance(hull.boundary)/scale for v in reflex),default=0.),
                extra_perimeter=max(0.,p.length/hull.length-1),
                min_segment_ratio=min(lengths)/scale)


def long_edge_eligible(room, length, obstacle_count):
    limits = room.shape_limits
    return (limits.allow_long_edge_split and obstacle_count >= 2
            and length/math.sqrt(room.target_area) >= limits.long_edge_ratio)


def check_shape(polygon, room, fixed, tolerance=1e-7):
    metrics = shape_metrics(polygon)
    limits = room.shape_limits
    errors = []
    if metrics['reflex_count']:
        contacts = sum(polygon.distance(p) <= tolerance for p in polygon_parts(fixed))
        x0,y0,x1,y1 = polygon.bounds
        if room.shape_policy != 'limited_recess' and not long_edge_eligible(room,max(x1-x0,y1-y0),contacts):
            errors.append('recess_not_allowed')
    for name,bad in (
        ('reflex_count',metrics['reflex_count'] > limits.max_reflex),
        ('rectangularity',metrics['rectangularity'] < limits.min_rectangularity-tolerance),
        ('notch_depth_ratio',metrics['notch_depth_ratio'] > limits.max_notch_depth_ratio+tolerance),
        ('extra_perimeter',metrics['extra_perimeter'] > limits.max_extra_perimeter+tolerance),
        ('min_segment_ratio',metrics['min_segment_ratio'] < limits.min_segment_ratio-tolerance)):
        if bad:
            errors.append(f'shape_{name}')
    width = room.min_width
    if width is None and not polygon.equals(polygon.envelope):
        width = limits.min_width_ratio*math.sqrt(room.target_area)
    metrics['required_width'] = width
    metrics['width_method'] = 'mitred_clearance_core'
    if width is not None and not clearance_width_ok(polygon,width,tolerance):
        errors.append('width_below_minimum')
    return errors,metrics
