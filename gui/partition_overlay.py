"""Optional read-only pink inference layer, separate from editable inputs."""
from common.plan_styles import DOOR_COLOR
from core.floor_partition import run_residential_floor_partition
from core.floor_partition.walls import wall_geometry
from gui.structure_canvas import viewport


def partition_overlay(config,width,height,visible=False,transparency=0):
    if not visible or config.get('Training',{}).get('training_stage')!='floor_partition':
        return []
    problem,result=run_residential_floor_partition(config)
    physical=wall_geometry(problem,result.corridor,result.unit_polygons,result.doors)
    scale,ox,oy=viewport(config['ExistingBuilding']['boundary'],width,height)
    objects=[]
    for geometry,color in ((physical.corridor,'#f9a8d4'),(physical.solid,'#db2777')):
        parts=[geometry] if geometry.geom_type=='Polygon' else list(getattr(geometry,'geoms',[]))
        for p in parts:
            if p.is_empty or p.geom_type!='Polygon': continue
            path=[]
            for ring in [p.exterior,*p.interiors]:
                for i,(x,y) in enumerate(list(ring.coords)[:-1]):
                    path.append(['M' if i==0 else 'L',ox+x*scale,oy-y*scale])
                path.append(['Z'])
            objects.append(dict(type='path',path=path,fill=color,fillRule='evenodd',strokeWidth=0,
                                opacity=.65*(1-transparency/100),selectable=False,evented=False))
    for d in result.doors:
        points=[dict(x=ox+x*scale,y=oy-y*scale) for x,y in d.points]
        objects.append(dict(type='polyline',points=points,left=min(p['x'] for p in points),
            top=min(p['y'] for p in points),stroke=DOOR_COLOR,fill='',strokeWidth=4,
            opacity=1-transparency/100,selectable=False,evented=False))
    return objects
