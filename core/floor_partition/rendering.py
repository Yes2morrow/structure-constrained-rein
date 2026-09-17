"""Shared, training-independent floor partition rendering."""
from common.plan_styles import DOOR_COLOR
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry.polygon import orient
from shapely.geometry import Polygon
from core.envs.structure_geometry import fixed_polygon
from .contracts import build_floor_partition_problem
from .walls import wall_geometry
from common.plan_styles import PARTITION_FILL, STRUCTURE_FILL, CORE_FILL, CORE_EDGE, CORE_HATCH

def _polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for part in geometry.geoms if part.geom_type == "Polygon"]
    return []


def _draw_geometry(axis, geometry, facecolor, edgecolor, linewidth=1.5, alpha=0.7, hatch=None, zorder=1):
    for polygon in _polygon_parts(geometry):
        polygon=orient(polygon,sign=1.)
        vertices=[]; codes=[]
        for ring in [polygon.exterior,*polygon.interiors]:
            coords=list(ring.coords)
            vertices.extend(coords)
            codes.extend([MplPath.MOVETO]+[MplPath.LINETO]*(len(coords)-2)+[MplPath.CLOSEPOLY])
        axis.add_patch(
            PathPatch(
                MplPath(vertices,codes),
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                hatch=hatch,
                zorder=zorder,
            )
        )


def render_partition(config: dict, result, output_dir: Path) -> None:
    boundary = Polygon(config["ExistingBuilding"]["boundary"])
    figure, axis = plt.subplots(figsize=(10, 6.5))
    from core.floor_partition.quality import facade
    from core.floor_partition.structured import lines
    problem=build_floor_partition_problem(config)
    physical=wall_geometry(problem,result.corridor,result.unit_polygons,result.doors)
    daylight_facade=facade(problem)
    bx, by = boundary.exterior.xy
    axis.plot(bx, by, color="#0f172a", linewidth=2.4, zorder=5)

    colors = ["#90caf9", "#a5d6a7", "#ffcc80", "#ce93d8", "#80cbc4", "#ef9a9a", "#b39ddb", "#f48fb1"]
    for index, (unit_id, geometry) in enumerate(sorted(physical.units.items())):
        color = colors[index % len(colors)]
        _draw_geometry(axis, geometry, color, "#1e3a5f", linewidth=1.8, alpha=0.82, zorder=3)
        point = geometry.representative_point()
        axis.text(
            point.x,
            point.y,
            f"{unit_id}\n{geometry.area:.1f}/{result.target_areas[unit_id]:.1f} m2\nFacade: {geometry.boundary.intersection(daylight_facade).length:.1f} m",
            ha="center",
            va="center",
            fontsize=8.5,
            zorder=6,
        )

    _draw_geometry(axis, physical.corridor, "#009df5", "#0571ad", linewidth=.8, alpha=.85, zorder=2)
    _draw_geometry(axis, physical.solid, PARTITION_FILL, PARTITION_FILL, linewidth=.2, alpha=1., zorder=5)
    _draw_geometry(axis, physical.thresholds, "white", "none", linewidth=0, alpha=1., zorder=6)

    for item in config["ExistingBuilding"].get("fixed_objects", []):
        core=item.get('type') in ('traffic_core','core')
        _draw_geometry(axis, fixed_polygon(item), CORE_FILL if core else STRUCTURE_FILL,
                       CORE_EDGE if core else STRUCTURE_FILL,linewidth=.8,alpha=1.,
                       hatch=CORE_HATCH if core else None,zorder=5)

    for door in result.doors:
        xs = [door.points[0][0], door.points[1][0]]
        ys = [door.points[0][1], door.points[1][1]]
        axis.plot(xs, ys, color=DOOR_COLOR, linewidth=3.0, solid_capstyle="butt", zorder=7)
        axis.annotate(door.unit_id, door.center, textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5)

    for line in lines(daylight_facade.difference(result.corridor)):
        axis.plot(*line.xy,color='#168354',linewidth=3,zorder=6)
    axis.set_title(f"Net floor areas | walls: {problem.profile.wall_thickness:.2f} m | corridor: {physical.corridor.area:.2f} m2\nGrey: partitions; dark: existing structure; hatching: traffic core")
    axis.set_aspect("equal", adjustable="box")
    axis.autoscale_view()
    figure.tight_layout()
    figure.savefig(output_dir / "partition_preview.png", dpi=180)
    figure.savefig(output_dir / "partition_preview.svg")
    plt.close(figure)

