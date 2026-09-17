"""楼层分区结果导出。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from shapely.geometry import Polygon
from core.envs.structure_geometry import fixed_polygon

from .growth import PartitionResult


def _round_coords(coords) -> list[list[float]]:
    return [[round(float(x), 6), round(float(y), 6)] for x, y in coords]


def _polygon_boundary(geometry) -> list[list[float]]:
    if geometry.geom_type != 'Polygon' or not geometry.is_valid or geometry.is_empty:
        raise ValueError('拒绝导出不连通或无效区域，不能丢弃小分量')
    polygon = geometry
    coords = list(polygon.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    return _round_coords(coords)


def _crop_fixed_objects(fixed_objects: list[dict[str, Any]], unit_polygon) -> list[dict[str, Any]]:
    result = []
    for item in fixed_objects:
        original = fixed_polygon(item)
        clipped = original.intersection(unit_polygon)
        if clipped.is_empty or clipped.area <= 1e-9:
            continue
        if clipped.equals(original):
            result.append(deepcopy(item))
            continue
        if clipped.geom_type == "Polygon":
            result.append(
                {
                    "id": item["id"],
                    "type": item.get("type", "fixed"),
                    "polygon": _round_coords(list(clipped.exterior.coords)),
                    "rect": [round(float(value), 6) for value in clipped.bounds],
                    "parametric": False,
                    "source_id": item["id"],
                    "clipped": True,
                }
            )
            continue
        for index, part in enumerate(getattr(clipped, "geoms", []), start=1):
            if part.geom_type != "Polygon" or part.area <= 1e-9:
                continue
            result.append(
                {
                    "id": f"{item['id']}_part{index}",
                    "type": item.get("type", "fixed"),
                    "polygon": _round_coords(list(part.exterior.coords)),
                    "rect": [round(float(value), 6) for value in part.bounds],
                    "parametric": False,
                    "source_id": item["id"],
                    "clipped": True,
                }
            )
    return result


def _intersect_original_spaces(original_spaces: list[dict[str, Any]], unit_polygon) -> list[dict[str, Any]]:
    result = []
    for item in original_spaces:
        polygon = Polygon(_round_coords([
            (item["rect"][0], item["rect"][1]),
            (item["rect"][2], item["rect"][1]),
            (item["rect"][2], item["rect"][3]),
            (item["rect"][0], item["rect"][3]),
        ]))
        if polygon.intersection(unit_polygon).area <= 1e-9:
            continue
        result.append(deepcopy(item))
    return result


def _door_positions(door) -> list[list[float]]:
    return [
        [round(float(door.points[0][0]), 6), round(float(door.points[0][1]), 6)],
        [round(float(door.points[1][0]), 6), round(float(door.points[1][1]), 6)],
    ]


def export_unit_configs(config: dict[str, Any], result: PartitionResult, output_dir: str | Path) -> list[Path]:
    """按户型边界导出下游训练配置。"""
    from .contracts import build_floor_partition_problem
    from .quality import validate_partition
    report=validate_partition(build_floor_partition_problem(config),result)
    if not report['valid']:
        raise ValueError('分户结果未通过独立验收，拒绝导出: '+', '.join(report['errors']))
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    fixed_objects = list(config.get("ExistingBuilding", {}).get("fixed_objects", []))
    original_spaces = list(config.get("ExistingBuilding", {}).get("original_spaces", []))
    doors = {door.unit_id: door for door in result.doors}
    floor_outline = _polygon_boundary(config["ExistingBuilding"]["boundary"] if isinstance(config["ExistingBuilding"]["boundary"], Polygon) else Polygon(config["ExistingBuilding"]["boundary"]))
    written = []
    for unit_id, unit_polygon in result.unit_polygons.items():
        unit_dir = base_dir / unit_id
        unit_dir.mkdir(parents=True, exist_ok=True)
        exported = deepcopy(config)
        exported['ConfigID']=unit_id
        exported['ProjectName']=f"{config.get('ProjectName','Floor')} / {unit_id}"
        exported.setdefault("FloorPartitionResult", {})
        exported["FloorPartitionResult"].update(
            {
                "unit_id": unit_id,
                "unit_boundary": _polygon_boundary(unit_polygon),
                "door_positions": _door_positions(doors[unit_id]),
                "door_edge": doors[unit_id].edge,
                "target_area": round(float(result.target_areas[unit_id]), 6),
                "requested_area": round(float(result.requested_areas[unit_id]), 6),
                "corridor": _polygon_boundary(result.corridor),
                "opening_side": result.opening_side,
                "floor_outline": floor_outline,
            }
        )
        exported["ExistingBuilding"]["boundary"] = _polygon_boundary(unit_polygon)
        # Unit geometry is net free space. Use its gross shell to recover column
        # holes; clipping structures against net space would silently erase them.
        gross=Polygon(unit_polygon.exterior)
        exported["ExistingBuilding"]["fixed_objects"] = _crop_fixed_objects(fixed_objects, gross)
        from shapely.ops import unary_union
        retained=unary_union([fixed_polygon(i) for i in exported['ExistingBuilding']['fixed_objects']])
        for index,hole in enumerate(unit_polygon.interiors):
            void=Polygon(hole).difference(retained)
            if void.area>1e-6:
                parts=[void] if void.geom_type=='Polygon' else list(void.geoms)
                for j,part in enumerate(parts):
                    exported['ExistingBuilding']['fixed_objects'].append(dict(
                        id=f'{unit_id}_void_{index}_{j}',type='fixed',polygon=_round_coords(part.exterior.coords),
                        rect=list(part.bounds),parametric=False))
        exported["ExistingBuilding"]["original_spaces"] = _intersect_original_spaces(original_spaces, unit_polygon)
        exported["ExistingBuilding"]["door_positions"] = _door_positions(doors[unit_id])
        exported["Training"]["ckpt_path"] = ""
        exported['Training']['training_stage']='room_training'
        exported.setdefault('FloorPartition',{})['enabled']=False
        exported.setdefault('SeedGrowth',{})['enabled']=False
        exported['TargetSpaces']=[]
        exported['FunctionalRelations']=[]
        exported.pop('InteriorTrainingEnvironment',None)
        exported['FloorPartitionResult']['quality']=report['units'][unit_id]
        exported['FloorPartitionResult']['interior_program_required']=True
        exported['FloorPartitionResult']['unit_holes']=[_round_coords(h.coords) for h in unit_polygon.interiors]
        unit_path = unit_dir / "config.yaml"
        unit_path.write_text(yaml.safe_dump(exported, allow_unicode=True, sort_keys=False), encoding="utf-8")
        written.append(unit_path)
    summary = {
        "units": [
            {
                "unit_id": unit_id,
                "area": round(float(unit_polygon.area), 6),
                "target_area": round(float(result.target_areas[unit_id]), 6),
                "requested_area": round(float(result.requested_areas[unit_id]), 6),
            }
            for unit_id, unit_polygon in result.unit_polygons.items()
        ],
        "area_scale": round(float(result.area_scale), 6),
    }
    (base_dir / "summary.yaml").write_text(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return written
