"""楼层联合分区入口：图策略或普通局部搜索，独立验收后导出。"""

from __future__ import annotations

# Allow direct script execution from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PatchPolygon
import yaml
from shapely.geometry import Polygon

from common.config_manager import get_config_path
from common.project_paths import RESULTS_DIR
from core.envs.structure_geometry import fixed_polygon
from core.floor_partition import export_unit_configs, run_residential_floor_partition, build_floor_partition_problem
from core.floor_partition.quality import validate_partition
from core.floor_partition.training import train_partition_policy
from common.project_paths import STOP_REQUEST_FILE


def _polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for part in geometry.geoms if part.geom_type == "Polygon"]
    return []


def _draw_geometry(axis, geometry, facecolor, edgecolor, linewidth=1.5, alpha=0.7, hatch=None, zorder=1):
    for polygon in _polygon_parts(geometry):
        axis.add_patch(
            PatchPolygon(
                list(polygon.exterior.coords),
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                hatch=hatch,
                zorder=zorder,
            )
        )


def _render_partition(config: dict, result, output_dir: Path) -> None:
    boundary = Polygon(config["ExistingBuilding"]["boundary"])
    figure, axis = plt.subplots(figsize=(10, 6.5))
    from core.floor_partition.quality import facade
    from core.floor_partition.structured import lines
    daylight_facade=facade(build_floor_partition_problem(config))
    bx, by = boundary.exterior.xy
    axis.plot(bx, by, color="#0f172a", linewidth=2.4, zorder=5)

    colors = ["#90caf9", "#a5d6a7", "#ffcc80", "#ce93d8", "#80cbc4", "#ef9a9a", "#b39ddb", "#f48fb1"]
    for index, (unit_id, geometry) in enumerate(sorted(result.unit_polygons.items())):
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

    _draw_geometry(axis, result.corridor, "#f8c471", "#b9770e", linewidth=1.8, alpha=0.7, hatch="//", zorder=2)

    for item in config["ExistingBuilding"].get("fixed_objects", []):
        _draw_geometry(axis, fixed_polygon(item), "#455a64", "#263238", linewidth=1.2, alpha=0.75, zorder=4)

    for door in result.doors:
        xs = [door.points[0][0], door.points[1][0]]
        ys = [door.points[0][1], door.points[1][1]]
        axis.plot(xs, ys, color="#c62828", linewidth=3.0, solid_capstyle="round", zorder=7)
        axis.annotate(door.unit_id, door.center, textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5)

    for line in lines(daylight_facade.difference(result.corridor)):
        axis.plot(*line.xy,color='#168354',linewidth=3,zorder=6)
    axis.set_title(f"Residential Floor Partition | opening: {result.opening_side}\nGreen: available external facade; red: unit doors")
    axis.set_aspect("equal", adjustable="box")
    axis.autoscale_view()
    figure.tight_layout()
    figure.savefig(output_dir / "partition_preview.png", dpi=180)
    figure.savefig(output_dir / "partition_preview.svg")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="住宅楼层分区规则链")
    parser.add_argument("--config-id", default="retrofit")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument('--episodes',type=int,default=None)
    parser.add_argument('--no-rl',action='store_true',help='仅运行结构约束搜索与验收')
    parser.add_argument('--output-dir',type=Path,default=None)
    parser.add_argument('--search-steps',type=int,default=None,help='普通联合搜索的修改次数')
    parser.add_argument('--checkpoint',type=Path,default=None,help='加载联合PPO权重，仅执行推理')
    parser.add_argument('--stop-file',type=Path,default=Path(STOP_REQUEST_FILE))
    return parser.parse_args()


def main():
    args = parse_args()
    matplotlib.use("Agg", force=True)
    config_path = args.config or Path(get_config_path(args.config_id))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    output_dir = args.output_dir or Path(RESULTS_DIR) / "floor_partition" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f'结果目录: {output_dir}',flush=True)
    problem=build_floor_partition_problem(config)
    rl_summary=None
    if args.checkpoint:
        import torch
        torch.set_num_threads(1)
        from core.floor_partition.training import GraphPolicy, policy_rollout
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if checkpoint.get('schema')!='floor-partition-joint-ppo-v2':
            raise ValueError('checkpoint 不是联合分户PPO模型')
        model=GraphPolicy(); model.load_state_dict(checkpoint['state_dict']); model.eval()
        result,quality,history=policy_rollout(problem,model,steps=args.search_steps or 8)
        (output_dir/'search_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    elif not args.no_rl and config.get('FloorPartition',{}).get('rl',{}).get('enabled',True):
        result,quality,rl_summary=train_partition_policy(problem,config,output_dir,args.episodes,args.stop_file)
    else:
        from core.floor_partition.joint import search_joint
        if args.search_steps is not None and args.search_steps<1: raise ValueError('search-steps must be positive')
        result,quality,search_summary=search_joint(problem,steps=args.search_steps,stop_file=args.stop_file)
        (output_dir/'search_history.json').write_text(json.dumps(search_summary,indent=2),encoding='utf-8')
    (output_dir/'validation.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding='utf-8')
    export_paths = export_unit_configs(config, result, output_dir / "exports")
    _render_partition(config, result, output_dir)

    summary = {
        "program_type": problem.program_type,
        "unit_count": len(problem.targets),
        "opening_side": result.opening_side,
        "area_scale": result.area_scale,
        "doors": [
            {
                "unit_id": door.unit_id,
                "door_positions": [
                    [round(float(value), 6) for value in door.points[0]],
                    [round(float(value), 6) for value in door.points[1]],
                ],
                "edge": door.edge,
            }
            for door in result.doors
        ],
        "units": [
            {
                "unit_id": unit_id,
                "area": round(float(geometry.area), 6),
                "target_area": round(float(result.target_areas[unit_id]), 6),
            }
            for unit_id, geometry in sorted(result.unit_polygons.items())
        ],
        "exported_configs": [str(path) for path in export_paths],
        'quality':quality,
        'rl_training':rl_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"楼层分区结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
