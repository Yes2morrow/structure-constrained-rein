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
import yaml

from common.config_manager import get_config_path
from common.project_paths import RESULTS_DIR
from core.floor_partition import export_unit_configs, build_floor_partition_problem
from core.floor_partition.rendering import render_partition
from common.project_paths import STOP_REQUEST_FILE


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
        if checkpoint.get('schema')!='floor-partition-joint-ppo-v3':
            raise ValueError('需要含墙厚、中轴和联合门厅动作的 v3 模型；旧权重需重新训练')
        model=GraphPolicy(); model.load_state_dict(checkpoint['state_dict']); model.eval()
        result,quality,history=policy_rollout(problem,model,steps=args.search_steps or 8)
        (output_dir/'search_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    elif not args.no_rl and config.get('FloorPartition',{}).get('rl',{}).get('enabled',True):
        from core.floor_partition.training import train_partition_policy
        result,quality,rl_summary=train_partition_policy(problem,config,output_dir,args.episodes,args.stop_file)
    else:
        from core.floor_partition.joint import search_joint
        if args.search_steps is not None and args.search_steps<1: raise ValueError('search-steps must be positive')
        result,quality,search_summary=search_joint(problem,steps=args.search_steps,stop_file=args.stop_file)
        (output_dir/'search_history.json').write_text(json.dumps(search_summary,indent=2),encoding='utf-8')
    (output_dir/'validation.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding='utf-8')
    export_paths = export_unit_configs(config, result, output_dir / "exports")
    render_partition(config, result, output_dir)

    summary = {
        'source_kind': 'policy_inference' if args.checkpoint else 'rl_training' if rl_summary else 'search',
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
                "area": round(float(quality['units'][unit_id]['area']), 6),
                "territory_area": round(float(geometry.area), 6),
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
