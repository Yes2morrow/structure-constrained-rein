"""结构交互后的初始布局自动修复。

画布/参数表交互移动固定结构（如交通核）或边界后，把与固定结构、边界或其他
智能体冲突的初始矩形推到最近合法位置，并把落在固定结构内/边界外的种子吸附回
初始矩形中心，然后落盘。矩形修复复用 AdaptiveReuseEnv.reset() 的搜索逻辑。
"""
from copy import deepcopy

from core.envs import AdaptiveReuseEnv
from core.envs.structure_geometry import fixed_polygon, normalize_structures
from gui import config_store


def private_rectangle_config(config: dict) -> dict:
    """环境硬约束只作用于独立智能体矩形；剩余公共节点不参与。"""
    result = deepcopy(config)
    result['TargetSpaces'] = [r for r in result['TargetSpaces'] if r.get('role') != 'residual']
    ids = {r['id'] for r in result['TargetSpaces']}
    result['FunctionalRelations'] = [e for e in result.get('FunctionalRelations', []) if e['from'] in ids and e['to'] in ids]
    return result


def apply_env_repairs_to_targets(config: dict, env: AdaptiveReuseEnv) -> list[dict]:
    """把环境 reset() 自动修复后的初始矩形回写到前端配置。"""
    repaired_by_id = {space.space_id: space for space in env.agent_spaces}
    repairs: list[dict] = []
    updated_targets = []
    for item in config.get("TargetSpaces", []):
        updated = deepcopy(item)
        repaired = repaired_by_id.get(str(updated.get("id", "")))
        if repaired is None:
            updated_targets.append(updated)
            continue
        old_rect = list(map(float, updated.get("initial_rect", [repaired.x1, repaired.y1, repaired.x2, repaired.y2])))
        new_rect = [repaired.x1, repaired.y1, repaired.x2, repaired.y2]
        if any(abs(old - new) > 1e-9 for old, new in zip(old_rect, new_rect)):
            if "seed" in updated:
                old_center = ((old_rect[0] + old_rect[2]) / 2.0, (old_rect[1] + old_rect[3]) / 2.0)
                new_center = ((new_rect[0] + new_rect[2]) / 2.0, (new_rect[1] + new_rect[3]) / 2.0)
                updated["seed"] = [
                    float(updated["seed"][0]) + (new_center[0] - old_center[0]),
                    float(updated["seed"][1]) + (new_center[1] - old_center[1]),
                ]
            updated["initial_rect"] = new_rect
            repairs.append({
                "space_id": updated["id"],
                "from_rect": old_rect,
                "to_rect": new_rect,
            })
        updated_targets.append(updated)
    config["TargetSpaces"] = updated_targets
    return repairs


def auto_repair_targets_config(config: dict, seed: int) -> tuple[dict, list[dict], dict]:
    """用环境硬约束自动修复初始智能体布局，返回修复后的整份配置。"""
    repaired_config = deepcopy(config)
    repaired_config["AdaptiveReuseEnvironment"]["randomize_initial"] = False
    env = AdaptiveReuseEnv(private_rectangle_config(repaired_config))
    _, info = env.reset(seed=seed)
    repairs = apply_env_repairs_to_targets(repaired_config, env)
    return repaired_config, repairs, info


def _snap_stray_seeds(config: dict) -> list[dict]:
    """把落在固定结构内或边界外的种子吸附回其初始矩形中心。"""
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union

    building = config['ExistingBuilding']
    boundary = Polygon(building['boundary'])
    polygons = [fixed_polygon(item) for item in normalize_structures(building.get('fixed_objects', []))]
    fixed = unary_union(polygons) if polygons else Polygon()
    fixes = []
    for item in config.get('TargetSpaces', []):
        if item.get('role') == 'residual' or 'seed' not in item:
            continue
        point = Point(float(item['seed'][0]), float(item['seed'][1]))
        if boundary.contains(point) and not fixed.intersects(point):
            continue
        rect = list(map(float, item['initial_rect']))
        old = list(map(float, item['seed']))
        item['seed'] = [(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2]
        fixes.append({'space_id': item['id'], 'from': old, 'to': list(item['seed'])})
    return fixes


def _layout_conflicts(config: dict) -> bool:
    """按环境硬约束快速判断初始布局是否已冲突；无冲突时跳过高开销的 reset 搜索。"""
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    building = config['ExistingBuilding']
    boundary = Polygon(building['boundary'])
    polygons = [fixed_polygon(item) for item in normalize_structures(building.get('fixed_objects', []))]
    fixed = unary_union(polygons) if polygons else Polygon()
    grid = float(config.get('AdaptiveReuseEnvironment', {}).get('grid_size', 0.5))
    shapes = []
    for item in config.get('TargetSpaces', []):
        if item.get('role') == 'residual':
            continue
        rect = list(map(float, item.get('initial_rect', [0, 0, 0, 0])))
        minx, maxx = min(rect[0], rect[2]), max(rect[0], rect[2])
        miny, maxy = min(rect[1], rect[3]), max(rect[1], rect[3])
        shape = box(minx, miny, maxx, maxy)
        if maxx - minx < grid - 1e-9 or maxy - miny < grid - 1e-9:
            return True
        if not boundary.covers(shape):
            return True
        if not fixed.is_empty and shape.intersection(fixed).area > 1e-9:
            return True
        shapes.append(shape)
    for index, shape in enumerate(shapes):
        for other in shapes[index + 1:]:
            if shape.intersection(other).area > 1e-9:
                return True
    return False


def repair_conflicts_and_save(config: dict, config_id: str, seed: int | None = None):
    """结构/边界交互后调用：修复与固定结构冲突的智能体布局并保存。

    返回 (rect_repairs, seed_fixes, error)。error 非 None 表示本次未能完成修复
    （例如找不到合法位置），已保存的结构保持不变，等待用户手动调整。
    """
    if seed is None:
        seed = int(config.get('Training', {}).get('seed', 42))
    try:
        if _layout_conflicts(config):
            repaired_config, rect_repairs, _info = auto_repair_targets_config(config, seed)
        else:
            repaired_config, rect_repairs = deepcopy(config), []
    except Exception as exc:
        return [], [], exc
    seed_fixes = _snap_stray_seeds(repaired_config)
    if rect_repairs or seed_fixes:
        saved = config_store.load_config(config_id)
        saved['TargetSpaces'] = repaired_config['TargetSpaces']
        config_store.save_config(saved, config_id)
        config['TargetSpaces'] = repaired_config['TargetSpaces']
    return rect_repairs, seed_fixes, None


def queue_repair_notice(config_id: str, rect_repairs: list[dict], seed_fixes: list[dict], error=None) -> None:
    """把修复结果写进页面顶部通知；无变化且无错误时不打扰。"""
    if error is not None:
        level, message = 'warning', f'自动修复冲突未完成（可手动调整后重试）：{error}'
    elif rect_repairs or seed_fixes:
        parts = [f"{item['space_id']}: {item['from_rect']} -> {item['to_rect']}" for item in rect_repairs]
        parts += [f"{item['space_id']} 种子: {item['from']} -> {item['to']}" for item in seed_fixes]
        level, message = 'success', '已自动调整与固定结构冲突的房间并保存：' + '；'.join(parts)
    else:
        return
    try:
        import streamlit as st

        st.session_state[f'{config_id}_auto_repair_notice'] = {'level': level, 'message': message}
    except Exception:
        pass
