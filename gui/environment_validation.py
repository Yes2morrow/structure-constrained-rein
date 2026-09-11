from __future__ import annotations

from typing import Optional


def infer_door_direction(door_positions: list[list[float]], tolerance: float = 1e-6) -> Optional[str]:
    """根据门点几何分布推断外门方向。"""
    if len(door_positions) < 2:
        return None

    xs = [float(point[0]) for point in door_positions]
    ys = [float(point[1]) for point in door_positions]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)

    if span_x <= tolerance and span_y <= tolerance:
        return None
    if span_x > span_y:
        return "水平"
    if span_y > span_x:
        return "垂直"
    return None


def validate_door_direction(door_positions: list[list[float]], selected_direction: str) -> dict:
    """校验门点分布与门方向是否一致。"""
    inferred_direction = infer_door_direction(door_positions)
    if len(door_positions) < 2:
        return {
            "valid": False,
            "level": "info",
            "message": "门点数量不足 2 个，当前无法判断门方向。",
            "inferred_direction": inferred_direction,
        }

    if inferred_direction is None:
        return {
            "valid": False,
            "level": "warning",
            "message": "门点分布无法稳定判断方向，请检查门点坐标是否正确。",
            "inferred_direction": inferred_direction,
        }

    if inferred_direction == selected_direction:
        return {
            "valid": True,
            "level": "success",
            "message": f"门点分布与门方向一致，当前判定为{inferred_direction}门。",
            "inferred_direction": inferred_direction,
        }

    return {
        "valid": False,
        "level": "warning",
        "message": (
            f"门点坐标推断为{inferred_direction}门，但当前选择的是{selected_direction}门。"
            f"保存环境时将自动修正为{inferred_direction}。"
        ),
        "inferred_direction": inferred_direction,
    }
