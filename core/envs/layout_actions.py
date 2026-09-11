"""布局类环境（住区楼栋/既有建筑功能空间）共享的动作定义与几何操作。"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import yaml


ACTION_NAMES = (
    "north", "south", "west", "east", "stay",
    "north_expand", "north_shrink", "south_expand", "south_shrink",
    "west_expand", "west_shrink", "east_expand", "east_shrink",
)


def rect_changes_for_action(
    x1: float, y1: float, x2: float, y2: float,
    action: int, move_step: float, resize_step: float,
) -> dict[str, float]:
    """计算动作对应的矩形坐标变化，返回可直接用于 dataclass.replace 的字段字典。"""
    changes = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    if action == 0:
        changes["y1"] += move_step; changes["y2"] += move_step
    elif action == 1:
        changes["y1"] -= move_step; changes["y2"] -= move_step
    elif action == 2:
        changes["x1"] -= move_step; changes["x2"] -= move_step
    elif action == 3:
        changes["x1"] += move_step; changes["x2"] += move_step
    elif action == 5:
        changes["y2"] += resize_step
    elif action == 6:
        changes["y2"] -= resize_step
    elif action == 7:
        changes["y1"] -= resize_step
    elif action == 8:
        changes["y1"] += resize_step
    elif action == 9:
        changes["x1"] -= resize_step
    elif action == 10:
        changes["x1"] += resize_step
    elif action == 11:
        changes["x2"] += resize_step
    elif action == 12:
        changes["x2"] -= resize_step
    return changes


def load_layout_config(config_path: str | Path) -> dict[str, Any]:
    """读取布局环境的 YAML 配置。"""
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)
