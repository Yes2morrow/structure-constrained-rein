"""既有建筑空间适应性转换的多智能体强化学习环境。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any

import gym
from gym import spaces
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from .layout_actions import ACTION_NAMES, load_layout_config, rect_changes_for_action
from .structure_geometry import fixed_polygon, normalize_structures


AREA_BUDGET_OBJECT_TYPES = {"traffic_core", "retained_circulation"}


def calculate_area_budget(boundary: list[tuple[float, float]] | list[list[float]], fixed_objects: list[dict[str, Any]]) -> dict[str, float]:
    """按“空间型约束”估算可分配面积，不把柱墙这类细碎结构计入预算扣减。"""
    boundary_polygon = Polygon([tuple(map(float, point)) for point in boundary])
    if not boundary_polygon.is_valid or boundary_polygon.area <= 0:
        raise ValueError("ExistingBuilding.boundary 必须是有效且面积大于 0 的多边形")
    blocked_parts = []
    for item in normalize_structures(fixed_objects):
        if str(item.get("type")) not in AREA_BUDGET_OBJECT_TYPES:
            continue
        blocked = fixed_polygon(item).intersection(boundary_polygon)
        if not blocked.is_empty and blocked.area > 1e-9:
            blocked_parts.append(blocked)
    blocked_area = unary_union(blocked_parts).area if blocked_parts else 0.0
    allocatable_area = max(boundary_polygon.area - blocked_area, 0.0)
    return {
        "boundary_area": float(boundary_polygon.area),
        "constraint_area": float(blocked_area),
        "allocatable_area": float(allocatable_area),
    }


def summarize_target_area_budget(config: dict[str, Any]) -> dict[str, float]:
    """汇总当前目标空间的最小/目标/最大面积，并和可分配面积做对比。"""
    budget = calculate_area_budget(
        config["ExistingBuilding"]["boundary"],
        config["ExistingBuilding"].get("fixed_objects", []),
    )
    targets = list(config.get("TargetSpaces", []))
    min_sum = sum(float(item.get("area_range", [item.get("target_area", 0.0), item.get("target_area", 0.0)])[0]) for item in targets)
    target_sum = sum(float(item.get("target_area", 0.0)) for item in targets)
    max_sum = sum(float(item.get("area_range", [item.get("target_area", 0.0), item.get("target_area", 0.0)])[1]) for item in targets)
    budget.update({
        "min_area_sum": float(min_sum),
        "target_area_sum": float(target_sum),
        "max_area_sum": float(max_sum),
        "max_area_gap": float(budget["allocatable_area"] - max_sum),
    })
    return budget


def redistribute_target_max_areas(config: dict[str, Any]) -> list[dict[str, Any]]:
    """把所有智能体的最大面积按 target_area 比例重新分配到可分配面积上。"""
    updated = []
    budget = summarize_target_area_budget(config)
    targets = list(config.get("TargetSpaces", []))
    lower_bounds = []
    weights = []
    for item in targets:
        target_area = float(item.get("target_area", 0.0))
        min_area = float(item.get("area_range", [target_area, target_area])[0])
        lower_bounds.append(max(min_area, target_area))
        weights.append(max(target_area, 0.0))
    lower_sum = sum(lower_bounds)
    if budget["allocatable_area"] + 1e-9 < lower_sum:
        raise ValueError(f"可分配面积 {budget['allocatable_area']:.2f} 小于各空间最低可接受最大面积总和 {lower_sum:.2f}")
    extra_area = max(budget["allocatable_area"] - lower_sum, 0.0)
    weight_sum = sum(weights) or float(len(weights) or 1)
    for item, lower_bound, weight in zip(targets, lower_bounds, weights):
        updated_item = dict(item)
        current = list(updated_item.get("area_range", [updated_item.get("target_area", 0.0), updated_item.get("target_area", 0.0)]))
        max_area = lower_bound + extra_area * ((weight or 1.0) / weight_sum)
        updated_item["area_range"] = [float(current[0]), float(max_area)]
        updated.append(updated_item)
    return updated


@dataclass
class FunctionalSpace:
    """一个目标功能空间智能体，采用与规则网格对齐的轴对齐矩形。"""

    space_id: str
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    target_area: float
    min_area: float
    max_area: float
    aspect_min: float
    aspect_max: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def aspect_ratio(self) -> float:
        return max(self.width, self.height) / max(min(self.width, self.height), 1e-9)

    def polygon(self) -> Polygon:
        return box(self.x1, self.y1, self.x2, self.y2)

    def copy(self, **changes) -> "FunctionalSpace":
        return replace(self, **changes)


class AdaptiveReuseEnv(gym.Env):
    """从既有平面出发，在固定结构约束下调整多个目标功能空间。"""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        building = config["ExistingBuilding"]
        environment = config["AdaptiveReuseEnvironment"]
        self.boundary = [tuple(map(float, point)) for point in building["boundary"]]
        self.boundary_polygon = Polygon(self.boundary)
        if not self.boundary_polygon.is_valid or self.boundary_polygon.area <= 0:
            raise ValueError("ExistingBuilding.boundary 必须是有效且面积大于 0 的多边形")

        self.grid_size = float(environment.get("grid_size", 1.0))
        if self.grid_size <= 0:
            raise ValueError("AdaptiveReuseEnvironment.grid_size 必须大于 0")
        self.move_step = float(environment.get("move_step", self.grid_size))
        self.resize_step = float(environment.get("resize_step", self.grid_size))
        self.max_steps = int(config.get("Training", {}).get("max_steps", 300))
        self.success_patience = int(environment.get("success_patience", 8))
        self.area_tolerance = float(environment.get("area_tolerance", 0.12))
        self.adjacency_distance = float(environment.get("adjacency_distance", self.grid_size))
        self.separation_distance = float(environment.get("separation_distance", 2 * self.grid_size))
        self.randomize_initial = bool(environment.get("randomize_initial", True))
        self.initial_jitter_steps = int(environment.get("initial_jitter_steps", 1))

        self.original_spaces = {
            str(item["id"]): box(*map(float, item["rect"]))
            for item in building.get("original_spaces", [])
        }
        self.fixed_objects = normalize_structures(building.get("fixed_objects", []))
        fixed_polygons = [fixed_polygon(item) for item in self.fixed_objects]
        self.fixed_union = unary_union(fixed_polygons) if fixed_polygons else Polygon()
        self.area_budget = calculate_area_budget(self.boundary, self.fixed_objects)

        self.space_specs = list(config["TargetSpaces"])
        if not self.space_specs:
            raise ValueError("TargetSpaces 至少需要一个目标功能空间")
        ids = [str(item["id"]) for item in self.space_specs]
        if len(ids) != len(set(ids)):
            raise ValueError("TargetSpaces.id 不可重复")
        self.num_agents = len(self.space_specs)
        self.id_to_index = {space_id: index for index, space_id in enumerate(ids)}
        self.relations = list(config.get("FunctionalRelations", []))
        self.reward_weights = {
            key: float(value) for key, value in config.get("RewardWeights", {}).items()
        }
        defaults = {
            "area": 3.0, "shape": 1.0, "adjacency": 1.5, "separation": 1.0,
            "structure_alignment": 0.8, "original_reuse": 1.5,
            "intervention": 1.0, "invalid_action": 4.0, "team_coordination": 0.5,
        }
        for key, value in defaults.items():
            self.reward_weights.setdefault(key, value)

        self.action_space = spaces.MultiDiscrete([len(ACTION_NAMES)] * self.num_agents)
        # 14 个自身/全局特征，每个其他智能体增加 dx、dy、距离和功能关系 4 个特征。
        self.local_state_size = 14 + 4 * (self.num_agents - 1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents, self.local_state_size),
            dtype=np.float32,
        )
        self.agent_spaces: list[FunctionalSpace] = []
        self.total_steps = 0
        self.success_steps = 0
        self.invalid_action_count = 0
        self.last_invalid_actions = [False] * self.num_agents
        self.last_metrics: dict[str, float] = {}
        self._last_figure = None
        self.initial_repairs: list[dict[str, Any]] = []

    def _make_space(self, spec: dict[str, Any]) -> FunctionalSpace:
        x1, y1, x2, y2 = map(float, spec["initial_rect"])
        target_area = float(spec["target_area"])
        area_range = spec.get("area_range", [target_area * 0.85, target_area * 1.15])
        aspect_range = spec.get("aspect_range", [1.0, 2.5])
        return FunctionalSpace(
            str(spec["id"]), str(spec.get("name", spec["id"])), x1, y1, x2, y2,
            target_area, float(area_range[0]), float(area_range[1]),
            float(aspect_range[0]), float(aspect_range[1]),
        )

    def reset(self, *, seed=None, options=None, **kwargs):
        super().reset(seed=seed)
        self.total_steps = 0
        self.success_steps = 0
        self.invalid_action_count = 0
        self.last_invalid_actions = [False] * self.num_agents
        self.initial_repairs = []
        self.agent_spaces = [self._make_space(spec) for spec in self.space_specs]
        if self.randomize_initial and self.initial_jitter_steps > 0:
            rng = np.random.default_rng(seed)
            for index in range(self.num_agents):
                for _ in range(self.initial_jitter_steps):
                    action = int(rng.integers(0, 5))
                    candidate = self._candidate_for_action(self.agent_spaces[index], action)
                    if self._is_hard_valid(candidate, index):
                        self.agent_spaces[index] = candidate
        self._repair_initial_layout()
        self._validate_initial_layout()
        self.last_metrics = self.calculate_metrics()
        return self.get_state(), self._build_info()

    def _validate_initial_layout(self) -> None:
        for index, space in enumerate(self.agent_spaces):
            if not self._is_hard_valid(space, index):
                raise ValueError(f"目标空间 {space.space_id} 的初始矩形违反边界、固定构件或空间冲突约束")

    def _space_signature(self, space: FunctionalSpace) -> tuple[float, float, float, float]:
        """用量化坐标去重搜索状态，避免自动修复陷入重复遍历。"""
        return tuple(round(value, 9) for value in (space.x1, space.y1, space.x2, space.y2))

    def _repair_initial_layout(self) -> None:
        """在每轮开始前尝试把非法初始矩形推到最近的合法状态。"""
        for index, space in enumerate(self.agent_spaces):
            if self._is_hard_valid(space, index):
                continue
            repaired, steps = self._find_nearest_valid_space(space, index)
            if repaired is None:
                continue
            self.agent_spaces[index] = repaired
            self.initial_repairs.append({
                "space_id": space.space_id,
                "from_rect": [space.x1, space.y1, space.x2, space.y2],
                "to_rect": [repaired.x1, repaired.y1, repaired.x2, repaired.y2],
                "action_steps": steps,
            })

    def _find_nearest_valid_space(
        self, start: FunctionalSpace, agent_index: int, max_depth: int = 12, max_nodes: int = 12000
    ) -> tuple[FunctionalSpace | None, int]:
        """先做全局平移搜索，再按动作步数做局部广搜，找到最近的合法初始状态。"""
        translated, translation_steps = self._find_valid_translation(start, agent_index)
        if translated is not None:
            return translated, translation_steps
        queue = deque([(start, 0)])
        visited = {self._space_signature(start)}
        seen_nodes = 0
        # 先平移再缩放，优先保持原始尺度和位置语义。
        search_actions = (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12)
        while queue and seen_nodes < max_nodes:
            current, depth = queue.popleft()
            seen_nodes += 1
            if depth > 0 and self._is_hard_valid(current, agent_index):
                return current, depth
            if depth >= max_depth:
                continue
            for action in search_actions:
                candidate = self._candidate_for_action(current, action)
                if candidate.width < self.grid_size or candidate.height < self.grid_size:
                    continue
                signature = self._space_signature(candidate)
                if signature in visited:
                    continue
                visited.add(signature)
                queue.append((candidate, depth + 1))
        return None, -1

    def _find_valid_translation(self, start: FunctionalSpace, agent_index: int) -> tuple[FunctionalSpace | None, int]:
        """优先保持房间尺度不变，只通过整体平移寻找最近合法位置。"""
        min_x, min_y, max_x, max_y = self.boundary_polygon.bounds
        min_dx = math.ceil((min_x - start.x1) / self.move_step)
        max_dx = math.floor((max_x - start.x2) / self.move_step)
        min_dy = math.ceil((min_y - start.y1) / self.move_step)
        max_dy = math.floor((max_y - start.y2) / self.move_step)
        offsets = [
            (dx, dy)
            for dx in range(min_dx, max_dx + 1)
            for dy in range(min_dy, max_dy + 1)
            if dx or dy
        ]
        offsets.sort(key=lambda item: (abs(item[0]) + abs(item[1]), max(abs(item[0]), abs(item[1])), abs(item[0]), abs(item[1])))
        for dx_steps, dy_steps in offsets:
            candidate = start.copy(
                x1=start.x1 + dx_steps * self.move_step,
                y1=start.y1 + dy_steps * self.move_step,
                x2=start.x2 + dx_steps * self.move_step,
                y2=start.y2 + dy_steps * self.move_step,
            )
            if self._is_hard_valid(candidate, agent_index):
                return candidate, abs(dx_steps) + abs(dy_steps)
        return None, -1

    def _candidate_for_action(self, space: FunctionalSpace, action: int) -> FunctionalSpace:
        changes = rect_changes_for_action(
            space.x1, space.y1, space.x2, space.y2,
            action, self.move_step, self.resize_step,
        )
        return space.copy(**changes)

    def _is_hard_valid(self, candidate: FunctionalSpace, agent_index: int) -> bool:
        if candidate.width < self.grid_size or candidate.height < self.grid_size:
            return False
        polygon = candidate.polygon()
        if not self.boundary_polygon.covers(polygon):
            return False
        if not self.fixed_union.is_empty and polygon.intersection(self.fixed_union).area > 1e-9:
            return False
        return all(
            other_index == agent_index or polygon.intersection(other.polygon()).area <= 1e-9
            for other_index, other in enumerate(self.agent_spaces)
        )

    def _get_allow_actions(self) -> list[list[int]]:
        allowed = []
        for index, current in enumerate(self.agent_spaces):
            choices = [4]
            for action in range(len(ACTION_NAMES)):
                if action != 4 and self._is_hard_valid(self._candidate_for_action(current, action), index):
                    choices.append(action)
            allowed.append(sorted(choices))
        return allowed

    def step(self, actions):
        if len(actions) != self.num_agents:
            raise ValueError(f"需要 {self.num_agents} 个动作，实际收到 {len(actions)} 个")
        self.total_steps += 1
        allowed = self._get_allow_actions()
        invalid = [False] * self.num_agents
        proposals = list(self.agent_spaces)
        for index, action_value in enumerate(actions):
            action = int(action_value)
            if action not in allowed[index]:
                invalid[index] = True
                continue
            proposals[index] = self._candidate_for_action(self.agent_spaces[index], action)

        # 同步动作可能产生相互冲突；所有涉及冲突的提案都回退，避免智能体顺序偏置。
        conflict_indices: set[int] = set()
        for first in range(self.num_agents):
            for second in range(first + 1, self.num_agents):
                if proposals[first].polygon().intersection(proposals[second].polygon()).area > 1e-9:
                    conflict_indices.update((first, second))
        for index in conflict_indices:
            proposals[index] = self.agent_spaces[index]
            invalid[index] = True

        self.agent_spaces = proposals
        self.last_invalid_actions = invalid
        self.invalid_action_count += sum(invalid)
        rewards, components = self._calculate_rewards(invalid)
        self.last_metrics = self.calculate_metrics()
        success_now = self._meets_success_condition(self.last_metrics)
        self.success_steps = self.success_steps + 1 if success_now else 0
        success = self.success_steps >= self.success_patience
        truncated = self.total_steps >= self.max_steps
        done = success or truncated
        info = self._build_info()
        info.update({
            "reward_components": components,
            "done_list": [done] * self.num_agents,
            "terminated_by_success": success,
            "truncated_by_max_steps": truncated and not success,
        })
        return self.get_state(), rewards, done, info

    def _relation_value(self, first_id: str, second_id: str) -> float:
        for relation in self.relations:
            pair = {str(relation.get("from")), str(relation.get("to"))}
            if pair == {first_id, second_id}:
                relation_type = str(relation.get("type", "none"))
                if relation_type == "adjacent":
                    return 1.0
                if relation_type == "separate":
                    return -1.0
                return 0.0
        return 0.0

    def _original_polygon(self, space: FunctionalSpace) -> Polygon:
        return self.original_spaces.get(space.space_id, Polygon())

    @staticmethod
    def _iou(first: Polygon, second: Polygon) -> float:
        if first.is_empty or second.is_empty:
            return 0.0
        union_area = first.union(second).area
        return first.intersection(second).area / max(union_area, 1e-9)

    def _grid_alignment_score(self, space: FunctionalSpace) -> float:
        min_x, min_y, _, _ = self.boundary_polygon.bounds
        values = [space.x1 - min_x, space.x2 - min_x, space.y1 - min_y, space.y2 - min_y]
        residuals = [abs(value / self.grid_size - round(value / self.grid_size)) for value in values]
        return max(0.0, 1.0 - 2.0 * float(np.mean(residuals)))

    def _space_scores(self, index: int) -> dict[str, float]:
        space = self.agent_spaces[index]
        polygon = space.polygon()
        area_error = abs(space.area - space.target_area) / max(space.target_area, 1e-9)
        area_score = 1.0 - min(area_error, 1.5)
        if space.aspect_min <= space.aspect_ratio <= space.aspect_max:
            shape_score = 1.0
        else:
            nearest = min(abs(space.aspect_ratio - space.aspect_min), abs(space.aspect_ratio - space.aspect_max))
            shape_score = max(-1.0, 1.0 - nearest / max(space.aspect_max - space.aspect_min, 0.5))

        adjacent_values, separation_values = [], []
        for relation in self.relations:
            first, second = str(relation.get("from")), str(relation.get("to"))
            if space.space_id not in (first, second):
                continue
            other_id = second if space.space_id == first else first
            if other_id not in self.id_to_index:
                continue
            distance = polygon.distance(self.agent_spaces[self.id_to_index[other_id]].polygon())
            if relation.get("type") == "adjacent":
                adjacent_values.append(1.0 - min(distance / max(self.adjacency_distance, 1e-9), 2.0))
            elif relation.get("type") == "separate":
                separation_values.append(min(distance / max(self.separation_distance, 1e-9), 1.0))

        original = self._original_polygon(space)
        reuse_score = self._iou(polygon, original)
        intervention = polygon.symmetric_difference(original).area / max(original.area, polygon.area, 1e-9)
        return {
            "area": area_score,
            "shape": shape_score,
            "adjacency": float(np.mean(adjacent_values)) if adjacent_values else 0.0,
            "separation": float(np.mean(separation_values)) if separation_values else 0.0,
            "structure_alignment": self._grid_alignment_score(space),
            "original_reuse": reuse_score,
            "intervention": intervention,
        }

    def _calculate_rewards(self, invalid: list[bool]):
        scores = [self._space_scores(index) for index in range(self.num_agents)]
        team_score = float(np.mean([
            (score["area"] + score["shape"] + score["original_reuse"]) / 3.0
            for score in scores
        ]))
        rewards, all_components = [], []
        for index, score in enumerate(scores):
            components = {
                "area": self.reward_weights["area"] * score["area"],
                "shape": self.reward_weights["shape"] * score["shape"],
                "adjacency": self.reward_weights["adjacency"] * score["adjacency"],
                "separation": self.reward_weights["separation"] * score["separation"],
                "structure_alignment": self.reward_weights["structure_alignment"] * score["structure_alignment"],
                "original_reuse": self.reward_weights["original_reuse"] * score["original_reuse"],
                "intervention": -self.reward_weights["intervention"] * score["intervention"],
                "invalid_action": -self.reward_weights["invalid_action"] if invalid[index] else 0.0,
                "team_coordination": self.reward_weights["team_coordination"] * team_score,
            }
            all_components.append(components)
            rewards.append(float(sum(components.values())))
        return rewards, all_components

    def calculate_metrics(self) -> dict[str, float]:
        scores = [self._space_scores(index) for index in range(self.num_agents)]
        area_compliant = sum(
            abs(space.area - space.target_area) / max(space.target_area, 1e-9) <= self.area_tolerance
            for space in self.agent_spaces
        ) / self.num_agents
        shape_compliant = sum(
            space.aspect_min <= space.aspect_ratio <= space.aspect_max for space in self.agent_spaces
        ) / self.num_agents
        hard_conflicts = 0
        for index, space in enumerate(self.agent_spaces):
            polygon = space.polygon()
            hard_conflicts += int(not self.boundary_polygon.covers(polygon))
            hard_conflicts += int(
                not self.fixed_union.is_empty and polygon.intersection(self.fixed_union).area > 1e-9
            )
            hard_conflicts += sum(
                polygon.intersection(other.polygon()).area > 1e-9
                for other in self.agent_spaces[index + 1:]
            )
        return {
            "area_compliance": float(area_compliant),
            "shape_compliance": float(shape_compliant),
            "adjacency_score": float(np.mean([score["adjacency"] for score in scores])),
            "separation_score": float(np.mean([score["separation"] for score in scores])),
            "original_reuse": float(np.mean([score["original_reuse"] for score in scores])),
            "intervention_ratio": float(np.mean([score["intervention"] for score in scores])),
            "structure_alignment": float(np.mean([score["structure_alignment"] for score in scores])),
            "hard_conflicts": float(hard_conflicts),
            "invalid_action_rate": self.invalid_action_count / max(self.total_steps * self.num_agents, 1),
        }

    def _meets_success_condition(self, metrics: dict[str, float]) -> bool:
        return (
            metrics["hard_conflicts"] == 0
            and metrics["area_compliance"] == 1.0
            and metrics["shape_compliance"] == 1.0
            and metrics["intervention_ratio"] <= float(
                self.config["AdaptiveReuseEnvironment"].get("max_intervention_ratio", 0.45)
            )
        )

    def get_state(self) -> np.ndarray:
        min_x, min_y, max_x, max_y = self.boundary_polygon.bounds
        x_span, y_span = max(max_x - min_x, 1e-9), max(max_y - min_y, 1e-9)
        diagonal = math.hypot(x_span, y_span)
        metrics = self.calculate_metrics()
        states = []
        for index, space in enumerate(self.agent_spaces):
            polygon = space.polygon()
            fixed_distance = polygon.distance(self.fixed_union) if not self.fixed_union.is_empty else diagonal
            original = self._original_polygon(space)
            own = [
                (space.center[0] - min_x) / x_span,
                (space.center[1] - min_y) / y_span,
                space.width / x_span,
                space.height / y_span,
                space.area / max(self.boundary_polygon.area, 1e-9),
                space.area / max(space.target_area, 1e-9),
                space.aspect_ratio / max(space.aspect_max, 1e-9),
                (space.x1 - min_x) / x_span,
                (space.y1 - min_y) / y_span,
                (max_x - space.x2) / x_span,
                (max_y - space.y2) / y_span,
                fixed_distance / max(diagonal, 1e-9),
                self._iou(polygon, original),
                metrics["intervention_ratio"],
            ]
            for other_index, other in enumerate(self.agent_spaces):
                if other_index == index:
                    continue
                own.extend([
                    (other.center[0] - space.center[0]) / x_span,
                    (other.center[1] - space.center[1]) / y_span,
                    polygon.distance(other.polygon()) / max(diagonal, 1e-9),
                    self._relation_value(space.space_id, other.space_id),
                ])
            states.append(own)
        return np.asarray(states, dtype=np.float32)

    def get_grid_matrix(self) -> np.ndarray:
        """返回矩阵化平面：0 可用、1 固定构件、2..N+1 为功能空间。"""
        min_x, min_y, max_x, max_y = self.boundary_polygon.bounds
        columns = int(math.ceil((max_x - min_x) / self.grid_size))
        rows = int(math.ceil((max_y - min_y) / self.grid_size))
        matrix = np.full((rows, columns), -1, dtype=np.int16)
        for row in range(rows):
            for column in range(columns):
                cell = box(
                    min_x + column * self.grid_size, min_y + row * self.grid_size,
                    min_x + (column + 1) * self.grid_size, min_y + (row + 1) * self.grid_size,
                )
                if self.boundary_polygon.covers(cell):
                    matrix[row, column] = 0
                if not self.fixed_union.is_empty and cell.intersects(self.fixed_union):
                    matrix[row, column] = 1
                for index, space in enumerate(self.agent_spaces):
                    if cell.intersection(space.polygon()).area > cell.area * 0.5:
                        matrix[row, column] = index + 2
        return matrix

    def _build_info(self) -> dict[str, Any]:
        return {
            "allow_actions": self._get_allow_actions(),
            "action_names": ACTION_NAMES,
            "metrics": self.last_metrics or self.calculate_metrics(),
            "grid_matrix": self.get_grid_matrix(),
            "initial_repairs": list(self.initial_repairs),
            "area_budget": {
                **self.area_budget,
                "target_area_sum": float(sum(space.target_area for space in self.agent_spaces)),
                "min_area_sum": float(sum(space.min_area for space in self.agent_spaces)),
                "max_area_sum": float(sum(space.max_area for space in self.agent_spaces)),
            },
        }

    def render(self, render=False, show_original=True, **kwargs):
        if str(self.config.get("Language", "zh")).lower() == "zh":
            # macOS 自带 STHeiti；Windows/Linux 会依次尝试后续字体。
            plt.rcParams["font.sans-serif"] = ["STHeiti", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
        if self._last_figure is not None:
            plt.close(self._last_figure)
        figure, axis = plt.subplots(figsize=(11, 7.5))
        self._last_figure = figure
        min_x, min_y, max_x, max_y = self.boundary_polygon.bounds
        for x in np.arange(min_x, max_x + self.grid_size, self.grid_size):
            axis.axvline(x, color="#e6ebf1", linewidth=0.45, zorder=0)
        for y in np.arange(min_y, max_y + self.grid_size, self.grid_size):
            axis.axhline(y, color="#e6ebf1", linewidth=0.45, zorder=0)
        bx, by = self.boundary_polygon.exterior.xy
        axis.plot(bx, by, color="#172b4d", linewidth=2.2, label="Building boundary")

        if show_original:
            for original in self.original_spaces.values():
                ox, oy = original.exterior.xy
                axis.plot(ox, oy, color="#91a4b7", linestyle="--", linewidth=1.0, alpha=0.9)

        fixed_colors = {"column": "#5f6b76", "load_bearing_wall": "#263238", "shear_wall": "#c62828", "core": "#7b1fa2", "traffic_core": "#5e35b1", "retained_circulation": "#f9a825"}
        fixed_labels = {"column": "Column", "load_bearing_wall": "Load-bearing wall", "shear_wall": "Shear wall", "core": "Core", "traffic_core": "Traffic core", "retained_circulation": "Retained circulation"}
        seen_fixed = set()
        for item in self.fixed_objects:
            x1, y1, x2, y2 = map(float, item["rect"])
            kind = str(item.get("type", "fixed"))
            label = fixed_labels.get(kind, kind) if kind not in seen_fixed else None
            seen_fixed.add(kind)
            axis.add_patch(patches.Polygon(
                list(fixed_polygon(item).exterior.coords),
                facecolor=fixed_colors.get(kind, "#455a64"), edgecolor="#263238",
                alpha=0.72, hatch="//" if kind == "retained_circulation" else None,
                label=label, zorder=2,
            ))

        colors = plt.cm.Set3(np.linspace(0, 1, self.num_agents))
        for space, color in zip(self.agent_spaces, colors):
            axis.add_patch(patches.Rectangle(
                (space.x1, space.y1), space.width, space.height,
                facecolor=color, edgecolor="#174a73", linewidth=1.7, alpha=0.88, zorder=3,
            ))
            axis.text(
                *space.center, f"{space.name}\n{space.area:.1f}/{space.target_area:.1f} m²",
                ha="center", va="center", fontsize=8.5, zorder=4,
            )
        metrics = self.calculate_metrics()
        axis.set_title(
            "Existing-building adaptive reuse | "
            f"Area {metrics['area_compliance']:.0%} | Reuse {metrics['original_reuse']:.0%} | "
            f"Intervention {metrics['intervention_ratio']:.0%}"
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(min_x - self.grid_size, max_x + self.grid_size)
        axis.set_ylim(min_y - self.grid_size, max_y + self.grid_size)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=8)
        figure.tight_layout()
        if render:
            plt.show(block=False)
            plt.pause(0.001)
        return figure


def make_adaptive_reuse_env(config_path: str | Path):
    config = load_layout_config(config_path)
    return AdaptiveReuseEnv(config), config
