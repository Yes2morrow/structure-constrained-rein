"""固定楼栋数量的多智能体住区布局环境。"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import gym
from gym import spaces
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from .building import Building
from .layout_actions import ACTION_NAMES, load_layout_config, rect_changes_for_action


class ResidentialLayoutEnv(gym.Env):
    """一个楼栋对应一个智能体的住区总平面环境。"""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        site = config["Site"]
        buildings = config["Buildings"]
        targets = config["PlanningTargets"]

        self.boundary = [tuple(map(float, point)) for point in site["boundary"]]
        self.boundary_polygon = Polygon(self.boundary)
        if not self.boundary_polygon.is_valid or self.boundary_polygon.area <= 0:
            raise ValueError("Site.boundary 必须是有效且面积大于0的多边形")
        self.entrance = tuple(map(float, site["entrance"]))
        self.num_agents = int(buildings["count"])
        self.building_types = list(buildings.get("types", ["residential"] * self.num_agents))
        if len(self.building_types) != self.num_agents:
            raise ValueError("Buildings.types 的数量必须等于 Buildings.count")

        self.initial_width = float(buildings.get("initial_width", 22.0))
        self.initial_depth = float(buildings.get("initial_depth", 12.0))
        self.width_range = tuple(map(float, buildings.get("width_range", [18.0, 35.0])))
        self.depth_range = tuple(map(float, buildings.get("depth_range", [10.0, 16.0])))
        floors = buildings.get("floors", 12)
        self.floors = [int(floors)] * self.num_agents if isinstance(floors, (int, float)) else list(map(int, floors))
        if len(self.floors) != self.num_agents:
            raise ValueError("Buildings.floors 的数量必须等于 Buildings.count")
        self.floor_height = float(buildings.get("floor_height", 3.0))
        self.move_step = float(buildings.get("move_step", 1.0))
        self.resize_step = float(buildings.get("resize_step", 1.0))
        self.nearest_neighbors = min(int(buildings.get("nearest_neighbors", 3)), max(self.num_agents - 1, 0))

        self.target_far = float(targets["target_far"])
        self.target_density = float(targets["target_density"])
        self.minimum_spacing = float(targets["minimum_spacing"])
        self.boundary_setback = float(targets["boundary_setback"])
        self.max_steps = int(config.get("Training", {}).get("max_steps", 300))
        self.convergence_patience = int(targets.get("convergence_patience", 80))
        self.reward_weights = {key: float(value) for key, value in config["RewardWeights"].items()}

        self.buildable_polygon = self.boundary_polygon.buffer(-self.boundary_setback)
        if self.buildable_polygon.is_empty:
            raise ValueError("建筑退界过大，场地内不存在可建设区域")

        self.action_space = spaces.MultiDiscrete([len(ACTION_NAMES)] * self.num_agents)
        self.local_state_size = 16 + 4 * self.nearest_neighbors
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents, self.local_state_size),
            dtype=np.float32,
        )
        self.buildings: list[Building] = []
        self.total_steps = 0
        self.no_improvement_steps = 0
        self.best_step_reward = -np.inf
        self.last_metrics: dict[str, float] = {}
        self._last_figure = None

    def reset(self, *, seed=None, options=None, **kwargs):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.total_steps = 0
        self.no_improvement_steps = 0
        self.best_step_reward = -np.inf
        self.buildings = self._initialize_buildings()
        self.last_metrics = self.calculate_metrics()
        return self.get_state(), {"allow_actions": self._get_allow_actions(), "metrics": self.last_metrics}

    def _initialize_buildings(self) -> list[Building]:
        min_x, min_y, max_x, max_y = self.buildable_polygon.bounds
        result: list[Building] = []
        # 优先使用满足最小间距的规则候选点，避免纯随机顺序放置在后期陷入死角。
        x_step = self.initial_width + self.minimum_spacing
        y_step = self.initial_depth + self.minimum_spacing
        grid_centers = [
            (x, y)
            for y in np.arange(min_y + self.initial_depth / 2, max_y - self.initial_depth / 2 + 1e-9, y_step)
            for x in np.arange(min_x + self.initial_width / 2, max_x - self.initial_width / 2 + 1e-9, x_step)
        ]
        random.shuffle(grid_centers)
        for index in range(self.num_agents):
            candidate = None
            while grid_centers:
                cx, cy = grid_centers.pop()
                grid_candidate = Building(
                    index,
                    self.building_types[index],
                    cx - self.initial_width / 2,
                    cy - self.initial_depth / 2,
                    cx + self.initial_width / 2,
                    cy + self.initial_depth / 2,
                    self.floors[index],
                    self.floor_height,
                )
                if self.buildable_polygon.covers(grid_candidate.get_polygon()) and all(
                    grid_candidate.get_polygon().distance(other.get_polygon()) >= self.minimum_spacing
                    for other in result
                ):
                    candidate = grid_candidate
                    break
            for _ in range(1000):
                if candidate is not None:
                    break
                cx = random.uniform(min_x + self.initial_width / 2, max_x - self.initial_width / 2)
                cy = random.uniform(min_y + self.initial_depth / 2, max_y - self.initial_depth / 2)
                candidate = Building(
                    index,
                    self.building_types[index],
                    cx - self.initial_width / 2,
                    cy - self.initial_depth / 2,
                    cx + self.initial_width / 2,
                    cy + self.initial_depth / 2,
                    self.floors[index],
                    self.floor_height,
                )
                if self.buildable_polygon.covers(candidate.get_polygon()) and all(
                    candidate.get_polygon().distance(other.get_polygon()) >= self.minimum_spacing
                    for other in result
                ):
                    break
            else:
                raise RuntimeError("无法在当前场地中初始化全部楼栋，请减少楼栋数量、尺寸或最小间距")
            result.append(candidate)
        return result

    def get_state(self) -> np.ndarray:
        min_x, min_y, max_x, max_y = self.boundary_polygon.bounds
        x_range = max(max_x - min_x, 1e-9)
        y_range = max(max_y - min_y, 1e-9)
        diagonal = math.hypot(x_range, y_range)
        max_height = max((building.height for building in self.buildings), default=1.0)
        metrics = self.calculate_metrics()
        states = []

        for building in self.buildings:
            cx, cy = building.center
            state = [
                1.0,
                (cx - min_x) / x_range,
                (cy - min_y) / y_range,
                building.width / x_range,
                building.depth / y_range,
                building.footprint_area / self.boundary_polygon.area,
                building.floors / max(max(self.floors), 1),
                (building.x1 - min_x) / x_range,
                (building.y1 - min_y) / y_range,
                (max_x - building.x2) / x_range,
                (max_y - building.y2) / y_range,
                math.dist(building.center, self.entrance) / diagonal,
            ]

            neighbors = sorted(
                (other for other in self.buildings if other.building_id != building.building_id),
                key=lambda other: building.get_polygon().distance(other.get_polygon()),
            )[:self.nearest_neighbors]
            for other in neighbors:
                ox, oy = other.center
                state.extend([
                    building.get_polygon().distance(other.get_polygon()) / diagonal,
                    (ox - cx) / x_range,
                    (oy - cy) / y_range,
                    (other.height - building.height) / max(max_height, 1e-9),
                ])

            state.extend([
                metrics["far"] / max(self.target_far, 1e-9),
                metrics["density"] / max(self.target_density, 1e-9),
                metrics["overlap_ratio"],
                metrics["spacing_compliance"],
            ])
            states.append(state)
        return np.asarray(states, dtype=np.float32)

    def _candidate_for_action(self, building: Building, action: int) -> Building:
        changes = rect_changes_for_action(
            building.x1, building.y1, building.x2, building.y2,
            action, self.move_step, self.resize_step,
        )
        return building.copy(**changes)

    def _is_geometrically_valid(self, building: Building) -> bool:
        return (
            self.width_range[0] <= building.width <= self.width_range[1]
            and self.depth_range[0] <= building.depth <= self.depth_range[1]
            and self.buildable_polygon.covers(building.get_polygon())
        )

    def _get_allow_actions(self) -> list[list[int]]:
        allowed = []
        for building in self.buildings:
            actions = [4]
            actions.extend(
                action for action in range(len(ACTION_NAMES))
                if action != 4 and self._is_geometrically_valid(self._candidate_for_action(building, action))
            )
            allowed.append(actions)
        return allowed

    def step(self, actions):
        if len(actions) != self.num_agents:
            raise ValueError(f"需要 {self.num_agents} 个动作，实际收到 {len(actions)} 个")
        self.total_steps += 1
        invalid_actions = [False] * self.num_agents
        proposals = []
        for index, (building, action) in enumerate(zip(self.buildings, actions)):
            candidate = self._candidate_for_action(building, int(action))
            if self._is_geometrically_valid(candidate):
                proposals.append(candidate)
            else:
                proposals.append(building)
                invalid_actions[index] = True
        self.buildings = proposals

        reward_info = self._calculate_rewards(invalid_actions)
        rewards = [value for value, _ in reward_info]
        mean_reward = float(np.mean(rewards))
        if mean_reward > self.best_step_reward + 1e-6:
            self.best_step_reward = mean_reward
            self.no_improvement_steps = 0
        else:
            self.no_improvement_steps += 1
        self.last_metrics = self.calculate_metrics()
        done = self.total_steps >= self.max_steps or self.no_improvement_steps >= self.convergence_patience
        info = {
            "reward_components": [components for _, components in reward_info],
            "done_list": [done] * self.num_agents,
            "allow_actions": self._get_allow_actions(),
            "metrics": self.last_metrics,
        }
        return self.get_state(), rewards, done, info

    def calculate_metrics(self) -> dict[str, float]:
        site_area = self.boundary_polygon.area
        footprint = sum(building.footprint_area for building in self.buildings)
        gross_floor_area = sum(building.gross_floor_area for building in self.buildings)
        overlap_area = 0.0
        compliant_pairs = 0
        pair_count = 0
        for index, building in enumerate(self.buildings):
            for other in self.buildings[index + 1:]:
                overlap_area += building.get_polygon().intersection(other.get_polygon()).area
                pair_count += 1
                if building.get_polygon().distance(other.get_polygon()) >= self.minimum_spacing:
                    compliant_pairs += 1
        return {
            "far": gross_floor_area / site_area,
            "density": footprint / site_area,
            "overlap_ratio": overlap_area / site_area,
            "spacing_compliance": compliant_pairs / max(pair_count, 1),
        }

    def _calculate_rewards(self, invalid_actions: list[bool]):
        metrics = self.calculate_metrics()
        far_reward = -self.reward_weights["far"] * abs(metrics["far"] - self.target_far)
        density_reward = -self.reward_weights["density"] * abs(metrics["density"] - self.target_density)
        diagonal = math.hypot(
            self.boundary_polygon.bounds[2] - self.boundary_polygon.bounds[0],
            self.boundary_polygon.bounds[3] - self.boundary_polygon.bounds[1],
        )
        rewards = []
        for index, building in enumerate(self.buildings):
            overlap_penalty = 0.0
            spacing_penalty = 0.0
            for other in self.buildings:
                if other.building_id == building.building_id:
                    continue
                intersection = building.get_polygon().intersection(other.get_polygon()).area
                overlap_penalty -= self.reward_weights["overlap"] * intersection / max(building.footprint_area, 1e-9)
                distance = building.get_polygon().distance(other.get_polygon())
                spacing_penalty -= self.reward_weights["spacing"] * (
                    max(0.0, self.minimum_spacing - distance) / max(self.minimum_spacing, 1e-9)
                ) ** 2

            boundary_penalty = -self.reward_weights["boundary"] if invalid_actions[index] else 0.0
            orientation_reward = self.reward_weights["orientation"] * min(
                building.width / max(building.depth, 1e-9), 2.0
            ) / 2.0
            accessibility_reward = -self.reward_weights["accessibility"] * (
                math.dist(building.center, self.entrance) / max(diagonal, 1e-9)
            )
            components = {
                "boundary": boundary_penalty,
                "overlap": overlap_penalty,
                "spacing": spacing_penalty,
                "orientation": orientation_reward,
                "accessibility": accessibility_reward,
                "far": far_reward,
                "density": density_reward,
            }
            rewards.append((sum(components.values()), components))
        return rewards

    def render(self, render=False, **kwargs):
        if self._last_figure is not None:
            plt.close(self._last_figure)
        figure, axis = plt.subplots(figsize=(10, 8))
        self._last_figure = figure
        bx, by = self.boundary_polygon.exterior.xy
        axis.plot(bx, by, color="#263238", linewidth=2.0)
        buildable_x, buildable_y = self.buildable_polygon.exterior.xy
        axis.plot(buildable_x, buildable_y, color="#90a4ae", linestyle="--", linewidth=1.2)
        axis.scatter([self.entrance[0]], [self.entrance[1]], marker="v", s=120, color="#d32f2f", label="Entrance")

        colors = plt.cm.Set3(np.linspace(0, 1, max(self.num_agents, 1)))
        for building, color in zip(self.buildings, colors):
            rectangle = patches.Rectangle(
                (building.x1, building.y1), building.width, building.depth,
                facecolor=color, edgecolor="#37474f", linewidth=1.5, alpha=0.9,
            )
            axis.add_patch(rectangle)
            axis.text(
                *building.center,
                f"B{building.building_id + 1}\n{building.floors}F",
                ha="center", va="center", fontsize=9,
            )
        metrics = self.calculate_metrics()
        axis.set_title(
            f"Residential layout | FAR {metrics['far']:.3f} | Density {metrics['density']:.3f} | "
            f"Spacing {metrics['spacing_compliance']:.0%}"
        )
        axis.set_aspect("equal", adjustable="box")
        axis.legend(loc="upper right")
        axis.grid(alpha=0.15)
        figure.tight_layout()
        if render:
            plt.show(block=False)
            plt.pause(0.001)
        return figure


def make_residential_env(config_path: str | Path):
    config = load_layout_config(config_path)
    return ResidentialLayoutEnv(config), config
