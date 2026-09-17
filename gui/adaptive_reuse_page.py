"""既有建筑适应性改造环境的 Web 参数页。"""

from __future__ import annotations

import math
import base64
import hashlib
import json
from html import escape
from io import BytesIO
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps
try:
    from gui.responsive_canvas import st_canvas
except ImportError:
    st_canvas = None

from common.config_manager import get_config_dir
from core.envs import AdaptiveReuseEnv, summarize_target_area_budget, redistribute_target_max_areas
from core.envs.structure_geometry import fixed_polygon, normalize_structures
from core.floor_partition import build_floor_partition_problem, run_residential_floor_partition
from core.floor_partition.walls import wall_geometry
from gui.partition_overlay import partition_overlay
from gui.partition_settings import render_partition_constraints, net_unit_archive
from core.floor_partition.quality import validate_partition
from shapely.geometry import Polygon
from shapely.ops import unary_union
from gui.config_store import load_config, save_config
from gui.layout_repair import (
    private_rectangle_config as _private_rectangle_config,
    queue_repair_notice,
    repair_conflicts_and_save,
)
from gui.adaptive_reuse_support import (
    consume_auto_repair_notice,
    render_collapsible_subheader,
    run_auto_repair_and_rerun,
)
from gui.structure_editor import render_structure_editor, save_structures
from gui.boundary_editor import boundary_handles, parse_boundary_handles, save_boundary
from gui.environment_canvas import scene, parse_scene, controls as environment_controls, color as control_color, entrance_points, DISPLAY_LAYERS, FUNCTION_ROOM_COLORS
from gui.structure_canvas import canvas_objects, parse_canvas, signature, viewport, identity_colors


REWARD_LABELS = {
    "area": "目标面积与基本使用奖励",
    "shape": "空间形态奖励",
    "adjacency": "功能邻近奖励",
    "separation": "干扰空间分离奖励",
    "structure_alignment": "结构网格顺应奖励",
    "original_reuse": "原有空间利用奖励",
    "intervention": "改造干预惩罚",
    "invalid_action": "硬约束无效动作惩罚",
    "team_coordination": "多智能体整体协调奖励",
}

CONSTRAINT_STYLES = {
    "column": ("柱", "rgba(69,90,100,0.72)", "#263238"),
    "load_bearing_wall": ("承重墙", "rgba(38,50,56,0.72)", "#172126"),
    "shear_wall": ("剪力墙", "rgba(198,40,40,0.68)", "#8e0000"),
    "core": ("核心筒", "rgba(96,108,118,0.28)", "#263238"),
    "traffic_core": ("交通核区域", "rgba(126,87,194,0.34)", "#5e35b1"),
    "retained_circulation": ("保留交通空间", "rgba(249,168,37,0.48)", "#c17900"),
    "fixed": ("其他不可调整构件", "rgba(55,71,79,0.64)", "#263238"),
}

# 修改默认案例时同步升级此值，防止浏览器端旧组件状态覆盖新的平面配置。
REFERENCE_PLAN_UI_VERSION = "input_only_v5"

ROLE_OPTIONS = ["agent", "residual"]
RELATION_TYPE_OPTIONS = ["none", "adjacent", "separate", "connected"]
RELATION_TYPE_LABELS = {
    "none": "无所谓",
    "adjacent": "相邻/共边",
    "separate": "被其他空间隔开",
    "connected": "直接连通（占位）",
}
ROLE_LABELS = {
    "agent": "独立智能体",
    "residual": "剩余公共空间节点",
}
RELATION_GRAPH_STYLES = {
    "none": {"stroke": "#c7ced6", "dash": "7 7", "width": 1.5, "opacity": 0.55},
    "adjacent": {"stroke": "#ef6c00", "dash": "", "width": 3.0, "opacity": 0.95},
    "separate": {"stroke": "#c62828", "dash": "10 6", "width": 2.5, "opacity": 0.95},
    "connected": {"stroke": "#2e7d32", "dash": "4 4", "width": 2.5, "opacity": 0.95},
}


def _plan_objects(config: dict, width: int, height: int, display=None) -> list[dict]:
    """Embed the plan in Fabric JSON, avoiding the legacy background URL adapter."""
    buffer = BytesIO()
    if display is None:
        image=_load_plan_image(config,width,height)
    else:
        image=_create_tangwu_plan_image(config,width,height,grid_only=True)
        if config['ExistingBuilding'].get('plan_image') and display.get('background',(True,0))[0]:
            image=Image.blend(image,_load_plan_image(config,width,height),1-display.get('background',(True,0))[1]/100)
    image.save(buffer, format="PNG")
    return [{
        "type": "image", "left": 0, "top": 0,
        "width": width, "height": height,
        "src": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        "selectable": False, "evented": False,
        "lockMovementX": True, "lockMovementY": True,
    }]


def _clean(value, fallback=""):
    return fallback if value is None or (isinstance(value, float) and math.isnan(value)) else value


def _normalize_role(value) -> str:
    role = str(_clean(value, "agent")).strip() or "agent"
    return role if role in ROLE_OPTIONS else "agent"


def _normalize_relation_type(value) -> str:
    kind = str(_clean(value, "none")).strip() or "none"
    if kind in ("via_circular", "via_circulation"):
        kind = "none"
    return kind if kind in RELATION_TYPE_OPTIONS else "none"


def _ordered_target_ids(items: list[dict]) -> list[str]:
    ids = []
    for item in items:
        identifier = str(_clean(item.get("id"))).strip()
        if identifier:
            ids.append(identifier)
    return ids


def _pair_in_target_order(order: dict[str, int], first: str, second: str) -> tuple[str, str]:
    return (first, second) if order.get(first, math.inf) <= order.get(second, math.inf) else (second, first)


def _target_records(items: list[dict]) -> list[dict]:
    records = []
    for item in items:
        rect = item.get("initial_rect", [0, 0, 1, 1])
        areas = item.get("area_range", [item.get("target_area", 1), item.get("target_area", 1)])
        aspects = item.get("aspect_range", [1, 2.5])
        records.append({
            "id": item.get("id", ""), "name": item.get("name", ""),
            "role": _normalize_role(item.get("role", "agent")),
            "x1": rect[0], "y1": rect[1], "x2": rect[2], "y2": rect[3],
            "target_area": item.get("target_area", 1), "min_area": areas[0], "max_area": areas[1],
            "aspect_min": aspects[0], "aspect_max": aspects[1],
            "seed_x": item.get("seed",[(rect[0]+rect[2])/2,(rect[1]+rect[3])/2])[0],
            "seed_y": item.get("seed",[(rect[0]+rect[2])/2,(rect[1]+rect[3])/2])[1],
        })
    return records


def _records_to_targets(frame: pd.DataFrame, previous=()) -> list[dict]:
    from copy import deepcopy
    originals={str(item['id']):item for item in previous}
    result = []
    for raw in frame.to_dict("records"):
        identifier = str(_clean(raw.get("id"))).strip()
        if not identifier:
            continue
        item=deepcopy(originals.get(identifier,{}))
        item.update({
            "id": identifier,
            "name": str(_clean(raw.get("name"), identifier)),
            "role": _normalize_role(raw.get("role")),
            "initial_rect": [float(raw[key]) for key in ("x1", "y1", "x2", "y2")],
            "target_area": float(raw["target_area"]),
            "area_range": [float(raw["min_area"]), float(raw["max_area"])],
            "aspect_range": [float(raw["aspect_min"]), float(raw["aspect_max"])],
        })
        if 'seed_x' in raw and 'seed_y' in raw:
            seed=[float(raw['seed_x']),float(raw['seed_y'])]
            old=originals.get(identifier,{})
            r=old.get('initial_rect',item['initial_rect'])
            default_seed=[(r[0]+r[2])/2,(r[1]+r[3])/2]
            if 'seed' in old or seed!=default_seed: item['seed']=seed
        result.append(item)
    return result


def _relation_editor_rows(targets: list[dict], relations: list[dict]) -> list[dict]:
    # 统一补全全部节点对，便于直接编辑 C(n,2) 关系矩阵。
    ids = _ordered_target_ids(targets)
    order = {identifier: index for index, identifier in enumerate(ids)}
    names = {str(item.get("id")): str(item.get("name", item.get("id", ""))) for item in targets}
    mapped: dict[tuple[str, str], dict] = {}
    for edge in relations or []:
        source = str(_clean(edge.get("from"))).strip()
        target = str(_clean(edge.get("to"))).strip()
        if source not in order or target not in order or source == target:
            continue
        source, target = _pair_in_target_order(order, source, target)
        pair = (source, target)
        kind = _normalize_relation_type(edge.get("type"))
        record = {
            "from": source,
            "to": target,
            "from_name": names.get(source, source),
            "to_name": names.get(target, target),
            "type": kind,
            "min_shared_length": None,
            "min_clear_width": None,
            "min_distance": None,
        }
        if kind != "none":
            for key in ("min_shared_length", "min_clear_width", "min_distance"):
                value = edge.get(key)
                if pd.notna(value):
                    record[key] = float(value)
        current = mapped.get(pair)
        if current is None or current.get("type") == "none":
            mapped[pair] = record
    rows = []
    for source, target in combinations(ids, 2):
        rows.append(mapped.get((source, target), {
            "from": source,
            "to": target,
            "from_name": names.get(source, source),
            "to_name": names.get(target, target),
            "type": "none",
            "min_shared_length": None,
            "min_clear_width": None,
            "min_distance": None,
        }))
    return rows


def _records_to_relations(frame: pd.DataFrame, targets: list[dict]) -> list[dict]:
    ids = _ordered_target_ids(targets)
    order = {identifier: index for index, identifier in enumerate(ids)}
    records_by_pair: dict[tuple[str, str], dict] = {}
    for raw in frame.to_dict("records"):
        source = str(_clean(raw.get("from"))).strip()
        target = str(_clean(raw.get("to"))).strip()
        if source not in order or target not in order or source == target:
            continue
        source, target = _pair_in_target_order(order, source, target)
        kind = _normalize_relation_type(raw.get("type"))
        item = {"from": source, "to": target, "type": kind}
        if kind != "none":
            for key in ("min_shared_length", "min_clear_width", "min_distance"):
                value = raw.get(key)
                if pd.notna(value):
                    item[key] = float(value)
        records_by_pair[(source, target)] = item
    result = []
    for source, target in combinations(ids, 2):
        result.append(records_by_pair.get((source, target), {"from": source, "to": target, "type": "none"}))
    return result


def _render_relation_graph(config: dict, graph_height: int = 260) -> None:
    targets = [item for item in config.get("TargetSpaces", []) if str(_clean(item.get("id"))).strip()]
    if len(targets) < 2:
        st.info("至少需要两个图节点，才会显示节点关系图。")
        return
    ids = _ordered_target_ids(targets)
    names = {str(item.get("id")): str(item.get("name", item.get("id", ""))) for item in targets}
    roles = {str(item.get("id")): _normalize_role(item.get("role")) for item in targets}
    residual_ids = [identifier for identifier in ids if roles.get(identifier) == "residual"]
    width, height = 760, max(220, graph_height)
    cx, cy = width / 2.0, height / 2.0
    radius = max(min(height * 0.32, width * 0.26), 85.0)
    positions: dict[str, tuple[float, float]] = {}
    orbital_ids = ids[:]
    if residual_ids:
        anchor = residual_ids[0]
        positions[anchor] = (cx, cy)
        orbital_ids = [identifier for identifier in ids if identifier != anchor]
    if orbital_ids:
        for index, identifier in enumerate(orbital_ids):
            angle = (2.0 * math.pi * index / max(len(orbital_ids), 1)) - math.pi / 2.0
            positions[identifier] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    relation_rows = _relation_editor_rows(targets, config.get("FunctionalRelations", []))
    edge_svg = []
    for row in relation_rows:
        kind = _normalize_relation_type(row.get("type"))
        style = RELATION_GRAPH_STYLES[kind]
        x1, y1 = positions[row["from"]]
        x2, y2 = positions[row["to"]]
        detail = [f"{names.get(row['from'], row['from'])} ({row['from']})", f"{names.get(row['to'], row['to'])} ({row['to']})", f"关系：{RELATION_TYPE_LABELS[kind]}"]
        if pd.notna(row.get("min_shared_length")):
            detail.append(f"最小共边：{float(row['min_shared_length']):.2f} m")
        if pd.notna(row.get("min_clear_width")):
            detail.append(f"最小净宽：{float(row['min_clear_width']):.2f} m")
        if pd.notna(row.get("min_distance")):
            detail.append(f"最小距离：{float(row['min_distance']):.2f} m")
        tooltip = escape("\n".join(detail))
        edge_svg.append(
            f"<line class='edge' data-source='{escape(row['from'])}' data-target='{escape(row['to'])}' "
            f"x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' "
            f"stroke='{style['stroke']}' stroke-width='{style['width']}' stroke-dasharray='{style['dash']}' "
            f"stroke-opacity='{style['opacity']}'><title>{tooltip}</title></line>"
        )
    node_svg = []
    for identifier in ids:
        x, y = positions[identifier]
        name = names.get(identifier, identifier)
        role = roles.get(identifier, "agent")
        title = escape(f"{name} ({identifier})\n角色：{ROLE_LABELS.get(role, role)}")
        if role == "residual":
            points = "0,-20 20,0 0,20 -20,0"
            marker = f"<polygon points='{points}' fill='#ffca28' stroke='#8d6e63' stroke-width='2'><title>{title}</title></polygon>"
        else:
            marker = f"<circle cx='0' cy='0' r='18' fill='#64b5f6' stroke='#1565c0' stroke-width='2'><title>{title}</title></circle>"
        node_svg.append(
            f"<g class='node' data-node='{escape(identifier)}' data-x='{x:.2f}' data-y='{y:.2f}' transform='translate({x:.2f} {y:.2f})'>{marker}"
            f"<text x='0' y='34' text-anchor='middle' class='node-label'>{escape(name)}</text>"
            f"<text x='0' y='49' text-anchor='middle' class='node-sub'>{escape(identifier)}</text></g>"
        )
    legend = "".join(
        f"<div class='legend-item'><span class='legend-line' "
        f"style='border-top:3px {'dashed' if style['dash'] else 'solid'} {style['stroke']};"
        f"opacity:{style['opacity']};'></span>{label}</div>"
        for kind, label in RELATION_TYPE_LABELS.items()
        for style in [RELATION_GRAPH_STYLES[kind]]
    )
    html = f"""
<div class="relation-graph-shell">
  <div class="relation-toolbar">
    <div class="legend">{legend}</div>
    <div class="buttons">
      <button type="button" id="zoom-in">+</button>
      <button type="button" id="zoom-out">-</button>
      <button type="button" id="zoom-reset">重置</button>
    </div>
  </div>
  <div class="graph-tip">拖拽节点可手动整理关系图；滚轮缩放、空白区域可平移，单击节点高亮相关边；客厅若为 residual，会以菱形显示。</div>
  <svg id="relation-graph" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc" rx="16"></rect>
    <g id="viewport">
      {''.join(edge_svg)}
      {''.join(node_svg)}
    </g>
  </svg>
</div>
<style>
  .relation-graph-shell {{
    width: 100%;
    border: 1px solid #d9dee6;
    border-radius: 16px;
    background: #ffffff;
    overflow: hidden;
    box-sizing: border-box;
  }}
  .relation-toolbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 14px 6px 14px;
    border-bottom: 1px solid #edf1f5;
    flex-wrap: wrap;
    font: 13px/1.4 sans-serif;
  }}
  .legend {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    color: #455a64;
  }}
  .legend-item {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  .legend-line {{
    width: 20px;
    height: 0;
    border-top: 3px solid #90a4ae;
    display: inline-block;
  }}
  .buttons button {{
    border: 1px solid #c7d0db;
    background: #ffffff;
    border-radius: 8px;
    padding: 4px 10px;
    cursor: pointer;
  }}
  .graph-tip {{
    padding: 6px 14px 0 14px;
    color: #607080;
    font: 12px/1.4 sans-serif;
  }}
  #relation-graph {{
    width: 100%;
    height: {height}px;
    cursor: grab;
    touch-action: none;
  }}
  #relation-graph.dragging {{
    cursor: grabbing;
  }}
  .node {{
    cursor: move;
  }}
  .node-label {{
    font: 600 13px/1.2 sans-serif;
    fill: #243447;
    pointer-events: none;
  }}
  .node-sub {{
    font: 11px/1.2 sans-serif;
    fill: #6b7c93;
    pointer-events: none;
  }}
  .edge.fade {{
    opacity: 0.12 !important;
  }}
  .edge.active {{
    opacity: 1 !important;
    stroke-width: 4.5 !important;
  }}
  .node.fade {{
    opacity: 0.3;
  }}
  .node.active circle,
  .node.active polygon {{
    stroke: #111827;
    stroke-width: 3.5;
  }}
</style>
<script>
  (() => {{
    const svg = document.getElementById('relation-graph');
    const viewport = document.getElementById('viewport');
    const edges = Array.from(svg.querySelectorAll('.edge'));
    const nodes = Array.from(svg.querySelectorAll('.node'));
    const positions = Object.fromEntries(nodes.map((node) => [node.dataset.node, {{
      x: Number(node.dataset.x),
      y: Number(node.dataset.y),
    }}]));
    const state = {{ scale: 1, tx: 0, ty: 0, panning: false, dragNode: null, px: 0, py: 0 }};
    function render() {{
      viewport.setAttribute('transform', `translate(${{state.tx}} ${{state.ty}}) scale(${{state.scale}})`);
      nodes.forEach((node) => {{
        const point = positions[node.dataset.node];
        node.setAttribute('transform', `translate(${{point.x}} ${{point.y}})`);
      }});
      edges.forEach((edge) => {{
        const source = positions[edge.dataset.source];
        const target = positions[edge.dataset.target];
        edge.setAttribute('x1', source.x.toFixed(2));
        edge.setAttribute('y1', source.y.toFixed(2));
        edge.setAttribute('x2', target.x.toFixed(2));
        edge.setAttribute('y2', target.y.toFixed(2));
      }});
    }}
    function zoom(factor) {{
      state.scale = Math.max(0.55, Math.min(2.4, state.scale * factor));
      render();
    }}
    function eventPoint(event) {{
      const rect = svg.getBoundingClientRect();
      return {{
        x: (event.clientX - rect.left - state.tx) / state.scale,
        y: (event.clientY - rect.top - state.ty) / state.scale,
      }};
    }}
    document.getElementById('zoom-in').onclick = () => zoom(1.15);
    document.getElementById('zoom-out').onclick = () => zoom(1 / 1.15);
    document.getElementById('zoom-reset').onclick = () => {{
      state.scale = 1;
      state.tx = 0;
      state.ty = 0;
      render();
    }};
    svg.addEventListener('wheel', (event) => {{
      event.preventDefault();
      zoom(event.deltaY < 0 ? 1.08 : 1 / 1.08);
    }}, {{ passive: false }});
    nodes.forEach((node) => {{
      node.addEventListener('pointerdown', (event) => {{
        event.stopPropagation();
        state.dragNode = node.dataset.node;
        const point = eventPoint(event);
        state.px = point.x;
        state.py = point.y;
        svg.classList.add('dragging');
      }});
    }});
    svg.addEventListener('pointerdown', (event) => {{
      if (state.dragNode) return;
      state.panning = true;
      state.px = event.clientX;
      state.py = event.clientY;
      svg.classList.add('dragging');
    }});
    window.addEventListener('pointerup', () => {{
      state.panning = false;
      state.dragNode = null;
      svg.classList.remove('dragging');
    }});
    window.addEventListener('pointermove', (event) => {{
      if (state.dragNode) {{
        const point = eventPoint(event);
        const node = positions[state.dragNode];
        node.x += point.x - state.px;
        node.y += point.y - state.py;
        node.x = Math.max(26, Math.min({width - 26}, node.x));
        node.y = Math.max(26, Math.min({height - 26}, node.y));
        state.px = point.x;
        state.py = point.y;
        render();
        return;
      }}
      if (!state.panning) return;
      state.tx += event.clientX - state.px;
      state.ty += event.clientY - state.py;
      state.px = event.clientX;
      state.py = event.clientY;
      render();
    }});
    nodes.forEach((node) => {{
      node.addEventListener('click', (event) => {{
        event.stopPropagation();
        const current = node.dataset.node;
        nodes.forEach((item) => item.classList.toggle('active', item.dataset.node === current));
        nodes.forEach((item) => item.classList.toggle('fade', item.dataset.node !== current));
        edges.forEach((edge) => {{
          const active = edge.dataset.source === current || edge.dataset.target === current;
          edge.classList.toggle('active', active);
          edge.classList.toggle('fade', !active);
        }});
      }});
    }});
    svg.addEventListener('click', () => {{
      nodes.forEach((item) => item.classList.remove('active', 'fade'));
      edges.forEach((edge) => edge.classList.remove('active', 'fade'));
    }});
    render();
  }})();
</script>
"""
    components.html(html, height=height + 86, scrolling=False)


def _plan_font(size: int):
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _create_tangwu_plan_image(config: dict, width: int, height: int, grid_only=False):
    """根据原始空间数据生成可直接标注的既有住宅二维底图。"""
    building = config["ExistingBuilding"]
    boundary = building["boundary"]
    xs, ys = [float(point[0]) for point in boundary], [float(point[1]) for point in boundary]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    x_span, y_span = max(max_x - min_x, 1e-9), max(max_y - min_y, 1e-9)
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    title_font, small_font = _plan_font(22), _plan_font(14)

    def point(x: float, y: float) -> tuple[float, float]:
        scale, ox, oy = viewport(boundary, width, height)
        return (ox+x*scale, oy-y*scale)

    # 1m×1m 参考网格只作为坐标参考，结构约束由上层彩色图层表达。
    grid = 1.0
    for x in np.arange(min_x, max_x + grid, grid):
        px, _ = point(x, min_y)
        draw.line([(px, 0), (px, height)], fill="#edf0f2", width=1)
    for y in np.arange(min_y, max_y + grid, grid):
        _, py = point(min_x, y)
        draw.line([(0, py), (width, py)], fill="#edf0f2", width=1)

    if grid_only: return image

    # 在底图中画出中央楼梯/电梯核心区的内部符号，约束图层会半透明覆盖其上。
    core_item = next((
        item for item in building.get("fixed_objects", [])
        if item.get("type") in ("traffic_core", "core") and not item.get('derived')
    ), None)
    if core_item:
        x1, y1, x2, y2 = map(float, core_item["rect"])
        left, bottom = point(x1, y1)
        right, top = point(x2, y2)
        draw.rectangle([left, top, right, bottom], fill="#eceff1", outline="#263238", width=4)
        third = (right - left) / 3
        draw.line([(left + third, top), (left + third, bottom)], fill="#37474f", width=3)
        draw.line([(left + 2 * third, top), (left + 2 * third, bottom)], fill="#37474f", width=3)
        for start in (left, left + third):
            draw.line([(start, top), (start + third, bottom)], fill="#65727b", width=2)
            draw.line([(start + third, top), (start, bottom)], fill="#65727b", width=2)
        stair_left = left + 2 * third
        for step_y in np.linspace(top + 5, bottom - 5, 8):
            draw.line([(stair_left + 4, step_y), (right - 4, step_y)], fill="#65727b", width=2)
        label = "交通核" if core_item.get("type") == "traffic_core" else "核心筒"
        draw.text((left + third + 8, (top + bottom) / 2 - 10), label, fill="#263238", font=small_font)

    boundary_pixels = [point(float(x), float(y)) for x, y in boundary]
    draw.line(boundary_pixels + [boundary_pixels[0]], fill="#182c3d", width=7, joint="curve")

    # 住宅主入口符号，帮助辨识既有住宅的中轴与朝向。
    opening=[point(x,y) for x,y in entrance_points(building)]
    if len(opening)>=2:
        draw.line(opening,fill='#facc15',width=7)
    draw.text((12, 10), "既有住宅·原始平面", fill="#1f3448", font=title_font)
    return image


def _load_plan_image(config: dict, width: int, height: int):
    image_path = str(config.get("ExistingBuilding", {}).get("plan_image", "")).strip()
    if image_path and Path(image_path).is_file():
        with Image.open(image_path) as source:
            boundary = config['ExistingBuilding']['boundary']
            scale, ox, oy = viewport(boundary, width, height)
            xs, ys = zip(*boundary)
            size = (max(1, round((max(xs)-min(xs))*scale)), max(1, round((max(ys)-min(ys))*scale)))
            image = Image.new('RGB', (width,height), '#ffffff')
            image.paste(ImageOps.fit(source.convert('RGB'), size, method=Image.Resampling.LANCZOS),
                        (round(ox+min(xs)*scale),round(oy-max(ys)*scale)))
            return image
    return _create_tangwu_plan_image(config, width, height)


def _render_residential_inputs(config, config_id):
    st.markdown("### 既有住宅输入")
    scenario_col1, scenario_col2 = st.columns(2)
    with scenario_col1:
        config["BuildingTypology"] = st.text_input(
            "既有建筑类型", value=config.get("BuildingTypology", "traditional_tangwu"),
            help="当前研究场景为既有住宅更新。", key=f"{config_id}_ar_typology",
        )
    with scenario_col2:
        config["ConversionGoal"] = st.text_input(
            "转换目标", value=config.get("ConversionGoal", "modern_residence"),
            help="目标是现代住宅功能布局，而不是公共服务或社区中心。", key=f"{config_id}_ar_goal",
        )

    if st.button("保存住宅输入", type="primary", use_container_width=True):
        saved = load_config(config_id)
        for field in ("BuildingTypology", "ConversionGoal"):
            saved[field] = config[field]
        save_config(saved, config_id)
        st.success("住宅类型与转换目标已保存。")

    return True


def _save_scene(config, config_id, updated):
    saved=load_config(config_id)
    for field in ('ExistingBuilding','TargetSpaces','FunctionalRelations'):
        saved[field]=updated[field]
        config[field]=updated[field]
    save_config(saved,config_id)
    # 结构/边界交互可能把智能体房间压进固定结构：自动挪回合法位置并保存。
    rect_repairs, seed_fixes, repair_error = repair_conflicts_and_save(config, config_id)
    queue_repair_notice(config_id, rect_repairs, seed_fixes, repair_error)
    key=f'{config_id}_ar_canvas_revision'
    st.session_state[key]=st.session_state.get(key,0)+1


def _render_target_editor(config, config_id):
    revision=st.session_state.get(f'{config_id}_ar_canvas_revision',0)
    baseline=deepcopy([config.get('TargetSpaces'),config.get('FunctionalRelations')])
    st.markdown("### 目标功能空间智能体")
    st.caption("下表列出全部目标空间图节点。`role=agent` 会创建可训练智能体；`role=residual` 仅作为剩余公共空间/关系节点存在，常用于客厅。")
    target_df = st.data_editor(
        pd.DataFrame(_target_records(config.get("TargetSpaces", [])),columns=[
            'id','name','role','x1','y1','x2','y2','target_area','min_area','max_area',
            'aspect_min','aspect_max','seed_x','seed_y']),
        num_rows="dynamic", use_container_width=True, key=f"{config_id}_ar_targets_{REFERENCE_PLAN_UI_VERSION}_{revision}",
        column_config={
            "role": st.column_config.SelectboxColumn("role", options=ROLE_OPTIONS, help="agent=独立智能体；residual=剩余公共空间节点"),
            **{key: st.column_config.NumberColumn(key, format="%.4f") for key in ("x1", "y1", "x2", "y2", "seed_x", "seed_y")},
            **{key: st.column_config.NumberColumn(key, min_value=.001, format="%.4f") for key in ("target_area", "min_area", "max_area", "aspect_min", "aspect_max")},
        },
    )
    config["TargetSpaces"] = _records_to_targets(target_df,config.get('TargetSpaces',[]))
    if not config['TargetSpaces']:
        st.info('此户尚未配置房间。请在表格中添加房间及面积要求，保存后再开始第二阶段训练。坐标沿用原楼层，边界已扣除分户隔墙厚度。')
        return True
    residual_rooms=[r for r in config.get('TargetSpaces',[]) if _normalize_role(r.get('role'))=='residual']
    if residual_rooms:
        public=residual_rooms[0]
        st.info(f"{public['id']}：当前被定义为剩余公共活动空间节点。它不创建独立矩形策略/种子，面积由扣除其他房间后的余量形成，并继续参与图关系。")
        public['residual_min_area']=float(st.number_input('客厅公共空间最小面积（㎡）',min_value=.1,value=float(public.get('residual_min_area',public['area_range'][0])),key=f'{config_id}_public_area'))
        public['min_width']=float(st.number_input('公共空间连通净宽检查（m）',min_value=.1,value=float(public.get('min_width',.9)),step=.1,key=f'{config_id}_public_width'))
    if residual_rooms:
        st.caption('公共空间承接扣除固定结构和独立房间后的全部余量；不再要求各房间最大面积相加等于可分配面积。')
    else:
        try:
            budget = summarize_target_area_budget(config)
            st.caption(
                f"面积验算：场地 {budget['boundary_area']:.2f} ㎡ - 交通约束 {budget['constraint_area']:.2f} ㎡"
                f" = 可分配 {budget['allocatable_area']:.2f} ㎡；当前 min/target/max 总和 = "
                f"{budget['min_area_sum']:.2f} / {budget['target_area_sum']:.2f} / {budget['max_area_sum']:.2f} ㎡"
            )
            if abs(budget["max_area_gap"]) <= 1e-6:
                st.success("当前最大面积总和与可分配面积一致。")
            elif budget["max_area_gap"] > 0:
                st.warning(f"当前最大面积总和偏小 {budget['max_area_gap']:.2f} ㎡，还有可分配余量。")
            else:
                st.error(f"当前最大面积总和超出可分配面积 {-budget['max_area_gap']:.2f} ㎡。")
            if st.button("按可分配面积自动分布最大面积", use_container_width=True, key=f"{config_id}_auto_max_area"):
                updated_targets = redistribute_target_max_areas(config)
                saved = load_config(config_id)
                saved["TargetSpaces"] = updated_targets
                save_config(saved, config_id)
                config["TargetSpaces"] = updated_targets
                st.session_state[f'{config_id}_ar_canvas_revision'] = revision + 1
                st.success("已按目标面积比例自动分布最大面积，并保存到 YAML。")
                st.rerun()
        except ValueError as exc:
            st.error(f"面积验算失败：{exc}")
    seed_valid=True
    if config.get('SeedGrowth',{}).get('enabled',False):
        from gui.seed_page import render_seed_settings
        seed_valid=render_seed_settings(config,config_id)

    st.markdown("### 功能关系")
    st.caption(f"系统会自动列出全部节点对，共 {len(config.get('TargetSpaces', []))} 个节点、默认 {len(_relation_editor_rows(config.get('TargetSpaces', []), config.get('FunctionalRelations', [])))} 组组合关系。")
    relation_df = st.data_editor(
        pd.DataFrame(_relation_editor_rows(config.get("TargetSpaces", []), config.get("FunctionalRelations", []))),
        hide_index=True, use_container_width=True, num_rows="fixed", disabled=["from", "from_name", "to", "to_name"],
        key=f"{config_id}_ar_relations_{REFERENCE_PLAN_UI_VERSION}_{hashlib.sha256(json.dumps(_ordered_target_ids(config.get('TargetSpaces', []))).encode()).hexdigest()[:8]}",
        column_config={
            "from": st.column_config.TextColumn("from"),
            "from_name": st.column_config.TextColumn("from_name"),
            "to": st.column_config.TextColumn("to"),
            "to_name": st.column_config.TextColumn("to_name"),
            "type": st.column_config.SelectboxColumn("type", options=RELATION_TYPE_OPTIONS, help="none=无特殊位置约束"),
            **{k:st.column_config.NumberColumn(k,min_value=.001, format="%.4f") for k in ('min_shared_length','min_clear_width','min_distance')},
        },
    )
    config["FunctionalRelations"] = _records_to_relations(relation_df, config.get("TargetSpaces", []))
    st.caption('关系说明：`none`=无所谓，不对空间位置产生特殊约束；`adjacent`=要求实际共边；`connected`=两个房间紧邻且有门直接连通，但当前尚未建模房间门，所以先作为占位目标保留；`separate`=中间应有客厅或其他房间隔开，当前前端与训练阶段先近似为“非直接相邻/不贴邻”。')

    for r in config['TargetSpaces']:
        if r.get('role')=='residual': continue
        rect=r['initial_rect']
        if not all(math.isfinite(v) for v in rect) or rect[2]<=rect[0] or rect[3]<=rect[1]:
            st.error('初始智能体区域宽高必须为正，坐标必须有效。')
            return False
        min_area, max_area = map(float, r.get('area_range', [r.get('target_area', 0.0), r.get('target_area', 0.0)]))
        target_area = float(r.get('target_area', 0.0))
        if min_area <= 0 or max_area <= 0 or target_area <= 0:
            st.error('目标面积、最小面积、最大面积必须为正数。')
            return False
        if min_area - 1e-9 > max_area:
            st.error(f"{r['id']} 的最小面积不能大于最大面积。")
            return False
        if target_area + 1e-9 < min_area or target_area - 1e-9 > max_area:
            st.error(f"{r['id']} 的 target_area 必须落在 area_range 内。")
            return False
    if baseline!=[config.get('TargetSpaces'),config.get('FunctionalRelations')]:
        ids=[r['id'] for r in config['TargetSpaces']]
        if len(ids)!=len(set(ids)) or any(e[k] not in ids for e in config['FunctionalRelations'] for k in ('from','to')):
            st.error('智能体 ID 必须唯一；删除节点前请更新其图关系。')
            return False
        residual_count=sum(_normalize_role(r.get('role'))=='residual' for r in config['TargetSpaces'])
        if residual_count > 1 or residual_count == len(config['TargetSpaces']):
            st.error('最多只能有一个 residual 公共空间节点，且至少保留一个 agent 智能体。')
            return False
        _save_scene(config,config_id,config)
        st.rerun()
    return seed_valid


def _render_all_points(config,config_id):
    with st.expander('全部对象控制点坐标（含门和其他结构）',expanded=False):
        values=environment_controls(config)
        keys=list(values)
        rows=[dict(group=k[0],id=k[1],point=k[2],x=p[0],y=p[1]) for k,p in values.items()]
        revision=st.session_state.get(f'{config_id}_ar_canvas_revision',0)
        frame=st.data_editor(pd.DataFrame(rows),hide_index=True,disabled=['group','id','point'],height=260,
            key=f'{config_id}_all_points_{revision}',column_config={k:st.column_config.NumberColumn(k,format='%.4f') for k in ('x','y')})
        edited=frame.to_dict('records')
        if edited!=rows:
            w,h=760,480
            objects=scene(config,w,h,CONSTRAINT_STYLES)
            lookup={o['stroke']:o for o in objects if o['type']=='circle'}
            scale,ox,oy=viewport(config['ExistingBuilding']['boundary'],w,h)
            try:
                for k,row,old in zip(keys,edited,rows):
                    if row==old: continue
                    lookup[control_color(k)].update(left=ox+float(row['x'])*scale,top=oy-float(row['y'])*scale)
                updated=parse_scene(objects,config,w,h)
                _save_scene(config,config_id,updated)
                st.rerun()
            except (ValueError,TypeError,KeyError) as exc:
                st.error(f'控制点未保存：{exc}')
                return False
    return True


def _render_plan_constraint_editor(
    config: dict,
    config_id: str,
    *,
    show_heading: bool = True,
) -> bool:
    building = config["ExistingBuilding"]
    environment = config["AdaptiveReuseEnvironment"]
    boundary = building["boundary"]
    if show_heading:
        st.markdown("### 二维原始平面与前期约束标注")
    st.caption('参数表与画布共用一份构件数据。有效修改自动保存到 YAML 并双向同步；新增表格行请先补全必填数据。')
    stage_valid = _render_floor_partition_section(config, config_id, preview=False)
    stage_is_floor_partition = _is_floor_partition_stage(config)
    st.markdown("---")

    upload_col = st.container()
    type_col = st.container()
    with upload_col, st.expander("平面底图（可选）", expanded=False):
        uploaded = st.file_uploader(
            "上传原始建筑平面图", type=["png", "jpg", "jpeg", "webp"],
            key=f"{config_id}_ar_plan_upload_{REFERENCE_PLAN_UI_VERSION}",
        )
        if uploaded is not None and st.button("载入为二维标注底图", use_container_width=True):
            try:
                with Image.open(uploaded) as source:
                    source.verify()
                suffix = Path(uploaded.name).suffix.lower() or ".png"
                image_path = get_config_dir(config_id) / f"existing_plan{suffix}"
                image_path.write_bytes(uploaded.getvalue())
                building["plan_image"] = str(image_path)
                save_config(config, config_id)
                st.session_state[f"{config_id}_ar_canvas_revision"] = st.session_state.get(f"{config_id}_ar_canvas_revision", 0) + 1
                st.success("二维底图已载入。")
                st.rerun()
            except Exception as exc:
                st.error(f"无法读取该图片：{exc}")
        if building.get("plan_image"):
            st.caption(f"当前底图：{building['plan_image']}")
            if st.button("恢复系统生成的住宅原始平面", use_container_width=True):
                building.pop("plan_image", None)
                save_config(config, config_id)
                st.session_state[f"{config_id}_ar_canvas_revision"] = st.session_state.get(f"{config_id}_ar_canvas_revision", 0) + 1
                st.rerun()
        else:
            st.caption("当前底图：系统根据既有住宅自动生成的二维平面")
    with type_col:
        tools_row=st.columns(4)
        type_options = [k for k in CONSTRAINT_STYLES if k != 'core']
        selected_type = tools_row[0].selectbox(
            "当前标注类型", type_options,
            format_func=lambda key: CONSTRAINT_STYLES[key][0],
            key=f"{config_id}_ar_constraint_type",
        )
        operation_mode = tools_row[1].radio(
            "画布操作", ["统一点编辑", "绘制约束", "选择/调整", "调整建筑边界"], horizontal=True,
            key=f"{config_id}_ar_canvas_mode",
        )
        layers={'全部对象':'all','建筑边界':'boundary','智能体初始区域':'agent','智能体种子':'seed','柱':'column','墙线与左右厚度':'wall','其他矩形构件':'fixed_rect','其他多边形构件':'fixed','门位置':'door'}
        layer=layers[tools_row[2].selectbox('编辑图层（控制可选点）',list(layers),key=f'{config_id}_environment_layer')]
        entries=environment_controls(config)
        ids=sorted({k[1] for k in entries if layer=='all' or k[0]==layer})
        object_id=tools_row[3].selectbox('编辑对象 ID（重叠时用于选取）',['全部']+ids,key=f'{config_id}_environment_object')
        object_id=None if object_id=='全部' else object_id
        wall_left, wall_right = .12, .12
        if selected_type in ('shear_wall', 'load_bearing_wall'):
            wall_left = st.number_input('新墙左侧厚度（m）', min_value=0.0, value=.12, step=.01, key=f'{config_id}_new_wall_left')
            wall_right = st.number_input('新墙右侧厚度（m）', min_value=0.0, value=.12, step=.01, key=f'{config_id}_new_wall_right')
            st.caption('拖出墙线的起点与终点；左右按拖动方向定义。绘制后画布上仅可整体移动，端点与厚度请在“边界与结构”表格中输入。')
    xs = [float(point[0]) for point in boundary]
    ys = [float(point[1]) for point in boundary]
    aspect = max((max(xs) - min(xs)) / max(max(ys) - min(ys), 1e-9), 0.35)
    canvas_width = 680
    canvas_height = max(360, min(680, int(canvas_width / aspect)))
    relation_graph_height = max(240, min(360, int(canvas_height * 0.62)))
    # 右侧参数容器需要和左侧“画布说明 + 操作行 + 关系图标题/说明 + 关系图组件”总高度对齐。
    # 关系图组件本身会额外占用约 86 px 的工具栏与边距空间，这里一并计入。
    parameter_panel_height = canvas_height + relation_graph_height + 210

    canvas_column, parameters_column = st.columns([1.05, 1], gap="medium")
    with parameters_column, st.container(height=parameter_panel_height, border=False):
        panel_tabs=st.tabs(['显示','住宅输入','边界与结构','智能体与关系','控制点'])
        with panel_tabs[0]:
            with st.expander('图层显示与透明度', expanded=True):
                st.caption('勾选显示；透明度 0 为原始着色，100 为全透明。隐藏不会删除数据。')
                rows=[{'类别':label,'显示':True,'透明度':0} for key,label in DISPLAY_LAYERS.items()]
                rows += [{'类别':'对象名称','显示':True,'透明度':0},{'类别':'上传的平面底图','显示':True,'透明度':0}]
                layer_keys=list(DISPLAY_LAYERS)+['labels','background']
                if stage_is_floor_partition:
                    rows.insert(0,{'类别':'分户推演预览（粉色，非训练结果）','显示':False,'透明度':0})
                    layer_keys.insert(0,'partition_preview')
                view=st.data_editor(pd.DataFrame(rows),hide_index=True,disabled=['类别'],
                    column_config={'显示':st.column_config.CheckboxColumn('显示'),'透明度':st.column_config.NumberColumn('透明度 %',min_value=0,max_value=100,step=5)},
                    key=f'{config_id}_environment_display_v2_{stage_is_floor_partition}',use_container_width=True)
                display={key:(bool(row['显示']),float(row['透明度'] or 0)) for key,row in zip(layer_keys,view.to_dict('records'))}
                if stage_is_floor_partition:
                    for hidden_key in ('agent','seed','labels','background'):
                        display[hidden_key]=(False,100.0)
                st.caption('上传图片中的内容属于底图像素，可整体隐藏；各类环境对象可独立隐藏。')
        with panel_tabs[1]:
            _render_residential_inputs(config, config_id)
        with panel_tabs[2]:
            editor_valid = render_structure_editor(config, config_id)
        try:
            with panel_tabs[3]:
                if stage_is_floor_partition:
                    st.info("当前是楼层功能分区阶段，这里先不进入户型内部房间与功能关系编辑。切回“户型内部训练”后会恢复这一页。")
                else:
                    editor_valid = _render_target_editor(config, config_id) and editor_valid
            with panel_tabs[4]:
                if stage_is_floor_partition:
                    st.info("楼层功能分区阶段仍可在左侧画布编辑边界与结构；房间/种子控制点在户型内部训练阶段再开放。")
                else:
                    editor_valid = _render_all_points(config,config_id) and editor_valid
        except (ValueError,TypeError,KeyError) as exc:
            st.error(f"智能体数据未保存：{exc}")
            editor_valid=False
        if stage_is_floor_partition:
            st.caption('粉色图层是当前参数的自动推演，不是训练结果，也不是结构约束。默认关闭，可在「显示」中开启；黄色为推演户门。修改这些预览形状需调整参数或进行训练，不能当作结构拖动。')

    with canvas_column:
        if not editor_valid or not stage_valid:
            st.info('请先补全右侧参数行，画布编辑将在数据有效后恢复。')
            return False
        if st_canvas is None:
            st.warning("当前环境未安装 streamlit-drawable-canvas，暂时只能使用下方坐标表格。")
            return

        plan_objects = _plan_objects(config, canvas_width, canvas_height, display)
        plan_signature = hashlib.sha256(json.dumps([building,config.get("TargetSpaces"),display], sort_keys=True).encode()).hexdigest()[:12]
        _, fill_color, stroke_color = CONSTRAINT_STYLES[selected_type]
        revision = st.session_state.get(f"{config_id}_ar_canvas_revision", 0)
        canvas_key = f"{config_id}_ar_constraint_canvas_{REFERENCE_PLAN_UI_VERSION}_{plan_signature}_{selected_type}_{operation_mode}_{layer}_{object_id}_{revision}"
        editing_boundary = operation_mode == "调整建筑边界"
        display_layer='boundary' if editing_boundary else layer
        display_objects=scene(config,canvas_width,canvas_height,CONSTRAINT_STYLES,display_layer,object_id,display)
        if stage_is_floor_partition and display.get('partition_preview',(False,0))[0]:
            try:
                display_objects += partition_overlay(config,canvas_width,canvas_height,True,display['partition_preview'][1])
                st.info('粉色：自动推演，非训练结果；黄色：推演户门。隐藏预览不会改变建筑输入。')
            except ValueError as exc:
                st.warning(f'推演预览暂不可用，仍可编辑建筑输入：{exc}')
        if operation_mode=='绘制约束':
            for obj in display_objects: obj.update(selectable=False,evented=False)
        canvas = st_canvas(
            fill_color=fill_color,
            stroke_width=2,
            stroke_color=stroke_color,
            background_color="#ffffff",
            initial_drawing={
                "version": "4.4.0",
                "objects": plan_objects + display_objects,
            },
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=("line" if selected_type in ('shear_wall','load_bearing_wall') else "rect") if operation_mode == "绘制约束" else "transform",
            display_toolbar=True,
            key=canvas_key,
        )
        st.caption('观看操作：鼠标滚轮缩放画布，按住鼠标中键或 Alt + 左键拖动可平移，画布下方可缩小、放大或重置；这些操作不会改变实际尺寸。外轮廓尺寸线为只读标注，不参与结构约束。统一点编辑：橙色为边界，蓝色半透明为初始智能体，紫色为种子；中心点平移，尺寸请在右侧表格中输入。柱、墙等结构约束仅保留中心点用于移动，尺寸在“边界与结构”表中修改；交通核区域仍可通过角点拉伸。拖动智能体种子会整体平移该区域。重叠时选择图层和对象 ID。拖动不改变图节点关系。')
        if config.get('InteriorTrainingEnvironment'):
            legend_items=[]
            for room in config.get('TargetSpaces',[]):
                if room.get('role') == 'residual':
                    continue
                swatch=FUNCTION_ROOM_COLORS.get(room.get('id'),('#90a4ae',''))[0]
                legend_items.append(
                    f"<span style='display:inline-flex;align-items:center;gap:7px;margin:4px 18px 4px 0;'>"
                    f"<span style='width:18px;height:18px;border-radius:3px;background:{swatch};border:1px solid #456;'></span>"
                    f"{escape(str(room.get('name',room.get('id',''))))}</span>"
                )
            st.markdown("<div style='display:flex;flex-wrap:wrap;align-items:center'><b style='margin-right:18px'>功能空间图例</b>"+''.join(legend_items)+"</div>",unsafe_allow_html=True)
            st.caption('未着色的剩余可用空间作为客厅，不设置独立客厅色块。')
        ready_key = f'{config_id}_structure_canvas_ready'
        if canvas.json_data is not None and st.session_state.get(ready_key) != canvas_key:
            raw = canvas.json_data.get('objects', [])
            expected = set(identity_colors(building.get('fixed_objects', [])).values())
            expected.update(control_color(k) for k in environment_controls(config))
            observed = {o.get('stroke') for o in raw}
            # Fabric emits an empty frame before asynchronous initialDrawing hydration.
            # It must never be interpreted as a user deleting saved structures.
            if any(o.get('type') == 'image' for o in raw) and expected <= observed:
                st.session_state[ready_key] = canvas_key
        if canvas.json_data is not None and editor_valid and st.session_state.get(ready_key) == canvas_key:
            try:
                if operation_mode!='绘制约束':
                    updated=parse_scene(canvas.json_data.get('objects',[]),config,canvas_width,canvas_height)
                    if updated!=config:
                        _save_scene(config,config_id,updated)
                        st.rerun()
                else:
                    allowed=set(identity_colors(building.get('fixed_objects',[])).values())|{stroke_color}
                    raw=[o for o in canvas.json_data.get('objects',[]) if o.get('stroke') in allowed]
                    parsed=parse_canvas(raw,building.get('fixed_objects',[]),boundary,canvas_width,canvas_height,selected_type,wall_left,wall_right)
                    if signature(parsed)!=signature(building.get('fixed_objects',[])):
                        save_structures(config,config_id,parsed)
                        st.rerun()
            except (ValueError, TypeError, KeyError) as exc:
                st.error(f'画布修改未同步：{exc}')

        save_col, clear_col = st.columns(2)
        with save_col:
            st.caption(f"已同步 {len(building.get('fixed_objects', []))} 个构件（含自动核心筒与交通核区域）")
        with clear_col:
            if st.button("清空全部约束并同步参数表", use_container_width=True, disabled=not editor_valid):
                save_structures(config, config_id, [])
                st.rerun()
        if not stage_is_floor_partition:
            st.markdown("#### 图节点关系图")
            st.caption("下图与右侧“功能关系”表实时同步，便于在平面画布下方直接检查住宅内部的节点关系网络。")
            _render_relation_graph(config, graph_height=relation_graph_height)
        else:
            st.markdown("#### 分户推演图层说明")
            st.caption("开启时，浅粉色为自动走道，深粉色为分户隔墙，黄色为户门。它们随当前参数重新推演，仅供参考；训练后的方案请到「布局预览」查看。")
    return editor_valid and stage_valid


def _floor_usable_area_summary(config: dict) -> tuple[float, float, float] | None:
    """验算当前建筑环境下的总可用面积：总轮廓面积 − 全部结构约束面积（含交通核、柱、墙等）。"""
    building = config.get("ExistingBuilding", {})
    try:
        boundary = Polygon(building.get("boundary", []))
        if not boundary.is_valid or boundary.area <= 0:
            return None
        fixed_shapes = [fixed_polygon(item) for item in normalize_structures(building.get("fixed_objects", []))]
        fixed_union = unary_union(fixed_shapes) if fixed_shapes else None
        if fixed_union is None or fixed_union.is_empty:
            return boundary.area, 0.0, boundary.area
        usable = boundary.difference(fixed_union)
        return boundary.area, boundary.area - usable.area, usable.area
    except Exception:
        return None


def _ensure_floor_partition_defaults(config: dict) -> dict:
    floor = config.setdefault("FloorPartition", {})
    floor.setdefault("enabled", False)
    floor.setdefault("program_type", "residential")
    floor.setdefault(
        "grid_size",
        config.get("SeedGrowth", {}).get(
            "grid_size",
            config.get("AdaptiveReuseEnvironment", {}).get("grid_size", 0.5),
        ),
    )
    residential = floor.setdefault("residential", {})
    residential.setdefault("unit_count", 2)
    residential.setdefault("target_area_mode", "equal")
    residential.setdefault("target_areas", [80.0, 80.0])
    residential.setdefault("corridor_width", 1.5)
    residential.setdefault("min_door_spacing", 2.4)
    residential.setdefault("door_width", 0.9)
    residential.setdefault("wall_thickness", 0.2)
    residential.setdefault("opening_width", 1.5)
    residential.setdefault("opening_side", "auto")
    residential.setdefault("export_config_prefix", "unit")
    return floor


def _ensure_training_stage_defaults(config: dict) -> dict:
    """确保训练阶段配置存在。"""
    training = config.setdefault("Training", {})
    training.setdefault("training_stage", "room_training")
    return training


def _is_floor_partition_stage(config: dict) -> bool:
    return str(config.get("Training", {}).get("training_stage", "room_training")).strip() == "floor_partition"


def _render_floor_partition_section(config: dict, config_id: str, preview: bool = True) -> bool:
    """渲染楼层分区中间阶段配置，并做一次轻量校验。"""
    training = _ensure_training_stage_defaults(config)
    floor = _ensure_floor_partition_defaults(config)
    residential = floor["residential"]
    # 阶段切换入口在侧边栏一级分类，这里只读取当前阶段。
    training["training_stage"] = st.session_state.get(
        f"{config_id}_training_stage",
        training.get("training_stage", "room_training"),
    )
    floor["enabled"] = _is_floor_partition_stage(config)
    st.caption("训练阶段（楼层分区 / 户型内部训练）在左侧边栏的一级分类中切换，这里配置当前阶段的分区参数。")
    if not floor["enabled"]:
        interior = config.get("InteriorTrainingEnvironment", {})
        if interior:
            st.info(
                f"当前户型内部训练环境：{float(interior.get('width', 5)):g} × "
                f"{float(interior.get('height', 8)):g} 米（{float(interior.get('area', 40)):g} ㎡）"
            )
    floor["program_type"] = st.selectbox(
        "楼层功能分区类型",
        ["residential", "office"],
        index=0 if str(floor.get("program_type", "residential")) == "residential" else 1,
        format_func=lambda value: "住宅分区" if value == "residential" else "办公分区（预留）",
        key=f"{config_id}_floor_partition_type",
        disabled=not floor["enabled"],
    )
    st.caption("办公分区当前先预留入口；现阶段只实现住宅分区规则链。")
    cols = st.columns(3)
    with cols[0]:
        floor["grid_size"] = float(
            st.number_input(
                "分区网格（m）",
                min_value=0.1,
                value=float(floor.get("grid_size", 0.5)),
                step=0.05,
                key=f"{config_id}_floor_partition_grid",
            )
        )
        residential["unit_count"] = int(
            st.number_input(
                "户数",
                min_value=1,
                value=int(residential.get("unit_count", 2)),
                step=1,
                key=f"{config_id}_floor_partition_unit_count",
            )
        )
    with cols[1]:
        equal_area_mode = st.checkbox(
            "等面积（平均分配给各户）",
            value=str(residential.get("target_area_mode", "equal")) == "equal",
            key=f"{config_id}_floor_partition_area_equal",
            help="勾选后禁止逐户填写面积，按剩余可分配面积平均分配；取消勾选可在下方表格中逐户填写目标面积。",
        )
        residential["corridor_width"] = float(
            st.number_input(
                "走道净宽（m）",
                min_value=0.8,
                value=float(residential.get("corridor_width", 1.5)),
                step=0.1,
                key=f"{config_id}_floor_partition_corridor_width",
            )
        )
        residential["min_door_spacing"] = float(
            st.number_input(
                "最小门间距（m）",
                min_value=0.1,
                value=float(residential.get("min_door_spacing", 2.4)),
                step=0.1,
                key=f"{config_id}_floor_partition_door_spacing",
                help="不同户门洞口线段之间的最短直线净距，也适用于转角两侧；过大会迫使门厅加深。这是设计参数，并非规范值。",
            )
        )
    with cols[2]:
        residential['wall_thickness']=float(st.number_input(
            '分户隔墙总厚度（m）',min_value=0.,max_value=1.,
            value=float(residential.get('wall_thickness',.2)),step=.05,
            key=f'{config_id}_floor_partition_wall_thickness',
            help='分界线为墙中线，两侧各扣除一半厚度。贴齐结构侧面的是完成墙面，不是中线；户内训练使用扣墙后的净边界。'))
        residential["door_width"] = float(
            st.number_input(
                "门宽（m）",
                min_value=0.6,
                value=float(residential.get("door_width", 0.9)),
                step=0.1,
                key=f"{config_id}_floor_partition_door_width",
            )
        )
        residential["opening_width"] = float(
            st.number_input(
                "交通核开口宽度（m）",
                min_value=0.6,
                value=float(residential.get("opening_width", 1.5)),
                step=0.1,
                key=f"{config_id}_floor_partition_opening_width",
            )
        )
        residential["opening_side"] = st.selectbox(
            "交通核开口朝向",
            ["auto", "north", "south", "east", "west"],
            index=["auto", "north", "south", "east", "west"].index(str(residential.get("opening_side", "auto"))),
            format_func=lambda value: {"auto": "自动", "north": "北", "south": "南", "east": "东", "west": "西"}[value],
            key=f"{config_id}_floor_partition_opening_side",
        )
    residential["export_config_prefix"] = st.text_input(
        "导出户型配置前缀",
        value=str(residential.get("export_config_prefix", "unit")),
        key=f"{config_id}_floor_partition_prefix",
    ).strip() or "unit"
    if any(c in residential['export_config_prefix'] for c in '/\\:'):
        st.error('导出配置前缀不能包含路径分隔符或冒号。')
        return False
    residential["target_area_mode"] = "equal" if equal_area_mode else "custom"
    usable_summary = _floor_usable_area_summary(config)
    if usable_summary is not None:
        boundary_area, fixed_area, usable_area = usable_summary
        st.info(
            f"验算：总轮廓面积 {boundary_area:.2f} ㎡ − 结构约束面积 {fixed_area:.2f} ㎡"
            f"（交通核、柱、原有固定墙等）= 待分配面积 {usable_area:.2f} ㎡；生成的隔墙另行扣除。"
        )
    if equal_area_mode:
        if usable_summary is not None:
            st.caption(
                f"已勾选等面积：逐户面积输入被禁用，系统在扣除结构、走道与隔墙后把净面积平均分给 "
                f"{residential['unit_count']} 户，约 {usable_area / max(int(residential['unit_count']), 1):.2f} ㎡/户"
                f"（此处估算尚未扣走道和隔墙，最终以求解净面积为准）。"
            )
        else:
            st.caption("已勾选等面积：逐户面积输入被禁用，系统在扣除结构、走道与隔墙后把净面积平均分给各户。")
    else:
        prefix = str(residential.get("export_config_prefix", "unit"))
        base_areas = [float(value) for value in (residential.get("target_areas") or [])[: int(residential["unit_count"])]]
        base_areas += [80.0] * (int(residential["unit_count"]) - len(base_areas))
        area_frame = pd.DataFrame(
            {
                "户型": [f"{prefix}_{index + 1:02d}" for index in range(int(residential["unit_count"]))],
                "目标面积（㎡）": base_areas,
            }
        )
        edited_frame = st.data_editor(
            area_frame,
            num_rows="fixed",
            hide_index=True,
            use_container_width=True,
            key=f"{config_id}_floor_partition_area_table_{residential['unit_count']}",
            column_config={"目标面积（㎡）": st.column_config.NumberColumn(min_value=0.1, step=0.5, format="%.2f")},
        )
        try:
            values = [float(value) for value in edited_frame["目标面积（㎡）"]]
            if len(values) != int(residential["unit_count"]):
                raise ValueError("每户都需要填写目标面积。")
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError("目标面积必须是大于 0 的有限数值。")
            residential["target_areas"] = values
        except (TypeError, ValueError) as exc:
            st.error(f"逐户面积无效：{exc}")
            return False
        if usable_summary is not None:
            boundary_area, _fixed_area, usable_area = usable_summary
            total_requested = sum(values)
            gap = total_requested - usable_area
            if abs(gap) <= 0.05:
                st.success(
                    f"验算通过：逐户面积合计 {total_requested:.2f} ㎡，与总可用面积 {usable_area:.2f} ㎡ 一致"
                    f"（实际分配会再扣除走道与隔墙后按比例调整）。"
                )
            elif gap > 0:
                st.warning(
                    f"验算：逐户面积合计 {total_requested:.2f} ㎡ 超过总可用面积 {usable_area:.2f} ㎡"
                    f"（超出 {gap:.2f} ㎡），分区时会按比例缩小到实际可分配面积。"
                )
            else:
                st.info(
                    f"验算：逐户面积合计 {total_requested:.2f} ㎡ 少于总可用面积 {usable_area:.2f} ㎡"
                    f"（差 {-gap:.2f} ㎡），分区时会按比例放大到实际可分配面积。"
                )

    if not render_partition_constraints(floor, config_id):
        return False
    if not floor["enabled"]:
        return True
    if str(floor.get("program_type", "residential")) != "residential":
        st.info("办公分区模块还没接入具体规则与训练环境，这里先保留选项位。")
        return False

    try:
        if not preview:
            from core.floor_partition.quality import settings
            problem = build_floor_partition_problem(config)
            settings(problem)
            st.caption('这里只编辑楼层输入：外轮廓、结构约束和交通核开口。显示图层可开启粉色分户推演；它不是训练结果，也不属于结构约束。正式方案在「布局预览」中选择。')
            return True
        problem, result = run_residential_floor_partition(config)
        physical=wall_geometry(problem,result.corridor,result.unit_polygons,result.doors)
        report=validate_partition(problem,result)
        st.success(
            f"分区预校验通过：{len(problem.targets)} 户，开口朝向 {result.opening_side}，"
            f"走道净面积 {physical.corridor.area:.2f} ㎡，户型净面积合计 {sum(p.area for p in physical.units.values()):.2f} ㎡，"
            f"隔墙实体占地 {physical.solid.area:.2f} ㎡。"
        )
        st.caption('图例：灰色为生成隔墙，深色为原有结构，黑色斜线为交通核，蓝色为公共走道。'
                   '当前为有限搜索预览，不代表训练完成或全局最优。')
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "户型": unit_id,
                        "目标面积": round(float(result.target_areas[unit_id]), 2),
                        "实际净面积": round(float(geometry.area), 2),
                        "面积偏差(%)": round(report['units'][unit_id]['area_error_ratio']*100, 2),
                        "采光机会覆盖(%)": round(report['units'][unit_id]['daylight_coverage_proxy']*100, 1),
                        "有效外墙(m)": round(geometry.boundary.intersection(problem.boundary.exterior.difference(problem.fixed_union)).length,2),
                        "所需外墙(m)": round(max(float(floor.get('quality',{}).get('min_facade_length',3.)),geometry.area*float(floor.get('quality',{}).get('facade_per_area',.12))),2),
                        "门洞中线坐标": [
                            [round(float(value), 2) for value in next(door.points for door in result.doors if door.unit_id == unit_id)[0]],
                            [round(float(value), 2) for value in next(door.points for door in result.doors if door.unit_id == unit_id)[1]],
                        ],
                    }
                    for unit_id, geometry in sorted(physical.units.items())
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        with st.expander('导出户内训练净边界', expanded=False):
            st.caption('下载每户的配置：边界已扣除墙厚，门位在户内完成面。分户隔墙不会变成结构约束；'
                       '导入后仍需设置房间需求。此下载对应当前预览，不会覆盖已有配置或模型。')
            st.download_button('下载户内净边界配置（ZIP）', net_unit_archive(config,result),
                file_name=f'{config_id}_net_units.zip', mime='application/zip',
                key=f'{config_id}_partition_net_download')
            st.download_button('下载分户验收报告（JSON）', json.dumps(report,ensure_ascii=False,indent=2),
                file_name=f'{config_id}_partition_validation.json', mime='application/json',
                key=f'{config_id}_partition_report_download')
        return True
    except Exception as exc:
        st.error(f"楼层分区预校验失败：{exc}")
        return False


def render_adaptive_reuse_config_page(
    config: dict,
    config_id: str,
    *,
    include_environment: bool = True,
) -> None:
    consume_auto_repair_notice(config_id)

    stage = st.session_state.get(f'{config_id}_training_stage', config.get('Training', {}).get('training_stage', 'room_training'))
    with st.expander('当前阶段的求解方法', expanded=False):
        if stage == 'floor_partition':
            st.caption('户型、户门和走道共同构成状态。先生成可行起点，再由空间图 PPO 选择交界与修改范围，'
                       '通过局部搜索调整布局。每一步按墙厚后的净面积、通行、结构对齐和采光机会验收。')
        else:
            st.markdown(
                "- **状态**：功能空间位置、尺度、目标面积差、形态、与固定构件/其他智能体的距离、功能关系、原空间复用率与整体改造率。\n"
                "- **基础动作**：矩形直接优化使用移动、保持及四边伸缩，共 13 类动作；图种子加速将位置决策压缩为移动/保持 5 类动作，轮廓交给约束生长。\n"
                "- **硬约束**：建筑边界、柱、承重墙、核心筒、交通核区域、保留交通空间和智能体重叠；非法动作被屏蔽并回退。\n"
                "- **奖励**：面积、形态、邻近/分离、结构网格顺应、原空间利用、改造干预、非法动作与团队协调。\n"
                "- **验收**：矩形直接优化按连续满足基础条件终止；加速训练按回合步数采样，每 250 轮及结束/停止时精确成图，检查实际轮廓和共边关系。"
            )

    training = _ensure_training_stage_defaults(config)
    environment = config.setdefault("AdaptiveReuseEnvironment", {})
    floor_partition = _ensure_floor_partition_defaults(config)
    training["training_stage"] = st.session_state.get(f"{config_id}_training_stage", training.get("training_stage", "room_training"))
    floor_partition["program_type"] = st.session_state.get(f"{config_id}_floor_partition_type", floor_partition.get("program_type", "residential"))
    floor_partition["enabled"] = _is_floor_partition_stage(config)
    stage_is_floor_partition = _is_floor_partition_stage(config)
    seed_settings=config.setdefault('SeedGrowth',{})
    has_residual=any(r.get('role')=='residual' for r in config.get('TargetSpaces',[]))
    if stage_is_floor_partition:
        seed_mode = False
        seed_settings['enabled'] = False
        st.info("当前选择的是楼层功能分区阶段，下面的房间级图种子加速与户内矩形优化参数会自动隐藏。")
    else:
        requires_polygons=bool(config.get('FloorWorkflow'))
        seed_mode=st.checkbox('在原矩形基础上启用图种子加速优化',value=True if has_residual or requires_polygons else bool(seed_settings.get('enabled',True)),key=f'{config_id}_residual_accel' if has_residual else f'{config_id}_seed_enabled',disabled=has_residual or requires_polygons)
        if requires_polygons:
            st.caption('逐户合成使用精确多边形结果，因此保持图种子模式；完成后到「整体合成」选择本户结果。')
        seed_settings['enabled']=seed_mode
    if seed_mode and not stage_is_floor_partition:
        st.info('原始空间/初始矩形与功能关系 → 移动种子并快速估算共同优化目标 → 约束生长精炼轮廓。精确阶段按图关系联合调整共边，再将余量分配给客厅并验收连通及户门。每 250 轮、结束或停止时成图；实际邻接以共边验收为准。')
        st.caption('两种求解方式共用房间、结构、目标面积、功能关系及奖励权重。加速训练从已配置种子（默认初始矩形中心）出发；原空间矩形作为复用与改造干预的固定参照。')
    left, right = st.columns(2)
    with left:
        st.markdown("### 训练设置")
        if stage_is_floor_partition:
            st.text('算法模型：空间图 PPO（多步联合修改户型、户门与走道）')
        else:
            training["agent_name"] = st.selectbox("算法模型", ["mappo"], key=f"{config_id}_ar_agent")
        training["episodes"] = int(st.number_input("训练轮数", 1, value=int(training.get("episodes", 1000)), key=f"{config_id}_ar_episodes"))
        training["max_steps"] = int(st.number_input("每轮最大决策步数", 1, value=int(training.get("max_steps", 240)), key=f"{config_id}_ar_steps",disabled=stage_is_floor_partition))
        minimum_batch = 2 if seed_mode else 1
        training["batch_size"] = int(st.number_input("批量大小", minimum_batch, value=max(minimum_batch, int(training.get("batch_size", 64))), key=f"{config_id}_ar_batch",disabled=stage_is_floor_partition))
        if stage_is_floor_partition:
            rl=floor_partition.setdefault('rl',{})
            rl['enabled']=True
            rl['algorithm']='graph_ppo'
            rl['steps_per_episode']=int(st.number_input('每轮分户修改次数',min_value=1,max_value=128,value=int(rl.get('steps_per_episode',8)),key=f'{config_id}_partition_steps'))
            rl['learning_rate']=float(st.number_input('分户策略学习率',min_value=.000001,value=float(rl.get('learning_rate',.0003)),format='%.6f',key=f'{config_id}_partition_rl_lr'))
            rl['seed']=int(st.number_input('分户策略随机种子',value=int(rl.get('seed',42)),key=f'{config_id}_partition_rl_seed'))
            joint=floor_partition.setdefault('joint_search',{})
            joint['enabled']=True
            joint['max_area']=float(st.number_input('一次修改最大面积（㎡）',min_value=1.,value=float(joint.get('max_area',100.)),key=f'{config_id}_partition_max_area'))
            joint['seconds_per_action']=float(st.number_input('每次局部搜索时间预算（秒）',min_value=.1,value=float(joint.get('seconds_per_action',1.5)),key=f'{config_id}_partition_seconds'))
            joint['max_cells']=int(st.number_input('一次修改最多几何单元数',min_value=1,value=int(joint.get('max_cells',160)),key=f'{config_id}_partition_max_cells'))
            joint['proposals']=int(st.number_input('每次局部修改最多候选数',min_value=1,value=int(joint.get('proposals',32)),key=f'{config_id}_partition_proposals'))
            st.caption('局部范围同时受面积和几何单元数限制；候选数与时间预算限制每次搜索的计算量。环境中的门前净空、墙厚和采光设置也会用于训练验收。')
        else:
            training["lr"] = float(st.number_input("学习率", min_value=0.000001, value=float(training.get("lr", 0.0003)), format="%.6f", key=f"{config_id}_ar_lr"))
            training["seed"] = int(st.number_input("随机种子", value=int(training.get("seed", 42)), key=f"{config_id}_ar_seed"))
    with right:
        if stage_is_floor_partition:
            st.markdown("### 当前阶段说明")
            st.caption("先获得可行初始方案，再由图策略多步选择修改交界和范围。局部搜索共同调整空间归属与户门，每步检查面积、外墙采光机会、进深、形态和通行；保留已验收的最好方案。当前初始方案仍依赖原有生成器，尚不支持分离交通核，也未证明跨建筑泛化。")
        else:
            st.markdown("### 矩形直接优化参数")
            if seed_mode:
                st.caption("以下参数保留给矩形直接优化；加速方式使用下方种子网格，按每轮最大步数采样，不以估算结果提前宣告成功。规则网格尺寸仍用于共同的结构顺应奖励。")
            environment["grid_size"] = float(st.number_input("规则网格尺寸（m）", 0.1, value=float(environment.get("grid_size", 1.0)), step=0.1, key=f"{config_id}_ar_grid"))
            environment["move_step"] = float(st.number_input("单次移动距离（m）", 0.1, value=float(environment.get("move_step", 1.0)), step=0.1, key=f"{config_id}_ar_move", disabled=seed_mode))
            environment["resize_step"] = float(st.number_input("单次尺度变化（m）", 0.1, value=float(environment.get("resize_step", 1.0)), step=0.1, key=f"{config_id}_ar_resize", disabled=seed_mode))
            environment["area_tolerance"] = float(st.slider("目标面积容差", 0.01, 0.50, float(environment.get("area_tolerance", 0.12)), 0.01, key=f"{config_id}_ar_area_tol", disabled=seed_mode))
            environment["max_intervention_ratio"] = float(st.slider("终止允许的最大改造率", 0.0, 1.0, float(environment.get("max_intervention_ratio", 0.45)), 0.01, key=f"{config_id}_ar_intervention", disabled=seed_mode))
            environment["success_patience"] = int(st.number_input("连续满足步数", 1, value=int(environment.get("success_patience", 8)), key=f"{config_id}_ar_patience", disabled=seed_mode))
            environment["randomize_initial"] = st.checkbox("训练时随机扰动初始位置", value=bool(environment.get("randomize_initial", True)), key=f"{config_id}_ar_random", disabled=seed_mode)

    config_valid = True
    if include_environment:
        config_valid = _render_plan_constraint_editor(config, config_id)

    if stage_is_floor_partition:
        st.markdown('### 分户评价')
        st.caption('初始方案、局部搜索和训练共用评分：净面积偏差、走道占比、隔墙转折、短边、采光机会和结构对齐一致性。'
                   '墙厚、门前净空、面积免罚范围等在「环境搭建」中设置并保存。')
    else:
        st.markdown("### 奖励权重")
        st.caption("这里只保留论文方法对应的基础奖励项，不设“额外奖励”分组。")
        if seed_mode:
            st.caption('加速训练使用下列共同权重：复用/改造率按原矩形参照快速估算，邻接增加接近过程反馈，形态包含轮廓规整与碎片惩罚；最终质量以精确成图为准。')
        weights = config.setdefault("RewardWeights", {})
        reward_columns = st.columns(2)
        for index, (key, label) in enumerate(REWARD_LABELS.items()):
            with reward_columns[index % 2]:
                weights[key] = float(st.slider(label, 0.0, 10.0, float(weights.get(key, 1.0)), 0.1, key=f"{config_id}_ar_reward_{key}"))

    if not include_environment:
        if st.button("保存参数配置", type="primary", use_container_width=True):
            try:
                saved_path = save_config(config, config_id)
                st.success(f"参数配置已保存：{saved_path}")
            except Exception as exc:
                st.error(f"参数配置未保存：{exc}")
        return

    repair_col, save_col, preview_col = st.columns(3)
    with repair_col:
        if st.button("自动修复初始布局并保存", use_container_width=True, disabled=(not config_valid) or stage_is_floor_partition):
            try:
                run_auto_repair_and_rerun(config, config_id, int(training.get("seed", 42)))
            except Exception as exc:
                st.error(f"自动修复失败：{exc}")
    with save_col:
        if st.button("保存既有建筑环境", type="primary", use_container_width=True,disabled=not config_valid):
            try:
                if config.get("FloorPartition", {}).get("enabled", False):
                    run_residential_floor_partition(config)
                elif seed_mode:
                    from core.seed_growth.environment import SeedLayoutEnv
                    SeedLayoutEnv(config)
                else:
                    if any(e['type'] not in ('adjacent','separate','none') for e in config['FunctionalRelations']):
                        raise ValueError('connected 请启用图种子加速优化；矩形直接优化仅支持 adjacent / separate / none')
                    AdaptiveReuseEnv(config).reset(seed=int(training.get("seed", 42)))
                saved_path = save_config(config, config_id)
                st.success(f"环境校验通过并已保存：{saved_path}")
            except Exception as exc:
                st.error(f"环境未保存：{exc}")
    with preview_col:
        preview_requested = st.button("刷新环境预览", use_container_width=True)

    st.markdown("### 环境预览")
    if stage_is_floor_partition:
        if not config_valid:
            st.info('请先处理分户参数或几何验收问题，再查看当前楼层预览。')
            return
        st.markdown("#### 楼层分区中间结果预览")
        try:
            partition_problem, partition_result = run_residential_floor_partition(config)
            net=wall_geometry(partition_problem,partition_result.corridor,partition_result.unit_polygons,partition_result.doors)
            metric_columns = st.columns(4)
            metric_columns[0].metric("户数", len(partition_result.unit_polygons))
            metric_columns[1].metric("走道净面积", f"{net.corridor.area:.1f} ㎡")
            metric_columns[2].metric("户型净面积合计", f"{sum(p.area for p in net.units.values()):.1f} ㎡")
            metric_columns[3].metric("面积缩放系数", f"{partition_result.area_scale:.3f}")
        except Exception as exc:
            st.error(f"楼层分区预览失败：{exc}")
        return
    if not seed_mode:
        with st.expander('查看已有加速优化成图'):
            from gui.seed_page import render_seed_results
            render_seed_results(config_id)
    if seed_mode:
        from gui.seed_page import render_seed_preview,render_seed_results
        if config_valid:
            render_seed_preview(config)
        render_seed_results(config_id)
        st.markdown("#### 原矩形基础布局（与上方加速优化共用配置）")
    try:
        preview_config = yaml.safe_load(yaml.safe_dump(config, allow_unicode=True))
        preview_config["AdaptiveReuseEnvironment"]["randomize_initial"] = False
        env = AdaptiveReuseEnv(_private_rectangle_config(preview_config))
        _, info = env.reset(seed=int(training.get("seed", 42)))
        if info.get("initial_repairs"):
            st.info(
                "预览已自动修复初始布局："
                + "；".join(
                    f"{item['space_id']} ({item['action_steps']} 步)"
                    for item in info["initial_repairs"]
                )
            )
        st.pyplot(env.render(show_original=True), use_container_width=True)
        metrics = info["metrics"]
        metric_columns = st.columns(4)
        metric_columns[0].metric("面积达标率", f"{metrics['area_compliance']:.0%}")
        metric_columns[1].metric("形态达标率", f"{metrics['shape_compliance']:.0%}")
        metric_columns[2].metric("原空间利用率", f"{metrics['original_reuse']:.0%}")
        metric_columns[3].metric("硬约束冲突", int(metrics["hard_conflicts"]))
        st.caption(f"网格矩阵尺寸：{info['grid_matrix'].shape[1]} × {info['grid_matrix'].shape[0]}；智能体数：{env.num_agents}；动作数：{len(info['action_names'])}")
    except Exception as exc:
        st.error(f"当前参数无法构成有效环境：{exc}")
def render_adaptive_reuse_environment_page(config: dict, config_id: str) -> None:
    """Render environment construction and all environment-derived previews."""
    st.subheader("环境搭建")
    st.caption("在同一工作区完成原始平面标注、图节点关系检查、环境校验与成图预览。")
    consume_auto_repair_notice(config_id)

    training = _ensure_training_stage_defaults(config)
    floor_partition = _ensure_floor_partition_defaults(config)
    training["training_stage"] = st.session_state.get(
        f"{config_id}_training_stage",
        training.get("training_stage", "room_training"),
    )
    floor_partition["program_type"] = st.session_state.get(
        f"{config_id}_floor_partition_type",
        floor_partition.get("program_type", "residential"),
    )
    floor_partition["enabled"] = _is_floor_partition_stage(config)

    editor_open = render_collapsible_subheader(
        "二维原始平面与前期约束标注",
        f"{config_id}_environment_editor_collapsed",
    )
    config_valid = (
        _render_plan_constraint_editor(config, config_id, show_heading=False)
        if editor_open
        else True
    )
    stage_is_floor_partition = _is_floor_partition_stage(config)
    seed_settings = config.setdefault("SeedGrowth", {})
    seed_mode = False if stage_is_floor_partition else bool(seed_settings.get("enabled", True))

    repair_col, save_col, preview_col = st.columns(3)
    with repair_col:
        if st.button(
            "自动修复初始布局并保存",
            use_container_width=True,
            disabled=(not config_valid) or stage_is_floor_partition,
            key=f"{config_id}_environment_auto_repair",
        ):
            try:
                run_auto_repair_and_rerun(config, config_id, int(training.get("seed", 42)))
            except Exception as exc:
                st.error(f"自动修复失败：{exc}")
    with save_col:
        if st.button(
            "保存既有建筑环境",
            type="primary",
            use_container_width=True,
            disabled=not config_valid,
            key=f"{config_id}_environment_save",
        ):
            try:
                if config.get("FloorPartition", {}).get("enabled", False):
                    run_residential_floor_partition(config)
                elif seed_mode:
                    from core.seed_growth.environment import SeedLayoutEnv

                    SeedLayoutEnv(config)
                else:
                    if any(
                        edge["type"] not in ("adjacent", "separate", "none")
                        for edge in config["FunctionalRelations"]
                    ):
                        raise ValueError(
                            "connected 请启用图种子加速优化；矩形直接优化仅支持 adjacent / separate / none"
                        )
                    AdaptiveReuseEnv(config).reset(seed=int(training.get("seed", 42)))
                saved_path = save_config(config, config_id)
                st.success(f"环境校验通过并已保存：{saved_path}")
            except Exception as exc:
                st.error(f"环境未保存：{exc}")
    with preview_col:
        st.button(
            "刷新环境预览",
            use_container_width=True,
            key=f"{config_id}_environment_refresh",
        )

    preview_open = render_collapsible_subheader(
        "环境预览",
        f"{config_id}_environment_preview_collapsed",
    )
    if not preview_open:
        return
    if stage_is_floor_partition:
        st.info('楼层输入编辑完成后，请保存环境；到「布局预览」查看未训练预览或已保存的第一阶段结果。')
        return
    if not config.get('TargetSpaces'):
        st.info('请先在「智能体与关系」中添加本户房间，再预览或训练户内布局。')
        return

    from gui.seed_page import render_seed_preview, render_seed_results

    if seed_mode and config_valid:
        render_seed_preview(config)

    render_seed_results(config_id,config)
    st.markdown("#### 原矩形基础布局")
    try:
        preview_config = yaml.safe_load(yaml.safe_dump(config, allow_unicode=True))
        preview_config["AdaptiveReuseEnvironment"]["randomize_initial"] = False
        env = AdaptiveReuseEnv(_private_rectangle_config(preview_config))
        _, info = env.reset(seed=int(training.get("seed", 42)))
        if info.get("initial_repairs"):
            st.info(
                "预览已自动修复初始布局："
                + "；".join(
                    f"{item['space_id']} ({item['action_steps']} 步)"
                    for item in info["initial_repairs"]
                )
            )
        st.pyplot(env.render(show_original=True), use_container_width=True)
        metrics = info["metrics"]
        metric_columns = st.columns(4)
        metric_columns[0].metric("面积达标率", f"{metrics['area_compliance']:.0%}")
        metric_columns[1].metric("形态达标率", f"{metrics['shape_compliance']:.0%}")
        metric_columns[2].metric("原空间利用率", f"{metrics['original_reuse']:.0%}")
        metric_columns[3].metric("硬约束冲突", int(metrics["hard_conflicts"]))
        st.caption(
            f"网格矩阵尺寸：{info['grid_matrix'].shape[1]} × {info['grid_matrix'].shape[0]}；"
            f"智能体数：{env.num_agents}；动作数：{len(info['action_names'])}"
        )
    except Exception as exc:
        st.error(f"当前参数无法构成有效环境：{exc}")
