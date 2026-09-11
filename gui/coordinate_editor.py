# Adapted from community_site_marl/ui/coordinate_editor.py.
# Parent page owns saving; rerun the page so edits are available to its save button.
from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None


PLOT_PADDING_LEFT = 46
PLOT_PADDING_RIGHT = 64
PLOT_PADDING_TOP = 44
PLOT_PADDING_BOTTOM = 42


def _clean_points(points) -> list[list[float]]:
    return [
        [round(float(point[0]), 2), round(float(point[1]), 2)]
        for point in points
    ]


def _signature(
    groups: list[dict[str, Any]],
    north_angle_deg: float = 0.0,
) -> tuple:
    return (
        round(float(north_angle_deg), 4),
        tuple(
        (
            group["name"],
            tuple(tuple(point) for point in _clean_points(group["points"])),
            tuple(
                (
                    tuple(segment[0]),
                    tuple(segment[1]),
                )
                for segment in group.get("preview_segments", [])
            ),
        )
        for group in groups
        ),
    )


def _nice_step(span: float) -> float:
    target = max(span / 10.0, 0.1)
    magnitude = 10 ** math.floor(math.log10(target))
    normalized = target / magnitude
    factor = 1 if normalized < 2 else 2 if normalized < 5 else 5
    return factor * magnitude


def _view_bounds(
    groups: list[dict[str, Any]],
) -> tuple[float, float, float, float]:
    points = [point for group in groups for point in group.get("points", [])]
    for group in groups:
        for segment in group.get("preview_segments", []):
            points.extend(segment)
    if not points:
        return -10.0, -10.0, 10.0, 10.0
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    span_x = max(max(xs) - min(xs), 1.0)
    span_y = max(max(ys) - min(ys), 1.0)
    padding_x = max(span_x * 0.12, 2.0)
    padding_y = max(span_y * 0.12, 2.0)
    return (
        min(xs) - padding_x,
        min(ys) - padding_y,
        max(xs) + padding_x,
        max(ys) + padding_y,
    )


def _primary_closed_group(groups: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for group in groups:
        points = group.get("points", [])
        if group.get("closed") and len(points) >= 2:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            candidates.append((span_x * span_y, group))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _group_span(group: dict[str, Any]) -> tuple[float, float]:
    points = group.get("points", [])
    if len(points) < 2:
        return 0.0, 0.0
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return max(xs) - min(xs), max(ys) - min(ys)


def _plot_frame(width: int, height: int) -> tuple[float, float, float, float]:
    return (
        float(PLOT_PADDING_LEFT),
        float(PLOT_PADDING_TOP),
        float(width - PLOT_PADDING_RIGHT),
        float(height - PLOT_PADDING_BOTTOM),
    )


def _to_canvas(point, bounds, width, height):
    min_x, min_y, max_x, max_y = bounds
    left, top, right, bottom = _plot_frame(width, height)
    plot_width = max(right - left, 1.0)
    plot_height = max(bottom - top, 1.0)
    x = left + (float(point[0]) - min_x) / (max_x - min_x) * plot_width
    y = bottom - (
        (float(point[1]) - min_y) / (max_y - min_y) * plot_height
    )
    return x, y


def _from_canvas(x, y, bounds, width, height):
    min_x, min_y, max_x, max_y = bounds
    left, top, right, bottom = _plot_frame(width, height)
    plot_width = max(right - left, 1.0)
    plot_height = max(bottom - top, 1.0)
    clamped_x = min(max(float(x), left), right)
    clamped_y = min(max(float(y), top), bottom)
    coordinate_x = min_x + (clamped_x - left) / plot_width * (max_x - min_x)
    coordinate_y = min_y + (bottom - clamped_y) / plot_height * (max_y - min_y)
    return [round(coordinate_x, 2), round(coordinate_y, 2)]


def _static_line(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    width: int,
    role: str,
):
    return {
        "type": "line",
        "x1": round(start[0], 2),
        "y1": round(start[1], 2),
        "x2": round(end[0], 2),
        "y2": round(end[1], 2),
        "stroke": color,
        "strokeWidth": width,
        "selectable": False,
        "evented": False,
        "hasControls": False,
        "lockMovementX": True,
        "lockMovementY": True,
        "lockRotation": True,
        "lockScalingX": True,
        "lockScalingY": True,
        "excludeFromExport": True,
        "role": role,
    }


def _static_text(
    text: str,
    position: tuple[float, float],
    *,
    color: str,
    font_size: int = 11,
    role: str = "label",
):
    return {
        "type": "text",
        "text": text,
        "left": round(position[0], 2),
        "top": round(position[1], 2),
        "fontSize": font_size,
        "fill": color,
        "selectable": False,
        "evented": False,
        "hasControls": False,
        "lockMovementX": True,
        "lockMovementY": True,
        "lockRotation": True,
        "lockScalingX": True,
        "lockScalingY": True,
        "excludeFromExport": True,
        "role": role,
    }


def _compass_objects(
    bounds,
    width: int,
    height: int,
    north_angle_deg: float,
):
    objects = []
    angle = math.radians(float(north_angle_deg))
    right = float(width - 26)
    top = float(22)
    center = (right, top)
    radius = 18.0
    tip_radius = 13.0
    label_radius = 24.0
    labels = (
        ("N", 0.0, "#dc2626", 12),
        ("E", 90.0, "#475569", 11),
        ("S", 180.0, "#475569", 11),
        ("W", 270.0, "#475569", 11),
    )
    objects.append(
        {
            "type": "circle",
            "left": round(center[0] - radius, 2),
            "top": round(center[1] - radius, 2),
            "radius": radius,
            "width": round(radius * 2, 2),
            "height": round(radius * 2, 2),
            "fill": "rgba(255,255,255,0.0)",
            "stroke": "#94a3b8",
            "strokeWidth": 1.5,
            "selectable": False,
            "evented": False,
            "hasControls": False,
            "lockMovementX": True,
            "lockMovementY": True,
            "lockRotation": True,
            "lockScalingX": True,
            "lockScalingY": True,
            "excludeFromExport": True,
            "role": "compass",
        }
    )
    for text, base_deg, color, font_size in labels:
        label_angle = angle + math.radians(base_deg)
        dx = math.sin(label_angle)
        dy = -math.cos(label_angle)
        position = (
            center[0] + dx * label_radius - font_size * 0.35,
            center[1] + dy * label_radius - font_size * 0.55,
        )
        objects.append(
            _static_text(
                text,
                position,
                color=color,
                font_size=font_size,
                role="compass",
            )
        )
        if text == "N":
            tip = (
                center[0] + dx * tip_radius,
                center[1] + dy * tip_radius,
            )
            objects.append(
                _static_line(
                    center,
                    tip,
                    color="#dc2626",
                    width=3,
                    role="compass",
                )
            )
    return objects


def _static_objects(groups, bounds, width, height, north_angle_deg: float = 0.0):
    objects = []
    min_x, min_y, max_x, max_y = bounds
    left, top, right, bottom = _plot_frame(width, height)
    step = _nice_step(max(max_x - min_x, max_y - min_y) / 1.8)

    first_x = math.ceil(min_x / step) * step
    x = first_x
    while x <= max_x + 1e-9:
        px, _ = _to_canvas((x, min_y), bounds, width, height)
        color = "#94a3b8" if abs(x) < step * 0.01 else "#d1d5db"
        objects.append(
            _static_line(
                (px, top),
                (px, bottom),
                color=color,
                width=1,
                role="grid",
            )
        )
        objects.append(
            _static_text(
                f"{x:g}",
                (px - 10, bottom + 8),
                color="#64748b",
                font_size=10,
            )
        )
        x += step

    first_y = math.ceil(min_y / step) * step
    y = first_y
    while y <= max_y + 1e-9:
        _, py = _to_canvas((min_x, y), bounds, width, height)
        color = "#94a3b8" if abs(y) < step * 0.01 else "#d1d5db"
        objects.append(
            _static_line(
                (left, py),
                (right, py),
                color=color,
                width=1,
                role="grid",
            )
        )
        objects.append(
            _static_text(
                f"{y:g}",
                (4, py - 8),
                color="#64748b",
                font_size=10,
            )
        )
        y += step

    objects.append(
        _static_line(
            (left, bottom),
            (right, bottom),
            color="#475569",
            width=2,
            role="axis",
        )
    )
    objects.append(
        _static_line(
            (left, bottom),
            (left, top),
            color="#475569",
            width=2,
            role="axis",
        )
    )
    objects.append(
        _static_text("X", (right - 8, bottom + 18), color="#334155", font_size=11)
    )
    objects.append(
        _static_text("Y", (10, top - 8), color="#334155", font_size=11)
    )
    objects.extend(_compass_objects(bounds, width, height, north_angle_deg))

    for group in groups:
        pixels = [
            _to_canvas(point, bounds, width, height)
            for point in group["points"]
        ]
        if len(pixels) >= 2:
            path = pixels + [pixels[0]] if group.get("closed") else pixels
            for start, end in zip(path, path[1:]):
                objects.append(
                    _static_line(
                        start,
                        end,
                        color=group["color"],
                        width=3,
                        role="outline",
                    )
                )
        for segment in group.get("preview_segments", []):
            start = _to_canvas(segment[0], bounds, width, height)
            end = _to_canvas(segment[1], bounds, width, height)
            objects.append(
                _static_line(
                    start,
                    end,
                    color=group.get("preview_color", group["color"]),
                    width=int(group.get("preview_width", 4)),
                    role="preview",
                )
            )
    return objects


def _initial_objects(groups, bounds, width, height, north_angle_deg: float = 0.0):
    objects = _static_objects(groups, bounds, width, height, north_angle_deg)
    radius = 8
    for group in groups:
        for index, point in enumerate(group["points"]):
            x, y = _to_canvas(point, bounds, width, height)
            objects.append(
                {
                    "type": "circle",
                    "left": x - radius,
                    "top": y - radius,
                    "radius": radius,
                    "width": radius * 2,
                    "height": radius * 2,
                    "fill": group["color"],
                    "stroke": "#111827",
                    "strokeWidth": 1,
                    "lockScalingX": True,
                    "lockScalingY": True,
                    "lockRotation": True,
                    "hasControls": False,
                    "groupName": group["name"],
                    "pointIndex": index,
                    "role": "control",
                }
            )
    return objects


def _normalize_color(value: str | None) -> str:
    if not value:
        return ""
    value = value.lower().replace(" ", "")
    if value.startswith("rgba(") or value.startswith("rgb("):
        numbers = value[value.find("(") + 1 : value.find(")")].split(",")
        if len(numbers) >= 3:
            return "#{:02x}{:02x}{:02x}".format(
                int(float(numbers[0])),
                int(float(numbers[1])),
                int(float(numbers[2])),
            )
    return value[:7]


def _parse_objects(objects, groups, bounds, width, height):
    colors = {
        _normalize_color(group["color"]): group["name"] for group in groups
    }
    expected_counts = {
        group["name"]: len(group.get("points", [])) for group in groups
    }
    indexed_points = {group["name"]: {} for group in groups}
    unordered_points = {group["name"]: [] for group in groups}
    control_count = 0
    for obj in objects:
        if obj.get("type") != "circle":
            continue
        group_name = obj.get("groupName") or colors.get(
            _normalize_color(obj.get("fill"))
        )
        if group_name is None:
            continue
        control_count += 1
        scale_x = float(obj.get("scaleX", 1.0))
        scale_y = float(obj.get("scaleY", 1.0))
        width_value = float(obj.get("width", 16.0))
        height_value = float(obj.get("height", 16.0))
        center_x = float(obj.get("left", 0.0)) + width_value * scale_x / 2
        center_y = float(obj.get("top", 0.0)) + height_value * scale_y / 2
        point = _from_canvas(center_x, center_y, bounds, width, height)
        point_index = obj.get("pointIndex")
        if point_index is None:
            unordered_points[group_name].append(point)
        else:
            indexed_points[group_name][int(point_index)] = point
    if control_count == 0:
        return None
    result = {}
    total_expected = sum(expected_counts.values())
    total_parsed = 0
    for group in groups:
        name = group["name"]
        points = [
            point
            for _, point in sorted(indexed_points[name].items())
        ] + unordered_points[name]
        result[name] = points
        total_parsed += len(points)
        if len(points) != expected_counts[name]:
            return None
    if total_expected > 0 and total_parsed != total_expected:
        return None
    return result


def _rows_from_points(groups):
    rows = []
    for group in groups:
        for index, point in enumerate(group["points"], 1):
            rows.append(
                {
                    "对象": group["name"],
                    "点序号": index,
                    "X": float(point[0]),
                    "Y": float(point[1]),
                }
            )
    return rows


def _points_from_rows(frame: pd.DataFrame, groups):
    valid_names = {group["name"] for group in groups}
    result = {group["name"]: [] for group in groups}
    for _, row in frame.iterrows():
        name = str(row["对象"])
        if name not in valid_names:
            continue
        result[name].append([round(float(row["X"]), 2), round(float(row["Y"]), 2)])
    return result


def _validate_group_sizes(
    points_by_group: dict[str, list[list[float]]],
    groups: list[dict[str, Any]],
) -> list[str]:
    errors = []
    for group in groups:
        minimum = int(group.get("min_points", 0))
        actual = len(points_by_group.get(group["name"], []))
        if actual < minimum:
            errors.append(
                f"{group['name']} 至少保留 {minimum} 个坐标点，当前为 {actual} 个。"
            )
    return errors


def coordinate_editor(
    label: str,
    groups: list[dict[str, Any]],
    key: str,
    north_angle_deg: float = 0.0,
) -> dict[str, list[list[float]]]:
    """Edit several named point sets by dragging or by numeric table."""
    st.markdown(f"#### {label}")
    st.caption(
        "拖动彩色控制点可修改坐标；下方表格可直接输入数值、增减坐标点。"
        "数值修改后点击“应用数值并刷新画布”。"
    )
    if st_canvas is None:
        st.error(
            "当前环境缺少 streamlit-drawable-canvas，无法显示交互画布。"
        )
        st.code(
            "python -m pip install "
            "streamlit-drawable-canvas"
        )
        return {
            group["name"]: _clean_points(group["points"]) for group in groups
        }

    state_key = f"{key}_coordinate_groups"
    source_key = f"{key}_source_signature"
    revision_key = f"{key}_canvas_revision"
    incoming_signature = _signature(groups, north_angle_deg)
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            group["name"]: _clean_points(group["points"]) for group in groups
        }
        st.session_state[source_key] = incoming_signature
        st.session_state[revision_key] = 0
    elif st.session_state.get(source_key) != incoming_signature:
        # 数据源变化时强制重置，避免切模板后画布不刷新或丢失。
        st.session_state[state_key] = {
            group["name"]: _clean_points(group["points"]) for group in groups
        }
        st.session_state[source_key] = incoming_signature
        st.session_state[revision_key] = (
            st.session_state.get(revision_key, 0) + 1
        )
        st.session_state.pop(f"{key}_numeric_editor", None)

    current = st.session_state[state_key]
    working_groups = [
        {
            **group,
            "points": current.get(
                group["name"], _clean_points(group["points"])
            ),
        }
        for group in groups
    ]
    bounds = _view_bounds(working_groups)
    plot_width = 560
    coordinate_ratio = (bounds[3] - bounds[1]) / max(
        bounds[2] - bounds[0], 1e-6
    )
    plot_height = int(max(360, min(560, plot_width * coordinate_ratio)))
    width = plot_width + PLOT_PADDING_LEFT + PLOT_PADDING_RIGHT
    height = plot_height + PLOT_PADDING_TOP + PLOT_PADDING_BOTTOM
    grid_step = _nice_step(
        max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 1.8
    )
    primary_group = _primary_closed_group(working_groups)
    if primary_group is not None:
        span_x, span_y = _group_span(primary_group)
        st.caption(
            f"网格辅助线间距约 {grid_step:.2f} m；"
            f"主轮廓包络尺寸约为 {span_x:.2f} m × {span_y:.2f} m。"
        )
    else:
        st.caption(f"网格辅助线间距约 {grid_step:.2f} m。")

    legend_columns = st.columns(min(max(len(groups), 1), 4))
    for index, group in enumerate(groups):
        legend_columns[index % len(legend_columns)].markdown(
            f"<span style='color:{group['color']};font-size:22px'>●</span> "
            f"{group['name']}",
            unsafe_allow_html=True,
        )

    canvas_column, table_column = st.columns([1.15, 0.95], gap="medium")
    result = None
    with canvas_column:
        with st.container(border=True):
            try:
                result = st_canvas(
                    fill_color="#2563eb",
                    stroke_width=1,
                    stroke_color="#111827",
                    background_color="#ffffff",
                    update_streamlit=True,
                    height=height,
                    width=width,
                    drawing_mode="transform",
                    key=f"{key}_canvas_{st.session_state[revision_key]}",
                    initial_drawing={
                        "objects": _initial_objects(
                            working_groups,
                            bounds,
                            width,
                            height,
                            north_angle_deg,
                        )
                    },
                    display_toolbar=False,
                )
                st.caption(
                    f"左侧画布可直接拖动控制点，右侧坐标表可精确输入数值。"
                    f"当前指北针 {float(north_angle_deg):.1f}°，默认 0° 为上北。"
                )
            except Exception as error:
                if "PyArrow" in str(error) or "pyarrow" in str(error):
                    st.error(
                        "当前环境缺少 `pyarrow`，交互画布暂时无法加载。"
                        "请执行 `python -m pip install pyarrow` 后重启 Streamlit。"
                    )
                else:
                    st.error(f"交互画布加载失败：{error}")

    if result and result.json_data and "objects" in result.json_data:
        parsed = _parse_objects(
            result.json_data["objects"],
            working_groups,
            bounds,
            width,
            height,
        )
        if parsed is not None and parsed != current:
            st.session_state[state_key] = parsed
            st.session_state[revision_key] += 1
            st.session_state.pop(f"{key}_numeric_editor", None)
            # 仅重跑当前片段，避免拖动画布时整页进入灰化状态。
            st.rerun()

    frame = pd.DataFrame(_rows_from_points(working_groups))
    with table_column:
        st.markdown("##### 坐标表")
        table_height = min(
            52 + max(len(frame), 1) * 31,
            max(180, min(height, 360)),
        )
        edited = st.data_editor(
            frame,
            key=f"{key}_numeric_editor",
            num_rows="dynamic",
            hide_index=True,
            disabled=["点序号"],
            height=table_height,
            row_height=28,
            column_config={
                "对象": st.column_config.SelectboxColumn(
                    "对象",
                    options=[group["name"] for group in groups],
                    required=True,
                    width="small",
                ),
                "点序号": st.column_config.NumberColumn(
                    "点序号", format="%d", width="small"
                ),
                "X": st.column_config.NumberColumn(
                    "X", format="%.2f", width="small"
                ),
                "Y": st.column_config.NumberColumn(
                    "Y", format="%.2f", width="small"
                ),
            },
            use_container_width=True,
        )
        st.caption("可增删行，也可修改对象归属与坐标值。")
    numeric_points = _points_from_rows(edited, groups)
    controls = st.columns([1, 1, 3])
    if controls[0].button(
        "应用数值并刷新画布",
        key=f"{key}_apply_numeric",
        use_container_width=True,
    ):
        errors = _validate_group_sizes(numeric_points, groups)
        if errors:
            for message in errors:
                st.error(message)
        else:
            st.session_state[state_key] = numeric_points
            st.session_state[revision_key] += 1
            st.session_state.pop(f"{key}_numeric_editor", None)
            st.rerun()
    if controls[1].button(
        "恢复配置值",
        key=f"{key}_reset",
        use_container_width=True,
    ):
        st.session_state[state_key] = {
            group["name"]: _clean_points(group["points"]) for group in groups
        }
        st.session_state[source_key] = incoming_signature
        st.session_state[revision_key] += 1
        st.session_state.pop(f"{key}_numeric_editor", None)
        st.rerun()
    return {
        group["name"]: _clean_points(
            st.session_state[state_key].get(group["name"], [])
        )
        for group in groups
    }
