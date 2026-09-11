import ast
import os
import time

import pandas as pd
import streamlit as st
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None

from gui.config_store import parse_room_types, save_config, select_folder
from gui.environment_validation import validate_door_direction
from gui.layout_canvas import (
    ROOM_COLORS,
    build_initial_objects,
    compute_grid_size,
    create_boundary_image,
    get_room_color,
    parse_canvas_objects,
)

DEFAULT_PRIOR_LAYOUT_SCALE = 2.0


def _calculate_polygon_area(points: list[list[float]]) -> float:
    """使用鞋带公式计算轮廓面积。"""
    if not points or len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _get_effective_room_max_area(room_conf: dict, room_name: str, boundary_area: float) -> float:
    """计算当前配置下某类房间的实际最大面积。"""
    if room_conf.get("UseAreasRatio", True):
        ratio = float(room_conf.get("MaxAreasRatio", {}).get(room_name, 0.25))
        return boundary_area * ratio
    return float(room_conf.get("MaxAreasValue", {}).get(room_name, 10.0))


def _parse_float_list(value: str, fallback: list[float]) -> list[float]:
    """解析浮点数列表。"""
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
    except Exception:
        pass
    return fallback


def _parse_literal(value: str, fallback):
    """解析字面量对象。"""
    try:
        return ast.literal_eval(value)
    except Exception:
        return fallback


def render_config_page(config: dict, config_id: str) -> None:
    """渲染参数配置页。"""
    st.subheader("参数配置")
    st.caption("当前页面直接编辑当前配置文件，训练控制仅以配置文件中的数据为准。")

    render_checkpoint_section(config, config_id)
    render_training_section(config, config_id)
    render_reward_section(config, config_id)
    render_room_section(config, config_id)
    render_advanced_section(config, config_id)

    st.markdown("---")
    if st.button("保存当前配置", type="primary", use_container_width=True):
        saved_path = save_config(config, config_id)
        st.success(f"配置已保存到: {saved_path}")

    st.markdown("---")
    render_environment_editor(config, config_id)


def render_checkpoint_section(config: dict, config_id: str) -> None:
    """渲染检查点路径配置。"""
    training_conf = config.setdefault("Training", {})
    input_key = f"{config_id}_ckpt_path"
    pending_key = f"{input_key}__pending"
    pending_folder = st.session_state.pop(pending_key, None)
    if pending_folder:
        training_conf["ckpt_path"] = pending_folder
        st.session_state[input_key] = pending_folder
        save_config(config, config_id)

    st.markdown("### 续训模型目录")
    st.info("仅“真正续训”或“基于模型新训练”需要填写该项；“新训练”会自动清空它并从头开始训练。这里填写的是模型目录，例如 best_model、final_model 或 checkpoints 下的具体目录。")
    st.caption("续训模型目录（新训练可留空）")

    col_path_1, col_path_2 = st.columns([5, 1], vertical_alignment="center")
    with col_path_1:
        training_conf["ckpt_path"] = st.text_input(
            "续训模型目录（新训练可留空）",
            value=training_conf.get("ckpt_path", ""),
            placeholder="例如：d:/.../results2/mappo/17348_xxx/best_model",
            help="这里填写目录而不是单个 .pth 文件。",
            key=input_key,
            label_visibility="collapsed",
        )
    with col_path_2:
        st.markdown("<div style='height: 2px;'></div>", unsafe_allow_html=True)
        if st.button("浏览目录", use_container_width=True):
            folder = select_folder()
            if folder:
                st.session_state[pending_key] = folder
                st.rerun()


def render_training_section(config: dict, config_id: str) -> None:
    """渲染训练与环境基础参数。"""
    training_conf = config.setdefault("Training", {})
    env_conf = config.setdefault("Environment", {})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 训练设置")
        agent_options = ["mappo", "dqn", "madqn"]
        current_agent = training_conf.get("agent_name", "mappo")
        if current_agent not in agent_options:
            current_agent = "mappo"
        training_conf["agent_name"] = st.selectbox("算法模型", agent_options, index=agent_options.index(current_agent), key=f"{config_id}_agent_name")
        training_conf["episodes"] = st.number_input("训练轮数", min_value=1, value=int(training_conf.get("episodes", 10000)), key=f"{config_id}_episodes")
        training_conf["max_steps"] = st.number_input("每轮步数", min_value=1, value=int(training_conf.get("max_steps", 512)), key=f"{config_id}_max_steps")
        training_conf["batch_size"] = st.number_input("批量大小", min_value=1, value=int(training_conf.get("batch_size", 32)), key=f"{config_id}_batch_size")
        training_conf["lr"] = st.number_input("学习率", value=float(training_conf.get("lr", 3e-4)), format="%.5f", key=f"{config_id}_lr")
        training_conf["seed"] = st.number_input(
            "随机种子",
            value=int(training_conf.get("seed", -1)),
            help="-1 表示每次随机；设置为固定整数后可以帮助复现实验结果。",
            key=f"{config_id}_seed",
        )
        training_conf["train"] = st.checkbox("训练模式", value=training_conf.get("train", True), key=f"{config_id}_train")
        training_conf["save_threshold"] = st.number_input(
            "模型保存阈值",
            value=float(training_conf.get("save_threshold", 240)),
            help="当回合平均奖励达到该阈值时，会额外保存 checkpoint 和奖励布局图；它不控制 latest_layout 的按轮更新。",
            key=f"{config_id}_save_threshold",
        )
        training_conf["threshold_save_interval"] = st.number_input(
            "达标保存间隔",
            min_value=1,
            value=int(training_conf.get("threshold_save_interval", 1)),
            help="当回合平均奖励达到保存阈值后，每隔多少轮保存一组 checkpoint 和 images 布局图。例如 20 表示第 20/40/60... 轮保存。",
            key=f"{config_id}_threshold_save_interval",
        )
        training_conf["final_model_save_interval"] = st.number_input(
            "final_model 保存间隔",
            min_value=0,
            value=int(training_conf.get("final_model_save_interval", 1)),
            help="每隔多少个回合更新一次 final_model；0 表示关闭自动更新。",
            key=f"{config_id}_final_model_save_interval",
        )
        training_conf["latest_layout_save_interval"] = st.number_input(
            "latest_layout 保存间隔",
            min_value=0,
            value=int(training_conf.get("latest_layout_save_interval", 1)),
            help="每隔多少个回合更新一次 latest_layout.png；0 表示关闭自动更新。",
            key=f"{config_id}_latest_layout_save_interval",
        )
        training_conf["num_workers"] = st.number_input(
            "并行训练进程数",
            min_value=1,
            max_value=os.cpu_count() or 4,
            value=int(training_conf.get("num_workers", 1)),
            key=f"{config_id}_num_workers",
        )

    with col2:
        st.markdown("### 环境设置")
        room_types = env_conf.get("room_type", [])
        room_types_str = st.text_area(
            "房间类型列表",
            value=str(room_types),
            help="填写需要训练的房间类型列表，例如: ['卧室', '卧室', '厨房', '浴室', '阳台']。",
            key=f"{config_id}_room_type",
        )
        try:
            env_conf["room_type"] = parse_room_types(room_types_str)
        except Exception:
            st.error("房间类型格式错误，请保持 Python 列表格式。")

        reward_options = ["extra", "base"]
        current_reward = env_conf.get("reward_method", "extra")
        if current_reward not in reward_options:
            current_reward = "extra"
        env_conf["reward_method"] = st.selectbox(
            "奖励方式",
            reward_options,
            index=reward_options.index(current_reward),
            help="`base` 表示只使用基础奖励；`extra` 表示基础奖励和额外奖励一起参与训练。",
            key=f"{config_id}_reward_method",
        )
        language_options = ["zh", "en"]
        current_language = config.get("Language", "zh")
        if current_language not in language_options:
            current_language = "zh"
        config["Language"] = st.selectbox(
            "界面语言",
            language_options,
            index=language_options.index(current_language),
            help="控制环境绘图与文字标注的语言。",
            key=f"{config_id}_language",
        )
        init_col1, init_col2 = st.columns(2)
        with init_col1:
            env_conf["init_width"] = st.number_input(
                "初始房间宽度",
                min_value=0.1,
                value=float(env_conf.get("init_width", 1.0)),
                step=0.1,
                help="每个非客厅房间在初始化时的起始宽度，不是环境总轮廓宽度。",
                key=f"{config_id}_init_width",
            )
        with init_col2:
            env_conf["init_height"] = st.number_input(
                "初始房间高度",
                min_value=0.1,
                value=float(env_conf.get("init_height", 1.0)),
                step=0.1,
                help="每个非客厅房间在初始化时的起始高度，不是环境总轮廓高度。",
                key=f"{config_id}_init_height",
            )
        env_conf["prior_random"] = st.checkbox(
            "随机生成先验位置",
            value=bool(env_conf.get("prior_random", False)),
            help="关闭时使用配置中的固定先验点；开启后会在约束条件下随机生成先验位置。",
            key=f"{config_id}_prior_random",
        )
        env_conf["prior_interval"] = st.number_input(
            "先验最小间距",
            min_value=0.1,
            value=float(env_conf.get("prior_interval", 3.0)),
            step=0.1,
            help="仅在随机生成先验位置时生效，表示不同先验点之间的期望最小距离。",
            key=f"{config_id}_prior_interval",
        )
        config["Render"] = st.checkbox(
            "开启渲染",
            value=config.get("Render", False),
            help="训练过程中实时绘制环境与布局变化，便于观察，但会明显降低训练速度。通常调试时开启，正式批量训练时关闭。",
            key=f"{config_id}_render",
        )
        config["RenderEverySteps"] = st.number_input(
            "训练渲染步长",
            min_value=1,
            value=int(config.get("RenderEverySteps", 1)),
            help="训练时每隔多少步才触发一次环境渲染。数值越大越快，但实时画面更新会更慢。",
            key=f"{config_id}_render_every_steps",
            disabled=not config["Render"],
        )
        config["RenderInterval"] = st.number_input(
            "网页实时渲染频率（秒）",
            min_value=0.1,
            max_value=5.0,
            value=float(config.get("RenderInterval", 0.5)),
            step=0.1,
            format="%.1f",
            help="这是文件落盘的时间间隔；配合训练渲染步长一起决定 live 预览的更新频率。建议调试时使用 0.5~1.0，正式训练时使用 1.0 以上。",
            key=f"{config_id}_render_interval",
            disabled=not config["Render"],
        )
        config["MonitorImageAnnotations"] = st.checkbox(
            "训练监控显示面积/比例标注",
            value=bool(config.get("MonitorImageAnnotations", False)),
            help="开启后，训练监控里的实时预览图会在房间内部显示面积、比例，并在左上角显示总轮廓面积。需要同时开启渲染才会生成带标注的 live 预览图。",
            key=f"{config_id}_monitor_image_annotations",
        )
        config["SavedImageAnnotations"] = st.checkbox(
            "保存图片显示面积/比例标注",
            value=bool(config.get("SavedImageAnnotations", False)),
            help="开启后，latest_layout.png 和奖励达标保存的布局图都会带房间面积、比例和总轮廓面积标注。",
            key=f"{config_id}_saved_image_annotations",
        )
        config["Show_Prior"] = st.checkbox("显示先验位置", value=config.get("Show_Prior", False), key=f"{config_id}_show_prior")


def render_reward_section(config: dict, config_id: str) -> None:
    """渲染奖励权重设置。"""
    st.markdown("---")
    st.markdown("### 奖励权重")
    env_conf = config.setdefault("Environment", {})
    reward_advanced = config.setdefault("RewardAdvanced", {})
    base_weight = env_conf.setdefault("Reward_Base_Weight", {})
    reward_advanced.setdefault("prior_layout_scale", DEFAULT_PRIOR_LAYOUT_SCALE)

    base_keys = [
        "面积奖励",
        "长宽比奖励",
        "贴边奖励",
        "房间与房间之间的边界贴合奖励",
        "角落占领奖励",
        "外门遮挡惩罚",
        "先验知识布局奖励",
        "房间嵌套惩罚",
    ]

    st.markdown("#### 基础奖励")
    columns = st.columns(2)
    for index, key in enumerate(base_keys):
        with columns[index % len(columns)]:
            base_weight[key] = st.slider(
                key,
                0.0,
                5.0,
                float(base_weight.get(key, 1.0)),
                0.1,
                key=f"{config_id}_base_{key}",
            )

def render_room_section(config: dict, config_id: str) -> None:
    """渲染房间参数设置。"""
    st.markdown("---")
    st.markdown("### 房间参数")
    room_conf = config.setdefault("Room", {})
    std_rooms = ["卧室", "厨房", "浴室", "阳台", "客厅"]

    room_conf.setdefault("MinAreas", {})
    room_conf.setdefault("MaxAreasRatio", {})
    room_conf.setdefault("MaxAreasValue", {})
    room_conf.setdefault("ASPECT_RATIO_RANGES", {})
    room_conf.setdefault("NOISE_WEIGHTS", {})
    room_conf.setdefault("NESTING_WEIGHTS", {})
    room_conf.setdefault("NESTING_PRIORITY", {})
    boundary_area = _calculate_polygon_area(config.setdefault("Environment", {}).get("boundary", []))

    st.markdown("#### 最小面积")
    cols_min = st.columns(len(std_rooms))
    for index, room in enumerate(std_rooms):
        with cols_min[index]:
            room_conf["MinAreas"][room] = st.number_input(f"{room} 最小面积", value=int(room_conf["MinAreas"].get(room, 4)), key=f"{config_id}_min_{room}")

    st.markdown("#### 最大面积策略")
    room_conf["UseAreasRatio"] = st.checkbox(
        "使用面积比例",
        value=room_conf.get("UseAreasRatio", True),
        help="开启后按“轮廓面积 × 房间比例”计算最大面积；关闭后直接使用固定最大面积值。训练动作约束和基础奖励都会共用这套规则。",
        key=f"{config_id}_use_area_ratio",
    )
    col_ratio, col_value = st.columns(2)
    with col_ratio:
        for room in std_rooms[:-1]:
            room_conf["MaxAreasRatio"][room] = st.number_input(
                f"{room} 最大比例",
                min_value=0.0,
                max_value=1.0,
                value=float(room_conf["MaxAreasRatio"].get(room, 0.25)),
                step=0.05,
                key=f"{config_id}_ratio_{room}",
            )
    with col_value:
        for room in std_rooms[:-1]:
            room_conf["MaxAreasValue"][room] = st.number_input(
                f"{room} 最大面积",
                min_value=1,
                value=int(room_conf["MaxAreasValue"].get(room, 10)),
                help="关闭“使用面积比例”后，训练会直接使用这里的固定最大面积值。",
                key=f"{config_id}_value_{room}",
            )

    if boundary_area > 0:
        st.caption(f"当前环境轮廓面积: {boundary_area:.2f} ㎡")
        derived_lines = []
        for room in std_rooms[:-1]:
            effective_max_area = _get_effective_room_max_area(room_conf, room, boundary_area)
            if room_conf.get("UseAreasRatio", True):
                ratio = float(room_conf["MaxAreasRatio"].get(room, 0.25))
                derived_lines.append(f"{room}: {boundary_area:.2f} × {ratio:.2f} = {effective_max_area:.2f} ㎡")
            else:
                derived_lines.append(f"{room}: 固定最大面积 = {effective_max_area:.2f} ㎡")
        st.info("当前最大面积实际值:\n\n" + "\n".join(f"- {line}" for line in derived_lines))

    st.markdown("#### 长宽比")
    for room in std_rooms[:-1]:
        ratio = room_conf["ASPECT_RATIO_RANGES"].get(room, [1.0, 2.0])
        if not isinstance(ratio, list) or len(ratio) != 2:
            ratio = [1.0, 2.0]
        col_a, col_b = st.columns(2)
        with col_a:
            min_ratio = st.number_input(
                f"{room} 最小长宽比",
                min_value=0.1,
                value=float(ratio[0]),
                step=0.1,
                key=f"{config_id}_ratio_min_{room}",
            )
        with col_b:
            max_ratio = st.number_input(
                f"{room} 最大长宽比",
                min_value=0.1,
                value=float(ratio[1]),
                step=0.1,
                key=f"{config_id}_ratio_max_{room}",
            )
        room_conf["ASPECT_RATIO_RANGES"][room] = [min_ratio, max_ratio]

    st.markdown("#### 噪声与嵌套")
    col_noise, col_nesting, col_priority = st.columns(3)
    with col_noise:
        for room in std_rooms:
            room_conf["NOISE_WEIGHTS"][room] = st.number_input(
                f"{room} 噪声权重",
                min_value=0.0,
                value=float(room_conf["NOISE_WEIGHTS"].get(room, 1.0)),
                step=0.5,
                help="这里配置的是各类房间在主噪声奖励中的基础噪声指数，数值越大表示该空间本身越吵。",
                key=f"{config_id}_noise_{room}",
            )
    with col_nesting:
        for room in std_rooms:
            room_conf["NESTING_WEIGHTS"][room] = st.number_input(
                f"{room} 嵌套权重",
                min_value=0.0,
                value=float(room_conf["NESTING_WEIGHTS"].get(room, 0.1)),
                step=0.05,
                key=f"{config_id}_nesting_{room}",
            )
    with col_priority:
        for room in std_rooms:
            room_conf["NESTING_PRIORITY"][room] = st.number_input(
                f"{room} 嵌套优先级",
                min_value=1,
                value=int(room_conf["NESTING_PRIORITY"].get(room, 1)),
                key=f"{config_id}_priority_{room}",
            )


def render_advanced_section(config: dict, config_id: str) -> None:
    """渲染环境/奖励/算法高级参数。"""
    st.markdown("---")
    st.markdown("### 高级参数")

    env_conf = config.setdefault("Environment", {})
    room_conf = config.setdefault("Room", {})
    env_adv = config.setdefault("EnvironmentAdvanced", {})
    reward_adv = config.setdefault("RewardAdvanced", {})
    algo_adv = config.setdefault("AlgorithmAdvanced", {})
    base_weight = env_conf.setdefault("Reward_Base_Weight", {})
    extra_weight = env_conf.setdefault("Reward_Extra_Weight", {})
    boundary_area = _calculate_polygon_area(env_conf.get("boundary", []))

    show_param_reference = st.checkbox(
        "显示参数来源说明",
        value=bool(st.session_state.get(f"{config_id}_show_param_reference", False)),
        help="开启后，会在高级参数下面显示它对应的函数、公式摘要、核心变量和当前配置代入关系，但不直接展示整段源码。",
        key=f"{config_id}_show_param_reference",
    )
    if show_param_reference:
        st.caption("说明内容基于当前配置和源码逻辑生成，重点展示“这个参数参与了什么计算、乘了哪些核心变量、当前值会怎样代入”。")

    def _format_value(value) -> str:
        if isinstance(value, float):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)

    def _build_param_reference(key: str, current_value) -> dict | None:
        current_grid_size = float(env_adv.get("grid_size", 15.0))
        current_grid_resolution = max(float(env_adv.get("grid_resolution", 0.2)), 1e-9)
        current_grid_cells = int(current_grid_size / current_grid_resolution)

        area_weight = float(base_weight.get("面积奖励", 1.0))
        aspect_weight = float(base_weight.get("长宽比奖励", 1.0))
        edge_weight = float(base_weight.get("贴边奖励", 1.0))
        shared_wall_weight = float(base_weight.get("房间与房间之间的边界贴合奖励", 1.0))
        corner_weight = float(base_weight.get("角落占领奖励", 1.0))
        door_block_weight = float(base_weight.get("外门遮挡惩罚", 1.0))
        prior_weight = float(base_weight.get("先验知识布局奖励", 1.0))
        nesting_weight = float(base_weight.get("房间嵌套惩罚", 1.0))
        adjacency_weight = float(extra_weight.get("邻接关系奖励", 1.0))
        privacy_weight = float(extra_weight.get("隐私保护奖励", 1.0))
        south_weight = float(extra_weight.get("南向采光奖励", 1.0))
        noise_weight = float(extra_weight.get("噪声干扰度奖励", 1.0))

        if key in {"grid_size", "grid_resolution"}:
            return {
                "source": "make_env() -> HouseEnv.__init__()",
                "formula": "grid_cells = int(grid_size / grid_resolution)，同时 observation_space 的上界会使用 grid_size。",
                "variables": "`grid_size`, `grid_resolution`, `grid_cells`, `observation_space`",
                "current": f"当前代入: grid_size={_format_value(current_grid_size)}, grid_resolution={_format_value(current_grid_resolution)}, 预计网格边长={current_grid_cells}",
            }
        if key in {"prior_match_threshold", "prior_random_max_attempts", "prior_random_wall_clearance", "prior_random_door_avoidance", "prior_corner_region_size", "prior_corner_attempt_limit", "prior_distance_relax_attempts"}:
            return {
                "source": "HouseEnv 随机先验生成逻辑",
                "formula": "这些值不直接乘进奖励，而是控制先验点搜索、避墙、避门、角落优先和放宽次数。",
                "variables": "`prior_positions`, `attempts`, `min_distance`, `door_clearance`, `wall_clearance`",
                "current": f"当前值: {_format_value(current_value)}；会影响随机先验是否容易生成成功、以及生成出的先验点分布。",
            }
        if key in {"living_room_min_ratio", "living_room_max_ratio"}:
            min_area = boundary_area * float(env_adv.get("living_room_min_ratio", 0.3))
            max_area = boundary_area * float(env_adv.get("living_room_max_ratio", 0.5))
            return {
                "source": "HouseEnv 初始化客厅面积范围",
                "formula": "living_room_area_range = [boundary_area × min_ratio, boundary_area × max_ratio]",
                "variables": "`boundary_area`, `living_room_min_ratio`, `living_room_max_ratio`",
                "current": f"当前代入: boundary_area={_format_value(boundary_area)} ㎡，客厅面积范围约为 {_format_value(min_area)} ~ {_format_value(max_area)} ㎡",
            }
        if key in {"room_init_max_attempts", "door_center_offset", "vertical_door_default_length", "horizontal_door_default_length"}:
            return {
                "source": "HouseEnv.set_boundary() / 初始布局生成逻辑",
                "formula": "这些值控制入口门几何默认值和房间初始化搜索上限，不直接参与 reward 加减。",
                "variables": "`entrance_door`, `door_center_offset`, `door_length`, `room_init_attempts`",
                "current": f"当前值: {_format_value(current_value)}",
            }

        if key in {"area_under_penalty_scale", "area_over_penalty_scale", "area_reward_bonus"}:
            if key == "area_under_penalty_scale":
                formula = "面积不足项 = -系数 × |min_area - area| × 基础权重(面积奖励)"
            elif key == "area_over_penalty_scale":
                formula = "面积超限项 = -系数 × |area - max_area| × 基础权重(面积奖励)"
            else:
                formula = "面积达标项 = bonus × 基础权重(面积奖励)"
            return {
                "source": "HouseEnv._cal_base_reward()",
                "formula": formula,
                "variables": "`area`, `min_area`, `max_area`, `self.base_reward_weight['面积奖励']`",
                "current": f"当前代入: 系数/奖励={_format_value(current_value)}, 面积奖励权重={_format_value(area_weight)}",
            }
        if key in {"aspect_under_penalty_scale", "aspect_over_penalty_scale", "aspect_reward_bonus"}:
            if key == "aspect_under_penalty_scale":
                formula = "长宽比不足项 = -系数 × |min_ratio - aspect_ratio| × 基础权重(长宽比奖励)"
            elif key == "aspect_over_penalty_scale":
                formula = "长宽比超限项 = -系数 × |aspect_ratio - max_ratio| × 基础权重(长宽比奖励)"
            else:
                formula = "长宽比达标项 = bonus × 基础权重(长宽比奖励)，且面积先达标。"
            return {
                "source": "HouseEnv._cal_base_reward()",
                "formula": formula,
                "variables": "`aspect_ratio`, `min_ratio`, `max_ratio`, `area_success`, `self.base_reward_weight['长宽比奖励']`",
                "current": f"当前代入: 系数/奖励={_format_value(current_value)}, 长宽比奖励权重={_format_value(aspect_weight)}",
            }
        if key in {"edge_touch_threshold", "edge_near_threshold", "edge_touch_reward", "edge_near_reward_scale"}:
            return {
                "source": "HouseEnv._cal_base_reward()",
                "formula": "贴边项先根据 distance_to_boundary 判断完全贴边或近邻，再乘基础权重(贴边奖励)。近邻区公式约为 edge_near_reward_scale × (1 - distance_to_boundary)。",
                "variables": "`distance_to_boundary`, `edge_touch_threshold`, `edge_near_threshold`, `self.base_reward_weight['贴边奖励']`",
                "current": f"当前代入: 当前值={_format_value(current_value)}, 贴边奖励权重={_format_value(edge_weight)}",
            }
        if key == "shared_wall_reward_per_meter":
            return {
                "source": "HouseEnv._cal_base_reward()",
                "formula": "共享墙项 = 每米值 × shared_length × 基础权重(房间与房间之间的边界贴合奖励)",
                "variables": "`shared_length`, `shared_wall_reward_per_meter`, `self.base_reward_weight['房间与房间之间的边界贴合奖励']`",
                "current": f"当前代入: 每米值={_format_value(current_value)}, 共享墙奖励权重={_format_value(shared_wall_weight)}",
            }
        if key in {"corner_touch_threshold", "corner_near_threshold", "corner_touch_reward", "corner_near_reward_scale", "corner_dot_threshold"}:
            return {
                "source": "HouseEnv._cal_base_reward()",
                "formula": "先用 corner_dot_threshold 从轮廓点识别拐角，再按房间中心到角点的距离计算完全贴角或近邻奖励，最后乘基础权重(角落占领奖励)。",
                "variables": "`actual_corners`, `room.center`, `corner_touch_threshold`, `corner_near_threshold`, `corner_dot_threshold`",
                "current": f"当前代入: 当前值={_format_value(current_value)}, 角落奖励权重={_format_value(corner_weight)}",
            }
        if key in {"door_to_door_threshold", "door_to_door_penalty_scale", "door_face_to_face_threshold", "door_face_to_face_extra_penalty"}:
            return {
                "source": "HouseEnv._cal_base_reward()",
                "formula": "门对门项先算两门中心距离 dist；若 dist < door_to_door_threshold，则按 -penalty_scale × penalty_factor 扣分；若再小于正对阈值，会额外减去 face_to_face_extra_penalty。",
                "variables": "`door_centers`, `dist`, `penalty_factor`, `door_to_door_threshold`, `door_face_to_face_threshold`",
                "current": f"当前代入: 当前值={_format_value(current_value)}；该项当前代码里不再额外乘基础奖励权重。",
            }
        if key in {"door_proximity_threshold", "vertical_required_clearance", "horizontal_required_clearance", "door_block_penalty_scale"}:
            return {
                "source": "HouseEnv._cal_base_reward()",
                "formula": "外门遮挡项 = -door_block_penalty_scale × (overlap_length - (door_width - required_clearance))，最后再乘基础权重(外门遮挡惩罚)。",
                "variables": "`overlap_length`, `door_width`, `required_clearance`, `door_proximity_threshold`, `self.base_reward_weight['外门遮挡惩罚']`",
                "current": f"当前代入: 当前值={_format_value(current_value)}, 外门遮挡权重={_format_value(door_block_weight)}",
            }
        if key == "overlap_penalty_scale":
            return {
                "source": "HouseEnv._cal_base_reward()",
                "formula": "重叠项 = -overlap_penalty_scale × (overlap_ratio^2)",
                "variables": "`overlap_area`, `room.area`, `overlap_ratio`",
                "current": f"当前代入: 惩罚倍率={_format_value(current_value)}；这里直接进入 total reward，不再额外乘基础权重。",
            }
        if key in {"nesting_equal_dimension_tolerance", "nesting_exact_match_tolerance", "nesting_equal_dimension_penalty_scale", "nesting_equal_dimension_exact_multiplier", "nesting_sigmoid_scale", "nesting_penalty_scale"}:
            return {
                "source": "HouseEnv._cal_base_reward()",
                "formula": "嵌套项分两段：先对“相交区域近似正方形”计算 -equal_dimension_penalty_scale × overlap_ratio；若宽高几乎完全相等，再乘 exact_multiplier；随后对超出 max_overlap 的嵌套比例走 Sigmoid 惩罚，并最终乘基础权重(房间嵌套惩罚)。",
                "variables": "`overlap_width`, `overlap_height`, `overlap_ratio`, `max_overlap`, `penalty_factor`, `self.base_reward_weight['房间嵌套惩罚']`",
                "current": f"当前代入: 当前值={_format_value(current_value)}, 嵌套惩罚权重={_format_value(nesting_weight)}",
            }
        if key == "prior_layout_scale":
            return {
                "source": "HouseEnv._cal_base_reward()",
                "formula": "先验布局项 = prior_reward × prior_layout_scale × 基础权重(先验知识布局奖励)",
                "variables": "`prior_reward`, `self.prior_layout_scale`, `self.base_reward_weight['先验知识布局奖励']`",
                "current": f"当前代入: 内部倍率={_format_value(current_value)}, 先验布局权重={_format_value(prior_weight)}",
            }
        if key in {"adjacency_synergy_reward", "adjacency_repellent_penalty"}:
            return {
                "source": "HouseEnv._cal_simplified_adjacency_reward() -> _cal_extra_reward()",
                "formula": "先按协同/排斥房间对给每对关系一个 score，再把所有涉及当前房间的 score 求和，最后乘额外权重(邻接关系奖励)。",
                "variables": "`adjacency_synergistic_pairs`, `adjacency_repellent_pairs`, `room_centers`, `distance`, `self.extra_reward_weight['邻接关系奖励']`",
                "current": f"当前代入: 当前值={_format_value(current_value)}, 邻接关系奖励权重={_format_value(adjacency_weight)}",
            }
        if key in {"privacy_sigmoid_k", "privacy_sigmoid_midpoint", "privacy_base_reward", "privacy_reward_scale"}:
            return {
                "source": "HouseEnv._cal_privacy_partition_reward() -> _cal_extra_reward()",
                "formula": "私密空间项 = (privacy_base_reward + sigmoid(k × (normalized_distance - midpoint))) × privacy_reward_scale，再乘额外权重(隐私保护奖励)。",
                "variables": "`door_center`, `room.center`, `normalized_distance`, `privacy_private_spaces`, `self.extra_reward_weight['隐私保护奖励']`",
                "current": f"当前代入: 当前值={_format_value(current_value)}, 隐私奖励权重={_format_value(privacy_weight)}",
            }
        if key in {"south_region_ratio", "south_position_reward_scale", "south_balcony_bonus"}:
            return {
                "source": "HouseEnv._cal_south_lighting_reward() -> _cal_extra_reward()",
                "formula": "先用 south_region_ratio 取轮廓下方南向区域，再按 room_center_y 到南边界的相对位置算位置奖励；若有朝南相邻阳台，再加 south_balcony_bonus；最后乘额外权重(南向采光奖励)。",
                "variables": "`boundary_polygon.bounds`, `south_threshold`, `room.center[1]`, `position_score`, `self.extra_reward_weight['南向采光奖励']`",
                "current": f"当前代入: 当前值={_format_value(current_value)}, 南向采光权重={_format_value(south_weight)}",
            }
        if key.startswith("noise_sensitivity_") or key == "noise_adjacent_distance_threshold":
            return {
                "source": "HouseEnv._cal_noise_partition_reward() -> _cal_extra_reward()",
                "formula": "噪声项先算 R_noise = -(weight × (NI_max - NI)^2) / N，再乘额外权重(噪声干扰度奖励)。其中 weight 就来自对应房间类型的 noise_sensitivity_*。",
                "variables": "`noise_indices`, `weight_factors`, `noise_difference_square`, `N`, `noise_adjacent_distance_threshold`, `self.extra_reward_weight['噪声干扰度奖励']`",
                "current": f"当前代入: 当前值={_format_value(current_value)}, 噪声奖励权重={_format_value(noise_weight)}",
            }

        if key in {"mappo_hidden_size", "mappo_ppo_epochs", "mappo_max_grad_norm", "mappo_update_step"}:
            return {
                "source": "MAPPO.__init__() -> 训练更新循环",
                "formula": "这些值作为 MAPPO 的初始化常量，分别影响网络宽度、每次 PPO 重复优化轮数、梯度裁剪上限和累计多少步后更新。",
                "variables": "`self.hidden_size`, `self.ppo_epochs`, `self.max_grad_norm`, `self.update_step`",
                "current": f"当前值: {_format_value(current_value)}",
            }
        if key == "mappo_gamma":
            return {
                "source": "MAPPO.compute_gae()",
                "formula": "GAE / return 递推里使用: delta = reward + gamma × V(next) × (1-done) - V(now)",
                "variables": "`global_rewards[t]`, `values[t]`, `values[t+1]`, `global_dones[t]`",
                "current": f"当前代入: gamma={_format_value(current_value)}",
            }
        if key == "mappo_gae_lambda":
            return {
                "source": "MAPPO.compute_gae()",
                "formula": "GAE 递推: gae = delta + gamma × gae_lambda × (1-done) × gae",
                "variables": "`delta`, `gae`, `self.gamma`, `global_dones[t]`",
                "current": f"当前代入: gae_lambda={_format_value(current_value)}",
            }
        if key == "mappo_clip_epsilon":
            return {
                "source": "MAPPO PPO 更新",
                "formula": "PPO 裁剪区间: clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon)",
                "variables": "`ratios`, `surr1`, `surr2`",
                "current": f"当前代入: clip_epsilon={_format_value(current_value)}",
            }
        if key == "mappo_value_coef":
            return {
                "source": "MAPPO.__init__()",
                "formula": "该参数会写入 `self.value_coef`，但当前版本训练更新里 critic_loss 是单独优化的，未看到再乘这个系数进入总损失。",
                "variables": "`self.value_coef`, `critic_loss`",
                "current": f"当前值: {_format_value(current_value)}",
                "status": "当前版本未直接参与损失加权，仅保存到成员变量。",
            }
        if key == "mappo_entropy_coef":
            return {
                "source": "MAPPO PPO 更新",
                "formula": "actor 总损失 = actor_loss - entropy_coef × entropy",
                "variables": "`actor_loss`, `entropy`, `loss`",
                "current": f"当前代入: entropy_coef={_format_value(current_value)}",
            }
        if key == "mappo_reward_clip_range":
            return {
                "source": "MAPPO.__init__() -> RewardNormalizer",
                "formula": "该值传入 RewardNormalizer(clip_range=...)，用于奖励归一化时限制极端值范围。",
                "variables": "`self.reward_normalizer.clip_range`, `reward_list`, `r_sum`",
                "current": f"当前代入: reward_clip_range={_format_value(current_value)}",
            }

        if key in {"qmix_hidden_size", "qmix_mixing_hidden_size", "qmix_hyper_hidden_size", "qmix_buffer_size", "qmix_update_frequency"}:
            return {
                "source": "QMIX.__init__() -> learn()",
                "formula": "这些值作为 QMIX 网络宽度、回放池容量和更新频率等初始化常量；其中 update_frequency 会控制每隔多少步触发目标网络更新。",
                "variables": "`self.hidden_size`, `mixing_net`, `hyper_net`, `buffer`, `self.update_frequency`",
                "current": f"当前值: {_format_value(current_value)}",
            }
        if key == "qmix_gamma":
            return {
                "source": "QMIX.learn()",
                "formula": "目标 Q 值: target_q = reward + gamma × target_global_q × (1-done)",
                "variables": "`rewards_tensor`, `target_global_q`, `dones_tensor`",
                "current": f"当前代入: gamma={_format_value(current_value)}",
            }
        if key == "qmix_tau":
            return {
                "source": "QMIX.update_target_networks()",
                "formula": "软更新: target = tau × online + (1 - tau) × target",
                "variables": "`target_param`, `param`, `self.tau`",
                "current": f"当前代入: tau={_format_value(current_value)}",
            }
        if key in {"qmix_epsilon", "qmix_epsilon_decay", "qmix_min_epsilon"}:
            return {
                "source": "QMIX.learn()",
                "formula": "探索率递推: epsilon = max(min_epsilon, epsilon × epsilon_decay)，初始值来自 qmix_epsilon。",
                "variables": "`epsilon`, `epsilon_decay`, `min_epsilon`",
                "current": f"当前代入: 初始={_format_value(algo_adv.get('qmix_epsilon', 1.0))}, 衰减={_format_value(algo_adv.get('qmix_epsilon_decay', 0.999))}, 下限={_format_value(algo_adv.get('qmix_min_epsilon', 0.01))}",
            }

        if key in {"madqn_hidden_dim", "madqn_lr", "madqn_buffer_capacity"}:
            return {
                "source": "MADQN.__init__() -> update()",
                "formula": "这些值控制 MADQN 的单体 Q 网络宽度、优化器学习率和经验回放容量。",
                "variables": "`DQNNetwork`, `optimizer`, `replay_buffer`",
                "current": f"当前值: {_format_value(current_value)}",
            }
        if key == "madqn_gamma":
            return {
                "source": "MADQN.update()",
                "formula": "目标 Q 值: target_q = reward + gamma × max_next_q × (1-done)",
                "variables": "`agent_rewards`, `max_next_q_values`, `agent_dones`",
                "current": f"当前代入: gamma={_format_value(current_value)}",
            }
        if key == "madqn_tau":
            return {
                "source": "MADQN.soft_update_target_networks()",
                "formula": "软更新: target = tau × online + (1 - tau) × target",
                "variables": "`target_param`, `param`, `self.tau`",
                "current": f"当前代入: tau={_format_value(current_value)}",
            }
        if key in {"madqn_epsilon_start", "madqn_epsilon_end", "madqn_epsilon_decay"}:
            return {
                "source": "MADQN.update()",
                "formula": "探索率递推: 若 epsilon > epsilon_end，则 epsilon *= epsilon_decay；初始值来自 epsilon_start。",
                "variables": "`self.epsilon`, `self.epsilon_end`, `self.epsilon_decay`",
                "current": f"当前代入: 初始={_format_value(algo_adv.get('madqn_epsilon_start', 1.0))}, 终止={_format_value(algo_adv.get('madqn_epsilon_end', 0.01))}, 衰减={_format_value(algo_adv.get('madqn_epsilon_decay', 0.9995))}",
            }

        return None

    def _render_param_reference(container, key: str, current_value) -> None:
        if not show_param_reference:
            return
        reference = _build_param_reference(key, current_value)
        if not reference:
            return
        container.caption(f"来源函数: {reference['source']}")
        container.caption(f"公式摘要: {reference['formula']}")
        container.caption(f"核心变量: {reference['variables']}")
        container.caption(reference["current"])
        if reference.get("status"):
            container.caption(f"状态说明: {reference['status']}")


    def render_number_field(container, state: dict, key: str, label: str, help_text: str, *, step=0.1, min_value=None, max_value=None, fmt=None):
        value = state[key]
        is_int_field = isinstance(value, int) and not isinstance(value, bool)
        normalized_value = int(value) if is_int_field else float(value)
        kwargs = {
            "label": label,
            "value": normalized_value,
            "help": help_text,
            "key": f"{config_id}_{key}",
        }
        if is_int_field:
            kwargs["step"] = int(step) if isinstance(step, int) else 1
            if min_value is not None:
                kwargs["min_value"] = int(min_value)
            if max_value is not None:
                kwargs["max_value"] = int(max_value)
        else:
            kwargs["step"] = float(step)
            if min_value is not None:
                kwargs["min_value"] = float(min_value)
            if max_value is not None:
                kwargs["max_value"] = float(max_value)
        if fmt is not None:
            kwargs["format"] = fmt
        state[key] = container.number_input(**kwargs)
        _render_param_reference(container, key, state[key])

    with st.expander("环境高级参数", expanded=False):
        env_defaults = {
            "grid_size": 15.0,
            "grid_resolution": 0.2,
            "prior_match_threshold": 1.0,
            "prior_random_max_attempts": 3000,
            "prior_random_wall_clearance": 0.8,
            "prior_random_door_avoidance": 1.2,
            "prior_corner_region_size": 1.5,
            "prior_corner_attempt_limit": 50,
            "prior_distance_relax_attempts": 100,
            "living_room_min_ratio": 0.3,
            "living_room_max_ratio": 0.5,
            "room_init_max_attempts": 100,
            "door_center_offset": 0.6,
            "vertical_door_default_length": 1.2,
            "horizontal_door_default_length": 1.0,
        }
        for key, default_value in env_defaults.items():
            env_adv.setdefault(key, default_value)
        st.caption("这部分控制环境内部的采样、初始化和自动推断逻辑，通常只在你明确知道影响时再修改。")

        with st.container(border=True):
            st.markdown("#### 网格与坐标")
            render_number_field(st, env_adv, "grid_size", "网格尺寸上限", "状态空间和渲染坐标使用的基准网格尺寸；调大后可容纳更大的场地范围。", step=1, min_value=1.0)
            render_number_field(st, env_adv, "grid_resolution", "网格分辨率", "内部离散网格的精度，越小越精细，但状态空间和计算量也会更大。", step=0.01, min_value=0.01, fmt="%.2f")
            env_adv["boundary_retry_scale_factors"] = _parse_float_list(
                st.text_input(
                    "边界回退缩放序列",
                    value=str(env_adv.get("boundary_retry_scale_factors", [0.8, 0.6, 0.4, 0.2, 0.1, 0.05])),
                    help="房间移动或拉伸后如果越界，会依次尝试按这些比例缩小动作，直到找到合法结果或放弃。",
                    key=f"{config_id}_adv_boundary_retry_scale_factors",
                ),
                env_adv.get("boundary_retry_scale_factors", [0.8, 0.6, 0.4, 0.2, 0.1, 0.05]),
            )
            if show_param_reference:
                st.caption("来源函数: HouseEnv 动作越界回退逻辑")
                st.caption("公式摘要: 房间移动或拉伸后若越界，会按缩放序列依次把动作幅度乘上这些比例，直到找到合法结果或放弃。")
                st.caption("核心变量: `scale_factors`, `shift_value`, `candidate_action`, `boundary_polygon`")
                st.caption(f"当前代入: boundary_retry_scale_factors={env_adv['boundary_retry_scale_factors']}")

        with st.container(border=True):
            st.markdown("#### 随机先验生成")
            render_number_field(st, env_adv, "prior_match_threshold", "先验匹配阈值", "房间中心距离先验点小于该值时，更容易被判定为命中先验位置。", step=0.1, min_value=0.0)
            render_number_field(st, env_adv, "prior_random_max_attempts", "随机先验最大尝试数", "随机生成先验位置时的总尝试上限，数值越大越容易找到可行解，但生成会更慢。", step=1, min_value=1)
            render_number_field(st, env_adv, "prior_random_wall_clearance", "随机先验离墙距离", "随机先验点距离外轮廓边界至少保留多远，避免先验过于贴墙。", step=0.1, min_value=0.0)
            render_number_field(st, env_adv, "prior_random_door_avoidance", "随机先验避门距离", "随机先验点会避开入口门附近区域，这个值是避让半径。", step=0.1, min_value=0.0)
            render_number_field(st, env_adv, "prior_corner_region_size", "角落优先区域尺寸", "先验优先往角落搜索时，每个角落候选区域的边长。", step=0.1, min_value=0.1)
            render_number_field(st, env_adv, "prior_corner_attempt_limit", "角落优先尝试次数", "前多少次随机先验尝试优先放在角落区域。", step=1, min_value=1)
            render_number_field(st, env_adv, "prior_distance_relax_attempts", "放宽阶段尝试次数", "先验点最小间距放不下时，每一档放宽距离会尝试多少次。", step=1, min_value=1)
            env_adv["prior_distance_relaxation"] = _parse_float_list(
                st.text_input(
                    "先验间距放宽序列",
                    value=str(env_adv.get("prior_distance_relaxation", [2.5, 2.0, 1.5, 1.0, 0.8])),
                    help="随机先验仍然放不下时，会按这个距离序列逐步降低先验点之间的最小间距要求。",
                    key=f"{config_id}_adv_prior_distance_relaxation",
                ),
                env_adv.get("prior_distance_relaxation", [2.5, 2.0, 1.5, 1.0, 0.8]),
            )
            if show_param_reference:
                st.caption("来源函数: HouseEnv 随机先验生成逻辑")
                st.caption("公式摘要: 若当前最小间距要求放不下，会依次取放宽序列中的距离重新尝试布点。")
                st.caption("核心变量: `distance_requirements`, `min_distance`, `attempts`, `prior_positions`")
                st.caption(f"当前代入: prior_distance_relaxation={env_adv['prior_distance_relaxation']}")

        with st.container(border=True):
            st.markdown("#### 客厅与房间初始化")
            render_number_field(st, env_adv, "living_room_min_ratio", "客厅最小面积比例", "客厅初始化时，相对总轮廓面积允许的最小比例。", step=0.05, min_value=0.0, max_value=1.0)
            render_number_field(st, env_adv, "living_room_max_ratio", "客厅最大面积比例", "客厅初始化时，相对总轮廓面积允许的最大比例。", step=0.05, min_value=0.0, max_value=1.0)
            render_number_field(st, env_adv, "room_init_max_attempts", "房间初始化最大尝试数", "初始随机放置各房间时的最大尝试次数，过小容易初始化失败，过大则会更慢。", step=1, min_value=1)

        with st.container(border=True):
            st.markdown("#### 自动门参数")
            render_number_field(st, env_adv, "door_center_offset", "默认门中心偏移", "自动推断入口门时，门中心相对候选边中点的偏移量。", step=0.1, min_value=0.0)
            render_number_field(st, env_adv, "vertical_door_default_length", "默认竖向门长度", "自动生成竖向门线段时使用的默认长度。", step=0.1, min_value=0.1)
            render_number_field(st, env_adv, "horizontal_door_default_length", "默认横向门长度", "自动生成横向门线段时使用的默认长度。", step=0.1, min_value=0.1)

    with st.expander("奖励高级参数", expanded=False):
        reward_defaults = {
            "area_under_penalty_scale": 5.0,
            "area_over_penalty_scale": 20.0,
            "area_reward_bonus": 10.0,
            "aspect_under_penalty_scale": 5.0,
            "aspect_over_penalty_scale": 20.0,
            "aspect_reward_bonus": 30.0,
            "edge_touch_threshold": 0.02,
            "edge_near_threshold": 0.5,
            "edge_touch_reward": 20.0,
            "edge_near_reward_scale": 8.0,
            "shared_wall_reward_per_meter": 2.0,
            "door_to_door_threshold": 2.0,
            "door_to_door_penalty_scale": 20.0,
            "door_face_to_face_threshold": 0.5,
            "door_face_to_face_extra_penalty": 30.0,
            "corner_touch_threshold": 0.02,
            "corner_near_threshold": 0.5,
            "corner_touch_reward": 20.0,
            "corner_near_reward_scale": 20.0,
            "corner_dot_threshold": 0.1,
            "door_proximity_threshold": 0.6,
            "vertical_required_clearance": 1.2,
            "horizontal_required_clearance": 1.0,
            "door_block_penalty_scale": 200.0,
            "overlap_penalty_scale": 50.0,
            "nesting_equal_dimension_tolerance": 0.1,
            "nesting_exact_match_tolerance": 0.01,
            "nesting_equal_dimension_penalty_scale": 20.0,
            "nesting_equal_dimension_exact_multiplier": 2.0,
            "nesting_sigmoid_scale": 10.0,
            "nesting_penalty_scale": 5.0,
            "adjacency_synergy_reward": 2.0,
            "adjacency_repellent_penalty": -2.0,
            "privacy_sigmoid_k": 10.0,
            "privacy_sigmoid_midpoint": 0.5,
            "privacy_base_reward": 2.0,
            "privacy_reward_scale": 5.0,
            "south_region_ratio": 0.3,
            "south_position_reward_scale": 5.0,
            "south_balcony_bonus": 2.0,
            "noise_sensitivity_bedroom": 0.5,
            "noise_sensitivity_bathroom": 0.2,
            "noise_sensitivity_kitchen": 0.1,
            "noise_sensitivity_living_room": 0.15,
            "noise_sensitivity_balcony": 0.05,
            "noise_adjacent_distance_threshold": 0.1,
            "adjacency_synergistic_pairs": [("BEDROOM", "BATHROOM"), ("BEDROOM", "BALCONY"), ("BATHROOM", "BALCONY")],
            "adjacency_repellent_pairs": [("BEDROOM", "KITCHEN")],
            "privacy_private_spaces": ["BEDROOM", "BATHROOM"],
            "south_lighting_room_types": ["BEDROOM", "LIVING_ROOM"],
            "noise_calc_room_types": ["BEDROOM", "BATHROOM", "KITCHEN", "LIVING_ROOM"],
        }
        for key, default_value in reward_defaults.items():
            reward_adv.setdefault(key, default_value)
        st.caption("这部分用于控制奖励公式内部的常数、阈值和结构规则；普通调参通常先改基础奖励权重，只有要精调公式时再改这里。")

        with st.container(border=True):
            st.markdown("#### 面积与比例")
            render_number_field(st, reward_adv, "area_under_penalty_scale", "面积不足惩罚倍率", "房间面积低于最小面积时的线性惩罚强度，越大表示面积不足时扣分越狠。")
            render_number_field(st, reward_adv, "area_over_penalty_scale", "面积超限惩罚倍率", "房间面积超过最大面积时的线性惩罚强度。")
            render_number_field(st, reward_adv, "area_reward_bonus", "面积达标奖励", "房间面积落在允许范围内时给予的基础奖励。")
            render_number_field(st, reward_adv, "aspect_under_penalty_scale", "长宽比不足惩罚倍率", "房间长宽比低于最小值时的惩罚强度。")
            render_number_field(st, reward_adv, "aspect_over_penalty_scale", "长宽比超限惩罚倍率", "房间长宽比高于最大值时的惩罚强度。")
            render_number_field(st, reward_adv, "aspect_reward_bonus", "长宽比达标奖励", "面积达标且长宽比也达标时给予的额外奖励。")

        with st.container(border=True):
            st.markdown("#### 贴边与共享墙")
            render_number_field(st, reward_adv, "edge_touch_threshold", "贴边判定阈值", "房间距离外轮廓小于该值时，视为完全贴边。")
            render_number_field(st, reward_adv, "edge_near_threshold", "贴边近邻阈值", "房间还未完全贴边，但距离边界小于该值时，给予衰减型贴边奖励。")
            render_number_field(st, reward_adv, "edge_touch_reward", "完全贴边奖励", "房间完全贴边时给予的固定奖励。")
            render_number_field(st, reward_adv, "edge_near_reward_scale", "贴边近邻奖励倍率", "房间靠近边界但未完全贴边时的奖励斜率。")
            render_number_field(st, reward_adv, "shared_wall_reward_per_meter", "共享墙奖励每米值", "房间之间形成共享墙时，每米共享边界增加多少奖励。")

        with st.container(border=True):
            st.markdown("#### 角落与门")
            render_number_field(st, reward_adv, "corner_touch_threshold", "角落命中阈值", "房间角点距离真实外轮廓角点小于该值时，视为命中角落。")
            render_number_field(st, reward_adv, "corner_near_threshold", "角落近邻阈值", "房间靠近角落但未完全命中时，给衰减型角落奖励。")
            render_number_field(st, reward_adv, "corner_touch_reward", "完全贴角奖励", "房间角点完全命中外轮廓角点时的固定奖励。")
            render_number_field(st, reward_adv, "corner_near_reward_scale", "贴角近邻奖励倍率", "房间接近角落时的奖励斜率。")
            render_number_field(st, reward_adv, "corner_dot_threshold", "角落识别点积阈值", "用于识别轮廓拐角的几何阈值，越小越严格。")
            render_number_field(st, reward_adv, "door_to_door_threshold", "门对门惩罚距离阈值", "两个房门中心距离小于该值时，开始施加门对门惩罚。")
            render_number_field(st, reward_adv, "door_to_door_penalty_scale", "门对门惩罚倍率", "门对门距离过近时的惩罚强度。")
            render_number_field(st, reward_adv, "door_face_to_face_threshold", "正对门额外惩罚阈值", "门中心更近到该值以内时，再追加一层额外惩罚。")
            render_number_field(st, reward_adv, "door_face_to_face_extra_penalty", "正对门额外惩罚", "两门几乎正对时追加的固定惩罚。")
            render_number_field(st, reward_adv, "door_proximity_threshold", "门近邻判定阈值", "房间边界接近入口门到多近时，会进入门遮挡检测。")
            render_number_field(st, reward_adv, "vertical_required_clearance", "竖门最小净空", "竖向入口门前至少保留的无遮挡净空。")
            render_number_field(st, reward_adv, "horizontal_required_clearance", "横门最小净空", "横向入口门前至少保留的无遮挡净空。")
            render_number_field(st, reward_adv, "door_block_penalty_scale", "外门遮挡惩罚倍率", "房间遮挡入口门通行空间时的惩罚强度。")

        with st.container(border=True):
            st.markdown("#### 重叠与嵌套")
            render_number_field(st, reward_adv, "overlap_penalty_scale", "重叠惩罚倍率", "房间直接重叠时的惩罚强度。")
            render_number_field(st, reward_adv, "nesting_equal_dimension_tolerance", "嵌套宽高相等容差", "判断嵌套区域宽高是否近似相等时使用的容差。")
            render_number_field(st, reward_adv, "nesting_exact_match_tolerance", "嵌套完全相等容差", "判断嵌套区域是否几乎完全重合时使用的更严格容差。")
            render_number_field(st, reward_adv, "nesting_equal_dimension_penalty_scale", "嵌套等宽高惩罚倍率", "嵌套区域近似正方形时的基础惩罚倍率。")
            render_number_field(st, reward_adv, "nesting_equal_dimension_exact_multiplier", "嵌套完全相等额外倍率", "嵌套区域几乎完全相等时，对基础惩罚再乘的倍数。")
            render_number_field(st, reward_adv, "nesting_sigmoid_scale", "嵌套Sigmoid斜率", "控制嵌套惩罚 Sigmoid 曲线陡峭程度。")
            render_number_field(st, reward_adv, "nesting_penalty_scale", "嵌套惩罚倍率", "嵌套比例超过允许范围后，整体惩罚的放大系数。")

        with st.container(border=True):
            st.markdown("#### 功能奖励")
            render_number_field(st, reward_adv, "prior_layout_scale", "先验布局内部倍率", "先验位置得分进入总奖励前的内部放大倍数，最终是“先验评分 × 该倍率 × 先验知识布局奖励权重”。")
            render_number_field(st, reward_adv, "adjacency_synergy_reward", "邻接协同奖励", "协同房间对靠近或相邻时的正向奖励基值。")
            render_number_field(st, reward_adv, "adjacency_repellent_penalty", "邻接排斥惩罚", "排斥房间对靠近或相邻时的基础惩罚值。")
            render_number_field(st, reward_adv, "privacy_sigmoid_k", "隐私 Sigmoid 斜率", "控制隐私奖励曲线变化快慢，越大表示在阈值附近变化越剧烈。")
            render_number_field(st, reward_adv, "privacy_sigmoid_midpoint", "隐私 Sigmoid 中点", "隐私奖励 Sigmoid 曲线的中心位置。")
            render_number_field(st, reward_adv, "privacy_base_reward", "隐私基础奖励", "私密空间与公共空间分离时的基础奖励底值。")
            render_number_field(st, reward_adv, "privacy_reward_scale", "隐私奖励放大倍数", "隐私奖励整体乘上的放大系数。")
            render_number_field(st, reward_adv, "south_region_ratio", "南向区域比例", "从轮廓底部往上，取多少比例范围作为南向区域。")
            render_number_field(st, reward_adv, "south_position_reward_scale", "南向位置奖励倍率", "房间位于南向区域时的位置奖励放大系数。")
            render_number_field(st, reward_adv, "south_balcony_bonus", "南向阳台额外奖励", "阳台位于南向区域时额外叠加的奖励。")
            render_number_field(st, reward_adv, "noise_sensitivity_bedroom", "卧室噪声敏感度", "卧室对相邻高噪声空间的敏感程度，越大表示越怕噪。")
            render_number_field(st, reward_adv, "noise_sensitivity_bathroom", "浴室噪声敏感度", "浴室对相邻高噪声空间的敏感程度。")
            render_number_field(st, reward_adv, "noise_sensitivity_kitchen", "厨房噪声敏感度", "厨房对相邻高噪声空间的敏感程度。")
            render_number_field(st, reward_adv, "noise_sensitivity_living_room", "客厅噪声敏感度", "客厅对相邻高噪声空间的敏感程度。")
            render_number_field(st, reward_adv, "noise_sensitivity_balcony", "阳台噪声敏感度", "阳台对相邻高噪声空间的敏感程度。")
            render_number_field(st, reward_adv, "noise_adjacent_distance_threshold", "噪声邻接距离阈值", "两个空间距离小于该值时，会被视为噪声上相邻。")

        with st.container(border=True):
            st.markdown("#### 规则列表")
            reward_adv["adjacency_synergistic_pairs"] = _parse_literal(
                st.text_input(
                    "协同房间对",
                    value=str(reward_adv["adjacency_synergistic_pairs"]),
                    help="定义哪些房间类型对彼此靠近有利，例如 [('BEDROOM','BATHROOM'), ('BEDROOM','BALCONY')]。",
                    key=f"{config_id}_reward_adv_adjacency_synergistic_pairs",
                ),
                reward_adv["adjacency_synergistic_pairs"],
            )
            reward_adv["adjacency_repellent_pairs"] = _parse_literal(
                st.text_input(
                    "排斥房间对",
                    value=str(reward_adv["adjacency_repellent_pairs"]),
                    help="定义哪些房间类型对彼此靠近不利，例如 [('BEDROOM','KITCHEN')]。",
                    key=f"{config_id}_reward_adv_adjacency_repellent_pairs",
                ),
                reward_adv["adjacency_repellent_pairs"],
            )
            reward_adv["privacy_private_spaces"] = _parse_literal(
                st.text_input(
                    "私密空间列表",
                    value=str(reward_adv["privacy_private_spaces"]),
                    help="这些房间类型会按私密空间参与公私分区奖励，例如 ['BEDROOM','BATHROOM']。",
                    key=f"{config_id}_reward_adv_privacy_private_spaces",
                ),
                reward_adv["privacy_private_spaces"],
            )
            reward_adv["south_lighting_room_types"] = _parse_literal(
                st.text_input(
                    "参与南向采光的房间类型",
                    value=str(reward_adv["south_lighting_room_types"]),
                    help="这些房间类型会参与南向采光奖励计算，例如 ['BEDROOM','LIVING_ROOM']。",
                    key=f"{config_id}_reward_adv_south_lighting_room_types",
                ),
                reward_adv["south_lighting_room_types"],
            )
            reward_adv["noise_calc_room_types"] = _parse_literal(
                st.text_input(
                    "参与噪声归一化的房间类型",
                    value=str(reward_adv["noise_calc_room_types"]),
                    help="这些房间类型会参与噪声奖励中的归一化计数，例如 ['BEDROOM','BATHROOM','KITCHEN','LIVING_ROOM']。",
                    key=f"{config_id}_reward_adv_noise_calc_room_types",
                ),
                reward_adv["noise_calc_room_types"],
            )
            if show_param_reference:
                st.caption("来源函数: HouseEnv._cal_simplified_adjacency_reward() / _cal_privacy_partition_reward() / _cal_south_lighting_reward() / _cal_noise_partition_reward()")
                st.caption("公式摘要: 这些列表参数不直接乘数值，但会改变哪些房间对参与邻接、哪些房间算私密空间、哪些房间参与南向采光、以及噪声归一化分母 N 的统计范围。")
                st.caption("核心变量: `adjacency_synergistic_pairs`, `adjacency_repellent_pairs`, `privacy_private_spaces`, `south_lighting_room_types`, `noise_calc_room_types`")

    with st.expander("算法高级参数", expanded=False):
        algo_defaults = {
            "mappo_hidden_size": 256,
            "mappo_gamma": 0.99,
            "mappo_gae_lambda": 0.95,
            "mappo_clip_epsilon": 0.2,
            "mappo_value_coef": 0.5,
            "mappo_entropy_coef": 0.01,
            "mappo_ppo_epochs": 5,
            "mappo_max_grad_norm": 0.5,
            "mappo_update_step": 1024,
            "mappo_reward_clip_range": 8.0,
            "qmix_hidden_size": 128,
            "qmix_mixing_hidden_size": 32,
            "qmix_hyper_hidden_size": 64,
            "qmix_gamma": 0.99,
            "qmix_tau": 0.01,
            "qmix_buffer_size": 20000,
            "qmix_update_frequency": 50,
            "qmix_epsilon": 1.0,
            "qmix_epsilon_decay": 0.999,
            "qmix_min_epsilon": 0.01,
            "madqn_hidden_dim": 256,
            "madqn_lr": 0.0001,
            "madqn_gamma": 0.99,
            "madqn_tau": 0.01,
            "madqn_epsilon_start": 1.0,
            "madqn_epsilon_end": 0.01,
            "madqn_epsilon_decay": 0.9995,
            "madqn_buffer_capacity": 100000,
            "madqn_device": "cuda:0",
        }
        for key, default_value in algo_defaults.items():
            algo_adv.setdefault(key, default_value)
        st.caption("这部分控制各算法的网络结构、优化超参数和探索参数；只有在你明确要调算法行为时再修改。")

        with st.container(border=True):
            st.markdown("#### MAPPO")
            render_number_field(st, algo_adv, "mappo_hidden_size", "隐藏层维度", "MAPPO actor/critic 主干网络的隐藏层宽度。", step=1, min_value=1)
            render_number_field(st, algo_adv, "mappo_gamma", "Gamma", "折扣因子，越大越重视长期回报。", step=0.01, min_value=0.0, max_value=1.0)
            render_number_field(st, algo_adv, "mappo_gae_lambda", "GAE Lambda", "优势估计的平滑系数，越大越偏向长期估计。", step=0.01, min_value=0.0, max_value=1.0)
            render_number_field(st, algo_adv, "mappo_clip_epsilon", "Clip Epsilon", "PPO 裁剪范围，越大更新越激进。", step=0.01, min_value=0.0)
            render_number_field(st, algo_adv, "mappo_value_coef", "Value Coef", "价值函数损失在总损失中的权重。", step=0.01, min_value=0.0)
            render_number_field(st, algo_adv, "mappo_entropy_coef", "Entropy Coef", "策略熵奖励权重，越大越鼓励探索。", step=0.001, min_value=0.0, fmt="%.3f")
            render_number_field(st, algo_adv, "mappo_ppo_epochs", "PPO Epochs", "每次更新时对同一批数据重复优化的轮数。", step=1, min_value=1)
            render_number_field(st, algo_adv, "mappo_max_grad_norm", "Max Grad Norm", "梯度裁剪上限，用来稳定训练。", step=0.1, min_value=0.0)
            render_number_field(st, algo_adv, "mappo_update_step", "Update Step", "累计多少步经验后触发一次 PPO 更新。", step=1, min_value=1)
            render_number_field(st, algo_adv, "mappo_reward_clip_range", "Reward Clip Range", "奖励归一化前的裁剪范围，避免极端奖励导致更新不稳定。", step=0.1, min_value=0.0)

        with st.container(border=True):
            st.markdown("#### QMIX")
            render_number_field(st, algo_adv, "qmix_hidden_size", "隐藏层维度", "每个智能体 Q 网络的隐藏层宽度。", step=1, min_value=1)
            render_number_field(st, algo_adv, "qmix_mixing_hidden_size", "混合层维度", "QMIX mixing network 的隐藏层维度。", step=1, min_value=1)
            render_number_field(st, algo_adv, "qmix_hyper_hidden_size", "超网络维度", "生成 mixing 权重的 hyper network 隐藏维度。", step=1, min_value=1)
            render_number_field(st, algo_adv, "qmix_gamma", "Gamma", "折扣因子，越大越重视长期回报。", step=0.01, min_value=0.0, max_value=1.0)
            render_number_field(st, algo_adv, "qmix_tau", "Tau", "目标网络软更新系数。", step=0.001, min_value=0.0, max_value=1.0, fmt="%.3f")
            render_number_field(st, algo_adv, "qmix_buffer_size", "Buffer Size", "经验回放池容量。", step=1, min_value=1)
            render_number_field(st, algo_adv, "qmix_update_frequency", "Update Frequency", "每隔多少环境步执行一次网络更新。", step=1, min_value=1)
            render_number_field(st, algo_adv, "qmix_epsilon", "初始 Epsilon", "QMIX 初始探索率。", step=0.01, min_value=0.0, max_value=1.0)
            render_number_field(st, algo_adv, "qmix_epsilon_decay", "Epsilon Decay", "每步对探索率的衰减倍率。", step=0.0001, min_value=0.0, max_value=1.0, fmt="%.4f")
            render_number_field(st, algo_adv, "qmix_min_epsilon", "最小 Epsilon", "探索率衰减后的下限。", step=0.01, min_value=0.0, max_value=1.0)

        with st.container(border=True):
            st.markdown("#### MADQN")
            render_number_field(st, algo_adv, "madqn_hidden_dim", "隐藏层维度", "MADQN 单体 Q 网络的隐藏层宽度。", step=1, min_value=1)
            render_number_field(st, algo_adv, "madqn_lr", "学习率", "MADQN 优化器学习率。", step=0.0001, min_value=0.0, fmt="%.5f")
            render_number_field(st, algo_adv, "madqn_gamma", "Gamma", "折扣因子。", step=0.01, min_value=0.0, max_value=1.0)
            render_number_field(st, algo_adv, "madqn_tau", "Tau", "目标网络软更新系数。", step=0.001, min_value=0.0, max_value=1.0, fmt="%.3f")
            render_number_field(st, algo_adv, "madqn_epsilon_start", "初始 Epsilon", "MADQN 训练开始时的探索率。", step=0.01, min_value=0.0, max_value=1.0)
            render_number_field(st, algo_adv, "madqn_epsilon_end", "最小 Epsilon", "MADQN 探索率衰减后的最小值。", step=0.01, min_value=0.0, max_value=1.0)
            render_number_field(st, algo_adv, "madqn_epsilon_decay", "Epsilon Decay", "每步对探索率的衰减倍率。", step=0.0001, min_value=0.0, max_value=1.0, fmt="%.4f")
            render_number_field(st, algo_adv, "madqn_buffer_capacity", "Buffer Capacity", "经验回放池容量。", step=1, min_value=1)
            algo_adv["madqn_device"] = st.text_input(
                "运行设备",
                value=str(algo_adv.get("madqn_device", "cuda:0")),
                help="指定 MADQN 运行设备，例如 `cuda:0` 或 `cpu`；如果设备不可用，代码仍可能自动回退。",
                key=f"{config_id}_algo_madqn_device",
            )
            if show_param_reference:
                st.caption("来源函数: MADQN.__init__()")
                st.caption("公式摘要: 该值只用于选择 `torch.device(device)`；如果 CUDA 不可用，代码会自动回退到 CPU。")
                st.caption("核心变量: `device`, `torch.cuda.is_available()`, `self.device`")
                st.caption(f"当前代入: madqn_device={algo_adv['madqn_device']}")
def render_environment_editor(config: dict, config_id: str) -> None:
    """渲染交互式环境编辑器。"""
    st.markdown("### 环境编辑")
    st.caption("编辑环境边界、门、先验位置后，请显式保存，确保训练读取的是最新配置文件。")

    if st_canvas is None:
        st.warning("当前环境未安装 `streamlit-drawable-canvas`，环境画布编辑功能暂不可用。")
        st.code("python -m pip install streamlit-drawable-canvas", language="bash")
        return

    env_conf = config.setdefault("Environment", {})
    boundary = env_conf.get("boundary", [[2, 2], [10, 2], [10, 12], [2, 12]])
    doors = env_conf.get("door_positions", [[8, 12], [9, 12]])
    room_types = env_conf.get("room_type", [])
    priors = env_conf.get("prior", [])
    limit_area = env_conf.get("prior_limit_area", [])

    if st.session_state.force_refresh_canvas:
        st.session_state.canvas_key = f"canvas_environment_edit_{int(time.time())}"
        st.session_state.force_refresh_canvas = False

    grid_size = compute_grid_size(boundary, priors, limit_area)
    canvas_width = 800
    canvas_height = 800

    col_tool1, col_tool2, col_tool3 = st.columns([2, 1, 2])
    with col_tool1:
        add_type = st.selectbox("添加对象", ["环境边界顶点", "门位置", "先验位置限制顶点"] + list(ROOM_COLORS.keys()))
    with col_tool2:
        if st.button("添加对象", use_container_width=True):
            center_x = grid_size / 2
            center_y = grid_size / 2
            if add_type == "环境边界顶点":
                env_conf.setdefault("boundary", []).append([center_x, center_y])
            elif add_type == "门位置":
                env_conf.setdefault("door_positions", []).append([center_x, center_y])
            elif add_type == "先验位置限制顶点":
                env_conf.setdefault("prior_limit_area", []).append([center_x, center_y])
            else:
                env_conf.setdefault("room_type", []).append(add_type)
                env_conf.setdefault("prior", []).append([center_x, center_y])
            save_config(config, config_id)
            st.session_state.force_refresh_canvas = True
            st.rerun()
    with col_tool3:
        if st.button("刷新画布", use_container_width=True):
            st.session_state.force_refresh_canvas = True
            st.rerun()

    background_image = create_boundary_image(grid_size, boundary, limit_area, canvas_width, canvas_height)
    initial_objects = build_initial_objects(boundary, doors, room_types, priors, limit_area, grid_size, canvas_width, canvas_height)

    legend_items = []
    unique_rooms = list(dict.fromkeys(room_types))
    for room in unique_rooms:
        legend_items.append(
            (
                room,
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"background:{get_room_color(room)};border-radius:3px;border:1px solid #8a94a6;'></span>",
            )
        )

    if doors:
        legend_items.append(
            (
                "门",
                "<span style='display:inline-block;width:14px;height:14px;"
                "background:brown;border-radius:3px;border:1px solid #8a94a6;'></span>",
            )
        )

    if limit_area:
        legend_items.append(
            (
                "先验位置限制区",
                "<span style='display:inline-block;width:16px;height:12px;"
                "background:rgba(255,165,0,0.18);border:2px solid orange;border-radius:2px;'></span>",
            )
        )

    if legend_items:
        legend_columns = st.columns(min(len(legend_items), 6))
        for index, (label, marker_html) in enumerate(legend_items):
            legend_columns[index % len(legend_columns)].markdown(
                (
                    f"<div style='display:flex;align-items:center;gap:8px;'>"
                    f"{marker_html}"
                    f"<span>{label}</span></div>"
                ),
                unsafe_allow_html=True,
            )

    current_direction = env_conf.get("door_direction", "水平")
    direction_options = ["水平", "垂直"]
    if current_direction not in direction_options:
        current_direction = "水平"
    new_direction = st.radio("门方向", direction_options, index=direction_options.index(current_direction), horizontal=True)

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=2,
        stroke_color="black",
        background_image=background_image,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="transform",
        key=st.session_state.canvas_key,
        initial_drawing={"objects": initial_objects},
        display_toolbar=True,
    )

    if canvas_result.json_data is None:
        return

    objects = canvas_result.json_data["objects"]
    new_boundary, new_doors, new_room_types, new_priors, new_limit_area = parse_canvas_objects(
        objects,
        grid_size,
        canvas_width,
        canvas_height,
    )

    st.markdown("#### 结构数据校正")
    col1, col2, col3 = st.columns(3)
    with col1:
        edited_boundary = st.data_editor(pd.DataFrame(new_boundary, columns=["X", "Y"]), key="edit_boundary", num_rows="dynamic", hide_index=True)
        final_boundary = edited_boundary.values.tolist()
    with col2:
        edited_doors = st.data_editor(pd.DataFrame(new_doors, columns=["X", "Y"]), key="edit_doors", num_rows="dynamic", hide_index=True)
        final_doors = edited_doors.values.tolist()
    with col3:
        room_data = [{"Type": room, "X": point[0], "Y": point[1]} for room, point in zip(new_room_types, new_priors)]
        room_frame = pd.DataFrame(room_data, columns=["Type", "X", "Y"])
        edited_rooms = st.data_editor(room_frame, key="edit_rooms", num_rows="dynamic", hide_index=True)
        final_room_types = []
        final_priors = []
        if not edited_rooms.empty:
            for _, row in edited_rooms.iterrows():
                final_room_types.append(str(row["Type"]))
                final_priors.append([float(row["X"]), float(row["Y"])])
    edited_limit = st.data_editor(pd.DataFrame(new_limit_area, columns=["X", "Y"]), key="edit_limit", num_rows="dynamic", hide_index=True)
    final_limit_area = edited_limit.values.tolist()

    door_validation = validate_door_direction(final_doors, new_direction)
    if door_validation["level"] == "success":
        st.success(door_validation["message"])
    elif door_validation["level"] == "warning":
        st.warning(door_validation["message"])
    else:
        st.info(door_validation["message"])

    if st.button("保存环境编辑结果", use_container_width=True):
        env_conf["boundary"] = final_boundary
        env_conf["door_positions"] = final_doors
        env_conf["prior"] = final_priors
        env_conf["room_type"] = final_room_types
        saved_direction = new_direction
        inferred_direction = door_validation.get("inferred_direction")
        if inferred_direction and inferred_direction != new_direction:
            saved_direction = inferred_direction
            st.warning(f"检测到门方向与门点分布不一致，已自动修正为{saved_direction}。")
        env_conf["door_direction"] = saved_direction
        env_conf["prior_limit_area"] = final_limit_area
        saved_path = save_config(config, config_id)
        st.success(f"环境配置已保存到: {saved_path}")
        st.session_state.force_refresh_canvas = True
        time.sleep(0.5)
        st.rerun()
