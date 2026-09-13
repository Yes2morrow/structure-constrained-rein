import os

import streamlit as st

from common.config_manager import normalize_config_id
from gui.config_store import get_config_summary, get_selected_config_id, persist_selected_config_id, save_config, get_structure_config_ids, load_config
from common.config_manager import get_available_config_ids
from gui.training_service import start_training, stop_training


def _training_stage_label(config: dict, config_id: str) -> str:
    """返回当前训练阶段标签。"""
    training = config.get("Training", {})
    stage = str(st.session_state.get(f"{config_id}_training_stage", training.get("training_stage", "room_training"))).strip()
    if stage == "floor_partition":
        program = str(st.session_state.get(f"{config_id}_floor_partition_type", config.get("FloorPartition", {}).get("program_type", "residential"))).strip()
        if program == "office":
            return "楼层功能分区（办公）"
        return "楼层功能分区（住宅）"
    return "户型内部训练"


def render_training_status_indicator(status_text: str, training_pid) -> None:
    """渲染训练状态灯与 PID。"""
    is_running = status_text in {"Running", "Stopping"}
    if status_text == "Stopping":
        status_color = "#f39c12"
        status_label = "停止中"
    elif is_running:
        status_color = "#2e8b57"
        status_label = "训练中"
    else:
        status_color = "#c0392b"
        status_label = "已停止"
    pid_text = str(training_pid) if training_pid is not None else "无"

    st.markdown(
        (
            "<div style='padding:10px 12px;border:1px solid #d8e0ea;border-radius:8px;"
            "background:#f8fafc;margin:6px 0 10px 0;'>"
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            f"background:{status_color};'></span>"
            f"<span style='font-weight:600;color:#243b53;'>{status_label}</span>"
            "</div>"
            f"<div style='margin-top:6px;color:#52667a;font-size:0.92rem;'>当前训练 PID: {pid_text}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_config_selector() -> str:
    """渲染配置选择区域，并返回当前配置编号。"""
    with st.sidebar:
        st.markdown("## 配置管理")
        current_config_id = get_selected_config_id(st.session_state)
        available_ids = get_structure_config_ids()
        if current_config_id not in available_ids:
            available_ids.append(current_config_id)
            available_ids = sorted(set(available_ids), key=str)

        selected_config_id = st.selectbox(
            "配置编号",
            options=available_ids,
            index=available_ids.index(current_config_id),
            help="切换后立即重新加载对应的 config/<编号>/config.yaml",
            key="config_selector",
        )
        normalized_config_id = normalize_config_id(selected_config_id)
        if normalized_config_id != current_config_id:
            persist_selected_config_id(normalized_config_id)
            st.session_state.selected_config_id = normalized_config_id
            st.session_state.force_refresh_canvas = True
            st.rerun()

        current_path, available_ids = get_config_summary(current_config_id)
        st.caption(f"当前配置: {current_path}")
        st.caption(f"已有编号: {available_ids}")
        st.markdown("---")

        new_config_id = st.text_input(
            "新增配置编号",
            value="",
            help="输入新编号后会创建对应目录，例如 i 或 2。",
            key="new_config_id",
        ).strip()
        if st.button("创建并切换", use_container_width=True):
            if not new_config_id:
                st.warning("请先输入新的配置编号。")
            else:
                normalized_new_id = normalize_config_id(new_config_id)
                if normalized_new_id not in get_available_config_ids():
                    save_config(load_config(current_config_id),normalized_new_id)
                elif normalized_new_id not in get_structure_config_ids():
                    st.error('该编号属于其他项目，请使用新的编号创建结构约束配置。')
                    return current_config_id
                persist_selected_config_id(normalized_new_id)
                st.session_state.selected_config_id = normalized_new_id
                st.session_state.force_refresh_canvas = True
                st.rerun()

    return get_selected_config_id(st.session_state)


def render_sidebar_controls(config: dict, config_id: str) -> None:
    """渲染侧边栏训练控制区域。"""
    with st.sidebar:
        st.markdown("## 训练控制")
        status_text = st.session_state.training_status
        training_pid = st.session_state.training_pid
        stage_label = _training_stage_label(config, config_id)
        render_training_status_indicator(status_text, training_pid)
        st.metric("运行状态", status_text)
        st.caption(f"算法类型: {config.get('Training', {}).get('agent_name', 'mappo')}")
        st.caption(f"当前训练阶段: {stage_label}")
        st.caption(f"续训模型目录: {config.get('Training', {}).get('ckpt_path', '') or '未设置（新训练无需设置）'}")
        st.caption("继续训练会恢复历史轮次与监控计数。")
        action_message = None
        action_message_type = None

        if st.button(f"新训练（{stage_label}）", use_container_width=True, disabled=status_text == "Running"):
            training_conf = config.setdefault("Training", {})
            training_conf["ckpt_path"] = ""
            training_conf["resume_mode"] = "fresh"
            save_config(config, config_id)
            success, message = start_training(config_id)
            if success:
                st.session_state.main_view = "训练监控"
                st.session_state.main_view_selector = "训练监控"
            action_message = f"已按新训练模式启动：{stage_label} 不会读取任何预训练模型。"
            action_message_type = "info"
            if message:
                action_message = f"{action_message}\n\n{message}"
                action_message_type = "success" if success else "error"
            if success:
                st.rerun()

        ckpt_path = config.get("Training", {}).get("ckpt_path", "")
        if st.button(f"继续训练（{stage_label}）", use_container_width=True, disabled=status_text == "Running"):
            if not ckpt_path:
                action_message = "当前配置未设置续训模型目录，请先在参数配置页填写并保存。"
                action_message_type = "error"
            elif not os.path.isdir(ckpt_path):
                action_message = "当前续训模型目录不是有效目录，请检查配置文件中的路径。"
                action_message_type = "error"
            else:
                config.setdefault("Training", {})["resume_mode"] = "resume"
                save_config(config, config_id)
                success, message = start_training(config_id)
                if success:
                    st.session_state.main_view = "训练监控"
                    st.session_state.main_view_selector = "训练监控"
                action_message = f"将从检查点恢复 {stage_label} 的历史轮次、训练状态和监控计数继续训练。"
                action_message_type = "info"
                if message:
                    action_message = f"{action_message}\n\n{message}"
                    action_message_type = "success" if success else "error"
                if success:
                    st.rerun()

        if action_message:
            if action_message_type == "success":
                st.success(action_message)
            elif action_message_type == "error":
                st.error(action_message)
            elif action_message_type == "warning":
                st.warning(action_message)
            else:
                st.info(action_message)

        if st.button("停止训练", use_container_width=True, disabled=status_text == "Stopped"):
            stop_training()

        st.markdown("---")
        st.markdown("## 运行说明")
        st.caption("TensorBoard 日志目录: results2")
