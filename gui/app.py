import time

import streamlit as st

from gui.config_store import load_config
from gui.patches import apply_streamlit_patches
from gui.sidebar import render_config_selector, render_sidebar_controls
from gui.state import init_session_state
from gui.theme import apply_theme, configure_page
from gui.training_service import get_latest_log, is_process_running


def _render_environment_tab(config: dict, config_id: str) -> None:
    """仅在进入环境页时加载既有住宅环境搭建模块。"""
    from gui.adaptive_reuse_page import render_adaptive_reuse_environment_page

    render_adaptive_reuse_environment_page(config, config_id)


def _render_config_tab(config: dict, config_id: str, is_adaptive_reuse: bool) -> None:
    """按当前项目类型延迟加载参数页模块。"""
    if is_adaptive_reuse:
        from gui.adaptive_reuse_page import render_adaptive_reuse_config_page

        render_adaptive_reuse_config_page(config, config_id, include_environment=False)
        return
    from gui.config_page import render_config_page

    render_config_page(config, config_id)


def _render_monitor_tab(agent_name: str, configured_episodes: int, render_enabled: bool) -> None:
    """仅在监控页加载训练监控模块。"""
    from gui.monitor_page import render_monitor_page

    render_monitor_page(get_latest_log(), agent_name, configured_episodes, render_enabled)


def _render_preview_tab() -> None:
    """仅在预览页加载结果展示模块。"""
    from gui.preview_page import render_preview_page

    render_preview_page()


def run_app() -> None:
    """运行可视化工作台。"""
    apply_streamlit_patches()
    configure_page()
    apply_theme()
    init_session_state(is_process_running)

    config_id = render_config_selector()
    config = load_config(config_id)
    if not config:
        st.error("无法加载当前配置，请检查 config/<编号>/config.yaml 是否存在。")
        st.stop()

    is_adaptive_reuse = config.get("ProjectType") == "adaptive_reuse"
    st.title("城市更新生成式设计工作平台" if is_adaptive_reuse else "房屋布局强化学习工作台")
    if not is_adaptive_reuse:
        st.caption("配置、训练、监控与结果预览统一收敛到同一工作流中。")

    render_sidebar_controls(config, config_id)

    view_options = (["环境搭建"] if is_adaptive_reuse else []) + ["参数配置", "训练监控", "布局预览"]
    current_view = st.session_state.get("main_view", "参数配置")
    if current_view not in view_options:
        current_view = "参数配置"
    st.session_state.main_view = st.radio(
        "主视图",
        view_options,
        index=view_options.index(current_view),
        horizontal=True,
        key="main_view_selector",
        label_visibility="collapsed",
    )

    is_running = st.session_state.training_status in {"Running", "Stopping"}
    training_conf = config.get("Training", {})
    agent_name = training_conf.get("agent_name", "mappo")
    configured_episodes = int(training_conf.get("episodes", 0))
    render_enabled = bool(config.get("Render", False))

    if st.session_state.main_view == "环境搭建":
        _render_environment_tab(config, config_id)
    elif st.session_state.main_view == "参数配置":
        _render_config_tab(config, config_id, is_adaptive_reuse)
    elif st.session_state.main_view == "训练监控":
        if hasattr(st, "fragment"):
            if is_running:
                @st.fragment(run_every="2s")
                def render_monitor_fragment() -> None:
                    _render_monitor_tab(agent_name, configured_episodes, render_enabled)
            else:
                @st.fragment
                def render_monitor_fragment() -> None:
                    _render_monitor_tab(agent_name, configured_episodes, render_enabled)

            render_monitor_fragment()
        else:
            _render_monitor_tab(agent_name, configured_episodes, render_enabled)
    else:
        _render_preview_tab()

    if is_running and not hasattr(st, "fragment"):
        time.sleep(2)
        st.rerun()
