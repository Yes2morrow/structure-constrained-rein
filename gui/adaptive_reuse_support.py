"""既有住宅页面共用的轻量交互辅助函数。"""

from __future__ import annotations

import streamlit as st

from gui.layout_repair import repair_conflicts_and_save


def consume_auto_repair_notice(config_id: str) -> None:
    """读取并展示一次性自动修复通知。"""
    notice = st.session_state.pop(f"{config_id}_auto_repair_notice", None)
    if not notice:
        return
    level = notice.get("level", "success")
    if level == "info":
        st.info(notice["message"])
    elif level == "warning":
        st.warning(notice["message"])
    else:
        st.success(notice["message"])


def set_auto_repair_notice(config_id: str, rect_repairs: list[dict], seed_fixes: list[dict]) -> None:
    """统一整理自动修复后的页面提示文案。"""
    if rect_repairs or seed_fixes:
        labels = "；".join(
            [f"{item['space_id']}: {item['from_rect']} -> {item['to_rect']}" for item in rect_repairs]
            + [f"{item['space_id']} 种子: {item['from']} -> {item['to']}" for item in seed_fixes]
        )
        notice = {
            "level": "success",
            "message": f"已自动修复冲突布局并保存到 YAML：\n{labels}",
        }
    else:
        notice = {
            "level": "info",
            "message": "当前初始布局已合法，无需修复。",
        }
    st.session_state[f"{config_id}_auto_repair_notice"] = notice


def run_auto_repair_and_rerun(config: dict, config_id: str, seed: int) -> None:
    """执行一次自动修复，更新提示后立即刷新页面。"""
    rect_repairs, seed_fixes, repair_error = repair_conflicts_and_save(config, config_id, seed)
    if repair_error is not None:
        raise repair_error
    st.session_state[f"{config_id}_ar_canvas_revision"] = st.session_state.get(f"{config_id}_ar_canvas_revision", 0) + 1
    set_auto_repair_notice(config_id, rect_repairs, seed_fixes)
    st.rerun()


def render_collapsible_subheader(title: str, state_key: str) -> bool:
    """渲染带显式展开/收起按钮的二级标题。"""
    collapsed = bool(st.session_state.get(state_key, False))
    title_column, toggle_column = st.columns([10, 1.35], vertical_alignment="center")
    with title_column:
        st.markdown(f"## {title}")
    with toggle_column:
        if st.button(
            "展开 ▼" if collapsed else "收起 ▲",
            key=f"{state_key}_toggle",
            use_container_width=True,
        ):
            st.session_state[state_key] = not collapsed
            st.rerun()
    return not collapsed
