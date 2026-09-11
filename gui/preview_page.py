import glob
import os

import streamlit as st

from gui.image_utils import load_image_safely
from common.project_paths import RESULTS_DIR

MAX_PREVIEW_IMAGES = 6


def _safe_get_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return float("-inf")


def _list_result_images() -> list[str]:
    """收集结果目录下可展示的图片，按更新时间倒序排列。"""
    image_paths = glob.glob(os.path.join(RESULTS_DIR, "**", "*.png"), recursive=True)
    image_paths = [image_path for image_path in image_paths if os.path.isfile(image_path)]
    return sorted(image_paths, key=_safe_get_mtime, reverse=True)


def _relative_image_label(image_path: str) -> str:
    try:
        return os.path.relpath(image_path, RESULTS_DIR).replace("\\", "/")
    except ValueError:
        return image_path


def render_preview_page() -> None:
    """渲染布局预览页。"""
    st.subheader("布局预览")
    st.caption("默认展示 results2 中最近生成的 6 张结果图，不自动刷新；你也可以手动指定任意 6 张结果图进行对比。")

    image_paths = _list_result_images()
    if not image_paths:
        st.info("暂未检测到布局图片，请先启动训练并产出结果图。")
        st.caption(f"搜索目录: {os.path.abspath(RESULTS_DIR)}")
        return

    default_images = image_paths[:MAX_PREVIEW_IMAGES]
    image_options = {_relative_image_label(image_path): image_path for image_path in image_paths}
    if "preview_selected_image_labels" not in st.session_state:
        st.session_state.preview_selected_image_labels = [
            _relative_image_label(image_path) for image_path in default_images
        ]

    control_col1, control_col2 = st.columns([1.2, 4.0])
    with control_col1:
        if st.button("恢复默认六张", use_container_width=True):
            st.session_state.preview_selected_image_labels = [
                _relative_image_label(image_path) for image_path in default_images
            ]
            st.rerun()
    with control_col2:
        selected_labels = st.multiselect(
            "手动选择要展示的结果图（最多 6 张）",
            options=list(image_options.keys()),
            default=[
                label for label in st.session_state.preview_selected_image_labels
                if label in image_options
            ],
            help="默认显示最新 6 张图。你可以从 results2 中手动勾选任意结果图进行六宫格对比。",
        )

    if len(selected_labels) > MAX_PREVIEW_IMAGES:
        st.warning("最多只展示 6 张图，已自动截取前 6 张选择结果。")
        selected_labels = selected_labels[:MAX_PREVIEW_IMAGES]

    if not selected_labels:
        selected_labels = [_relative_image_label(image_path) for image_path in default_images]

    st.session_state.preview_selected_image_labels = selected_labels
    selected_paths = [image_options[label] for label in selected_labels if label in image_options]

    st.caption(f"当前共展示 {len(selected_paths)} 张图；搜索目录: {os.path.abspath(RESULTS_DIR)}")

    preview_columns = st.columns(3)
    for index, image_path in enumerate(selected_paths):
        image = load_image_safely(image_path)
        with preview_columns[index % 3]:
            st.caption(_relative_image_label(image_path))
            if image is not None:
                st.image(image, caption=image_path, use_container_width=True)
            else:
                st.warning("该图像正在更新或暂不可读，请稍后手动刷新页面。")
