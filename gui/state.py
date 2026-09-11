import os

import streamlit as st

from gui.config_store import load_persisted_config_id, get_structure_config_ids
from common.project_paths import PID_FILE, STOP_REQUEST_FILE


def init_session_state(is_process_running) -> None:
    """初始化界面运行状态。"""
    if "training_pid" not in st.session_state:
        st.session_state.training_pid = None

    if "training_status" not in st.session_state:
        st.session_state.training_status = "Stopped"

    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = "canvas_environment_edit_v1"

    if "force_refresh_canvas" not in st.session_state:
        st.session_state.force_refresh_canvas = False

    if st.session_state.get('selected_config_id') not in get_structure_config_ids():
        st.session_state.selected_config_id = load_persisted_config_id()
        st.session_state.pop('config_selector',None)

    if "monitor_log_cleared_line_count" not in st.session_state:
        st.session_state.monitor_log_cleared_line_count = 0

    if "tensorboard_pid" not in st.session_state:
        st.session_state.tensorboard_pid = None

    if "tensorboard_url" not in st.session_state:
        st.session_state.tensorboard_url = ""

    if "tensorboard_logdir" not in st.session_state:
        st.session_state.tensorboard_logdir = ""

    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r", encoding="utf-8") as file:
                saved_pid = int(file.read().strip())

            if is_process_running(saved_pid):
                st.session_state.training_pid = saved_pid
                st.session_state.training_status = "Stopping" if os.path.exists(STOP_REQUEST_FILE) else "Running"
            else:
                os.remove(PID_FILE)
                if os.path.exists(STOP_REQUEST_FILE):
                    os.remove(STOP_REQUEST_FILE)
        except Exception:
            st.session_state.training_pid = None
            st.session_state.training_status = "Stopped"
    else:
        current_pid = st.session_state.training_pid
        if current_pid is None or not is_process_running(int(current_pid)):
            st.session_state.training_pid = None
            st.session_state.training_status = "Stopped"
