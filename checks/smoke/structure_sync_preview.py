"""Isolated browser QA: all saves stay in this browser session, never in YAML."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from copy import deepcopy
import streamlit as st
from gui.config_store import load_config
from core.envs.structure_geometry import column, wall, normalize_structures
import gui.structure_editor as editor
import gui.boundary_editor as boundary_editor
from gui.adaptive_reuse_page import _render_plan_constraint_editor

st.set_page_config(layout='wide')
if 'qa_config' not in st.session_state:
    st.session_state.qa_config = load_config('retrofit')
    st.session_state.qa_config['ExistingBuilding']['fixed_objects'] = normalize_structures([
        column('qa_column',5,5,.6,.8),wall('qa_wall','shear_wall',10,5,18,8,.2,.1)])
editor.load_config = lambda _: deepcopy(st.session_state.qa_config)
editor.save_config = lambda c,i: st.session_state.update(qa_config=deepcopy(c))
boundary_editor.load_config = editor.load_config
boundary_editor.save_config = editor.save_config
st.info('隔离验证页：编辑仅保存在当前浏览器会话。')
_render_plan_constraint_editor(deepcopy(st.session_state.qa_config),'qa')
