"""Parameter-based column and wall editor."""
import pandas as pd
import streamlit as st
from core.envs.structure_geometry import parameter_rows, build_structures
from gui.config_store import load_config, save_config
from gui.structure_canvas import signature
from gui.boundary_editor import render_boundary_editor


def save_structures(config, config_id, objects):
    saved = load_config(config_id)
    saved['ExistingBuilding']['fixed_objects'] = objects
    saved['ExistingBuilding']['structure_schema_version'] = 2
    save_config(saved, config_id)
    config['ExistingBuilding']['fixed_objects'] = objects
    config['ExistingBuilding']['structure_schema_version'] = 2
    key = f'{config_id}_ar_canvas_revision'
    st.session_state[key] = st.session_state.get(key, 0) + 1


def render_structure_editor(config, config_id):
    st.markdown('#### 建筑边界、柱与墙体精确输入')
    if not render_boundary_editor(config,config_id):
        return False
    st.caption('单位：米。柱按中心点及 X/Y 尺寸定义；墙线左右以起点→终点方向为准，左右厚度独立输入。表格可增删行。')
    columns, walls, other = parameter_rows(config['ExistingBuilding'].get('fixed_objects', []))
    revision = st.session_state.get(f'{config_id}_ar_canvas_revision', 0)
    with st.container():
        column_df = st.data_editor(pd.DataFrame(columns, columns=['id', 'cx', 'cy', 'width', 'depth']),
            num_rows='dynamic', use_container_width=True, key=f'{config_id}_columns_{revision}',
            column_config={k: st.column_config.NumberColumn(v, format='%.4f') for k, v in
                dict(cx='中心 X', cy='中心 Y', width='X 尺寸（x2−x1）', depth='Y 尺寸（y2−y1）').items()})
        wall_df = st.data_editor(pd.DataFrame(walls, columns=['id', 'type', 'sx', 'sy', 'ex', 'ey', 'left', 'right']),
            num_rows='dynamic', use_container_width=True, key=f'{config_id}_walls_{revision}',
            column_config={
                'type': st.column_config.SelectboxColumn('墙属性', options=['shear_wall', 'load_bearing_wall'], required=True),
                **{k: st.column_config.NumberColumn(v, format='%.4f') for k, v in
                   dict(sx='起点 X', sy='起点 Y', ex='终点 X', ey='终点 Y', left='左侧厚度', right='右侧厚度').items()}})
        st.caption('shear_wall＝剪力墙；load_bearing_wall＝承重墙。两者几何构造相同，属性分别保存。剪力墙中心线无缝闭合时自动识别核心筒；不跨越缺口。核心筒围合区域整体保留，不允许功能空间进入。')
    try:
        objects = build_structures(column_df.to_dict('records'), wall_df.to_dict('records'), other)
        if signature(objects) != signature(config['ExistingBuilding'].get('fixed_objects', [])):
            save_structures(config, config_id, objects)
            st.rerun()
    except (ValueError, TypeError, KeyError) as exc:
        st.warning(f'请补全有效的柱墙数据后自动同步：{exc}')
        return False
    if columns:
        st.caption('柱边界换算：x1=cx−X尺寸/2，x2=cx+X尺寸/2；y1=cy−Y尺寸/2，y2=cy+Y尺寸/2。')
    return True
