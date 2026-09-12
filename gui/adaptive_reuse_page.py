"""既有建筑适应性改造环境的 Web 参数页。"""

from __future__ import annotations

import math
import base64
import hashlib
import json
from io import BytesIO
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps
try:
    from gui.responsive_canvas import st_canvas
except ImportError:
    st_canvas = None

from common.config_manager import get_config_dir
from core.envs import AdaptiveReuseEnv
from gui.config_store import load_config, save_config
from gui.structure_editor import render_structure_editor, save_structures
from gui.boundary_editor import boundary_handles, parse_boundary_handles, save_boundary
from gui.environment_canvas import scene, parse_scene, controls as environment_controls, color as control_color, entrance_points, DISPLAY_LAYERS
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
    "retained_circulation": ("保留交通空间", "rgba(249,168,37,0.48)", "#c17900"),
    "fixed": ("其他不可调整构件", "rgba(55,71,79,0.64)", "#263238"),
}

# 修改默认案例时同步升级此值，防止浏览器端旧组件状态覆盖新的平面配置。
REFERENCE_PLAN_UI_VERSION = "structure_sync_v2"


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


def _rect_records(items: list[dict], include_name: bool = True) -> list[dict]:
    records = []
    for item in items:
        rect = list(item.get("rect", [0, 0, 1, 1]))
        row = {"id": item.get("id", ""), "x1": rect[0], "y1": rect[1], "x2": rect[2], "y2": rect[3]}
        if include_name:
            row["name"] = item.get("name", "")
        else:
            row["type"] = item.get("type", "column")
        records.append(row)
    return records


def _clean(value, fallback=""):
    return fallback if value is None or (isinstance(value, float) and math.isnan(value)) else value


def _records_to_rects(frame: pd.DataFrame, include_name: bool = True) -> list[dict]:
    result = []
    for raw in frame.to_dict("records"):
        identifier = str(_clean(raw.get("id"))).strip()
        if not identifier:
            continue
        item = {
            "id": identifier,
            "rect": [float(raw[key]) for key in ("x1", "y1", "x2", "y2")],
        }
        if include_name:
            item["name"] = str(_clean(raw.get("name"), identifier))
        else:
            item["type"] = str(_clean(raw.get("type"), "column"))
        result.append(item)
    return result


def _target_records(items: list[dict]) -> list[dict]:
    records = []
    for item in items:
        rect = item.get("initial_rect", [0, 0, 1, 1])
        areas = item.get("area_range", [item.get("target_area", 1), item.get("target_area", 1)])
        aspects = item.get("aspect_range", [1, 2.5])
        records.append({
            "id": item.get("id", ""), "name": item.get("name", ""),
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
    """根据原始空间数据生成可直接标注的传统堂屋二维底图。"""
    building = config["ExistingBuilding"]
    boundary = building["boundary"]
    xs, ys = [float(point[0]) for point in boundary], [float(point[1]) for point in boundary]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    x_span, y_span = max(max_x - min_x, 1e-9), max(max_y - min_y, 1e-9)
    image = Image.new("RGB", (width, height), "#faf9f5")
    draw = ImageDraw.Draw(image)
    title_font, room_font, small_font = _plan_font(22), _plan_font(18), _plan_font(14)

    def point(x: float, y: float) -> tuple[float, float]:
        scale, ox, oy = viewport(boundary, width, height)
        return (ox+x*scale, oy-y*scale)

    # 细网格只作为坐标参考，原有房间以浅色区分，结构约束由上层彩色图层表达。
    grid = float(config.get("AdaptiveReuseEnvironment", {}).get("grid_size", 0.5))
    for x in np.arange(min_x, max_x + grid, grid):
        px, _ = point(x, min_y)
        draw.line([(px, 0), (px, height)], fill="#edf0f2", width=1)
    for y in np.arange(min_y, max_y + grid, grid):
        _, py = point(min_x, y)
        draw.line([(0, py), (width, py)], fill="#edf0f2", width=1)

    if grid_only: return image

    room_colors = ["#f7dada", "#f6d5d7", "#f8dcdd", "#cfe8cf", "#f4d5d8", "#f7d8da"]
    for index, item in enumerate(building.get("original_spaces", [])):
        x1, y1, x2, y2 = map(float, item["rect"])
        left, bottom = point(x1, y1)
        right, top = point(x2, y2)
        draw.rectangle([left, top, right, bottom], fill=room_colors[index % len(room_colors)], outline="#7c8791", width=2)
        label = str(item.get("name", item.get("id", "原空间")))
        box_value = draw.textbbox((0, 0), label, font=room_font)
        text_width = box_value[2] - box_value[0]
        text_height = box_value[3] - box_value[1]
        cx, cy = (left + right) / 2, (top + bottom) / 2
        draw.rounded_rectangle(
            [cx - text_width / 2 - 7, cy - text_height / 2 - 5, cx + text_width / 2 + 7, cy + text_height / 2 + 5],
            radius=5, fill="#ffffffdd",
        )
        draw.text((cx - text_width / 2, cy - text_height / 2 - 1), label, fill="#263746", font=room_font)

    # 在底图中画出中央楼梯/电梯核心区的内部符号，约束图层会半透明覆盖其上。
    core_item = next((item for item in building.get("fixed_objects", []) if item.get("type") == "core" and not item.get('derived')), None)
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
        draw.text((left + third + 8, (top + bottom) / 2 - 10), "核心筒", fill="#263238", font=small_font)

    boundary_pixels = [point(float(x), float(y)) for x, y in boundary]
    draw.line(boundary_pixels + [boundary_pixels[0]], fill="#182c3d", width=7, joint="curve")

    # 堂屋主入口符号，帮助辨识传统住宅的中轴与朝向。
    opening=[point(x,y) for x,y in entrance_points(building)]
    if len(opening)>=2:
        draw.line(opening,fill='#2d6f9f',width=7)
    draw.text((12, 10), "传统堂屋住宅·原始平面", fill="#1f3448", font=title_font)
    draw.text((12, 40), "浅色区域：原有房间　彩色覆盖：保留结构约束", fill="#617181", font=small_font)
    return image


def _load_plan_image(config: dict, width: int, height: int):
    image_path = str(config.get("ExistingBuilding", {}).get("plan_image", "")).strip()
    if image_path and Path(image_path).is_file():
        with Image.open(image_path) as source:
            boundary = config['ExistingBuilding']['boundary']
            scale, ox, oy = viewport(boundary, width, height)
            xs, ys = zip(*boundary)
            size = (max(1, round((max(xs)-min(xs))*scale)), max(1, round((max(ys)-min(ys))*scale)))
            image = Image.new('RGB', (width,height), '#faf9f5')
            image.paste(ImageOps.fit(source.convert('RGB'), size, method=Image.Resampling.LANCZOS),
                        (round(ox+min(xs)*scale),round(oy-max(ys)*scale)))
            return image
    return _create_tangwu_plan_image(config, width, height)


def _render_residential_inputs(config, config_id):
    original_before=deepcopy(config["ExistingBuilding"].get("original_spaces",[]))
    revision=st.session_state.get(f"{config_id}_ar_canvas_revision",0)
    st.markdown("### 传统堂屋住宅输入")
    scenario_col1, scenario_col2 = st.columns(2)
    with scenario_col1:
        config["BuildingTypology"] = st.text_input(
            "既有建筑类型", value=config.get("BuildingTypology", "traditional_tangwu"),
            help="当前研究场景固定为传统堂屋住宅。", key=f"{config_id}_ar_typology",
        )
    with scenario_col2:
        config["ConversionGoal"] = st.text_input(
            "转换目标", value=config.get("ConversionGoal", "modern_residence"),
            help="目标是现代住宅功能布局，而不是公共服务或社区中心。", key=f"{config_id}_ar_goal",
        )
    with st.expander("坐标数据表（精确编辑）", expanded=False):
        st.markdown("#### 原始空间（改造前状态）")
        original_df = st.data_editor(
            pd.DataFrame(_rect_records(config["ExistingBuilding"].get("original_spaces", []))),
            num_rows="dynamic", use_container_width=True, key=f"{config_id}_ar_originals_{REFERENCE_PLAN_UI_VERSION}_{revision}",
        )
        try:
            originals = _records_to_rects(original_df)
            for item in originals:
                x1, y1, x2, y2 = item["rect"]
                if not all(math.isfinite(v) for v in item["rect"]) or x2 <= x1 or y2 <= y1:
                    raise ValueError("原始空间需满足 x2 > x1、y2 > y1，坐标必须为有限数值。")
            config["ExistingBuilding"]["original_spaces"] = originals
        except (TypeError, ValueError, KeyError) as exc:
            st.error(f"原始空间坐标无效：{exc}")
            return False

    if original_before!=config['ExistingBuilding']['original_spaces']:
        saved=load_config(config_id)
        saved['ExistingBuilding']['original_spaces']=config['ExistingBuilding']['original_spaces']
        save_config(saved,config_id)
        st.session_state[f'{config_id}_ar_canvas_revision']=revision+1
        st.rerun()

    if st.button("保存住宅输入并应用到二维标注", type="primary", use_container_width=True):
        saved = load_config(config_id)
        for field in ("BuildingTypology", "ConversionGoal"):
            saved[field] = config[field]
        saved["ExistingBuilding"]["original_spaces"] = config["ExistingBuilding"]["original_spaces"]
        save_config(saved, config_id)
        st.session_state[f"{config_id}_ar_canvas_revision"] = st.session_state.get(f"{config_id}_ar_canvas_revision", 0) + 1
        st.success("住宅类型、转换目标、建筑边界及坐标表已保存，并应用到下方二维标注。")

    return True


def _save_scene(config, config_id, updated):
    saved=load_config(config_id)
    for field in ('ExistingBuilding','TargetSpaces','FunctionalRelations'):
        saved[field]=updated[field]
        config[field]=updated[field]
    save_config(saved,config_id)
    key=f'{config_id}_ar_canvas_revision'
    st.session_state[key]=st.session_state.get(key,0)+1


def _render_target_editor(config, config_id):
    revision=st.session_state.get(f'{config_id}_ar_canvas_revision',0)
    baseline=deepcopy([config.get('TargetSpaces'),config.get('FunctionalRelations')])
    st.markdown("### 目标功能空间智能体")
    st.caption("每一行是一个独立智能体；initial_rect 是其从既有平面出发的初始状态。")
    target_df = st.data_editor(
        pd.DataFrame(_target_records(config.get("TargetSpaces", []))),
        num_rows="dynamic", use_container_width=True, key=f"{config_id}_ar_targets_{REFERENCE_PLAN_UI_VERSION}_{revision}",
    )
    config["TargetSpaces"] = _records_to_targets(target_df,config.get('TargetSpaces',[]))
    seed_valid=True
    if config.get('SeedGrowth',{}).get('enabled',False):
        from gui.seed_page import render_seed_settings
        seed_valid=render_seed_settings(config,config_id)

    st.markdown("### 功能关系")
    relation_df = st.data_editor(
        pd.DataFrame(config.get("FunctionalRelations", []), columns=["from", "to", "type","min_shared_length","min_clear_width","min_distance"]),
        num_rows="dynamic", use_container_width=True, key=f"{config_id}_ar_relations_{REFERENCE_PLAN_UI_VERSION}",
        column_config={"type": st.column_config.SelectboxColumn("type", options=["adjacent", "separate","connected","via_circulation"]),
                       **{k:st.column_config.NumberColumn(k,min_value=.001) for k in ('min_shared_length','min_clear_width','min_distance')}},
    )
    config["FunctionalRelations"] = [
        {"from": str(row["from"]).strip(), "to": str(row["to"]).strip(), "type": str(row["type"]),
         **{k:float(row[k]) for k in ('min_shared_length','min_clear_width','min_distance') if pd.notna(row.get(k))}}
        for row in relation_df.to_dict("records")
        if str(_clean(row.get("from"))).strip() and str(_clean(row.get("to"))).strip()
    ]
    st.caption('from / to 是稳定的图节点 ID；移动种子不会更改关系。adjacent=实际共边，connected=开口连通，via_circulation=经交通空间连通，separate=分离；连通仍需门/通路证明。')

    for r in config['TargetSpaces']:
        rect=r['initial_rect']
        if not all(math.isfinite(v) for v in rect) or rect[2]<=rect[0] or rect[3]<=rect[1]:
            st.error('初始智能体区域宽高必须为正，坐标必须有效。')
            return False
    if baseline!=[config.get('TargetSpaces'),config.get('FunctionalRelations')]:
        ids=[r['id'] for r in config['TargetSpaces']]
        if len(ids)!=len(set(ids)) or any(e[k] not in ids for e in config['FunctionalRelations'] for k in ('from','to')):
            st.error('智能体 ID 必须唯一；删除节点前请更新其图关系。')
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


def _render_plan_constraint_editor(config: dict, config_id: str) -> None:
    building = config["ExistingBuilding"]
    environment = config["AdaptiveReuseEnvironment"]
    boundary = building["boundary"]
    st.markdown("### 二维原始平面与前期约束标注")
    st.caption('参数表与画布共用一份构件数据。有效修改自动保存到 YAML 并双向同步；新增表格行请先补全必填数据。')

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
            if st.button("恢复系统生成的堂屋原始平面", use_container_width=True):
                building.pop("plan_image", None)
                save_config(config, config_id)
                st.session_state[f"{config_id}_ar_canvas_revision"] = st.session_state.get(f"{config_id}_ar_canvas_revision", 0) + 1
                st.rerun()
        else:
            st.caption("当前底图：系统根据原堂屋及各原有房间自动生成的二维平面")
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
        layers={'全部对象':'all','建筑边界':'boundary','原有房间':'original','智能体初始区域':'agent','智能体种子':'seed','柱':'column','墙线与左右厚度':'wall','其他矩形构件':'fixed_rect','其他多边形构件':'fixed','门位置':'door'}
        layer=layers[tools_row[2].selectbox('编辑图层（控制可选点）',list(layers),key=f'{config_id}_environment_layer')]
        entries=environment_controls(config)
        ids=sorted({k[1] for k in entries if layer=='all' or k[0]==layer})
        object_id=tools_row[3].selectbox('编辑对象 ID（重叠时用于选取）',['全部']+ids,key=f'{config_id}_environment_object')
        object_id=None if object_id=='全部' else object_id
        wall_left, wall_right = .12, .12
        if selected_type in ('shear_wall', 'load_bearing_wall'):
            wall_left = st.number_input('新墙左侧厚度（m）', min_value=0.0, value=.12, step=.01, key=f'{config_id}_new_wall_left')
            wall_right = st.number_input('新墙右侧厚度（m）', min_value=0.0, value=.12, step=.01, key=f'{config_id}_new_wall_right')
            st.caption('拖出墙线的起点与终点；左右按拖动方向定义。调整模式可移动、旋转、沿长度或厚度缩放。')
    canvas_column, parameters_column = st.columns([1.05, 1], gap="medium")
    with parameters_column, st.container(height=620, border=False):
        panel_tabs=st.tabs(['显示','原房间','边界与结构','智能体与关系','控制点'])
        with panel_tabs[0]:
            with st.expander('图层显示与透明度', expanded=True):
                st.caption('勾选显示；透明度 0 为原始着色，100 为全透明。隐藏不会删除数据。')
                rows=[{'类别':label,'显示':True,'透明度':0} for key,label in DISPLAY_LAYERS.items()]
                rows += [{'类别':'对象名称','显示':True,'透明度':0},{'类别':'上传的平面底图','显示':True,'透明度':0}]
                view=st.data_editor(pd.DataFrame(rows),hide_index=True,disabled=['类别'],
                    column_config={'显示':st.column_config.CheckboxColumn('显示'),'透明度':st.column_config.NumberColumn('透明度 %',min_value=0,max_value=100,step=5)},
                    key=f'{config_id}_environment_display',use_container_width=True)
                display={key:(bool(row['显示']),float(row['透明度'] or 0)) for key,row in zip(list(DISPLAY_LAYERS)+['labels','background'],view.to_dict('records'))}
                st.caption('上传图片中的内容属于底图像素，可整体隐藏；各类环境对象可独立隐藏。')
        with panel_tabs[1]:
            inputs_valid = _render_residential_inputs(config, config_id)
        with panel_tabs[2]:
            editor_valid = render_structure_editor(config, config_id) and inputs_valid
        try:
            with panel_tabs[3]:
                editor_valid = _render_target_editor(config, config_id) and editor_valid
            with panel_tabs[4]:
                editor_valid = _render_all_points(config,config_id) and editor_valid
        except (ValueError,TypeError,KeyError) as exc:
            st.error(f"智能体数据未保存：{exc}")
            editor_valid=False

    with canvas_column:
        if not editor_valid:
            st.info('请先补全右侧参数行，画布编辑将在数据有效后恢复。')
            return False
        if st_canvas is None:
            st.warning("当前环境未安装 streamlit-drawable-canvas，暂时只能使用下方坐标表格。")
            return

        xs = [float(point[0]) for point in boundary]
        ys = [float(point[1]) for point in boundary]
        aspect = max((max(xs) - min(xs)) / max(max(ys) - min(ys), 1e-9), 0.35)
        canvas_width = 680
        canvas_height = max(360, min(680, int(canvas_width / aspect)))
        plan_objects = _plan_objects(config, canvas_width, canvas_height, display)
        plan_signature = hashlib.sha256(json.dumps([building,config.get("TargetSpaces"),display], sort_keys=True).encode()).hexdigest()[:12]
        _, fill_color, stroke_color = CONSTRAINT_STYLES[selected_type]
        revision = st.session_state.get(f"{config_id}_ar_canvas_revision", 0)
        canvas_key = f"{config_id}_ar_constraint_canvas_{REFERENCE_PLAN_UI_VERSION}_{plan_signature}_{selected_type}_{operation_mode}_{layer}_{object_id}_{revision}"
        editing_boundary = operation_mode == "调整建筑边界"
        display_layer='boundary' if editing_boundary else layer
        display_objects=scene(config,canvas_width,canvas_height,CONSTRAINT_STYLES,display_layer,object_id,display)
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
        st.caption('统一点编辑：橙色为边界，棕色为原房间，蓝色半透明为初始智能体，紫色为种子。角点调尺寸、中心点平移；墙线端点与左右厚度点独立可拖。重叠时选择图层和对象 ID。原房间/初始区域保持矩形；拖动不改变图节点关系。')
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
            st.caption(f"已同步 {len(building.get('fixed_objects', []))} 个构件（含自动核心筒）")
        with clear_col:
            if st.button("清空全部约束并同步参数表", use_container_width=True, disabled=not editor_valid):
                save_structures(config, config_id, [])
                st.rerun()


def render_adaptive_reuse_config_page(config: dict, config_id: str) -> None:
    st.subheader("传统堂屋住宅适应性更新环境")
    st.caption("依据论文第 3.1—3.6 节：从传统堂屋住宅原始状态出发，保留结构参与每一步决策，现代住宅功能空间作为多个智能体协同调整。")

    with st.expander("原矩形模式：论文方法与环境的对应关系", expanded=False):
        st.markdown(
            "- **状态**：功能空间位置、尺度、目标面积差、形态、与固定构件/其他智能体的距离、功能关系、原空间复用率与整体改造率。\n"
            "- **动作**：上/下/左/右移动、保持，以及四条边的扩张与收缩，共 13 类离散网格动作。\n"
            "- **硬约束**：建筑边界、柱、承重墙、核心筒、保留交通空间和智能体重叠；非法动作被屏蔽并回退。\n"
            "- **奖励**：面积、形态、邻近/分离、结构网格顺应、原空间利用、改造干预、非法动作与团队协调。\n"
            "- **终止**：连续满足面积/形态/无冲突/最大改造率，或达到单回合最大步数。"
        )

    training = config.setdefault("Training", {})
    environment = config.setdefault("AdaptiveReuseEnvironment", {})
    seed_settings=config.setdefault('SeedGrowth',{})
    seed_mode=st.checkbox('实验模式：图种子训练与约束生长',value=bool(seed_settings.get('enabled',False)),key=f'{config_id}_seed_enabled')
    seed_settings['enabled']=seed_mode
    if seed_mode:
        st.info('训练只移动种子并快速估算；每 250 轮、结束或停止时精确成图。邻接收敛仍在验证，目标图边不代表已经共边。')
        st.caption('右侧尺度变化、改造率及随机扰动设置仅用于原矩形模式；种子网格和轮廓规则在下方单独设置。')
    left, right = st.columns(2)
    with left:
        st.markdown("### 训练设置")
        training["agent_name"] = st.selectbox("算法模型", ["mappo"], key=f"{config_id}_ar_agent")
        training["episodes"] = int(st.number_input("训练轮数", 1, value=int(training.get("episodes", 1000)), key=f"{config_id}_ar_episodes"))
        training["max_steps"] = int(st.number_input("每轮最大决策步数", 1, value=int(training.get("max_steps", 240)), key=f"{config_id}_ar_steps"))
        training["batch_size"] = int(st.number_input("批量大小", 2 if seed_mode else 1, value=max(2 if seed_mode else 1,int(training.get("batch_size", 64))), key=f"{config_id}_ar_batch"))
        training["lr"] = float(st.number_input("学习率", min_value=0.000001, value=float(training.get("lr", 0.0003)), format="%.6f", key=f"{config_id}_ar_lr"))
        training["seed"] = int(st.number_input("随机种子", value=int(training.get("seed", 42)), key=f"{config_id}_ar_seed"))
    with right:
        st.markdown("### 网格、动作与终止")
        environment["grid_size"] = float(st.number_input("规则网格尺寸（m）", 0.1, value=float(environment.get("grid_size", 1.0)), step=0.1, key=f"{config_id}_ar_grid"))
        environment["move_step"] = float(st.number_input("单次移动距离（m）", 0.1, value=float(environment.get("move_step", 1.0)), step=0.1, key=f"{config_id}_ar_move"))
        environment["resize_step"] = float(st.number_input("单次尺度变化（m）", 0.1, value=float(environment.get("resize_step", 1.0)), step=0.1, key=f"{config_id}_ar_resize"))
        environment["area_tolerance"] = float(st.slider("目标面积容差", 0.01, 0.50, float(environment.get("area_tolerance", 0.12)), 0.01, key=f"{config_id}_ar_area_tol"))
        environment["max_intervention_ratio"] = float(st.slider("终止允许的最大改造率", 0.0, 1.0, float(environment.get("max_intervention_ratio", 0.45)), 0.01, key=f"{config_id}_ar_intervention"))
        environment["success_patience"] = int(st.number_input("连续满足步数", 1, value=int(environment.get("success_patience", 8)), key=f"{config_id}_ar_patience"))
        environment["randomize_initial"] = st.checkbox("训练时随机扰动初始位置", value=bool(environment.get("randomize_initial", True)), key=f"{config_id}_ar_random")

    seed_valid = _render_plan_constraint_editor(config, config_id) is not False

    st.markdown("### 奖励权重")
    st.caption("这里只保留论文方法对应的基础奖励项，不设“额外奖励”分组。")
    if seed_mode: st.caption('下列权重用于原矩形模式。种子模式当前使用已验证接口的固定实验奖励，暂不开放调权。')
    weights = config.setdefault("RewardWeights", {})
    reward_columns = st.columns(2)
    for index, (key, label) in enumerate(REWARD_LABELS.items()):
        with reward_columns[index % 2]:
            weights[key] = float(st.slider(label, 0.0, 10.0, float(weights.get(key, 1.0)), 0.1, key=f"{config_id}_ar_reward_{key}",disabled=seed_mode))

    save_col, preview_col = st.columns(2)
    with save_col:
        if st.button("保存既有建筑环境", type="primary", use_container_width=True,disabled=not seed_valid):
            try:
                if seed_mode:
                    from core.seed_growth.environment import SeedLayoutEnv
                    SeedLayoutEnv(config)
                else:
                    if any(e['type'] not in ('adjacent','separate') for e in config['FunctionalRelations']):
                        raise ValueError('connected/via_circulation 请使用实验种子模式；原模式只支持 adjacent/separate')
                    AdaptiveReuseEnv(config).reset(seed=int(training.get("seed", 42)))
                saved_path = save_config(config, config_id)
                st.success(f"环境校验通过并已保存：{saved_path}")
            except Exception as exc:
                st.error(f"环境未保存：{exc}")
    with preview_col:
        preview_requested = st.button("刷新环境预览", use_container_width=True)

    st.markdown("### 环境预览")
    if not seed_mode:
        with st.expander('查看已有种子生长成图（无需开启实验训练）'):
            from gui.seed_page import render_seed_results
            render_seed_results(config_id)
    if seed_mode:
        from gui.seed_page import render_seed_preview,render_seed_results
        if seed_valid:
            render_seed_preview(config)
        render_seed_results(config_id)
        return
    try:
        preview_config = yaml.safe_load(yaml.safe_dump(config, allow_unicode=True))
        preview_config["AdaptiveReuseEnvironment"]["randomize_initial"] = False
        env = AdaptiveReuseEnv(preview_config)
        _, info = env.reset(seed=int(training.get("seed", 42)))
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
