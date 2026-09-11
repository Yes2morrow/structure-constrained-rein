"""Building boundary parameters and draggable Fabric vertex handles."""
import math
import yaml
import streamlit as st
from shapely.geometry import Polygon
from gui.structure_canvas import viewport
from gui.config_store import load_config, save_config


def validate_boundary(points):
    if not isinstance(points,list) or len(points)<3:
        raise ValueError('建筑边界至少需要三个顶点')
    if any(not isinstance(p,(list,tuple)) or len(p)!=2 for p in points):
        raise ValueError('每个顶点必须是 [x, y]')
    points=[[float(x),float(y)] for x,y in points]
    if not all(math.isfinite(v) for p in points for v in p):
        raise ValueError('边界坐标必须为有限数值')
    if len(set(map(tuple,points)))!=len(points):
        raise ValueError('顶点不能重复，末尾无需重复起点')
    poly=Polygon(points)
    if not poly.is_valid or poly.area<=1e-8:
        raise ValueError('边界不能自交或退化为直线')
    return points


def save_boundary(config,config_id,points):
    points=validate_boundary(points)
    saved=load_config(config_id)
    saved['ExistingBuilding']['boundary']=points
    save_config(saved,config_id)
    config['ExistingBuilding']['boundary']=points
    key=f'{config_id}_ar_canvas_revision'
    st.session_state[key]=st.session_state.get(key,0)+1


def render_boundary_editor(config,config_id):
    revision=st.session_state.get(f'{config_id}_ar_canvas_revision',0)
    st.caption('建筑边界：按顺序输入 [x, y] 顶点，单位米；可增删顶点。也可在下方“调整建筑边界”模式拖动橙色顶点。修改只改变边界，不移动房间或墙柱。')
    text=st.text_area('建筑边界坐标',value=yaml.safe_dump(config['ExistingBuilding']['boundary'],default_flow_style=True).strip(),
                      key=f'{config_id}_boundary_vertices_{revision}')
    try:
        points=validate_boundary(yaml.safe_load(text))
        if points!=config['ExistingBuilding']['boundary']:
            save_boundary(config,config_id,points)
            st.rerun()
    except (ValueError,TypeError,yaml.YAMLError) as exc:
        st.error(f'建筑边界未保存：{exc}')
        return False
    return True


def boundary_handles(boundary,width,height):
    scale,ox,oy=viewport(boundary,width,height)
    return [dict(type='circle',left=ox+x*scale,top=oy-y*scale,radius=6,
                 originX='center',originY='center',fill='#ff9800',stroke=f'#{0xF00000+i:06x}',strokeWidth=1,
                 hasControls=False,hasBorders=False,lockScalingX=True,lockScalingY=True,lockRotation=True)
            for i,(x,y) in enumerate(boundary)]


def parse_boundary_handles(objects,boundary,width,height):
    handles=boundary_handles(boundary,width,height)
    by_color={o.get('stroke'):o for o in objects if o.get('type')=='circle'}
    if any(h['stroke'] not in by_color for h in handles):
        raise ValueError('边界顶点未加载完整或被删除；增删顶点请使用坐标输入')
    scale,ox,oy=viewport(boundary,width,height)
    result=[]
    for point,h in zip(boundary,handles):
        o=by_color[h['stroke']]
        # Preserve precise coordinates during Fabric two-decimal hydration.
        delta=[]
        for key in ('left','top'):
            value=float(o.get(key,h[key]))
            if min(abs(value-h[key]),abs(value-round(h[key],2)))<1e-7: delta.append(0.)
            else: delta.append(value-h[key])
        result.append([point[0]+delta[0]/scale,point[1]-delta[1]/scale])
    return validate_boundary(result)
