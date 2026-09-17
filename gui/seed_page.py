"""Rectangular retrofit acceleration controls and read-only saved precise output."""
from common.plan_styles import DOOR_COLOR
from copy import deepcopy
import json
from pathlib import Path
import pandas as pd
import streamlit as st
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch
from shapely.geometry import shape
from shapely.geometry.polygon import orient

from core.seed_growth.contracts import build_problem
from core.seed_growth.environment import SeedLayoutEnv
from core.seed_growth.shape_rules import polygon_parts
from common.project_paths import RESULTS_DIR


def seed_rows(rooms):
    rows=[]
    for room in rooms:
        if room.get('role') == 'residual': continue
        rect=room.get('initial_rect',[0,0,1,1])
        x,y=room.get('seed',[(rect[0]+rect[2])/2,(rect[1]+rect[3])/2])
        rows.append(dict(node_id=room['id'],x=x,y=y,shape_policy=room.get('shape_policy','regular'),min_width=room.get('min_width')))
    return rows


def apply_seed_rows(rooms,rows,limits):
    result=deepcopy(rooms); identifiers={r['id'] for r in result if r.get('role') != 'residual'}
    if not isinstance(limits,dict) or set(limits)-identifiers:
        raise ValueError('轮廓参数须按已有节点 ID 填写')
    values={r['node_id']:r for r in rows}
    if set(values)!=identifiers or len(rows)!=len(identifiers): raise ValueError('种子行必须与独立房间节点一一对应')
    for room in result:
        if room.get('role') == 'residual': continue
        raw=values[room['id']]
        room.update(seed=[float(raw['x']),float(raw['y'])],shape_policy=raw['shape_policy'])
        if pd.notna(raw.get('min_width')): room['min_width']=float(raw['min_width'])
        else: room.pop('min_width',None)
        if room['id'] in limits: room['shape_limits']=limits[room['id']]
        else: room.pop('shape_limits',None)
    return result


def render_seed_settings(config,config_id):
    with st.expander('图节点种子与轮廓规则',expanded=True):
        st.caption('节点 ID 与上方房间 ID 相同；坐标可以移动，目标关系不会重新配对。留空最小净宽时，非矩形使用相对净宽过滤。')
        frame=st.data_editor(pd.DataFrame(seed_rows(config['TargetSpaces'])),disabled=['node_id'],
            hide_index=True,key=f"{config_id}_seeds_v2_{st.session_state.get(f'{config_id}_ar_canvas_revision',0)}",
            column_config={'shape_policy':st.column_config.SelectboxColumn('轮廓规则',options=['regular','limited_recess']),
                           'min_width':st.column_config.NumberColumn('最小净宽（m）',min_value=.01)})
        limits=st.text_area('轮廓约束参数（YAML，按节点 ID）',value=yaml.safe_dump(
            {r['id']:r['shape_limits'] for r in config['TargetSpaces'] if 'shape_limits' in r},allow_unicode=True),key=f'{config_id}_shape_limits_v2')
        settings=config['SeedGrowth']
        settings['fitting_area_tolerance']=float(st.slider('共边调整允许的目标面积偏差',0.,.5,float(settings.get('fitting_area_tolerance',.12)),.01,key=f'{config_id}_fitting_area_tol'))
        if any(r.get('role')=='residual' for r in config['TargetSpaces']):
            st.caption('客厅保留图节点，不设置种子、不创建独立策略；面积和轮廓由其他房间的剩余空间确定。')
            settings['entrance_depth']=float(st.number_input('户门内侧客厅预留深度（m）',min_value=.1,value=float(settings.get('entrance_depth',.9)),step=.1,key=f'{config_id}_entrance_depth'))
        settings['grid_size']=float(st.number_input('种子代理网格（m）',min_value=.1,
            value=float(settings.get('grid_size',config['AdaptiveReuseEnvironment'].get('grid_size',.5))),step=.1,key=f'{config_id}_seed_grid'))
        settings['resume_run']=st.text_input('续训运行目录（留空则开始新训练）',value=settings.get('resume_run',''),key=f'{config_id}_seed_resume')
        st.caption('续训沿用该目录保存的配置和节点 ID；训练轮数表示目标累计轮数。本页其他改动用于新训练。加速策略动作维度不同，矩形策略权重不能直接续训；原空间与初始矩形仍沿用同一份配置。')
        try:
            rooms=apply_seed_rows(config['TargetSpaces'],frame.to_dict('records'),yaml.safe_load(limits) or {})
        except (ValueError,TypeError,KeyError,yaml.YAMLError) as exc:
            st.error(f'种子参数未应用：{exc}')
            return False
        trial=dict(config,TargetSpaces=rooms)
        try:
            # Relations are validated by the final save, after their editor.
            build_problem(dict(trial,FunctionalRelations=[]))
        except (ValueError,KeyError,TypeError) as exc:
            # 与固定结构冲突等几何问题交给“自动修复初始布局并保存”处理，不阻塞画布交互。
            repairable=('种子位于边界外','种子占用客厅户门内侧预留区域','户门内侧预留区域被固定结构阻挡','房间最小面积总和超过扣除固定结构后的自由面积')
            if not any(key in str(exc) for key in repairable):
                st.error(f'种子参数未应用：{exc}')
                return False
            st.warning(f'{exc}。画布保持可编辑；请点击“自动修复初始布局并保存”，系统会把冲突房间挪到不冲突的位置后再保存。')
        config['TargetSpaces']=rooms
        return True


def _draw(ax,geometry,color):
    for p in polygon_parts(geometry):
        p=orient(p,sign=1); vertices=[]; codes=[]
        for ring in [p.exterior,*p.interiors]:
            coords=list(ring.coords)
            vertices.extend(coords); codes.extend([PlotPath.MOVETO]+[PlotPath.LINETO]*(len(coords)-2)+[PlotPath.CLOSEPOLY])
        ax.add_patch(PathPatch(PlotPath(vertices,codes),facecolor=color,edgecolor='#263238',lw=1))


@st.cache_data(max_entries=4,show_spinner=False)
def _preview(config):
    env=SeedLayoutEnv(config)
    return env.estimate,env.node_positions(),env.problem.boundary.bounds


def render_seed_preview(config):
    st.markdown('#### 代理预览（估计，不是精确成图）')
    try:
        estimate,seeds,bounds=_preview(config)
        problem=build_problem(config)
        fig,ax=plt.subplots(figsize=(9,5))
        h=problem.grid_size; owners=estimate['owners']; x0,y0,_,_=bounds
        cmap=ListedColormap(['#f5f5f5','#90caf9','#a5d6a7','#ffcc80','#ce93d8','#80cbc4','#ef9a9a'])
        ax.imshow((owners+1)%7,origin='lower',extent=[x0,x0+owners.shape[1]*h,y0,y0+owners.shape[0]*h],cmap=cmap,vmin=0,vmax=6,alpha=.65)
        for edge in problem.relations:
            a,b=seeds[edge.source],seeds[edge.target]
            ax.plot([a[0],b[0]],[a[1],b[1]],'--',color='#d84315' if edge.kind=='separate' else '#455a64',lw=1)
        for key,(x,y) in seeds.items():
            public=problem.residual_room and key==problem.residual_room.id
            ax.scatter(x,y,c='#00897b' if public else 'black',marker='D' if public else 'o',s=24 if public else 14)
            ax.annotate(key+' (region)' if public else key,(x,y),fontsize=8)
        if problem.entrance is not None:
            dx,dy=problem.entrance.xy; ax.plot(dx,dy,color=DOOR_COLOR,lw=5,label='入口→客厅')
        _draw(ax,problem.fixed,'#455a64')
        x,y=problem.boundary.exterior.xy; ax.plot(x,y,color='black'); ax.set_aspect('equal')
        _, preview_column, _ = st.columns([1, 2, 1])
        with preview_column:
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption('虚线是目标图关系，不表示已经共边；颜色区域只代表快速估算。绿色菱形是公共区域节点，黑点才是可移动种子。')
        st.dataframe(pd.DataFrame(estimate['relations']),hide_index=True)
    except Exception as exc:
        st.error(f'代理预览不可用：{exc}')


def read_precise(run):
    run=Path(run).resolve()
    pointer=json.loads((run/'latest_precise.json').read_text(encoding='utf-8'))
    path=(run/pointer['path']).resolve()
    if run not in path.parents: raise ValueError('成图路径不在运行目录内')
    result=json.loads(path.read_text(encoding='utf-8'))
    config=yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8'))
    return config,result


def render_seed_results(config_id, config=None):
    st.markdown('#### 已保存的精确成图')
    root=Path(RESULTS_DIR)/'seed_experimental'
    runs=sorted((p for p in root.glob('*') if p.is_dir() and (p/'latest_precise.json').exists()),reverse=True)
    if config and config.get('FloorWorkflow'):
        from core.floor_partition.workflow import matching_runs
        ref=config['FloorWorkflow']
        allowed=set(matching_runs(RESULTS_DIR,ref['plan_id'],ref['unit_id']))
        runs=[p for p in runs if p in allowed]
    if not runs:
        st.caption('尚无精确产物；在第 250 轮、训练结束或停止后生成。')
        return
    selected=st.selectbox('查看实验运行',runs,format_func=lambda p:p.name,key=f'{config_id}_precise_run')
    st.button('刷新已保存成图',key=f'{config_id}_refresh_precise')
    try:
        config,result=read_precise(selected); problem=build_problem(config)
        st.caption(f'训练结果目录：{selected.resolve()}')
        export=selected.resolve()
        if st.button('导出 PNG / SVG 到当前训练运行目录',key=f'{config_id}_export_precise'):
            from core.seed_growth.rendering import render_result
            render_result(config,result,export)
        if (export/'layout.png').exists():
            st.caption(f'成图文件：{export / "layout.png"}')
            st.download_button('下载成图 PNG',(export/'layout.png').read_bytes(),file_name='seed_growth.png',mime='image/png')
        if result['status']=='valid': st.success('几何与已声明关系通过当前检查')
        else: st.warning('方案未通过完整验收；请查看未满足或未知的关系。')
        snap=result['snapshot']
        st.caption(f"累计完成 {snap['completed_episodes']} 轮，回合内 {snap['step']} 步；使用该运行保存的配置。")
        fig,ax=plt.subplots(figsize=(9,5))
        for i,(key,geometry) in enumerate(result['polygons'].items()):
            _draw(ax,shape(geometry),plt.get_cmap('Pastel1')(i%9))
            point=shape(geometry).representative_point()
            x,y=snap['seeds'].get(key,[point.x,point.y]); ax.annotate(key,(x,y),fontsize=8)
            if key in snap['seeds']: ax.scatter(x,y,c='black',s=14)
        if problem.entrance is not None:
            dx,dy=problem.entrance.xy; ax.plot(dx,dy,color=DOOR_COLOR,lw=5,label='入口→客厅')
        _draw(ax,problem.fixed,'#455a64')
        x,y=problem.boundary.exterior.xy; ax.plot(x,y,c='black'); ax.autoscale_view(); ax.set_aspect('equal')
        st.pyplot(fig); plt.close(fig)
        for error in result['validation']['errors']: st.error(error)
        retrofit=result['validation'].get('retrofit')
        if retrofit and retrofit['rooms']:
            st.caption('精确成图相对原空间矩形的复用率与改造率（沿用基础方法口径）')
            st.dataframe(pd.DataFrame(retrofit['rooms']),hide_index=True)
        st.dataframe(pd.DataFrame(result['validation']['relations']),hide_index=True)
        st.caption('satisfied=true：已验证；false：未满足；空值：尚无门或通路证据。')
        st.dataframe(pd.DataFrame(result['comparison']['relations']),hide_index=True)
        st.download_button('下载精确多边形、种子和验收报告',json.dumps(result,ensure_ascii=False,indent=2),file_name='precise_layout.json',mime='application/json')
    except Exception as exc:
        st.error(f'读取精确产物失败：{exc}')
