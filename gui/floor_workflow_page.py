"""Explicit preview, per-unit handoff and whole-floor assembly pages."""
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import streamlit as st
import yaml
from common.config_manager import CONFIG_ROOT
from common.project_paths import RESULTS_DIR
from core.floor_partition.workflow import (read_floor_snapshot,prepare_units,matching_runs,
                                           read_unit_result,composition,composition_figure)

WORKFLOW_ROOT=Path(RESULTS_DIR)/'floor_workflows'


def render_floor_results(config,config_id):
    st.subheader('第一阶段：分户方案与交接')
    st.caption('输入画布的粉色覆盖仅为推演。这里分别查看未训练预览和第一阶段保存的结果；选择一个方案后再进入逐户训练。')
    if st.button('生成当前参数推演（非训练结果）',key=f'{config_id}_generate_floor_preview'):
        try:
            from core.floor_partition import run_residential_floor_partition,export_unit_configs
            from core.floor_partition.rendering import render_partition
            from core.floor_partition.quality import validate_partition
            with st.spinner('正在生成并验收分户推演…'):
                problem,result=run_residential_floor_partition(config)
                out=Path(RESULTS_DIR)/'floor_partition'/('preview_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                out.mkdir(parents=True)
                (out/'config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf8')
                export_unit_configs(config,result,out/'exports')
                render_partition(config,result,out)
                (out/'summary.json').write_text(json.dumps(dict(source_kind='preview',quality=validate_partition(problem,result))),encoding='utf8')
                st.success('推演已保存，可在下方查看；尚未训练。')
        except (ValueError,OSError) as exc:
            st.error(str(exc))
    runs=[]
    for p in (Path(RESULTS_DIR)/'floor_partition').glob('*/exports/floor_plan.json'):
        try:
            source=yaml.safe_load((p.parents[1]/'config.yaml').read_text(encoding='utf8'))
            if source.get('ConfigID','').lower()==str(config.get('ConfigID',config_id)).lower(): runs.append(p.parents[1])
        except (OSError,ValueError,yaml.YAMLError): continue
    runs.sort(key=lambda p:p.stat().st_mtime,reverse=True)
    if not runs:
        st.info('还没有新版交接文件。先运行第一阶段，或生成一个明确标注为“未训练”的推演。旧PNG不能直接用于第二阶段。')
        return
    selected=st.selectbox('选择第一阶段方案',runs,format_func=lambda p:p.name,key=f'{config_id}_handoff_run')
    try:
        summary=json.loads((selected/'summary.json').read_text(encoding='utf8'))
        kind=summary.get('source_kind','unknown')
        labels=dict(preview='未训练推演',search='普通搜索结果',rl_training='RL训练结果',policy_inference='模型推理结果')
        st.info('方案来源：'+labels.get(kind,'历史结果（来源未标注）'))
        floor=read_floor_snapshot(selected/'exports/floor_plan.json')
        if (selected/'partition_preview.png').exists(): st.image(str(selected/'partition_preview.png'))
        st.caption(f"{len(floor['units'])} 户；保留整层轮廓、各户净边界、隔墙、走道和原楼层坐标。采用后每户独立配置房间，第一阶段方案保持不变。")
        if st.button('采用此方案，创建逐户训练清单',disabled=st.session_state.get('training_status') in ('Running','Stopping')):
            path=prepare_units(selected/'exports',CONFIG_ROOT,WORKFLOW_ROOT)
            st.session_state['floor_workflow_selected']=str(path)
            st.session_state['workflow_route']=dict(view='整体合成')
            st.rerun()
    except (OSError,ValueError,KeyError) as exc:
        st.error(f'交接失败：{exc}')


def render_floor_workflow():
    st.subheader('整体合成 · 逐户训练清单')
    st.caption('采用第一阶段方案 → 配置当前户房间 → 训练并验收 → 下一户 → 按原坐标合成整层。当前工作台一次运行一个训练任务；不是算法只能串行。')
    manifests=sorted(WORKFLOW_ROOT.glob('*/workflow.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not manifests:
        st.info('请先在第一阶段「布局预览」中采用一个分户方案，创建逐户清单。')
        return
    selected=st.session_state.get('floor_workflow_selected')
    paths=[str(p) for p in manifests]
    chosen=st.selectbox('楼层方案清单',paths,index=paths.index(selected) if selected in paths else 0,
                        format_func=lambda s:Path(s).parent.name[:12],key='floor_workflow_picker')
    st.session_state['floor_workflow_selected']=chosen
    manifest_path=Path(chosen)
    try:
        manifest=json.loads(manifest_path.read_text(encoding='utf8'))
        floor=read_floor_snapshot(manifest_path.parent/'floor_plan.json')
        rooms={}; provenance={}; rows=[]; configs={}
        for uid,item in manifest['units'].items():
            c=yaml.safe_load(Path(item['config_path']).read_text(encoding='utf8'))
            configs[uid]=c
            status='待配置房间' if not c.get('TargetSpaces') else '待训练 / 待验收'
            candidates=[p for p in matching_runs(RESULTS_DIR,floor['plan_id'],uid) if (p/'last_valid.json').exists()]
            if candidates:
                run=st.selectbox(f'{uid}：选择合格记录',candidates,format_func=lambda p:p.name,
                                 key=f"{floor['plan_id']}_{uid}_room_run")
                try:
                    rooms[uid],provenance[uid]=read_unit_result(floor,uid,run,c)
                    status='已验收，可合成'
                except (OSError,ValueError,KeyError) as exc:
                    status='需处理：'+str(exc)
            rows.append(dict(户型=uid,配置编号=item['config_id'],状态=status))
        st.dataframe(rows,hide_index=True,width='stretch')
        st.progress(len(rooms)/len(floor['units']),text=f"可合成 {len(rooms)}/{len(floor['units'])} 户")
        pending=[uid for uid in floor['units'] if uid not in rooms]
        target=st.selectbox('准备处理的户型',list(floor['units']),index=list(floor['units']).index(pending[0]) if pending else 0)
        if st.button('进入该户：配置房间与训练',disabled=st.session_state.get('training_status') in ('Running','Stopping')):
            st.session_state['workflow_route']=dict(config_id=manifest['units'][target]['config_id'],view='环境搭建')
            st.rerun()
        st.caption('进入户型后在「智能体与关系」中添加房间，保存并训练；完成后回到「整体合成」处理下一户。尚未通过验收的房间结果不会自动成为完成项。')
        geo=composition(floor,rooms); geo['sources']=provenance
        fig=composition_figure(floor,rooms)
        st.pyplot(fig)
        if not geo['complete']:
            st.warning('当前是过程草图：灰色户型尚未完成，不可导出为完整楼层方案。')
        else:
            output=BytesIO()
            with ZipFile(output,'w',ZIP_DEFLATED) as archive:
                for ext in ('png','svg'):
                    image=BytesIO();fig.savefig(image,format=ext,dpi=180)
                    archive.writestr('whole_floor.'+ext,image.getvalue())
                archive.writestr('rooms.geojson',json.dumps(geo,ensure_ascii=False,indent=2))
                archive.writestr('floor_plan.json',json.dumps(floor,ensure_ascii=False,indent=2))
            st.success('所有户型均通过当前几何与已声明关系检查，可导出整层结果。')
            st.download_button('下载完整建筑平面（PNG / SVG / 坐标）',output.getvalue(),file_name='whole_floor.zip',mime='application/zip')
        st.caption('合成的是带房间功能分区的建筑平面；室内门窗及施工详图不在当前输出范围。')
    except (OSError,ValueError,KeyError) as exc:
        st.error(f'清单或合成结果不可用：{exc}')
