"""Floor-partition controls and net-boundary downloads used by the Web UI."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import streamlit as st

from core.floor_partition.export import export_unit_configs


def render_partition_constraints(floor, config_id):
    quality = floor.setdefault('quality', {})
    joint = floor.setdefault('joint_search', {})

    def number(field, label, default, minimum, step, maximum=None, help=None):
        value_type = int if isinstance(default, int) else float
        kwargs = dict(min_value=minimum, value=value_type(quality.get(field, default)), step=step,
                      key=f'{config_id}_partition_quality_{field}', help=help)
        if maximum is not None:
            kwargs['max_value'] = maximum
        quality[field] = st.number_input(label, **kwargs)

    with st.expander('门前通行与净空', expanded=False):
        number('door_clearance_width', '门前净空宽度（m）', 1.2, .1, .1)
        number('door_clearance_depth', '门前净空深度（m）', 1.2, .1, .1)
        number('entrance_depth', '门内净空深度（m）', .9, .1, .1)
        st.caption('净空从完成墙面起算；门前宽度至少取门宽。缩短走道仍须满足这些净空及走道净宽。')

    with st.expander('分户质量约束（结构对齐、面积与采光）', expanded=False):
        st.caption('对齐方式由搜索和训练比较选择：墙中线对柱轴，或隔墙完成面对齐结构侧面。'
                   '完成面贴边时，中线自动偏移半个墙厚；评分鼓励统一基准。隔墙不加入固定结构。')
        number('min_structure_alignment', '分户共边贴合结构参考线的最低比例', 1., .01, .05, 1.,
               '参考线包含柱轴、完成面贴边所需的中线位置及柱跨中线；这不是强制所有墙对柱轴。')
        number('area_tolerance', '目标面积相对误差上限', .30, .01, .01, .99)
        number('area_balance_deadband', '面积均衡免罚范围', .01, 0., .005, .99,
               '0.01表示1%。此范围内不再为面积均衡加分，给缩小走道留下空间；不能超过面积误差上限。')
        number('min_unit_width', '户型最小有效宽度（m）', 1.5, .1, .1)
        number('max_corners', '每户最多转角数', 24, 4, 1,
               help='原有结构形成的必要凹口不计为新增隔墙转角。')
        number('max_unusable_ratio', '窄小附属区域面积比例上限', .20, .01, .01, 1.)
        number('facade_per_area', '每平方米净面积所需有效外墙长度（m/㎡）', .12, .01, .01)
        number('min_facade_length', '每户最小有效外墙长度（m）', 3., .1, .5)
        number('daylight_depth', '采光机会检查深度（m）', 6., .1, .5)
        number('min_daylight_coverage', '每户采光机会覆盖比例下限', .25, .01, .05, 1.)
        st.caption('外墙长度和进深覆盖用于估计采光机会；尚未计算窗洞、朝向和室外遮挡。')

    with st.expander('分户预览搜索预算', expanded=False):
        joint['compact_lobby'] = st.checkbox('尝试集中紧凑门厅', value=bool(joint.get('compact_lobby', True)),
            key=f'{config_id}_partition_compact_lobby',
            help='增加户门围绕交通核前门厅的候选，不强制采用固定走道形状。')
        joint['preview_steps'] = st.number_input('预览局部修改次数', min_value=0,
            value=int(joint.get('preview_steps', 4)), step=1, key=f'{config_id}_partition_preview_steps',
            help='仅控制网页预览；训练的每轮修改次数在训练参数页设置。0表示只生成初始方案。')
        number('beam_width', '初始划分搜索宽度', 24, 1, 1,
               help='每轮保留的划分候选数。越大通常越慢，未找到候选不代表无解。')
        number('max_candidates', '最多保留初始方案数', 24, 1, 1)

    if quality['area_balance_deadband'] > quality['area_tolerance']:
        st.error('面积均衡免罚范围不能超过目标面积相对误差上限。')
        return False
    return True


def net_unit_archive(config, result):
    """Reuse the accepted downstream export, without modifying saved configs."""
    # IDs become paths in the exporter; a download must stay inside its temp dir.
    for uid in result.unit_polygons:
        if not uid or uid in ('.', '..') or any(c in uid for c in '/\\:'):
            raise ValueError('导出户型编号不能包含路径分隔符或冒号')
    output = BytesIO()
    with TemporaryDirectory(prefix='partition_net_') as folder:
        paths = export_unit_configs(config, result, folder)
        with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
            for path in paths:
                archive.writestr(path.relative_to(folder).as_posix(), path.read_bytes())
            summary = Path(folder) / 'summary.yaml'
            if summary.exists():
                archive.writestr('summary.yaml', summary.read_bytes())
            archive.writestr('floor_plan.json',(Path(folder)/'floor_plan.json').read_bytes())
    return output.getvalue()
