# 柱墙精确输入与训练约束

“柱与墙体精确输入”是“二维原始平面与前期约束标注”内部的可折叠子版块，单位统一为米。参数表和画布共享同一组构件。

- 柱：输入唯一 ID、中心 X/Y、X 尺寸（x2−x1）和 Y 尺寸（y2−y1）。边界由中心加减半尺寸得到。
- 墙：输入唯一 ID、起点 X/Y、终点 X/Y、左侧厚度、右侧厚度。左右按从起点朝终点的方向判断，正 Y 向上；总厚度为左右之和。支持斜墙。
- 剪力墙与承重墙分别保存为 shear_wall、load_bearing_wall，使用相同的墙体几何计算。
- 剪力墙中心线无缝围合形成的闭合面自动生成核心筒；不会自动跨越缺口，也不会把承重墙作为剪力墙参与围合。修改或删除围合墙后重新保存，会重新识别。
- 核心筒沿用项目现有语义：围合区域整体保留，目标功能空间不能进入；并非仅墙厚区域不可进入。
- 参数表编辑结束后（回车或离开单元格），有效数据自动保存并刷新画布；新增行未填完整时保留编辑内容，暂停同步。
- 画布绘制柱用矩形，绘制墙用起终点线，并指定新墙左右厚度。松开鼠标自动保存并回填参数表。在“选择/调整”模式可移动、缩放柱墙，墙还可旋转；厚度缩放保留左右比例。删除会同步删除参数行，并重新识别核心筒。
- 构件不吸附动作网格。Fabric 序列化造成的像素舍入不会改写未编辑构件的精确参数；初始空帧不会清空 YAML。

配置使用 ExistingBuilding.structure_schema_version: 2，ExistingBuilding.fixed_objects 是唯一构件列表，不区分“手绘”或“数值输入”。每个构件有稳定的 id。

柱以 center、size 为主数据；墙以 start、end、left_thickness、right_thickness 和 type 为主数据。rect、polygon 是自动重建的兼容/几何缓存，不能作为第二份独立编辑数据。核心筒标记 derived: true，每次由剪力墙重新计算。加载旧矩形配置时自动转为参数化构件，下次保存写入版本 2。训练加载也使用相同的归一化逻辑。

示例（rect、polygon 可由加载器补齐）：

```yaml
ExistingBuilding:
  structure_schema_version: 2
  fixed_objects:
  - id: column_01
    type: column
    center: [5.0, 5.0]
    size: [0.6, 0.8]
  - id: wall_01
    type: shear_wall
    start: [10.0, 5.0]
    end: [18.0, 8.0]
    left_thickness: 0.2
    right_thickness: 0.1
```

训练实现：

1. core/envs/adaptive_reuse_env.py 将固定构件真实轮廓合并为 fixed_union。
2. _is_hard_valid 检查功能空间与固定区域的交集面积，超过 1e-9 则非法，同时检查建筑边界、房间间重叠、最小尺寸。
3. _get_allow_actions 枚举有效动作；core/agents/mappo.py 将其余动作 logits 置为负无穷，采样时屏蔽。
4. step 再次校验动作，非法动作保持原位置并扣 invalid_action 奖励；同时动作导致的房间冲突也会回退。
5. get_state 包含到固定区域的归一化距离。柱墙类型目前共享不可侵入规则，不代表已经实现不同材料或结构承载能力计算。
6. structure_alignment 当前衡量规则网格对齐，不是柱网受力或结构安全计算。

这次只扩展结构几何读取与预览，未改变 MAPPO 网络、动作集合或奖励公式。新增构件可能与原有目标空间初始位置冲突，可在环境预览中检查硬约束冲突数并调整初始布局。

检查脚本：checks/check_structure_geometry.py、checks/check_structure_ui.py、checks/check_plan_ui.py。使用项目 .venv312/Scripts/python.exe 运行。
