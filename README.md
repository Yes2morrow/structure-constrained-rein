# 结构约束布局训练工作台

## 启动与目录

Windows 双击 `start_gui.bat`。根目录 `main.py` 已恢复，可继续使用原入口。启动器优先使用项目内 `.venv312`，其次使用上一级的 `.venv312`（当前实际环境位置）。

| 目录 | 用途 |
| --- | --- |
| `scripts/` | 训练、模型评估、离线生长和结果导出入口 |
| `common/` | 配置管理、项目路径、训练器构建 |
| `core/` | 环境、智能体与种子生长算法 |
| `gui/` | WebUI，入口 `gui/main.py` |
| `config/` | 配置和上次选中的配置编号 |
| `checks/` | 自动测试；smoke/ 界面检查、benchmarks/ 基准、evidence/ 历史证据，见 checks/README.md |
| `docs/` | 工作检查点、说明与整理记录 |
| `runtime/` | 日志、PID、停止请求（可重新生成） |
| `results2/` | 训练运行、模型、快照、指标；每个种子运行目录直接包含 layout.png / layout.svg / result.json |

在项目目录运行：

```powershell
& '..\.venv312\Scripts\python.exe' -m streamlit run main.py
& '..\.venv312\Scripts\python.exe' scripts/train_seed_layout.py --episodes 250
& '..\.venv312\Scripts\python.exe' scripts/render_seed_result.py --run-dir results2/seed_experimental/<运行名>
& '..\.venv312\Scripts\python.exe' -m unittest discover -s checks -p 'test_*.py' -q
```

边界编辑：二维原始平面与前期约束标注 → 建筑边界、柱与墙体精确输入。坐标输入有效时自动保存；在“画布操作”选择“调整建筑边界”，拖动橙色顶点即可同步坐标与 YAML。增删顶点使用坐标输入；自交、重复或无面积边界拒绝保存。

原根目录的训练命令现需加 `scripts/` 前缀。图种子工作的续作入口为 [工作文档](docs/SEED_GROWTH_WORKLOG.md)，目录迁移和备份见 [整理记录](docs/PROJECT_ORGANIZATION.md)。

## 配置文件
    # 修改注意事项:
        1. True 和 False， 开头首字母大写
        2. 括号均用中括号 [ ]
        3. 字符类型的均用 英文的单引号 ' ' 或双引号 " " 包裹
        4. 每行结尾不要有逗号
        5. 参数的具体含义请参配置文件中的注释
        6. 训练读取 config/<编号>/config.yaml；原户型默认 a，改造默认 retrofit；

    # 预训练模型
    ckpt_path: "" # 预训练模型路径, 默认是空, 表示不加载预训练模型
                    如需要添加，则需要填写其目录路径，如: ckpt_path: "xxxx/"；目录下是所有智能体的模型文件".pth" 或 ".pt" 文件

    P.S 每次训练均会产生一个目录，目录下有智能体的模型文件，以及训练配置文件;
        该配置文件亦可以作为下次训练的预训练模型，只需修改 ckpt_path 即可；
    
        支持多进程训练，修改 Training -> num_processes 即可，默认为 1
        多进程最大数量为CPU核数，不建议超过或等于CPU核数，会卡顿；

    # 添加智能体
    ** 已有的类型:
        如果是添加一个或多个已经有的智能体类型，只需要在配置环境中，Environments -> room_type 中添加即可，例如: ["卧室", "卧室", "厨房", "浴室", "阳台", "卧室"， "厨房"]; 客厅不需要添加;

    *** 未存在的类型:
        1. 需要修改room.py中的定义的变量
           RoomType中添加该房间类型的索引,
            例如: STORAGE_ROOM = 6 以此类推
        2. name_transform中添加该房间类型的名称，例如: "storage_room": "储物室"
        3. RoomColor中添加该房间类型的颜色，例如: "storage_room": (0, 0, 0)
        4. MinAreas中添加该房间类型的最小面积，例如: RoomType:STORAGE_ROOM: 10
        5. MaxAreasRatio中添加该房间类型的最大面积比例，例如: RoomType:STORAGE_ROOM: 0.2
        6. ASPECT_RATIO_RANGES中添加该房间类型的面积比例范围，例如: RoomType:STORAGE_ROOM: (0.5, 1.5)

        7. ADJACENCY_WEIGHTS中添加该房间类型的邻接权重，例如: 自行参照
        8. ADJACENCY_SCORES中添加该房间类型的邻接分数，例如: 自行参照
        9. RELATIONSHIP_TYPES, ....
        10. NESTING_PRIORITY, 例如: STORAGE_ROOM: 5
        11. NESTING_WEIGHTS, 例如: STORAGE_ROOM: 0.5
        12. NOISE_WEIGHTS, 例如: STORAGE_ROOM: 0.5

#* 配置路径由 project_paths.py 统一管理
   默认配置文件: 项目根目录下的 config.yaml


## 住区楼栋布局原型

项目新增了一个不影响原室内房间环境的住区布局环境。第一阶段固定为6栋轴对齐矩形住宅，
每栋住宅由一个智能体控制，支持移动、保持不动以及四条边的伸缩。

默认配置：`config/residential/config.yaml`

快速验证：

```bash
.venv312/bin/python -m unittest checks.test_residential_env
```

启动训练：

```bash
.venv312/bin/python scripts/train_residential.py
```

短回合试运行：

```bash
.venv312/bin/python scripts/train_residential.py --episodes 2 --max-steps 20
```

训练结果保存在 `results2/residential/<时间>/`，包含模型、训练指标和最新住区平面图。

---

## 本机（Windows）运行说明 · 2026-09 实测

下面所有命令都在项目根目录
`J:\man2\structure constrained rein\CUHK_Rein_2026.06_edit by MY` 下执行。

### 环境

机器上没有任何 Python 环境装齐了依赖，所以建了一个沿用 conda `communitymarl`
（Python 3.11.15，torch 2.5.1 / numpy 2.0.1 / matplotlib / pandas / shapely / streamlit 1.60）
的轻量虚拟环境 `.venv312`（`.gitignore` 已忽略，路径沿用 README 的命名）：

```bash
# 1) 建 venv（继承 conda 环境里已装好的重依赖，省掉 torch 的几 GB 下载）
"/c/Users/TAITAN16HX/anaconda3/envs/communitymarl/python.exe" -m venv --system-site-packages .venv312

# 2) 补装 conda 环境里缺的三个包
./.venv312/Scripts/python.exe -m pip install gym==0.26.2 streamlit_drawable_canvas==0.9.3 tensorboard
```

> Windows 下解释器是 `.venv312/Scripts/python.exe`，不是 README 里写的 `.venv312/bin/python`（那是 macOS/Linux 路径）。
> `gym 0.26.2` 会打印 "does not support NumPy 2.0" 的告警，实测不影响运行，可忽略。
> 若要严格按 `requirements.txt`（numpy==1.26.4 / torch==2.12.1 / streamlit 1.58）重建干净环境，需要重新下载 torch，约 2~3 GB。

### 三条训练入口

| 入口 | 用途 | 配置 |
| --- | --- | --- |
| `scripts/train.py --config-id <编号>` | **室内户型布局**（原主任务，多智能体 MAPPO 排房间） | `config/<编号>/config.yaml` |
| `scripts/train_residential.py` | **住区楼栋布局原型**（6 栋矩形住宅，移动+四边伸缩） | `config/residential/config.yaml` |
| `scripts/train_adaptive_reuse.py` | **既有建筑适应性转换**（结构约束下的功能置换） | `config/retrofit/config.yaml` |

```bash
# 室内户型：b~h 是空 ckpt 的干净配置，3 回合 600 步，约 1.5 分钟跑完
./.venv312/Scripts/python.exe scripts/train.py --config-id b

# 住区布局：短跑验证
./.venv312/Scripts/python.exe scripts/train_residential.py --episodes 2 --max-steps 20

# 既有建筑改造：默认读 retrofit 配置
./.venv312/Scripts/python.exe scripts/train_adaptive_reuse.py --episodes 1 --max-steps 20
```

⚠️ **不要用 `--config-id a`（默认配置）直接开跑**：它的 `ckpt_path` 指向
`D:/man/TO 丁-强化学习模型/...` 这个本机不存在的路径，且 `resume_mode: resume`，会加载失败。
新建/复制配置后记得把 `Training.ckpt_path` 清空、`resume_mode` 设为 `fresh`。
`config/1`、`config/2` 同理，指向 F 盘旧路径。
`config/retrofit` 是干净的（`ckpt_path: ''`、`resume_mode: fresh`）。

### 可视化工作台

```bash
./.venv312/Scripts/python.exe -m streamlit run gui/main.py
# 打开 http://localhost:8501
```

三个页签：参数配置 / 训练监控 / 布局预览。顶部可切换配置编号；当配置的
`ProjectType: adaptive_reuse` 时会切换到"传统堂屋住宅更新"界面。

### 自测

```bash
./.venv312/Scripts/python.exe -m unittest checks.test_residential_env    # 3 passed
./.venv312/Scripts/python.exe -m unittest checks.test_adaptive_reuse_env # 4 passed
```

### 输出目录

```
results2/
├── mappo/<时间>/                # 室内户型（scripts/train.py）
│   ├── best_model|final_model|checkpoints/   模型
│   ├── images/epsiode_XXXX_reward_YYY.png    布局图
│   ├── training_metrics.csv  tensorboard/  trainer_state.json
│   └── config.yaml, cpnfig.yaml            配置快照
├── residential/<时间>/          # 住区布局：best_layout.png / latest_layout.png / training_metrics.csv
└── adaptive_reuse/<时间>/       # 既有建筑改造：同上
```

### 实测速度（CPU）

- 室内户型 `scripts/train.py`：约 **25 step/s**，600 步/回合 ≈ 21 秒/回合
- 住区布局：20 步 2 回合约 10 秒
- 既有建筑改造：20 步 1 回合约 25 秒

当前配置以 retrofit 为唯一默认。旧项目配置已按用户要求删除，Git 历史可回查；训练归档不删除。二维标注采用同一张左侧画布＋右侧住宅/边界/柱墙表格，不再使用独立边界画布或边界 YAML 输入。
