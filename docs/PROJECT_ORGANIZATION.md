# 项目整理记录（2026-09-12）

## 迁移

根目录只保留 README.md、requirements.txt、.gitignore、start_gui.bat。

| 原文件 | 新位置 |
| --- | --- |
| main.py | gui/main.py |
| train.py、train_residential.py、train_adaptive_reuse.py、train_seed_layout.py | scripts/ 内同名文件 |
| decode_seed_snapshot.py、render_seed_result.py | scripts/ 内同名文件 |
| test.py（实际是模型评估入口） | scripts/evaluate_layout.py |
| config_manager.py、project_paths.py | common/ 内同名文件 |
| train_common.py | common/training.py |
| SEED_GROWTH_WORKLOG.md | docs/SEED_GROWTH_WORKLOG.md |
| .selected_config_id | config/.selected_config_id |
| 新产生的 training.log、running.pid、stop_requested.flag | runtime/ |

所有内部导入、GUI 子进程启动目录、独立 CLI 的项目根路径、停止测试入口均已同步更新。根目录不保留重复 Python 转发文件；外部命令需要使用新路径。WebUI 仍双击根目录 start_gui.bat。

## 清理范围

- 删除未使用的 numpy、fixed_polygon 导入，以及迁移产生的重复 pathlib/sys 导入。
- 评估入口不再使用 `from train import *`，只导入 Trainer、make_env。
- 清理根目录 Python 缓存；保留测试、基准证据及仍被调用的原模式代码。
- 原户型训练仍支持 MAPPO/QMIX/MADQN，人工评估仍调用 checks/manual_env.py，因此这些不是废代码。
- results、results2、模型检查点、用户 YAML 和历史基线保留原位置，不重写结果或配置指纹。

## 备份与检查

迁移前代码备份位于项目上一级 `reorganization_backup_20260912_000015.zip`，不包含训练结果和 Python 环境。

78 项 unittest 回归通过，记录 runtime/reorganization_tests.log。六个 CLI 在项目外工作目录调用 --help 成功；WebUI 主入口 AppTest 和配置/训练/结果路径核对通过，记录 runtime/reorganization_entrypoints.log。结构与平面三组旧回归通过。

当前 Python 环境为 `../.venv312/Scripts/python.exe`。启动器也支持将来恢复项目内 `.venv312`。目录迁移后请关闭旧 WebUI，再通过 start_gui.bat 启动新入口；已有模型和图像文件仍使用原路径。

补充：首次主界面测试在退出时触发 Tk 清理异常，已在 gui/main.py 导入页面前指定 Agg 后端；复测退出码 0，记录 runtime/reorganization_gui.log。

路径更正：种子成图直接位于 results2/seed_experimental/<运行名>/layout.png（同目录 SVG、JSON），不再使用单独 results/seed_growth 目录。

兼容恢复：用户要求恢复根目录 main.py，现已恢复且启动器使用该入口。checks 的图片/JSON/日志迁到 evidence/，独立检查迁到 smoke/，基准迁到 benchmarks/，人工评估移到 gui/manual_env.py。此条替代前文根目录只四个文件以及 manual_env.py 旧位置说明。

## 增量整理记录（2026-09-16 版本 v0.2.2）

- 新增根目录 `pyproject.toml`，把项目元数据、`pytest` 测试入口与 `ruff` 静态检查配置收拢到统一位置，便于后续按标准 Python 工程方式维护。
- 新增 `gui/adaptive_reuse_support.py`，把既有住宅页面重复出现的自动修复提示、自动修复执行与折叠标题逻辑提取成公共辅助模块，减少 `gui/adaptive_reuse_page.py` 中的重复代码。
- `gui/app.py` 改为按主视图延迟导入页面模块，只在真正进入“环境搭建 / 参数配置 / 训练监控 / 布局预览”时加载对应页面，减少启动阶段的一次性导入开销。
- `.gitignore` 补充 `.pytest_cache/`、`.ruff_cache/`、`build/`、`dist/`，避免新增工程化配置后把本地工具缓存误入库。

本次整理不改变现有训练方法、环境约束或页面交互语义，目标是让目录结构更清晰、入口更轻、后续维护成本更低。
