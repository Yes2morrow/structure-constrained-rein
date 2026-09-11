# 检查与验证

这些文件仍有用途，不是训练时需要运行的全部代码。

- 根目录 `test_*.py`：自动回归测试，验证训练、几何、保存与前端行为。
- `smoke/`：独立界面、进程停止/恢复和结构同步检查；structure_sync_preview.py 是不写用户 YAML 的隔离页面。
- `benchmarks/`：速度与几何案例基准生成器，不属于正式训练。
- `evidence/`：历史基准 JSON、图片、日志。seed_geometry_baseline.json 被回归测试用作已知假邻接案例，不能整体删除。
- audit_seed_run.py：核对已完成运行的模型、种子、精确结果与指标是否完整。

原人工模型评估工具已移到 gui/manual_env.py，由 scripts/train.py 调用。

在项目根目录运行：

```powershell
& '..\.venv312\Scripts\python.exe' -m unittest discover -s checks -p 'test_*.py' -q
& '..\.venv312\Scripts\python.exe' checks/smoke/check_structure_sync.py
```
