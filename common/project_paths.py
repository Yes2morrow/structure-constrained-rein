from __future__ import annotations

from pathlib import Path

from common.config_manager import DEFAULT_CONFIG_ID, get_config_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(*parts: str) -> str:
    """返回项目根目录下的绝对路径。"""
    return str(PROJECT_ROOT.joinpath(*parts))


(PROJECT_ROOT / "runtime").mkdir(exist_ok=True)

CONFIG_PATH = get_config_path(DEFAULT_CONFIG_ID)
RESULTS_DIR = project_path("results2")
LOG_FILE = project_path("runtime", "training.log")
PID_FILE = project_path("runtime", "running.pid")
STOP_REQUEST_FILE = project_path("runtime", "stop_requested.flag")
TRAIN_SCRIPT = project_path("scripts", "train.py")
