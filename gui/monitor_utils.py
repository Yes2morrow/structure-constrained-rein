import csv
import os
import re
import glob
from dataclasses import dataclass
from typing import Optional

from common.project_paths import LOG_FILE, RESULTS_DIR


@dataclass
class TrainingProgress:
    current_episode: int = 0
    total_episodes: int = 0
    total_step: Optional[int] = None
    mean_reward: Optional[float] = None
    episode_time_seconds: Optional[float] = None
    step_time_seconds: Optional[float] = None
    avg_episode_time_seconds: Optional[float] = None
    avg_step_time_seconds: Optional[float] = None
    last_pid: Optional[int] = None
    result_dir: str = ""

    @property
    def ratio(self) -> float:
        if self.total_episodes <= 0:
            return 0.0
        return min(1.0, max(0.0, self.current_episode / self.total_episodes))


def _safe_existing_paths(paths: list[str], path_type: str = "file") -> list[str]:
    """过滤训练过程中可能瞬间消失的文件或目录。"""
    if path_type == "dir":
        return [path for path in paths if os.path.isdir(path)]
    return [path for path in paths if os.path.isfile(path)]


def _safe_latest_path(paths: list[str], path_type: str = "file") -> str:
    """安全获取最新路径，避免 getmtime 命中已删除路径。"""
    existing_paths = _safe_existing_paths(paths, path_type=path_type)
    if not existing_paths:
        return ""
    try:
        return max(existing_paths, key=os.path.getmtime)
    except OSError:
        # 训练过程中可能仍有竞态，再做一次过滤兜底。
        existing_paths = _safe_existing_paths(existing_paths, path_type=path_type)
        if not existing_paths:
            return ""
        return max(existing_paths, key=os.path.getmtime)


def read_monitor_log(raw_log_content: str) -> str:
    """读取训练日志，必要时回退到 gbk 解码。"""
    log_content = raw_log_content
    if ("□" in raw_log_content or "\ufffd" in raw_log_content) and os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="gbk", errors="ignore") as file:
                log_content = file.read()
        except Exception:
            pass
    return log_content


def parse_training_progress(log_content: str) -> TrainingProgress:
    """从日志中提取训练进度。"""
    progress = TrainingProgress()
    progress_pattern = re.compile(
        r"PID:\s*(\d+),\s*Episode\[(\d+)/(\d+)\],\s*Total Step:\s*(\d+),\s*Rwd\(mean\):\s*([-\d.]+)"
    )
    time_pattern = re.compile(
        r"Time\(ep\):\s*([-\d.]+)s,\s*Time\(step\):\s*([-\d.]+)s,\s*Avg\(ep\):\s*([-\d.]+)s,\s*Avg\(step\):\s*([-\d.]+)s"
    )
    result_dir_pattern = re.compile(r"结果目录:\s*(.+)")

    for line in log_content.splitlines():
        match = progress_pattern.search(line)
        if match:
            progress.last_pid = int(match.group(1))
            progress.current_episode = int(match.group(2))
            progress.total_episodes = int(match.group(3))
            progress.total_step = int(match.group(4))
            progress.mean_reward = float(match.group(5))

            time_match = time_pattern.search(line)
            if time_match:
                progress.episode_time_seconds = float(time_match.group(1))
                progress.step_time_seconds = float(time_match.group(2))
                progress.avg_episode_time_seconds = float(time_match.group(3))
                progress.avg_step_time_seconds = float(time_match.group(4))

        result_match = result_dir_pattern.search(line)
        if result_match:
            progress.result_dir = result_match.group(1).strip()

    return progress


def format_seconds(seconds: Optional[float], precision: int = 2) -> str:
    """格式化秒数显示。"""
    if seconds is None:
        return "-"
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.{precision}f} s"


def load_training_metrics(result_dir: str, tail_size: int = 2000) -> list[dict]:
    """读取训练指标 CSV，用于监控图表展示。"""
    if not result_dir or not os.path.isdir(result_dir):
        return []

    metrics_csv_path = os.path.join(result_dir, "training_metrics.csv")
    if not os.path.isfile(metrics_csv_path):
        return []

    try:
        with open(metrics_csv_path, "r", encoding="utf-8", newline="") as metrics_file:
            rows = list(csv.DictReader(metrics_file))
    except Exception:
        return []

    numeric_fields = {
        "episode": int,
        "total_episodes": int,
        "episode_steps": int,
        "total_env_steps": int,
        "mean_reward": float,
        "median_reward": float,
        "episode_seconds": float,
        "step_seconds": float,
        "avg_episode_seconds": float,
        "avg_step_seconds": float,
        "steps_per_second": float,
        "avg_steps_per_second": float,
    }
    parsed_rows = []
    for row in rows[-tail_size:]:
        try:
            parsed_row = dict(row)
            for field_name, field_type in numeric_fields.items():
                parsed_row[field_name] = field_type(row[field_name])
            parsed_row["step_milliseconds"] = parsed_row["step_seconds"] * 1000.0
            parsed_row["avg_step_milliseconds"] = parsed_row["avg_step_seconds"] * 1000.0
            parsed_rows.append(parsed_row)
        except (KeyError, TypeError, ValueError):
            continue
    return parsed_rows


def get_latest_agent_run_dir(agent_name: str) -> str:
    """获取指定算法最近一次训练结果目录。"""
    agent_dir = os.path.join(RESULTS_DIR, agent_name.lower())
    if not os.path.isdir(agent_dir):
        return ""

    subdirs = [
        os.path.join(agent_dir, entry)
        for entry in os.listdir(agent_dir)
        if os.path.isdir(os.path.join(agent_dir, entry))
    ]
    return _safe_latest_path(subdirs, path_type="dir")


def get_latest_pid_run_dir(agent_name: str, pid: int) -> str:
    """根据训练 PID 查找最近一次对应的结果目录。"""
    agent_dir = os.path.join(RESULTS_DIR, agent_name.lower())
    if not os.path.isdir(agent_dir):
        return ""

    pid_prefix = f"{pid}_"
    matched_subdirs = [
        os.path.join(agent_dir, entry)
        for entry in os.listdir(agent_dir)
        if os.path.isdir(os.path.join(agent_dir, entry)) and entry.startswith(pid_prefix)
    ]
    return _safe_latest_path(matched_subdirs, path_type="dir")


def resolve_result_dir(log_content: str, agent_name: str) -> str:
    """优先根据日志解析结果目录，失败时回退到最近目录。"""
    progress = parse_training_progress(log_content)
    if progress.result_dir and os.path.isdir(progress.result_dir):
        return progress.result_dir
    if progress.last_pid is not None:
        pid_result_dir = get_latest_pid_run_dir(agent_name, progress.last_pid)
        if pid_result_dir:
            return pid_result_dir
    return get_latest_agent_run_dir(agent_name)


def collect_checkpoint_status(result_dir: str, use_live_render: bool = True) -> dict:
    """收集结果目录中的模型与检查点状态。"""
    if not result_dir or not os.path.isdir(result_dir):
        return {
            "result_dir": "",
            "best_model_exists": False,
            "final_model_exists": False,
            "checkpoint_count": 0,
            "latest_checkpoint": "",
            "latest_image": "",
            "live_render": "",
            "latest_layout": "",
            "preview_image": "",
        }

    checkpoints_dir = os.path.join(result_dir, "checkpoints")
    checkpoint_dirs = []
    if os.path.isdir(checkpoints_dir):
        checkpoint_dirs = [
            os.path.join(checkpoints_dir, entry)
            for entry in os.listdir(checkpoints_dir)
            if os.path.isdir(os.path.join(checkpoints_dir, entry))
        ]

    images_dir = os.path.join(result_dir, "images")
    image_files = []
    if os.path.isdir(images_dir):
        image_files = [
            os.path.join(images_dir, entry)
            for entry in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, entry))
        ]

    latest_checkpoint = _safe_latest_path(checkpoint_dirs, path_type="dir")
    latest_image = _safe_latest_path(image_files, path_type="file")
    live_render = ""
    if use_live_render:
        live_render_files = _safe_existing_paths(
            glob.glob(os.path.join(result_dir, "live_render_*.png")),
            path_type="file",
        )
        try:
            live_render_files = sorted(live_render_files, key=os.path.getmtime)
        except OSError:
            live_render_files = sorted(
                _safe_existing_paths(live_render_files, path_type="file"),
                key=os.path.getmtime,
            )
        live_render = live_render_files[-1] if live_render_files else ""
    latest_layout = os.path.join(result_dir, "latest_layout.png")
    if not os.path.isfile(latest_layout):
        latest_layout = ""
    preview_image = live_render or latest_layout or latest_image

    return {
        "result_dir": result_dir,
        "best_model_exists": os.path.isdir(os.path.join(result_dir, "best_model")),
        "final_model_exists": os.path.isdir(os.path.join(result_dir, "final_model")),
        "checkpoint_count": len(checkpoint_dirs),
        "latest_checkpoint": latest_checkpoint,
        "latest_image": latest_image,
        "live_render": live_render,
        "latest_layout": latest_layout,
        "preview_image": preview_image,
    }
