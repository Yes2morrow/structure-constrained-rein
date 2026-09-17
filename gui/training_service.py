import os
import signal
import subprocess
import sys
import time

import streamlit as st

from common.project_paths import LOG_FILE, PID_FILE, STOP_REQUEST_FILE, TRAIN_SCRIPT, PROJECT_ROOT
from common.config_manager import get_config_path
from gui.config_store import ConfigSafeLoader
import yaml


def _get_windows_detached_creationflags() -> int:
    """构造 Windows 下更彻底脱离父终端的创建标志。"""
    creationflags = 0
    if os.name != "nt":
        return creationflags

    # 说明：
    # 1. CREATE_NEW_PROCESS_GROUP：让训练进程脱离当前进程组；
    # 2. DETACHED_PROCESS：不再绑定当前控制台；
    # 3. CREATE_BREAKAWAY_FROM_JOB：若上层终端/IDE 使用 job object 管理子进程，
    #    尝试让训练进程跳出该约束，避免 Streamlit 被中断时训练一并结束。
    creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    creationflags |= getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    return creationflags


def is_process_running(pid: int) -> bool:
    """检查训练进程是否仍在运行。"""
    if os.name == "nt":
        try:
            output = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                stderr=subprocess.STDOUT,
            ).decode(errors="ignore")
            return str(pid) in output
        except Exception:
            return False

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def training_command(config, config_id):
    script=TRAIN_SCRIPT
    if config.get('ProjectType')=='adaptive_reuse':
        training_stage=str(config.get('Training',{}).get('training_stage','room_training')).strip()
        floor_partition=config.get('FloorPartition',{})
        if training_stage=='floor_partition':
            script=os.path.join(os.path.dirname(TRAIN_SCRIPT),'train_floor_partition.py')
            return [sys.executable,'-u',script,'--config-id',str(config_id)]
        seed=dict(config.get('SeedGrowth',{}))
        if config.get('FloorWorkflow'):
            if not config.get('TargetSpaces'):
                raise ValueError('请先在「环境搭建 → 智能体与关系」中添加房间并保存，再开始该户训练。')
            if not seed.get('enabled',False):
                raise ValueError('逐户合成需要精确多边形结果，请在参数配置中保存图种子模式。')
        if any(r.get('role')=='residual' for r in config.get('TargetSpaces',[])): seed['enabled']=True
        script=os.path.join(os.path.dirname(TRAIN_SCRIPT),'train_seed_layout.py' if seed.get('enabled',False) else 'train_adaptive_reuse.py')
        if seed.get('enabled',False) and str(seed.get('resume_run','')).strip():
            return [sys.executable,'-u',script,'--resume',str(seed['resume_run']).strip(),
                    '--episodes',str(int(config['Training']['episodes']))]
    return [sys.executable,'-u',script,'--config-id',str(config_id)]


def start_training(config_id: str) -> tuple[bool, str]:
    """按指定配置编号启动训练。"""
    if st.session_state.training_pid is not None:
        return False, "训练已经在运行中。"

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as file:
            file.write("")
    if os.path.exists(STOP_REQUEST_FILE):
        os.remove(STOP_REQUEST_FILE)

    try:
        with open(get_config_path(config_id), "r", encoding="utf-8") as config_stream:
            selected_config = yaml.load(config_stream, Loader=ConfigSafeLoader) or {}
        cmd=training_command(selected_config,config_id)
        selected_script=cmd[2]
        log_file = open(LOG_FILE, "w", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            creationflags=_get_windows_detached_creationflags(),
            bufsize=1,
            universal_newlines=True,
            close_fds=True,
            encoding="utf-8",
        )
        # 父进程不再持有日志文件句柄，避免网页端生命周期影响训练子进程。
        log_file.close()

        time.sleep(0.6)
        if process.poll() is not None:
            try:
                with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as file:
                    error_output = file.read().strip()
            except Exception:
                error_output = ""
            message = "训练进程启动后立即退出。"
            if error_output:
                last_lines = "\n".join(error_output.splitlines()[-12:])
                message = f"{message}\n\n{last_lines}"
            return False, message

        st.session_state.training_pid = process.pid
        st.session_state.training_status = "Running"

        with open(PID_FILE, "w", encoding="utf-8") as file:
            file.write(str(process.pid))

        return True, f"训练已启动，PID: {process.pid}"
    except Exception as exc:
        return False, f"启动失败: {exc}"


def stop_training() -> None:
    """停止训练进程。"""
    pid = st.session_state.training_pid
    if pid is None:
        st.warning("当前没有正在运行的训练进程。")
        return

    try:
        if os.path.exists(STOP_REQUEST_FILE):
            if os.name == "nt":
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)])
            else:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            st.session_state.training_pid = None
            st.session_state.training_status = "Stopped"
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
            if os.path.exists(STOP_REQUEST_FILE):
                os.remove(STOP_REQUEST_FILE)
            st.success("已执行强制停止。")
            st.rerun()
            return

        with open(STOP_REQUEST_FILE, "w", encoding="utf-8") as file:
            file.write(f"pid={pid}\nrequested_at={int(time.time())}\n")

        st.session_state.training_status = "Stopping"
        st.success("已发送停止请求：种子模式在完整步骤后保存并成图，原模式在回合结束后保存退出。若长时间没有停止，再点一次“停止训练”可强制结束。")
        st.rerun()
    except Exception as exc:
        st.error(f"停止失败: {exc}")


def get_latest_log() -> str:
    """读取最新日志文本。"""
    if not os.path.exists(LOG_FILE):
        return "等待日志生成..."

    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as file:
            return file.read()
    except Exception as exc:
        return f"日志读取失败: {exc}"


def start_tensorboard(log_dir: str, port: int = 6010) -> tuple[bool, str]:
    """启动 TensorBoard 服务。"""
    if not log_dir or not os.path.isdir(log_dir):
        return False, "TensorBoard 日志目录不存在。"

    running_pid = st.session_state.get("tensorboard_pid")
    if running_pid and is_process_running(int(running_pid)):
        return True, st.session_state.get("tensorboard_url", f"http://localhost:{port}")

    creationflags = _get_windows_detached_creationflags()
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", str(port), "--host", "0.0.0.0"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(TRAIN_SCRIPT),
            creationflags=creationflags,
            close_fds=True,
        )
    except Exception as exc:
        return False, f"TensorBoard 启动失败: {exc}"

    st.session_state.tensorboard_pid = process.pid
    st.session_state.tensorboard_url = f"http://localhost:{port}"
    st.session_state.tensorboard_logdir = log_dir
    return True, st.session_state.tensorboard_url
