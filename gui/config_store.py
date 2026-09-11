import ast
import copy
import tkinter as tk
from tkinter import filedialog

import yaml

from common.config_manager import DEFAULT_CONFIG_ID, PROJECT_ROOT, get_available_config_ids, get_config_path, normalize_config_id
from core.envs.loads import CompatibleSafeLoader as ConfigSafeLoader


SELECTED_CONFIG_FILE = PROJECT_ROOT / ".selected_config_id"


class FlowList(list):
    """用于控制 YAML 列表以内联格式输出。"""


def _flow_list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(FlowList, _flow_list_representer)


def get_selected_config_id(session_state) -> str:
    """获取当前选中的配置编号。"""
    return normalize_config_id(session_state.selected_config_id)


def load_persisted_config_id() -> str:
    """Load the last selected config id from disk."""
    try:
        if SELECTED_CONFIG_FILE.exists():
            return normalize_config_id(SELECTED_CONFIG_FILE.read_text(encoding="utf-8").strip())
        migrated = PROJECT_ROOT / 'config' / '.selected_config_id'
        if migrated.exists():
            return normalize_config_id(migrated.read_text(encoding='utf-8').strip())
    except Exception:
        pass
    return DEFAULT_CONFIG_ID


def persist_selected_config_id(config_id: str) -> None:
    """Persist selected config id so browser refresh keeps the same config."""
    normalized_config_id = normalize_config_id(config_id)
    SELECTED_CONFIG_FILE.write_text(normalized_config_id, encoding="utf-8")


def get_config_summary(config_id: str) -> tuple[str, str]:
    """返回当前配置路径和已有编号文本。"""
    current_config_path = get_config_path(config_id)
    available_ids = ", ".join(str(item) for item in get_available_config_ids())
    return current_config_path, available_ids


def load_config(config_id: str) -> dict:
    """加载指定编号的配置文件。"""
    config_path = get_config_path(config_id)
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=ConfigSafeLoader) or {}
    if config.get('ProjectType') == 'adaptive_reuse':
        from core.envs.structure_geometry import normalize_structures
        building = config['ExistingBuilding']
        building['fixed_objects'] = normalize_structures(building.get('fixed_objects', []))
        building['structure_schema_version'] = 2
    return config


def _convert_to_flow_list(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = _convert_to_flow_list(value)
        return data

    if isinstance(data, tuple):
        data = list(data)

    if isinstance(data, list):
        data = [_convert_to_flow_list(item) for item in data]
        if data and (
            isinstance(data[0], (int, float))
            or (isinstance(data[0], list) and data[0] and isinstance(data[0][0], (int, float)))
        ):
            return FlowList(data)
        if data and isinstance(data[0], str) and len(str(data[0])) < 10:
            return FlowList(data)
    return data


def save_config(config_data: dict, config_id: str) -> str:
    """保存配置文件并返回落盘路径。"""
    config_path = get_config_path(config_id)
    config_copy = copy.deepcopy(config_data)
    if config_copy.get('ProjectType') == 'adaptive_reuse':
        from core.envs.structure_geometry import normalize_structures
        building = config_copy['ExistingBuilding']
        building['fixed_objects'] = normalize_structures(building.get('fixed_objects', []))
        building['structure_schema_version'] = 2
    config_copy = _convert_to_flow_list(config_copy)
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(config_copy, file, allow_unicode=True, sort_keys=False)
    return config_path


def select_folder() -> str:
    """打开文件夹选择对话框。"""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_path


def parse_room_types(room_types_str: str):
    """安全解析房间类型列表。"""
    value = ast.literal_eval(room_types_str)
    if not isinstance(value, list):
        raise ValueError("房间类型必须为列表")
    return value
