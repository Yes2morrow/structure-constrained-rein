from __future__ import annotations

from pathlib import Path
import re
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"
DEFAULT_CONFIG_NAME = "config.yaml"
DEFAULT_CONFIG_ID = "retrofit"
LEGACY_CONFIG_NAMES = [
    "config a.yaml",
    "config b.yaml",
    "config c.yaml",
    "config d.yaml",
    "config e.yaml",
    "config f.yaml",
    "config g.yaml",
    "config h.yaml",
    "config.yaml",
    "config备份.yaml",
    "config - 副本.yaml",
    "config 2.yaml",
]


def normalize_config_id(config_id: int | str | None, default: str = DEFAULT_CONFIG_ID) -> str:
    """将外部传入的配置编号规范化为可用的目录名。"""
    if config_id is None or config_id == "":
        return default

    normalized_id = str(config_id).strip().lower()
    if not normalized_id:
        return default
    if not re.fullmatch(r"[a-z0-9_-]+", normalized_id):
        raise ValueError("config_id 仅支持字母、数字、下划线和短横线")
    return normalized_id


def get_config_dir(config_id: int | str | None = None) -> Path:
    """返回指定编号配置所在目录。"""
    normalized_id = normalize_config_id(config_id)
    return CONFIG_ROOT / normalized_id


def _iter_seed_candidates(config_id: str) -> list[Path]:
    """返回新配置初始化时可复用的种子配置候选。"""
    candidates: list[Path] = []

    # 新建非默认配置时，优先从默认配置复制，便于快速派生。
    if config_id != DEFAULT_CONFIG_ID:
        default_indexed_config = get_config_dir(DEFAULT_CONFIG_ID) / DEFAULT_CONFIG_NAME
        candidates.append(default_indexed_config)

    letter_legacy_config = PROJECT_ROOT / f"config {config_id}.yaml"
    candidates.append(letter_legacy_config)

    for legacy_name in LEGACY_CONFIG_NAMES:
        candidates.append(PROJECT_ROOT / legacy_name)

    return candidates


def ensure_indexed_config(config_id: int | str | None = None) -> Path:
    """确保 `config/编号/config.yaml` 存在，不存在时自动创建。"""
    normalized_id = normalize_config_id(config_id)
    config_dir = get_config_dir(normalized_id)
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / DEFAULT_CONFIG_NAME
    if config_path.exists():
        return config_path

    for candidate in _iter_seed_candidates(normalized_id):
        if candidate.exists() and candidate.resolve() != config_path.resolve():
            shutil.copyfile(candidate, config_path)
            return config_path

    config_path.touch()
    return config_path


def get_config_path(config_id: int | str | None = None, ensure_exists: bool = True) -> str:
    """返回指定编号配置文件路径。"""
    config_path = ensure_indexed_config(config_id) if ensure_exists else (
        get_config_dir(config_id) / DEFAULT_CONFIG_NAME
    )
    return str(config_path)


def get_available_config_ids() -> list[str]:
    """获取当前已存在的配置编号列表。"""
    if not CONFIG_ROOT.exists():
        return [DEFAULT_CONFIG_ID]

    available_ids: list[str] = []
    for child in CONFIG_ROOT.iterdir():
        if not child.is_dir():
            continue
        if (child / DEFAULT_CONFIG_NAME).exists():
            available_ids.append(child.name)

    if DEFAULT_CONFIG_ID not in available_ids:
        available_ids.append(DEFAULT_CONFIG_ID)

    return sorted(set(available_ids), key=lambda item: (not item.isdigit(), item))
