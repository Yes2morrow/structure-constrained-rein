# 加载yaml文件
import shutil
import yaml


class CompatibleSafeLoader(yaml.SafeLoader):
    """兼容旧配置中的 `!!python/tuple`。"""


def _python_tuple_constructor(loader, node):
    return list(loader.construct_sequence(node))


CompatibleSafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _python_tuple_constructor)


def load_yaml_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=CompatibleSafeLoader)
    return config


def copy_yaml_file(file_path, target_path):
    shutil.copyfile(file_path, target_path)
