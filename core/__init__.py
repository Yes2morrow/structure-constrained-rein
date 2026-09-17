"""Public compatibility exports without eager environment/agent imports."""
from importlib import import_module

_EXPORTS = {
    **{name: '.envs' for name in (
        'make_env', 'HouseEnv', 'Building', 'load_yaml_file', 'copy_yaml_file',
        'ResidentialLayoutEnv', 'make_residential_env', 'AdaptiveReuseEnv',
        'FunctionalSpace', 'calculate_area_budget', 'summarize_target_area_budget',
        'redistribute_target_max_areas', 'make_adaptive_reuse_env')},
    **{name: '.envs.room' for name in (
        'Enum', 'RoomType', 'Room', 'name_transform', 'RoomColor', 'DoorColor',
        'MinAreas', 'MaxAreasRatio', 'ASPECT_RATIO_RANGES', 'NESTING_PRIORITY',
        'NESTING_WEIGHTS', 'NOISE_WEIGHTS', 'MaxAreasValue')},
    'MAPPO': '.agents.mappo', 'QMIX': '.agents.qmix', 'MADQN': '.agents.madqn',
    'save_layout_image': '.utils',
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS: raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
