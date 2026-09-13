from .env import HouseEnv
from .room import *
from .loads import load_yaml_file, copy_yaml_file
from .building import Building
from .residential_env import ResidentialLayoutEnv, make_residential_env
from .adaptive_reuse_env import (
    AdaptiveReuseEnv,
    FunctionalSpace,
    calculate_area_budget,
    summarize_target_area_budget,
    redistribute_target_max_areas,
    make_adaptive_reuse_env,
)


def make_env(config_path:str):
    """ 通过配置文件创建环境 """
    config = load_yaml_file(config_path)
    config.setdefault('EnvironmentAdvanced', {})
    config.setdefault('RewardAdvanced', {})
    config.setdefault('AlgorithmAdvanced', {})
    config['EnvironmentAdvanced'].setdefault('grid_size', 15)
    config['EnvironmentAdvanced'].setdefault('grid_resolution', 0.2)
    config['EnvironmentAdvanced'].setdefault('prior_match_threshold', 1.0)
    config['EnvironmentAdvanced'].setdefault('prior_random_max_attempts', 3000)
    config['EnvironmentAdvanced'].setdefault('prior_random_wall_clearance', 0.8)
    config['EnvironmentAdvanced'].setdefault('prior_random_door_avoidance', 1.2)
    config['EnvironmentAdvanced'].setdefault('prior_corner_region_size', 1.5)
    config['EnvironmentAdvanced'].setdefault('prior_corner_attempt_limit', 50)
    config['EnvironmentAdvanced'].setdefault('prior_distance_relaxation', [2.5, 2.0, 1.5, 1.0, 0.8])
    config['EnvironmentAdvanced'].setdefault('prior_distance_relax_attempts', 100)
    config['EnvironmentAdvanced'].setdefault('boundary_retry_scale_factors', [0.8, 0.6, 0.4, 0.2, 0.1, 0.05])
    config['EnvironmentAdvanced'].setdefault('living_room_min_ratio', 0.3)
    config['EnvironmentAdvanced'].setdefault('living_room_max_ratio', 0.5)
    config['EnvironmentAdvanced'].setdefault('room_init_max_attempts', 100)
    config['EnvironmentAdvanced'].setdefault('door_center_offset', 0.6)
    config['EnvironmentAdvanced'].setdefault('vertical_door_default_length', 1.2)
    config['EnvironmentAdvanced'].setdefault('horizontal_door_default_length', 1.0)
    config['RewardAdvanced'].setdefault('prior_layout_scale', 2.0)
    
    # 修改环境基础参数
    shift_value = config['Environment']['move_value']
    num_agents = len(config['Environment']['room_type'])
    reward_method = config['Environment']['reward_method']
    house_env = HouseEnv(
        grid_size=config['EnvironmentAdvanced']['grid_size'],
        grid_resolution=config['EnvironmentAdvanced']['grid_resolution'],
        shift_value=shift_value,
        num_agents=num_agents,
        method=reward_method,
        language=config['Language'],
        show_prior=config['Show_Prior'],
        use_area_ratio=config['Room']['UseAreasRatio'],
        prior_interval=config['Environment'].get('prior_interval', 3),
        environment_advanced=config['EnvironmentAdvanced'],
        reward_advanced=config['RewardAdvanced'],
        prior_layout_scale=config['RewardAdvanced']['prior_layout_scale'],
        init_width=config['Environment'].get('init_width', 1),
        init_height=config['Environment'].get('init_height', 1))
    
    # 设置其他参数
    #* boundary
    if config['Environment']['door_direction'] == "水平" and \
        len(config['Environment']['door_positions']) > 0:
        house_env.set_boundary(config['Environment']['boundary'],
                               right_points=config['Environment']['door_positions'])
    elif config['Environment']['door_direction'] == "垂直" and \
        len(config['Environment']['door_positions']) > 0:
        house_env.set_boundary(config['Environment']['boundary'],
                               left_points=config['Environment']['door_positions'])
    else:
        house_env.set_boundary(config['Environment']['boundary'])
        
    #* room types
    roomtype_dict = {
        "卧室": RoomType.BEDROOM,
        "厨房": RoomType.KITCHEN,
        "浴室": RoomType.BATHROOM,
        "阳台": RoomType.BALCONY,
    }
    roomtype_input = [roomtype_dict[room] for room in config['Environment']['room_type']]
    house_env.set_room_type(roomtype_input)
    
    #* prior
    house_env.set_prior_limit_area(config['Environment']['prior_limit_area'])
    if config['Environment']['prior_random']:
        house_env.set_fixed_position(False)
        house_env.set_prior_positions(config['Environment']['prior'], randomize=True)
    else:
        house_env.set_fixed_position(True)
        house_env.set_prior_positions(config['Environment']['prior'])
    
    # 权重
    house_env.set_extra_reward_weight(
        {
            "邻接关系奖励": config['Environment']['Reward_Extra_Weight']['邻接关系奖励'],
            "噪声干扰度奖励": config['Environment']['Reward_Extra_Weight']['噪声干扰度奖励'],
            "隐私保护奖励": config['Environment']['Reward_Extra_Weight']['隐私保护奖励'],
            "南向采光奖励": config['Environment']['Reward_Extra_Weight']['南向采光奖励'],
            "最短路径奖励": config['Environment']['Reward_Extra_Weight']['最短路径奖励'],
        }
    )
    house_env.set_base_reward_weight(
        {
            "面积奖励": config['Environment']['Reward_Base_Weight']['面积奖励'],
            "长宽比奖励": config['Environment']['Reward_Base_Weight']['长宽比奖励'],
            "贴边奖励": config['Environment']['Reward_Base_Weight']['贴边奖励'],
            "房间与房间之间的边界贴合奖励": config['Environment']['Reward_Base_Weight']['房间与房间之间的边界贴合奖励'],
            "角落占领奖励": config['Environment']['Reward_Base_Weight']['角落占领奖励'],
            "外门遮挡惩罚": config['Environment']['Reward_Base_Weight']['外门遮挡惩罚'],
            "先验知识布局奖励": config['Environment']['Reward_Base_Weight']['先验知识布局奖励'],
            "房间嵌套惩罚": config['Environment']['Reward_Base_Weight']['房间嵌套惩罚'],
        }
    )
    
    # 修改房间配置
    global MinAreas
    MinAreas[RoomType.BEDROOM] = config['Room']['MinAreas']['卧室']
    MinAreas[RoomType.BATHROOM] = config['Room']['MinAreas']['浴室']
    MinAreas[RoomType.KITCHEN] = config['Room']['MinAreas']['厨房']
    MinAreas[RoomType.BALCONY] = config['Room']['MinAreas']['阳台']
    
    global MaxAreasRatio
    MaxAreasRatio[RoomType.BEDROOM] = config['Room']['MaxAreasRatio']['卧室']
    MaxAreasRatio[RoomType.BATHROOM] = config['Room']['MaxAreasRatio']['浴室']
    MaxAreasRatio[RoomType.KITCHEN] = config['Room']['MaxAreasRatio']['厨房']
    MaxAreasRatio[RoomType.BALCONY] = config['Room']['MaxAreasRatio']['阳台']

    global MaxAreasValue
    MaxAreasValue[RoomType.BEDROOM] = config['Room']['MaxAreasValue']['卧室']
    MaxAreasValue[RoomType.BATHROOM] = config['Room']['MaxAreasValue']['浴室']
    MaxAreasValue[RoomType.KITCHEN] = config['Room']['MaxAreasValue']['厨房']
    MaxAreasValue[RoomType.BALCONY] = config['Room']['MaxAreasValue']['阳台']
    
    global ASPECT_RATIO_RANGES
    ASPECT_RATIO_RANGES[RoomType.BEDROOM] = config['Room']['ASPECT_RATIO_RANGES']['卧室']
    ASPECT_RATIO_RANGES[RoomType.BATHROOM] = config['Room']['ASPECT_RATIO_RANGES']['浴室']
    ASPECT_RATIO_RANGES[RoomType.KITCHEN] = config['Room']['ASPECT_RATIO_RANGES']['厨房']
    ASPECT_RATIO_RANGES[RoomType.BALCONY] = config['Room']['ASPECT_RATIO_RANGES']['阳台']
    
    global NOISE_WEIGHTS
    NOISE_WEIGHTS["LIVING_ROOM"] = config['Room']['NOISE_WEIGHTS']['客厅']
    NOISE_WEIGHTS["BEDROOM"] = config['Room']['NOISE_WEIGHTS']['卧室']
    NOISE_WEIGHTS["BATHROOM"] = config['Room']['NOISE_WEIGHTS']['浴室']
    NOISE_WEIGHTS["KITCHEN"] = config['Room']['NOISE_WEIGHTS']['厨房']
    NOISE_WEIGHTS["BALCONY"] = config['Room']['NOISE_WEIGHTS']['阳台']
    
    global NESTING_WEIGHTS
    NESTING_WEIGHTS[RoomType.LIVING_ROOM] = config['Room']['NESTING_WEIGHTS']['客厅']
    NESTING_WEIGHTS[RoomType.BEDROOM] = config['Room']['NESTING_WEIGHTS']['卧室']
    NESTING_WEIGHTS[RoomType.BATHROOM] = config['Room']['NESTING_WEIGHTS']['浴室']
    NESTING_WEIGHTS[RoomType.KITCHEN] = config['Room']['NESTING_WEIGHTS']['厨房']
    NESTING_WEIGHTS[RoomType.BALCONY] = config['Room']['NESTING_WEIGHTS']['阳台']
    
    global NESTING_PRIORITY
    NESTING_PRIORITY[RoomType.LIVING_ROOM] = config['Room']['NESTING_PRIORITY']['客厅']
    NESTING_PRIORITY[RoomType.BEDROOM] = config['Room']['NESTING_PRIORITY']['卧室']
    NESTING_PRIORITY[RoomType.BATHROOM] = config['Room']['NESTING_PRIORITY']['浴室']
    NESTING_PRIORITY[RoomType.KITCHEN] = config['Room']['NESTING_PRIORITY']['厨房']
    NESTING_PRIORITY[RoomType.BALCONY] = config['Room']['NESTING_PRIORITY']['阳台']
    
    return house_env, config
