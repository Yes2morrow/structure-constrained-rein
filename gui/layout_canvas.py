import random

from PIL import Image, ImageDraw, ImageFont


ROOM_COLORS = {
    "卧室": "red",
    "厨房": "green",
    "浴室": "blue",
    "阳台": "orange",
    "客厅": "purple",
}

COLOR_TO_ROOM = {value: key for key, value in ROOM_COLORS.items()}


def normalize_color(color_str):
    """归一化颜色字符串。"""
    if not color_str:
        return ""

    color_str = color_str.lower()
    color_map = {
        "#ff0000": "red",
        "red": "red",
        "#008000": "green",
        "green": "green",
        "#0000ff": "blue",
        "blue": "blue",
        "#ffa500": "orange",
        "orange": "orange",
        "#800080": "purple",
        "purple": "purple",
        "#a52a2a": "brown",
        "brown": "brown",
    }
    return color_map.get(color_str, color_str)


def get_room_color(room_name):
    """获取房间颜色。"""
    if room_name in ROOM_COLORS:
        return ROOM_COLORS[room_name]

    random.seed(room_name)
    return "#%06x" % random.randint(0, 0xFFFFFF)


def compute_grid_size(boundary, priors, limit_area):
    """根据环境数据推导画布网格范围。"""
    max_coord = max([max(point) for point in boundary]) if boundary else 12

    if priors:
        max_prior = max([max(point) for point in priors if isinstance(point, list) and len(point) >= 2], default=max_coord)
        max_coord = max(max_coord, max_prior)

    if limit_area:
        max_limit = max([max(point) for point in limit_area if isinstance(point, list) and len(point) >= 2], default=max_coord)
        max_coord = max(max_coord, max_limit)

    return max(15, int(max_coord) + 2)


def env_to_canvas_coords(env_x, env_y, grid_size, width=800, height=800):
    """将环境坐标转换为画布坐标。"""
    canvas_x = (env_x / grid_size) * width
    canvas_y = height - (env_y / grid_size) * height
    return canvas_x, canvas_y


def canvas_to_env_coords(canvas_x, canvas_y, grid_size, width=800, height=800):
    """将画布坐标转换为环境坐标。"""
    env_x = (canvas_x / width) * grid_size
    env_y = grid_size - (canvas_y / height) * grid_size
    return round(env_x, 1), round(env_y, 1)


def create_boundary_image(grid_size, boundary_points, limit_area_points, width=800, height=800):
    """创建环境编辑背景图。"""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    step_x = width / grid_size
    step_y = height / grid_size
    for index in range(grid_size + 1):
        x = index * step_x
        y = index * step_y
        draw.line([(x, 0), (x, height)], fill="#E0E0E0", width=1)
        draw.line([(0, y), (width, y)], fill="#E0E0E0", width=1)

    def to_pixel(x, y):
        px = (x / grid_size) * width
        py = height - (y / grid_size) * height
        return px, py

    font_default = ImageFont.load_default()
    draw.text((5, height - 15), "(0,0)", fill="black", font=font_default)
    draw.text((width - 40, height - 15), f"({grid_size},0)", fill="black", font=font_default)
    draw.text((5, 5), f"(0,{grid_size})", fill="black", font=font_default)
    return image


def build_grid_objects(grid_size, canvas_width, canvas_height):
    """构建 1x1 浅色网格对象。"""
    objects = []
    step_x = canvas_width / grid_size
    step_y = canvas_height / grid_size

    for index in range(grid_size + 1):
        x = index * step_x
        y = index * step_y
        objects.append({
            "type": "line",
            "left": x,
            "top": 0,
            "x1": 0,
            "y1": 0,
            "x2": 0,
            "y2": canvas_height,
            "stroke": "#dfe5ee",
            "strokeWidth": 1,
            "selectable": False,
            "evented": False,
            "excludeFromExport": False,
        })
        objects.append({
            "type": "line",
            "left": 0,
            "top": y,
            "x1": 0,
            "y1": 0,
            "x2": canvas_width,
            "y2": 0,
            "stroke": "#dfe5ee",
            "strokeWidth": 1,
            "selectable": False,
            "evented": False,
            "excludeFromExport": False,
        })

    return objects


def build_initial_objects(boundary, doors, room_types, priors, limit_area, grid_size, canvas_width, canvas_height):
    """构建画布初始对象。"""
    initial_objects = build_grid_objects(grid_size, canvas_width, canvas_height)

    if boundary and len(boundary) > 2:
        boundary_pixels = [
            env_to_canvas_coords(point[0], point[1], grid_size, canvas_width, canvas_height)
            for point in boundary
        ]
        min_x = min(pixel[0] for pixel in boundary_pixels)
        min_y = min(pixel[1] for pixel in boundary_pixels)
        initial_objects.append({
            "type": "polygon",
            "left": min_x,
            "top": min_y,
            "points": [{"x": px - min_x, "y": py - min_y} for px, py in boundary_pixels],
            "fill": "rgba(0,0,0,0)",
            "stroke": "#2f3b52",
            "strokeWidth": 3,
            "strokeLineJoin": "round",
            "selectable": False,
            "evented": False,
            "objectCaching": False,
        })

    if limit_area and len(limit_area) > 2:
        limit_pixels = [
            env_to_canvas_coords(point[0], point[1], grid_size, canvas_width, canvas_height)
            for point in limit_area
        ]
        min_x = min(pixel[0] for pixel in limit_pixels)
        min_y = min(pixel[1] for pixel in limit_pixels)
        initial_objects.append({
            "type": "polygon",
            "left": min_x,
            "top": min_y,
            "points": [{"x": px - min_x, "y": py - min_y} for px, py in limit_pixels],
            "fill": "rgba(255,165,0,0.08)",
            "stroke": "orange",
            "strokeWidth": 2,
            "strokeLineJoin": "round",
            "selectable": False,
            "evented": False,
            "objectCaching": False,
        })

    for point in boundary:
        cx, cy = env_to_canvas_coords(point[0], point[1], grid_size, canvas_width, canvas_height)
        initial_objects.append({
            "type": "rect",
            "left": cx - 8,
            "top": cy - 8,
            "width": 16,
            "height": 16,
            "fill": "blue",
            "stroke": "black",
            "strokeWidth": 1,
            "angle": 0,
        })

    for point in doors:
        cx, cy = env_to_canvas_coords(point[0], point[1], grid_size, canvas_width, canvas_height)
        initial_objects.append({
            "type": "rect",
            "left": cx - 8,
            "top": cy - 8,
            "width": 16,
            "height": 16,
            "fill": "brown",
            "stroke": "black",
            "strokeWidth": 1,
            "angle": 0,
        })

    safe_priors = priors[:]
    if len(safe_priors) < len(room_types):
        for _ in range(len(room_types) - len(safe_priors)):
            safe_priors.append([grid_size / 2, grid_size / 2])
    elif len(safe_priors) > len(room_types):
        safe_priors = safe_priors[: len(room_types)]

    for index, room_name in enumerate(room_types):
        point = safe_priors[index]
        if isinstance(point, list) and len(point) >= 2:
            cx, cy = env_to_canvas_coords(point[0], point[1], grid_size, canvas_width, canvas_height)
            initial_objects.append({
                "type": "circle",
                "left": cx - 10,
                "top": cy - 10,
                "radius": 10,
                "width": 20,
                "height": 20,
                "fill": get_room_color(room_name),
                "stroke": "white",
                "strokeWidth": 2,
                "angle": 0,
            })

    for point in limit_area:
        if isinstance(point, list) and len(point) >= 2:
            cx, cy = env_to_canvas_coords(point[0], point[1], grid_size, canvas_width, canvas_height)
            initial_objects.append({
                "type": "rect",
                "left": cx - 6,
                "top": cy - 6,
                "width": 12,
                "height": 12,
                "fill": "orange",
                "stroke": "black",
                "strokeWidth": 1,
                "angle": 0,
            })

    return initial_objects


def parse_canvas_objects(objects, grid_size, canvas_width, canvas_height):
    """从画布结果反推环境对象。"""
    new_boundary = []
    new_doors = []
    new_priors = []
    new_room_types = []
    new_limit_area = []

    for obj in objects:
        fill = normalize_color(obj.get("fill"))
        scale_x = obj.get("scaleX", 1)
        scale_y = obj.get("scaleY", 1)
        center_x = obj["left"] + (obj["width"] * scale_x) / 2
        center_y = obj["top"] + (obj["height"] * scale_y) / 2
        env_x, env_y = canvas_to_env_coords(center_x, center_y, grid_size, canvas_width, canvas_height)

        if obj.get("type") == "rect":
            if fill == "blue":
                new_boundary.append([env_x, env_y])
            elif fill == "brown":
                new_doors.append([env_x, env_y])
            elif fill == "orange":
                new_limit_area.append([env_x, env_y])
        elif obj.get("type") == "circle":
            room_name = COLOR_TO_ROOM.get(fill)
            if room_name:
                new_room_types.append(room_name)
                new_priors.append([env_x, env_y])

    return new_boundary, new_doors, new_room_types, new_priors, new_limit_area
