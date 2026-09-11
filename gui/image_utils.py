import os
import time

from PIL import Image, UnidentifiedImageError


def load_image_safely(image_path: str, retries: int = 3, delay: float = 0.12):
    """安全读取正在被训练进程更新的图片，避免读到半张图。"""
    if not image_path or not os.path.isfile(image_path):
        return None

    for attempt in range(retries):
        try:
            with Image.open(image_path) as image:
                image.load()
                return image.copy()
        except (UnidentifiedImageError, OSError, ValueError):
            if attempt == retries - 1:
                return None
            time.sleep(delay)

    return None
