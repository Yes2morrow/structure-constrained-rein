import base64
from io import BytesIO

import streamlit.elements.image as st_image
from PIL import Image


def apply_streamlit_patches() -> None:
    """修复第三方画布组件在新版 Streamlit 下的兼容问题。"""
    if hasattr(st_image, "image_to_url"):
        return

    def image_to_url(image, width, clamp, channels, output_format, image_id, allow_emoji=False):
        if isinstance(image, Image.Image):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        return ""

    st_image.image_to_url = image_to_url
