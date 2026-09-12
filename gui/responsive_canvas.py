"""Project-local responsive adapter; keeps Fabric world coordinates unchanged."""
from pathlib import Path
import shutil
import types
import streamlit.components.v1 as components
import streamlit_drawable_canvas as original


def _build():
    source=Path(original.__file__).parent/'frontend/build'
    target=Path(__file__).resolve().parents[1]/'runtime/responsive_canvas'
    script=(Path(__file__).parent/'responsive_canvas.js').read_text(encoding='utf-8')
    marker=target/'adapter.js'
    if not marker.exists() or marker.read_text(encoding='utf-8')!=script:
        shutil.copytree(source,target,dirs_exist_ok=True)
        html=(source/'index.html').read_text(encoding='utf-8')
        (target/'index.html').write_text(html.replace('</body>','<script src="adapter.js"></script></body>'),encoding='utf-8')
        marker.write_text(script,encoding='utf-8')
    return target


_context=dict(original.st_canvas.__globals__)
_context['_component_func']=components.declare_component('responsive_environment_canvas',path=str(_build()))
st_canvas=types.FunctionType(original.st_canvas.__code__,_context,original.st_canvas.__name__,original.st_canvas.__defaults__,original.st_canvas.__closure__)
