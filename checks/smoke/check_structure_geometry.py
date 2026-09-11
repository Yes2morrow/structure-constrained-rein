"""Regression checks for precise geometry and real environment collisions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from copy import deepcopy
import math
import matplotlib
matplotlib.use('Agg')
from shapely.geometry import box
from core.envs.structure_geometry import column, wall, fixed_polygon, detect_cores, parameter_rows, build_structures
from core.envs import AdaptiveReuseEnv
from gui.config_store import load_config
from gui.structure_canvas import canvas_objects, parse_canvas, signature
from gui.adaptive_reuse_page import CONSTRAINT_STYLES


def test_geometry():
    c = column('c', 3, 4, 0.4, 0.6)
    assert c['rect'] == [2.8, 3.7, 3.2, 4.3]
    w = wall('w', 'shear_wall', 0, 0, 4, 0, .2, .1)
    assert fixed_polygon(w).bounds == (0, -.1, 4, .2)
    reverse = wall('r', 'load_bearing_wall', 4, 0, 0, 0, .1, .2)
    assert fixed_polygon(w).equals(fixed_polygon(reverse))
    diagonal = wall('d', 'shear_wall', 0, 0, 4, 4, .2, .1)
    assert math.isclose(fixed_polygon(diagonal).area, math.sqrt(32)*.3)
    assert not fixed_polygon(diagonal).intersects(box(0, 3, .5, 3.5))
    corners = [(10, 10), (14, 10), (14, 14), (10, 14), (10, 10)]
    ring = [wall(f'w{i}', 'shear_wall', *a, *b, .1, .2) for i, (a,b) in enumerate(zip(corners, corners[1:]))]
    assert len(detect_cores(ring)) == 1
    assert detect_cores(ring[:-1]) == []
    mixed = deepcopy(ring)
    mixed[-1]['type'] = 'load_bearing_wall'
    assert detect_cores(mixed) == []
    assert fixed_polygon(detect_cores(ring)[0]).contains(box(11,11,12,12))
    cc, ww, oo = parameter_rows([c, diagonal])
    rebuilt = build_structures(cc, ww, oo)
    assert fixed_polygon(rebuilt[1]).equals(fixed_polygon(diagonal))
    for args in [('bad', 0, 0, -1, 2), ('bad', float('nan'), 0, 1, 1)]:
        try:
            column(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid dimensions accepted')
    config = load_config('retrofit')
    config['ExistingBuilding']['boundary'] = [[0,0],[30,0],[30,30],[0,30]]
    config['ExistingBuilding']['fixed_objects'] = [diagonal]
    config['AdaptiveReuseEnvironment']['randomize_initial'] = False
    config['TargetSpaces'] = [config['TargetSpaces'][0]]
    env = AdaptiveReuseEnv(config)
    env.reset(seed=42)
    free = env.agent_spaces[0].copy(x1=0,y1=3,x2=.5,y2=3.5)
    blocked = free.copy(x1=1,y1=1,x2=2,y2=2)
    assert env._is_hard_valid(free,0), 'diagonal bounding box must not block free area'
    assert not env._is_hard_valid(blocked,0)
    env.render()
    canvas = canvas_objects([c, diagonal], config['ExistingBuilding']['boundary'], 900, 600, CONSTRAINT_STYLES)
    assert all(o['type']=='rect' for o in canvas)
    assert signature(parse_canvas(canvas,[c,diagonal],config['ExistingBuilding']['boundary'],900,600,'column'))==signature([c,diagonal])
    print('PASS: column dimensions, asymmetric/reversed/diagonal walls, closed/open/mixed rings, parameter roundtrip, actual environment collision, canvas locking')


if __name__ == '__main__':
    test_geometry()
