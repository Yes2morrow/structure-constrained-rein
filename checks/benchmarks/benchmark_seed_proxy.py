"""Measured environment cost, not evidence of equal quality/convergence."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import time,json
import numpy as np
from gui.config_store import load_config
from core.seed_growth.environment import SeedLayoutEnv
from core.envs import AdaptiveReuseEnv

c=load_config('retrofit')
t=time.perf_counter(); env=SeedLayoutEnv(c); init=time.perf_counter()-t
rng=np.random.default_rng(42)
t=time.perf_counter()
for _ in range(30): env.step([int(rng.choice(a)) for a in env._allowed()])
moving=(time.perf_counter()-t)*1000/30
t=time.perf_counter()
for _ in range(30): env.step([4]*env.num_agents)
cached=(time.perf_counter()-t)*1000/30
old=AdaptiveReuseEnv(c); old.reset(seed=42)
t=time.perf_counter()
for _ in range(30): old.step([4]*old.num_agents)
old_ms=(time.perf_counter()-t)*1000/30
r=dict(case='retrofit',rooms=env.num_agents,grid_size=env.problem.grid_size,seed_proxy_init_seconds=init,
       seed_moving_ms=moving,seed_cached_stay_ms=cached,rectangle_stay_ms=old_ms,n=30,
       note='Single-process environment timings; different actions; not convergence or quality evidence')
(ROOT/'checks/evidence/seed_proxy_benchmark.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps(r))
