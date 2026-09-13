"""Five-action acceleration of the rectangular retrofit objectives."""
import numpy as np
import gym
from gym import spaces
from shapely.geometry import LineString,Point
from .contracts import build_problem, SeedSnapshot
from .proxy import LayoutProxy
from .objectives import RetrofitObjectives


class SeedLayoutEnv(gym.Env):
    actions=((0,1),(0,-1),(-1,0),(1,0),(0,0))

    def __init__(self, config):
        self.config=config
        self.problem=build_problem(config)
        self.proxy=LayoutProxy(self.problem)
        self.objectives=RetrofitObjectives(config,self.proxy)
        self.ids=self.proxy.active_ids
        self.active_indices=self.proxy.active_indices
        self.num_agents=len(self.ids)
        self.max_steps=int(config.get('Training',{}).get('max_steps',240))
        self.action_space=spaces.MultiDiscrete([5]*self.num_agents)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(self.num_agents,9+10*(len(self.problem.rooms)-1)+(3 if self.problem.residual_room else 0)),dtype=np.float32)
        self.reset()

    def reset(self,seed=None,options=None,**kwargs):
        super().reset(seed=seed)
        self.seeds={r.id:r.seed for r in self.problem.active_rooms}
        self.total_steps=0
        self.estimate=self.proxy.evaluate(self.seeds)
        return self.get_state(),self._info()

    def _candidate(self,identifier,action):
        x,y=self.seeds[identifier]; dx,dy=self.actions[action]
        return (round(x+dx*self.problem.grid_size,9),round(y+dy*self.problem.grid_size,9))

    def _allowed(self):
        occupied={k:self.proxy.cell(v) for k,v in self.seeds.items()}
        result=[]
        for identifier in self.ids:
            valid=[4]
            for action in range(4):
                point=self._candidate(identifier,action)
                try: cell=self.proxy.cell(point)
                except ValueError: continue
                path=LineString([self.seeds[identifier],point])
                if cell in [v for k,v in occupied.items() if k!=identifier]: continue
                if (self.problem.boundary.contains(Point(point)) and self.problem.boundary.covers(path)
                        and not self.problem.fixed.intersects(path)): valid.append(action)
            result.append(sorted(valid))
        return result

    def get_state(self):
        x0,y0,x1,y1=self.problem.boundary.bounds
        out=[]
        positions=self.node_positions()
        for i,room in enumerate(self.problem.rooms):
            if room.role == 'residual': continue
            x,y=positions[room.id]
            own=[(x-x0)/(x1-x0),(y-y0)/(y1-y0),self.estimate['areas'][i]/room.target_area,
                 room.target_area/self.problem.free_space.area,self.estimate['rectangularity'][i],
                 min(self.estimate['fragmentation'][i],5)/5,self.estimate['reachable_capacity'][i]/self.problem.free_space.area,
                 float(self.estimate['uncertain']),self.estimate['aspects'][i]/room.aspect_range[1]]
            for j,other in enumerate(self.problem.rooms):
                if i==j: continue
                xx,yy=positions[other.id]
                edges=[e for e in self.problem.relations if {e.source,e.target}=={room.id,other.id}]
                flags=[float(any(e.kind==kind for e in edges))
                       for kind in ('adjacent','connected','via_circulation','separate')]
                scale=max(x1-x0,y1-y0)
                thresholds=[max((getattr(e,key) or 0 for e in edges),default=0)/scale
                            for key in ('min_shared_length','min_clear_width','min_distance')]
                own.extend([(xx-x)/(x1-x0),(yy-y)/(y1-y0),*flags,*thresholds,
                            self.estimate['shared'][i,j]/scale])
            if self.problem.residual_room:
                public=self.proxy.ids.index(self.problem.residual_room.id)
                own.extend([self.estimate['areas'][public]/self.problem.free_space.area,
                            min(self.estimate['fragmentation'][public],5)/5,
                            float(np.all(self.estimate['owners'][self.proxy.reserved]==public))])
            out.append(own)
        return np.asarray(out,dtype=np.float32)

    def node_positions(self):
        positions=dict(self.seeds)
        residual=self.problem.residual_room
        if residual:
            i=self.proxy.ids.index(residual.id)
            yy,xx=np.nonzero(self.estimate['owners']==i)
            if len(xx):
                positions[residual.id]=(self.proxy.x0+(float(xx.mean())+.5)*self.proxy.h,
                                        self.proxy.y0+(float(yy.mean())+.5)*self.proxy.h)
            else: positions[residual.id]=residual.seed
        return positions

    def _info(self,done=False):
        return dict(allow_actions=self._allowed(),done_list=[done]*self.num_agents,
                    estimated=True,metrics=self.calculate_metrics())

    def calculate_metrics(self):
        scores=self.objectives.scores(self.estimate)
        adjacent=[e for e in self.estimate['relations'] if e['kind']=='adjacent']
        public=self.problem.residual_room
        public_index=self.proxy.ids.index(public.id) if public else None
        return dict(estimated=True,
                    public_area=None if public is None else self.estimate['areas'][public_index],
                    public_components=None if public is None else self.estimate['fragmentation'][public_index],
                    entrance_reserved=None if public is None else bool(np.all(self.estimate['owners'][self.proxy.reserved]==public_index)),
                    original_reuse=float(np.mean([s['original_reuse'] for s in scores])),
                    intervention_ratio=float(np.mean([s['intervention'] for s in scores])),
                    structure_alignment=float(np.mean([s['structure_alignment'] for s in scores])),
                    area_compliance=float(np.mean([r.area_range[0]<=a<=r.area_range[1] for r,a in zip(self.problem.rooms,self.estimate['areas'])])),
                    rectangularity=float(np.mean(self.estimate['rectangularity'])),uncertain=bool(self.estimate['uncertain']),
                    adjacency_estimate=float(np.mean([e['estimated'] for e in adjacent])) if adjacent else None,
                    adjacency_proximity=float(np.mean([e['proximity'] for e in adjacent])) if adjacent else None)

    def step(self,actions):
        if len(actions)!=self.num_agents: raise ValueError('动作数量与种子数量不同')
        allowed=self._allowed(); proposed=dict(self.seeds); invalid=[False]*self.num_agents
        for i,k in enumerate(self.ids):
            if actions[i] not in allowed[i]: invalid[i]=True
            else: proposed[k]=self._candidate(k,int(actions[i]))
        cells=[self.proxy.cell(proposed[k]) for k in self.ids]
        for i,k in enumerate(self.ids):
            if cells.count(cells[i])>1: proposed[k]=self.seeds[k]; invalid[i]=True
        old_relations=self.estimate['relations']
        self.seeds=proposed
        self.total_steps+=1
        self.estimate=self.proxy.evaluate(self.seeds)
        rewards,self.reward_components=self.objectives.rewards(self.estimate,invalid,old_relations)
        done=self.total_steps>=self.max_steps
        return self.get_state(),rewards,done,self._info(done)

    def snapshot(self,completed_episodes,step=0):
        return SeedSnapshot.capture(self.problem,self.seeds,completed_episodes,step)
