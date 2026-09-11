"""Cheap, conservative raster estimates; never a precise floorplan decoder."""
from collections import OrderedDict, deque
import math
import numpy as np
from shapely.geometry import box


def label(mask):
    labels=np.zeros(mask.shape,dtype=np.int32); count=0
    rows,cols=mask.shape
    for y,x in zip(*np.nonzero(mask)):
        if labels[y,x]: continue
        count+=1; labels[y,x]=count; queue=[(y,x)]
        while queue:
            yy,xx=queue.pop()
            for dy,dx in ((0,1),(1,0),(0,-1),(-1,0)):
                ny,nx=yy+dy,xx+dx
                if 0<=ny<rows and 0<=nx<cols and mask[ny,nx] and not labels[ny,nx]:
                    labels[ny,nx]=count; queue.append((ny,nx))
    return labels,count


class LayoutProxy:
    def __init__(self, problem, cache_size=128):
        self.problem = problem
        self.h = problem.grid_size
        self.x0,self.y0,x2,y2 = problem.boundary.bounds
        self.cols = math.ceil((x2-self.x0)/self.h)
        self.rows = math.ceil((y2-self.y0)/self.h)
        if self.rows*self.cols > 100000:
            raise ValueError('代理网格超过 100000 格，请增大 SeedGrowth.grid_size')
        self.free = np.zeros((self.rows,self.cols),dtype=bool)
        for y in range(self.rows):
            for x in range(self.cols):
                cell=box(self.x0+x*self.h,self.y0+y*self.h,self.x0+(x+1)*self.h,self.y0+(y+1)*self.h)
                self.free[y,x]=problem.boundary.covers(cell) and cell.intersection(problem.fixed).area<=1e-12
        a,b=[],[]
        for y,x in zip(*np.nonzero(self.free)):
            for dy,dx in ((0,1),(1,0),(0,-1),(-1,0)):
                yy,xx=y+dy,x+dx
                if 0<=yy<self.rows and 0<=xx<self.cols and self.free[yy,xx]:
                    a.append(y*self.cols+x); b.append(yy*self.cols+xx)
        count=self.rows*self.cols
        self.graph=[[] for _ in range(count)]
        for first,second in zip(a,b): self.graph[first].append(second)
        self.components,self.component_count=label(self.free)
        self.distance_cache=OrderedDict()
        self.result_cache=OrderedDict()
        self.cache_size=cache_size
        self.ids=tuple(r.id for r in problem.rooms)

    def cell(self, point):
        x,y=point
        row,col=int(math.floor((y-self.y0)/self.h)),int(math.floor((x-self.x0)/self.h))
        if not (0<=row<self.rows and 0<=col<self.cols and self.free[row,col]):
            raise ValueError('种子所在粗网格被结构占用或越界；请调整种子/细化网格，不能静默穿墙')
        return row*self.cols+col

    def _remember(self, cache, key, value):
        cache[key]=value
        cache.move_to_end(key)
        while len(cache)>self.cache_size: cache.popitem(last=False)
        return value

    def distances(self, cell):
        if cell in self.distance_cache:
            self.distance_cache.move_to_end(cell)
            return self.distance_cache[cell]
        dist=np.full(len(self.graph),np.inf); dist[cell]=0; queue=deque([cell])
        while queue:
            current=queue.popleft()
            for neighbor in self.graph[current]:
                if np.isinf(dist[neighbor]):
                    dist[neighbor]=dist[current]+self.h; queue.append(neighbor)
        dist.setflags(write=False)
        return self._remember(self.distance_cache,cell,dist)

    def evaluate(self, seeds):
        cells=tuple(self.cell(seeds[k]) for k in self.ids)
        if len(set(cells))!=len(cells): raise ValueError('种子不能占用同一个代理网格')
        if cells in self.result_cache:
            self.result_cache.move_to_end(cells)
            return self.result_cache[cells]
        dist=np.stack([self.distances(c) for c in cells])
        targets=np.array([r.target_area for r in self.problem.rooms])
        costs=dist/np.sqrt(targets[:,None])
        owners=np.argmin(costs,axis=0)
        owners[~np.isfinite(costs.min(axis=0))]=-1
        # Bounded capacity estimate, not polygon growth: retain closest owned cells.
        for i,c in enumerate(cells):
            owners[c]=i
        for i,room in enumerate(self.problem.rooms):
            candidates=np.flatnonzero(owners==i)
            limit=max(1,int(math.floor(room.target_area/self.h**2)))
            if len(candidates)>limit:
                order=np.argsort(dist[i,candidates],kind='stable')
                owners[candidates[order[limit:]]]=-1
        owners=owners.reshape(self.rows,self.cols)
        areas=[]; rectangularity=[]; fragmentation=[]; capacity=[]; aspects=[]
        for i,c in enumerate(cells):
            yy,xx=np.nonzero(owners==i)
            areas.append(len(xx)*self.h**2)
            rectangularity.append(len(xx)/((xx.max()-xx.min()+1)*(yy.max()-yy.min()+1)) if len(xx) else 0.)
            aspects.append(max(xx.max()-xx.min()+1,yy.max()-yy.min()+1)/min(xx.max()-xx.min()+1,yy.max()-yy.min()+1) if len(xx) else 0.)
            fragmentation.append(label(owners==i)[1])
            capacity.append(np.isfinite(dist[i]).sum()*self.h**2)
        shared=np.zeros((len(cells),len(cells)))
        for first,second in ((owners[:,:-1],owners[:,1:]),(owners[:-1,:],owners[1:,:])):
            valid=(first>=0)&(second>=0)&(first!=second)
            for i,j in zip(first[valid],second[valid]):
                shared[i,j]+=self.h; shared[j,i]+=self.h
        relations=[]
        for edge in self.problem.relations:
            i,j=self.ids.index(edge.source),self.ids.index(edge.target)
            proximity = None
            if edge.kind=='adjacent':
                value=min(1.,shared[i,j]/(edge.min_shared_length or self.h))
                contact_scale=(math.sqrt(self.problem.rooms[i].target_area)+math.sqrt(self.problem.rooms[j].target_area))/2
                distance=dist[i,cells[j]]
                proximity=float(math.exp(-max(0.,distance-contact_scale)/contact_scale)) if math.isfinite(distance) else 0.
            elif edge.kind=='separate':
                # Empty regions or seeds alone cannot prove room separation.
                aa=np.argwhere(owners==i); bb=np.argwhere(owners==j)
                gap=min((np.linalg.norm(np.maximum(np.abs(bb-p)-1,0),axis=1).min() for p in aa),default=0)*self.h if len(bb) else 0
                value=min(1.,gap/(edge.min_distance or self.h))
            else:
                value=None  # No opening/corridor proof: explicitly unknown.
            relations.append(dict(source=edge.source,target=edge.target,kind=edge.kind,estimated=value,
                                  proximity=proximity,verified=False))
        owners.setflags(write=False); shared.setflags(write=False)
        result=dict(estimated=True,owners=owners,areas=tuple(areas),rectangularity=tuple(rectangularity),aspects=tuple(aspects),
                    fragmentation=tuple(fragmentation),reachable_capacity=tuple(capacity),shared=shared,
                    relations=tuple(relations),uncertain=any(f!=1 for f in fragmentation) or any(r['estimated'] is None or r['kind']=='adjacent' for r in relations))
        return self._remember(self.result_cache,cells,result)
