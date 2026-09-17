"""Bounded joint neighbourhood search on an exact, structure-aware cell graph.

The legacy beam search supplies feasible starts only. Every subsequent action is
an interface and a graph radius; local cuts reassign BOTH units and circulation.
No cell is discarded and no final plan is repaired by retaining its largest part.
"""
from dataclasses import dataclass, replace
from collections import Counter
import math
import time

import numpy as np
from shapely.geometry import box, LineString
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .circulation import CirculationLayout
from .contracts import target_areas_for_area
from .quality import structural_axes, validate_partition, settings, facade
from .structured import candidate_partitions, unit_doors, lines


def options(problem):
    q = dict(enabled=True, cell_size=1.5, radii=[2, 4, 8], max_cells=160,
             max_area=100., max_graph_cells=5000, max_actions=36,
             proposals=32, steps=8, seconds_per_action=1.5, seed=42)
    q.update(problem.search_settings)
    for k in ('cell_size', 'max_area', 'seconds_per_action'):
        if not math.isfinite(float(q[k])) or float(q[k]) <= 0:
            raise ValueError(f'joint_search.{k} must be positive and finite')
    for k in ('max_cells', 'max_graph_cells', 'max_actions', 'proposals', 'steps'):
        if isinstance(q[k], bool) or int(q[k]) != q[k] or q[k] <= 0:
            raise ValueError(f'joint_search.{k} must be a positive integer')
    if not q['radii'] or any(int(r) != r or r < 1 for r in q['radii']):
        raise ValueError('joint_search.radii must contain positive integers')
    return q


def objective(report):
    """Scale-normalized score, always subordinate to independent feasibility."""
    units = list(report['units'].values())
    free = sum(u['area'] for u in units) + report['corridor_area']
    return (10 - 8 * np.mean([u['area_error_ratio'] for u in units])
            - 6 * report['corridor_area'] / free
            - .08 * np.mean([max(0, u['corners'] - 4) for u in units])
            - .12 * np.mean([u.get('short_edges', 0) for u in units])
            + .4 * min(2., min(u['facade_length'] / u['required_facade_length'] for u in units))
            + .6 * min(u.get('daylight_coverage_proxy', 1.) for u in units)
            + .3 * report['structure_alignment_ratio'])


@dataclass(frozen=True)
class Action:
    a: int
    b: int
    radius: int


class JointPartitionEnv:
    """Single designer; labels 0=public circulation, 1..N=dwelling units."""
    def __init__(self, problem, initial=None):
        self.problem = problem
        self.q = options(problem)
        self.initial = initial if initial is not None else candidate_partitions(problem)[0][0]
        self.ids = list(self.initial.unit_polygons)
        self.free = problem.boundary.difference(problem.fixed_union)
        coords = [set(a) for a in structural_axes(problem)]
        for p in [problem.boundary, self.initial.corridor, *self.initial.unit_polygons.values()]:
            for x, y in p.exterior.coords:
                coords[0].add(x); coords[1].add(y)
        for d in (0, 1):
            lo, hi = problem.boundary.bounds[d], problem.boundary.bounds[d + 2]
            coords[d].update(np.arange(lo, hi, self.q['cell_size']).tolist())
        self.axes = [sorted(set(round(v, 7) for v in a)) for a in coords]
        nx, ny = len(self.axes[0]) - 1, len(self.axes[1]) - 1
        if nx * ny > self.q['max_graph_cells'] * 4:
            raise ValueError('Cell graph exceeds budget; increase joint_search.cell_size')
        self.cells = []
        for x0, x1 in zip(self.axes[0], self.axes[0][1:]):
            for y0, y1 in zip(self.axes[1], self.axes[1][1:]):
                g = box(x0, y0, x1, y1).intersection(self.free)
                parts = [g] if g.geom_type == 'Polygon' else getattr(g, 'geoms', [])
                self.cells.extend(p for p in parts if p.geom_type == 'Polygon' and p.area > 1e-9)
        if len(self.cells) > self.q['max_graph_cells']:
            raise ValueError('Cell graph exceeds joint_search.max_graph_cells')
        self.points = [p.representative_point() for p in self.cells]
        tree = STRtree(self.cells)
        self.edges = []
        self.neighbours = [set() for _ in self.cells]
        for i, p in enumerate(self.cells):
            for j in tree.query(p):
                j = int(j)
                if j <= i or p.boundary.intersection(self.cells[j].boundary).length < 1e-7:
                    continue
                self.edges.append((i, j))
                self.neighbours[i].add(j); self.neighbours[j].add(i)
        if self.free.symmetric_difference(unary_union(self.cells)).area > 1e-6:
            raise ValueError('Cell graph lost allocatable geometry')
        self.reset()

    def reset(self):
        self.current = self.initial
        self.report = validate_partition(self.problem, self.current)
        if not self.report['valid']:
            raise ValueError('Initial layout failed validation: ' + str(self.report['errors']))
        self.best, self.best_report = self.current, self.report
        self.history = []
        return self.current

    def owners(self):
        polygons = [self.current.corridor, *[self.current.unit_polygons[u] for u in self.ids]]
        return np.asarray([next(k for k, p in enumerate(polygons) if p.covers(pt))
                           for pt in self.points], dtype=np.int64)

    def actions(self):
        owner = self.owners()
        groups = {}
        for a, b in self.edges:
            if owner[a] != owner[b]:
                pair = tuple(sorted((int(owner[a]), int(owner[b]))))
                groups.setdefault(pair, []).append((a, b))
        # Round-robin across interfaces and spread anchors spatially within each.
        slots = max(1, self.q['max_actions'] // len(self.q['radii']))
        anchors = []
        for k in range(slots):
            for pair in sorted(groups):
                edges = groups[pair]
                at = round(k * (len(edges) - 1) / max(1, slots - 1))
                if edges[at] not in anchors:
                    anchors.append(edges[at])
                if len(anchors) >= slots:
                    break
            if len(anchors) >= slots:
                break
        return [Action(a, b, r) for a, b in anchors for r in self.q['radii']
                if self.q['max_cells']>=2 and self.cells[a].area+self.cells[b].area<=self.q['max_area']][:self.q['max_actions']]

    def neighbourhood(self, action):
        selected = set()
        front = {action.a, action.b}
        area = 0.
        for _ in range(action.radius + 1):
            nxt = set()
            for i in sorted(front):
                if i in selected:
                    continue
                if len(selected) >= self.q['max_cells'] or area + self.cells[i].area > self.q['max_area']:
                    continue
                selected.add(i); area += self.cells[i].area
                nxt.update(self.neighbours[i] - selected)
            front = nxt
        return sorted(selected), unary_union([self.cells[i] for i in sorted(selected)])

    def observation(self, actions):
        owner = self.owners()
        bounds = self.problem.boundary.bounds
        span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        exterior = facade(self.problem)
        q = settings(self.problem)
        door_lines={d.unit_id:LineString(d.points) for d in self.current.doors}
        all_doors=unary_union(list(door_lines.values()))
        xs = []
        for i, (cell, pt) in enumerate(zip(self.cells, self.points)):
            unit = self.report['units'].get(self.ids[owner[i] - 1]) if owner[i] else None
            xs.append([(pt.x - bounds[0]) / span, (pt.y - bounds[1]) / span,
                       cell.area / self.free.area, float(owner[i] == 0),
                       cell.boundary.intersection(exterior).length / max(cell.length, 1e-7),
                       cell.distance(self.problem.traffic_core) / span,
                       0. if unit is None else (unit['area'] / unit['target_area'] - 1),
                       0. if unit is None else min(2., unit['facade_length'] / unit['required_facade_length']),
                       0. if unit is None else unit['corners'] / q['max_corners'],
                       0. if unit is None else unit['daylight_coverage_proxy'],
                       self.report['corridor_area'] / self.free.area,
                       q['min_unit_width'] / span,
                       cell.distance(door_lines[self.ids[owner[i]-1]] if owner[i] else all_doors)/span,
                       float(cell.distance(all_doors)<1e-7),
                       cell.distance(self.current.opening)/span])
        return np.asarray(xs, dtype=np.float32), self.edges

    def _materialize(self, polygons, window):
        corridor = polygons[0].buffer(0).simplify(1e-7, preserve_topology=True)
        units = {u: polygons[i + 1].buffer(0).simplify(1e-7, preserve_topology=True)
                 for i, u in enumerate(self.ids)}
        if any(p.is_empty or p.geom_type != 'Polygon' for p in [corridor, *units.values()]):
            raise ValueError('disconnected')
        target, scale = target_areas_for_area(self.problem, sum(p.area for p in units.values()))
        candidate = replace(self.current, corridor=corridor, unit_polygons=units,
                            allocatable_space=self.free.difference(corridor),
                            target_areas=dict(zip(self.ids, target)), area_scale=scale)
        # Reject geometry/area/daylight before the more expensive door search.
        pre = validate_partition(self.problem, candidate)
        if any('door' not in err for err in pre['errors']):
            raise ValueError('geometry_or_quality')
        locked = [d for d in self.current.doors if LineString(d.points).distance(window) > 1e-7]
        circulation = CirculationLayout(corridor, candidate.opening, candidate.opening_side)
        candidate = replace(candidate, doors=unit_doors(self.problem, circulation, units, locked, window))
        report = validate_partition(self.problem, candidate)
        if not report['valid']:
            raise ValueError('door_or_validation')
        return candidate, report

    def step(self, action, stop_file=None):
        from pathlib import Path
        start = time.perf_counter()
        previous = objective(self.report)
        indices, window = self.neighbourhood(action)
        owner = self.owners()
        labels = sorted({int(owner[i]) for i in indices})
        polygons = [self.current.corridor, *[self.current.unit_polygons[u] for u in self.ids]]
        pair = (int(owner[action.a]), int(owner[action.b]))
        anchor = self.cells[action.a].union(self.cells[action.b]).centroid
        proposals = 0; accepted = 0; failures = Counter(); timeout = False; stopped = False
        chosen, chosen_report = self.current, self.report
        # One shared operation for units AND corridor: recut their local union.
        cut_options = []
        for dim in (0, 1):
            middle = (anchor.x, anchor.y)[dim]
            values = [v for v in self.axes[dim] if window.bounds[dim] + 1e-7 < v < window.bounds[dim+2] - 1e-7]
            values.sort(key=lambda v: abs(v - middle))
            cut_options.extend((dim, v, flip) for v in values[:8] for flip in (False, True))
        # Interleave dimensions so a bounded budget doesn't eliminate one axis.
        cut_options.sort(key=lambda x: abs(x[1] - (anchor.x, anchor.y)[x[0]]))
        seen = set()
        def edits():
            # Sweep a shared edge without reassigning unrelated parts of either
            # region. When shrinking circulation, distribute the released strip
            # among all adjacent units within this same action.
            contact=polygons[pair[0]].boundary.intersection(polygons[pair[1]].boundary).intersection(window)
            for line in lines(contact):
                for p,r in zip(line.coords,list(line.coords)[1:]):
                    dim=0 if abs(p[0]-r[0])<1e-7 else 1
                    if abs(p[1-dim]-r[1-dim])<1e-7: continue
                    base=p[dim]
                    values=sorted(self.axes[dim],key=lambda v:abs(v-base))
                    for v in [v for v in values if 1e-6<abs(v-base)<=self.problem.profile.corridor_width][:4]:
                        lo,hi=sorted((base,v)); t0,t1=sorted((p[1-dim],r[1-dim]))
                        strip=box(lo,t0,hi,t1) if dim==0 else box(t0,lo,t1,hi)
                        if 0 in pair:
                            # A single parallel trim may affect multiple doors
                            # and units, avoiding a forced one-unit-at-a-time move.
                            wb=window.bounds
                            band=box(lo,wb[1],hi,wb[3]) if dim==0 else box(wb[0],lo,wb[2],hi)
                            release=polygons[0].intersection(band).intersection(window)
                            if release.area>1e-7:
                                pieces=list(polygons); pieces[0]=pieces[0].difference(release)
                                pending=[self.cells[i].intersection(release) for i in indices]
                                pending=[g for g in pending if g.area>1e-7]
                                while pending:
                                    remaining=[]; changed=False
                                    for g in pending:
                                        contacts=[(pieces[k].boundary.intersection(g.boundary).length,k) for k in labels if k]
                                        length,k=max(contacts,default=(0,0))
                                        if length>1e-7:
                                            pieces[k]=pieces[k].union(g); changed=True
                                        else: remaining.append(g)
                                    if not changed: break
                                    pending=remaining
                                if not pending: yield pieces
                        for donor,recipient in (pair,pair[::-1]):
                            take=polygons[donor].intersection(strip).intersection(window)
                            if take.area<1e-7: continue
                            pieces=list(polygons)
                            pieces[donor]=pieces[donor].difference(take)
                            pieces[recipient]=pieces[recipient].union(take)
                            yield pieces
            for dim,cut,flip in cut_options:
                a,b=pair[::-1] if flip else pair
                local=polygons[a].union(polygons[b]).intersection(window)
                bx,by,ex,ey=self.problem.boundary.bounds
                mask=box(bx-1,by-1,cut,ey+1) if dim==0 else box(bx-1,by-1,ex+1,cut)
                pieces=list(polygons)
                pieces[a]=polygons[a].difference(window).union(local.intersection(mask))
                pieces[b]=polygons[b].difference(window).union(local.difference(mask))
                yield pieces
        for pieces in edits():
            if stop_file and Path(stop_file).exists(): stopped = True; break
            if proposals >= self.q['proposals']: break
            if time.perf_counter() - start >= self.q['seconds_per_action']: timeout = True; break
            key = tuple(p.wkb for p in pieces)
            if key in seen: continue
            seen.add(key); proposals += 1
            try:
                result, report = self._materialize(pieces, window)
            except ValueError as exc:
                failures[str(exc)] += 1; continue
            accepted += 1
            if objective(report) > objective(chosen_report) + 1e-9:
                chosen, chosen_report = result, report
        self.current, self.report = chosen, chosen_report
        if objective(self.report) > objective(self.best_report):
            self.best, self.best_report = self.current, self.report
        reward = float(objective(self.report) - previous)
        info = dict(step=len(self.history)+1, anchor=[action.a, action.b], radius=action.radius,
                    cells=len(indices), area=window.area, labels=labels, proposals=proposals,
                    accepted=accepted, failures=dict(failures), timed_out=timeout, stopped=stopped,
                    changed=reward > 1e-9, reward=reward, score=float(objective(self.report)),
                    corridor_area=self.current.corridor.area, seconds=time.perf_counter()-start)
        self.history.append(info)
        return self.current, reward, info


def search_joint(problem, initial=None, steps=None, stop_file=None):
    env = JointPartitionEnv(problem, initial)
    if steps is not None and (int(steps)!=steps or steps<1):
        raise ValueError('search steps must be a positive integer')
    rng = np.random.default_rng(env.q['seed'])
    used = set()
    for _ in range(steps or env.q['steps']):
        if stop_file:
            from pathlib import Path
            if Path(stop_file).exists(): break
        actions = env.actions()
        choices = [a for a in actions if a not in used]
        if not choices: break
        action = choices[int(rng.integers(len(choices)))]
        used.add(action)
        _, _, info = env.step(action, stop_file)
        if info['changed']: used.clear()
        if info['stopped']: break
    return env.best, env.best_report, dict(algorithm='joint_graph_local_search',
        graph_cells=len(env.cells), graph_edges=len(env.edges), history=env.history,
        initial_score=float(objective(validate_partition(problem, env.initial))),
        final_score=float(objective(env.best_report)))
