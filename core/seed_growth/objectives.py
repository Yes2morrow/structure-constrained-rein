"""Raster estimates of the rectangular retrofit method's shared objectives.

Original rectangles are fixed references, never replaced by a moving seed.
Only reference/cell intersections are computed at initialization; training uses
array sums and does not invoke the precise polygon decoder.
"""
import numpy as np
from shapely.geometry import box


DEFAULT_WEIGHTS = dict(area=3., shape=1., adjacency=1.5, separation=1.,
                       structure_alignment=.8, original_reuse=1.5,
                       intervention=1., invalid_action=4., team_coordination=.5)


def precise_retrofit_metrics(config, polygons):
    """Report the original method's overlap measures on final polygons."""
    originals = {str(r['id']): r['rect'] for r in config['ExistingBuilding'].get('original_spaces', [])}
    rows = []
    for room in config['TargetSpaces']:
        identifier = str(room['id'])
        rect = originals.get(identifier, room.get('initial_rect'))
        polygon = polygons.get(identifier)
        if rect is None or polygon is None or polygon.is_empty:
            continue
        reference = box(*rect)
        overlap = polygon.intersection(reference).area
        rows.append(dict(id=identifier,
            original_reuse=overlap/max(reference.area+polygon.area-overlap, 1e-9),
            intervention_ratio=(reference.area+polygon.area-2*overlap)/max(reference.area,polygon.area,1e-9)))
    return dict(estimated=False, rooms=rows,
        original_reuse=float(np.mean([r['original_reuse'] for r in rows])) if rows else None,
        intervention_ratio=float(np.mean([r['intervention_ratio'] for r in rows])) if rows else None)


class RetrofitObjectives:
    def __init__(self, config, proxy):
        self.proxy = proxy
        self.weights = dict(DEFAULT_WEIGHTS, **config.get('RewardWeights', {}))
        self.grid = float(config.get('AdaptiveReuseEnvironment', {}).get('grid_size', proxy.h))
        originals = {str(r['id']): r['rect'] for r in config['ExistingBuilding'].get('original_spaces', [])}
        self.references = []
        self.overlaps = []
        for room in config['TargetSpaces']:
            rect = originals.get(str(room['id']), room.get('initial_rect'))
            reference = box(*rect) if rect is not None else None
            overlap = np.zeros(proxy.free.shape)
            if reference is not None:
                for y, x in zip(*np.nonzero(proxy.free)):
                    cell = box(proxy.x0+x*proxy.h, proxy.y0+y*proxy.h,
                               proxy.x0+(x+1)*proxy.h, proxy.y0+(y+1)*proxy.h)
                    overlap[y, x] = reference.intersection(cell).area
            self.references.append(reference)
            self.overlaps.append(overlap)

    def scores(self, estimate, old_relations=()):
        scores = []
        for i, room in enumerate(self.proxy.problem.rooms):
            area = estimate['areas'][i]
            aspect = estimate['aspects'][i]
            aspect_penalty = max(room.aspect_range[0]-aspect, 0, aspect-room.aspect_range[1]) / max(room.aspect_range[1]-room.aspect_range[0], .5)
            mask = estimate['owners'] == i
            ref = self.references[i]
            overlap = float(self.overlaps[i][mask].sum())
            reuse = overlap / max(ref.area+area-overlap, 1e-9) if ref is not None else 0.
            intervention = (ref.area+area-2*overlap)/max(ref.area, area, 1e-9) if ref is not None else 0.
            yy, xx = np.nonzero(mask)
            values = np.array([xx.min(), xx.max()+1, yy.min(), yy.max()+1])*self.proxy.h/self.grid if len(xx) else np.array([.5])
            alignment = max(0., 1.-2*float(np.mean(np.abs(values-np.round(values)))))
            adjacent, separate = [], []
            for j, edge in enumerate(estimate['relations']):
                if room.id not in (edge['source'], edge['target']) or edge['estimated'] is None:
                    continue
                if edge['kind'] == 'separate':
                    separate.append(edge['estimated'])
                else:
                    value = edge['estimated']
                    if edge['kind'] == 'adjacent':
                        value = .75*value+.25*edge['proximity']
                        if old_relations:
                            value += 2*(edge['proximity']-old_relations[j]['proximity'])
                    adjacent.append(value)
            if room.role == 'residual':
                # Public space is judged by usable area and connectivity, not
                # a rectangular target or similarity to its former footprint.
                scores.append(dict(area=min(1.,area/room.area_range[0]),
                    shape=1.-max(0,estimate['fragmentation'][i]-1) if area else -1.,
                    adjacency=float(np.mean(adjacent)) if adjacent else 0.,
                    separation=float(np.mean(separate)) if separate else 0.,
                    structure_alignment=0.,original_reuse=0.,intervention=0.))
                continue
            scores.append(dict(area=1-min(abs(area-room.target_area)/room.target_area, 1.5),
                shape=estimate['rectangularity'][i]-aspect_penalty-max(0, estimate['fragmentation'][i]-1),
                adjacency=float(np.mean(adjacent)) if adjacent else 0.,
                separation=float(np.mean(separate)) if separate else 0.,
                structure_alignment=alignment, original_reuse=reuse, intervention=intervention))
        return scores

    def rewards(self, estimate, invalid, old_relations):
        scores = self.scores(estimate, old_relations)
        team = float(np.mean([(s['area']+s['shape']+s['original_reuse'])/3 for s in scores]))
        components = []
        public = self.proxy.problem.residual_room
        public_score = scores[self.proxy.ids.index(public.id)] if public else None
        for agent_index,i in enumerate(self.proxy.active_indices):
            score=scores[i]
            terms = {k: self.weights[k]*v*(-1 if k == 'intervention' else 1) for k, v in score.items()}
            terms.update(invalid_action=-self.weights['invalid_action']*invalid[agent_index],
                         team_coordination=self.weights['team_coordination']*team)
            if public_score:
                terms['public_space'] = self.weights['team_coordination']*(public_score['area']+public_score['shape']+public_score['adjacency']+public_score['separation'])
            components.append(terms)
        return [float(sum(c.values())) for c in components], components
