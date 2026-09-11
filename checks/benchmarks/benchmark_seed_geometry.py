"""Offline CP3 evidence: same seeds, proxy/continuous geometry comparison."""
import json
from pathlib import Path
import sys
import time
import argparse
from html import escape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import yaml
from core.seed_growth.contracts import build_problem, SeedSnapshot
from core.seed_growth.geometry import decode_layout
from core.seed_growth.proxy import LayoutProxy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['cp3','cp4'],default='cp3')
    args=parser.parse_args()
    config = yaml.safe_load((ROOT/'config/retrofit/config.yaml').read_text(encoding='utf-8'))
    problem = build_problem(config)
    seeds = {r.id:r.seed for r in problem.rooms}
    snapshot = SeedSnapshot.capture(problem,seeds,0)
    proxy = LayoutProxy(problem).evaluate(seeds)
    start = time.perf_counter()
    result = decode_layout(problem,snapshot,refine=args.stage=='cp4')
    seconds = time.perf_counter()-start
    polygons = result.pop('polygons')
    result['polygons'] = {k:list(p.exterior.coords) for k,p in polygons.items()}
    result['decode_seconds'] = seconds
    result['config_snapshot'] = config
    result['comparison'] = [dict(id=r.id, proxy_area=float(proxy['areas'][i]),
        precise_area=polygons[r.id].area, proxy_minus_precise=float(proxy['areas'][i])-polygons[r.id].area)
        for i,r in enumerate(problem.rooms)]
    result['proxy_relations'] = proxy['relations']
    out = ROOT/('checks/evidence/seed_geometry_baseline.json' if args.stage=='cp3' else 'checks/evidence/seed_edge_baseline.json')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    x0,y0,x1,y1 = problem.boundary.bounds
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0-1} {-y1-2} {x1-x0+2} {y1-y0+3}">',
           '<rect x="-10000" y="-10000" width="20000" height="20000" fill="white"/>']
    def draw(p,color):
        points = ' '.join(f'{x},{-y}' for x,y in p.exterior.coords)
        svg.append(f'<polygon points="{points}" fill="{color}" stroke="#263238" stroke-width="0.04"/>')
    draw(problem.boundary,'#f2f2f2')
    colors = ['#90caf9','#a5d6a7','#ffcc80','#ce93d8','#80cbc4','#ef9a9a']
    for i,(key,p) in enumerate(polygons.items()):
        draw(p,colors[i%len(colors)])
        x,y = seeds[key]
        svg.append(f'<circle cx="{x}" cy="{-y}" r="0.1" fill="black"/>')
        svg.append(f'<text x="{x+.15}" y="{-y}" font-size="0.3">{escape(key)}: {p.area:.2f} m²</text>')
    fixed = [problem.fixed] if problem.fixed.geom_type == 'Polygon' else list(problem.fixed.geoms)
    for p in fixed:
        if not p.is_empty:
            draw(p,'#455a64')
    svg.append(f'<text x="{x0}" y="{-y1-0.7}" font-size="0.4">{args.stage.upper()} precise geometry — {result["status"]}</text></svg>')
    out.with_suffix('.svg').write_text('\n'.join(svg),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],decode_seconds=seconds,
        errors=result['validation']['errors'],comparison=result['comparison'],
        retreat_attempts=result['retreat_attempts'],retreat_accepted=result['retreat_accepted']),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
