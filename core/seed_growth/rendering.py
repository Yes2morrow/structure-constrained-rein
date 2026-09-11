"""Render committed continuous polygons; never rerun growth or move seeds."""
from pathlib import Path
import json
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import PathPatch
from matplotlib.path import Path as PlotPath
from shapely.geometry import shape
from shapely.geometry.polygon import orient
from .contracts import build_problem
from .shape_rules import polygon_parts


def render_result(config, result, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    problem=build_problem(config)
    with matplotlib.rc_context({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False}):
        fig=Figure(figsize=(14,9),layout='constrained'); FigureCanvasAgg(fig)
        ax=fig.subplots()
        def draw(geometry,color):
            for poly in polygon_parts(geometry):
                poly=orient(poly,sign=1); vertices=[]; codes=[]
                for ring in [poly.exterior,*poly.interiors]:
                    coords=list(ring.coords); vertices.extend(coords)
                    codes.extend([PlotPath.MOVETO]+[PlotPath.LINETO]*(len(coords)-2)+[PlotPath.CLOSEPOLY])
                ax.add_patch(PathPatch(PlotPath(vertices,codes),facecolor=color,edgecolor='#263238',lw=1.3))
        draw(problem.boundary,'#fafafa')
        names={r['id']:r.get('name',r['id']) for r in config['TargetSpaces']}
        for i,(key,value) in enumerate(result['polygons'].items()):
            poly=shape(value); draw(poly,matplotlib.colormaps['Pastel1'](i%9))
            p=poly.representative_point()
            ax.text(p.x,p.y,f'{names[key]}\n{key}\n{poly.area:.2f} m²',ha='center',va='center',fontsize=9)
        draw(problem.fixed,'#455a64')
        seeds=result['snapshot']['seeds']
        for key,(x,y) in seeds.items(): ax.scatter(x,y,s=36,c='black',marker='+',zorder=5)
        relations=result['validation']['relations']
        for edge in relations:
            if edge['kind']!='adjacent': continue
            a,b=seeds[edge['source']],seeds[edge['target']]
            ax.plot([a[0],b[0]],[a[1],b[1]],'--',color='#238443' if edge['satisfied'] else '#d7301f',alpha=.8,lw=1.5)
        adjacent=[e for e in relations if e['kind']=='adjacent']
        satisfied=sum(e['satisfied'] is True for e in adjacent)
        snap=result['snapshot']
        ax.set_title(f"种子生长精确成图 · {snap['completed_episodes']} 轮 · {result['status']}\n"
                     f"邻接满足 {satisfied}/{len(adjacent)}；黑色十字为种子，虚线连接固定图节点（绿：满足，红：未满足）",fontsize=13)
        ax.set_xlabel('X / m'); ax.set_ylabel('Y / m'); ax.set_aspect('equal'); ax.autoscale_view()
        ax.grid(alpha=.12); ax.set_axisbelow(True)
        for suffix in ('png','svg'):
            fig.savefig(output/f'layout.{suffix}',dpi=180)
        (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return output/'layout.png'
