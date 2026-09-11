"""Reproducible CP4 examples, plotted only from computed polygons."""
from pathlib import Path
import sys
import json
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import box
from checks.test_seed_edge_growth import wall, problem_for
from core.seed_growth.contracts import RoomSpec,SeedSnapshot
from core.seed_growth.geometry import decode_layout
from core.seed_growth.edge_growth import refine_layout
from core.seed_growth.validation import validate_layout


def main():
    room=RoomSpec('r',(1,1),10.5,(10,11),(1,3),min_width=2)
    problem=problem_for(None,wall(),room)
    boundary=box(.5,0,4,5)
    problem=replace(problem,boundary=boundary,free_space=boundary.difference(wall()))
    snap=SeedSnapshot.capture(problem,{'r':room.seed},250)
    old=decode_layout(problem,snap,refine=False)
    new=decode_layout(problem,snap)
    samples=[('CP3: whole edge stops',problem,old['polygons']['r'],old['validation']),
             ('CP4: A, capped by target area',problem,new['polygons']['r'],new['validation'])]
    room_b=replace(room,target_area=12.4,area_range=(11,13))
    boundary_b=box(.5,0,4,4)
    pb=replace(problem,rooms=(room_b,),boundary=boundary_b,free_space=boundary_b.difference(wall()))
    polys_b,events_b=refine_layout(pb,{'r':room_b.seed},{'r':box(.5,0,4,2)})
    samples.append(('CP4: B, finite wall endpoint',pb,polys_b['r'],validate_layout(pb,{'r':room_b.seed},polys_b)))
    room_n=RoomSpec('r',(5,2),45,(40,50),(1,4),'limited_recess')
    pn=problem_for(None,box(4,4,6,5),room_n)
    boundary_n=box(0,0,10,5)
    pn=replace(pn,boundary=boundary_n,free_space=boundary_n.difference(pn.fixed))
    polys_n,events_n=refine_layout(pn,{'r':room_n.seed},{'r':box(0,0,10,4)})
    samples.append(('CP4: controlled column recess',pn,polys_n['r'],validate_layout(pn,{'r':room_n.seed},polys_n)))
    fig,axes=plt.subplots(2,2,figsize=(11,9))
    evidence=[]
    for ax,(title,p,poly,report) in zip(axes.flat,samples):
        x,y=poly.exterior.xy
        ax.fill(x,y,color='#a7d3eb',edgecolor='#193b50',linewidth=1.5)
        x,y=p.fixed.exterior.xy
        ax.fill(x,y,color='#455a64')
        seed=p.rooms[0].seed
        ax.scatter(*seed,color='#111111',s=20,zorder=3)
        ax.set_aspect('equal'); ax.grid(alpha=.2)
        ax.set_title(f'{title}\narea={poly.area:.3f}, geometry_valid={report["geometry_valid"]}',fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        evidence.append(dict(title=title,polygon=list(poly.exterior.coords),validation=report))
    fig.tight_layout()
    fig.savefig(ROOT/'checks/evidence/seed_edge_examples.png',dpi=140)
    plt.close(fig)
    out=dict(examples=evidence,events_A=new['refinement'],events_B=events_b,events_notch=events_n)
    (ROOT/'checks/evidence/seed_edge_examples.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([dict(title=s[0],area=s[2].area,valid=s[3]['geometry_valid']) for s in samples],indent=2))


if __name__=='__main__':
    main()
