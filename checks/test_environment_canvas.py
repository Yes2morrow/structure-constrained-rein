import copy
import unittest
from gui.environment_canvas import controls,scene,parse_scene,color
from gui.structure_canvas import viewport
from gui.adaptive_reuse_page import CONSTRAINT_STYLES, _auto_repair_targets_config
from gui.config_store import load_config
from core.envs.structure_geometry import column,wall,zone


class EnvironmentCanvasTest(unittest.TestCase):
    def setUp(self):
        import yaml
        from pathlib import Path
        self.c=yaml.safe_load((Path(__file__).parent/'fixtures/retrofit.yaml').read_text(encoding='utf-8'))
        self.c['ExistingBuilding']['fixed_objects']=[column('c',5,5,.4,.6),wall('w','shear_wall',10,5,16,8,.2,.1),zone('traffic_1','traffic_core',18,4,22,8)]
        self.c['TargetSpaces'][0]['seed']=[14,14]
        self.w,self.h=760,480

    def move(self,key,dx,dy):
        objects=scene(self.c,self.w,self.h,CONSTRAINT_STYLES)
        o=next(o for o in objects if o.get('type')=='circle' and o.get('stroke')==color(key))
        scale,_,_=viewport(self.c['ExistingBuilding']['boundary'],self.w,self.h)
        o['left']+=dx*scale; o['top']-=dy*scale
        return parse_scene(objects,self.c,self.w,self.h)

    def test_complete_scene_and_lossless_initial_load(self):
        objects=scene(self.c,self.w,self.h,CONSTRAINT_STYLES)
        self.assertEqual(parse_scene(objects,self.c,self.w,self.h),self.c)
        self.assertEqual({k[0] for k in controls(self.c)}, {'boundary','original','agent','seed','column','wall','fixed_rect','door'})
        self.assertEqual(sum(o.get('fill')=='rgba(66,165,245,0.30)' for o in objects),len(self.c['TargetSpaces']))
        for o in objects:
            if o['type']=='circle':
                o['left']=round(o['left'],2);o['top']=round(o['top'],2)
        self.assertEqual(parse_scene(objects,self.c,self.w,self.h),self.c)
        with self.assertRaises(ValueError): parse_scene([],self.c,self.w,self.h)

    def test_hidden_layers_preserve_geometry_and_graph(self):
        display={'original':(False,0),'column':(False,0),'shear_wall':(False,0),'agent':(True,65)}
        objects=scene(self.c,self.w,self.h,CONSTRAINT_STYLES,display=display)
        self.assertEqual(parse_scene(objects,self.c,self.w,self.h),self.c)
        hidden=next(o for o in objects if o.get('stroke')==color(('column','c','center')))
        self.assertFalse(hidden['visible'])
        self.assertFalse(hidden['selectable'])
        hidden['left']+=100
        self.assertEqual(parse_scene(objects,self.c,self.w,self.h),self.c)
        agent=next(o for o in objects if o.get('fill')=='rgba(66,165,245,0.30)')
        self.assertAlmostEqual(agent['opacity'],.35)
        key=self.c['TargetSpaces'][0]['id']
        seed=next(o for o in objects if o.get('stroke')==color(('seed',key,'point')))
        seed['left']+=10
        updated=parse_scene(objects,self.c,self.w,self.h)
        self.assertEqual(updated['ExistingBuilding'],self.c['ExistingBuilding'])
        self.assertEqual(updated['FunctionalRelations'],self.c['FunctionalRelations'])
        self.assertNotEqual(updated['TargetSpaces'][0]['seed'],self.c['TargetSpaces'][0]['seed'])

    def test_original_corner_only_changes_original(self):
        key=self.c['ExistingBuilding']['original_spaces'][0]['id']
        updated=self.move(('original',key,'ne'),1,1)
        self.assertEqual(updated['TargetSpaces'],self.c['TargetSpaces'])
        a=updated['ExistingBuilding']['original_spaces'][0]['rect']; b=self.c['ExistingBuilding']['original_spaces'][0]['rect']
        self.assertEqual(a,[b[0],b[1],b[2]+1,b[3]+1])

    def test_agent_center_moves_seed_without_rebinding_graph(self):
        key=self.c['TargetSpaces'][0]['id']
        updated=self.move(('agent',key,'center'),1,2)
        self.assertAlmostEqual(updated['TargetSpaces'][0]['seed'][0],15)
        self.assertAlmostEqual(updated['TargetSpaces'][0]['seed'][1],16)
        self.assertEqual(updated['FunctionalRelations'],self.c['FunctionalRelations'])
        self.assertEqual(updated['ExistingBuilding'],self.c['ExistingBuilding'])
        seeded=self.move(('seed',key,'point'),.5,0)
        original_rect=self.c['TargetSpaces'][0]['initial_rect']
        self.assertEqual(seeded['TargetSpaces'][0]['initial_rect'],[original_rect[0]+.5,original_rect[1],original_rect[2]+.5,original_rect[3]])
        self.assertEqual(seeded['TargetSpaces'][0]['seed'],[14.5,14])

    def test_auto_repair_helper_handles_traffic_core(self):
        repaired, repairs, info = _auto_repair_targets_config(self.c, 42)
        self.assertTrue(repairs)
        self.assertTrue(info['initial_repairs'])
        self.assertEqual({item['space_id'] for item in repairs}, {item['space_id'] for item in info['initial_repairs']})
        self.assertEqual(next(item for item in repaired['ExistingBuilding']['fixed_objects'] if item['id']=='traffic_1')['type'],'traffic_core')

    def test_column_and_wall_parameters(self):
        updated=self.move(('column','c','ne'),.2,.1)
        c=updated['ExistingBuilding']['fixed_objects'][0]
        self.assertAlmostEqual(c['size'][0],.6);self.assertAlmostEqual(c['size'][1],.7)
        updated=self.move(('wall','w','end'),1,0)
        w=updated['ExistingBuilding']['fixed_objects'][1]
        self.assertEqual(w['end'],[17,8]);self.assertEqual(w['type'],'shear_wall')
        self.assertEqual(w['left_thickness'],.2);self.assertEqual(w['right_thickness'],.1)
        updated=self.move(('wall','w','left'),-.1,.2)
        self.assertGreater(updated['ExistingBuilding']['fixed_objects'][1]['left_thickness'],.2)
        self.assertEqual(updated['ExistingBuilding']['fixed_objects'][1]['right_thickness'],.1)

    def test_traffic_core_rectangle_moves_as_fixed_zone(self):
        updated=self.move(('fixed_rect','traffic_1','center'),1,.5)
        core=next(item for item in updated['ExistingBuilding']['fixed_objects'] if item['id']=='traffic_1')
        self.assertEqual(core['type'],'traffic_core')
        self.assertEqual(core['rect'],[19,4.5,23,8.5])

    def test_corner_crossing_rejected(self):
        with self.assertRaises(ValueError): self.move(('column','c','ne'),-10,0)


if __name__=='__main__': unittest.main()
